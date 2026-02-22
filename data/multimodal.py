"""Dataset definitions for multimodal CT/WSI survival modeling."""

import os
import random
from itertools import product

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class MultimodalCTWSIDatasetSurv(Dataset):
    """Paired CT/WSI dataset with configurable missing-modality handling."""

    def __init__(
        self,
        fold: int,
        split: str,  # either "train" or "test"
        ct_path: str,
        wsi_path: str,
        labels_splits_path: str,
        missing_modality_prob: float = 0.0,  # Additional random masking probability
        missing_modality: str = "both",
        require_both_modalities: bool = False,  # Whether to only include patients
        # with both modalities
        pairing_mode: str = None,  # 'all_combinations, 'one_to_one', 'fixed_count'
        pairs_per_patient: int = None,  # For fixed_count_mode
        allow_repeats: bool = False,  # For fixed_count mode
        n_patches: int = 112,
        rad_dim: int = 1024,
        histo_dim: int = 768
    ):
        """
        Initialize the dataset and build the sample list.

        :param fold: Fold index used to select the split column (e.g., ``fold_0``).
        :type fold: int
        :param split: Split name, either ``"train"`` or ``"test"``.
        :type split: str
        :param ct_path: Root path containing per-patient CT feature folders.
        :type ct_path: str
        :param wsi_path: Root path containing WSI feature files.
        :type wsi_path: str
        :param labels_splits_path: TSV path with labels and fold splits.
        :type labels_splits_path: str
        :param missing_modality_prob: Probability of additional random masking.
        :type missing_modality_prob: float
        :param missing_modality: Which modality to mask when using random masking.
        :type missing_modality: str
        :param require_both_modalities: If True, drop patients lacking any modality.
        :type require_both_modalities: bool
        :param pairing_mode: Pairing strategy (``all_combinations``, ``one_to_one``, ``fixed_count``).
        :type pairing_mode: str
        :param pairs_per_patient: Number of pairs for ``fixed_count`` mode.
        :type pairs_per_patient: int
        :param allow_repeats: Whether pairing can repeat elements in ``fixed_count`` mode.
        :type allow_repeats: bool
        :param n_patches: Number of CT slices/patches to expect.
        :type n_patches: int
        :param rad_dim: Dimensionality of CT features.
        :type rad_dim: int
        :param histo_dim: Dimensionality of WSI features.
        :type histo_dim: int
        :return: None.
        :rtype: None
        """
        super().__init__()
        # assert split in ["train", "val", "overfit", "all"]
        assert split in ["train", "test"]
        assert pairing_mode in ["all_combinations", "one_to_one", "fixed_count"]
        assert 0 <= missing_modality_prob <= 1
        assert missing_modality in ["ct", "wsi", "both"]
        self.missing_modality = missing_modality
        self.fold = fold
        self.split = split
        self.ct_path = ct_path
        self.wsi_path = wsi_path
        self.missing_modality_prob = missing_modality_prob
        self.require_both_modalities = require_both_modalities
        self.labels_splits_path = labels_splits_path
        self.rad_dim = rad_dim
        self.histo_dim = histo_dim
        self.n_patches = n_patches
        self.pairing_mode = (
            pairing_mode  # 'all_combinations', 'one_to_one', or 'fixed_count'
        )
        self.pairs_per_patient = pairs_per_patient  # For fixed_count mode
        self.allow_repeats = allow_repeats  # For fixed_count mode
        labels_splits = pd.read_csv(labels_splits_path, sep="\t")
        labels_splits = labels_splits[labels_splits[f"fold_{self.fold}"] == self.split]
        self.labels_splits = labels_splits[
            ["case_id", "OS_days", "OS_event"]
        ].drop_duplicates("case_id")

        # Initialize data structures
        # Will store CT and WSI paths per patient
        self.patient_data = {}
        # Will store all valid combinations
        self.samples = []
        self.modality_stats = {"ct_only": 0, "wsi_only": 0, "both": 0}

        # Load split file
        self._load_split()

    def _get_max_pairs_for_patient(self, ct_scans, wsi_folders, allow_repeats):
        """
        Compute the maximum number of CT/WSI pairs for a patient.

        :param ct_scans: List of CT scan filenames.
        :type ct_scans: list
        :param wsi_folders: List of WSI folder or file names.
        :type wsi_folders: list
        :param allow_repeats: If True, allow all combinations; otherwise one-to-one.
        :type allow_repeats: bool
        :return: Maximum number of possible pairs for the patient.
        :rtype: int
        """
        if not ct_scans or not wsi_folders:
            return 0

        if allow_repeats:
            return len(ct_scans) * len(wsi_folders)
        else:  # one-to-one
            return min(len(ct_scans), len(wsi_folders))

    def _get_fixed_pairs(self, ct_scans, wsi_folders, n_pairs, allow_repeats=True):
        """
        Generate a fixed number of CT/WSI pairs.

        :param ct_scans: List of CT scan filenames.
        :type ct_scans: list
        :param wsi_folders: List of WSI folder or file names.
        :type wsi_folders: list
        :param n_pairs: Number of pairs to generate.
        :type n_pairs: int
        :param allow_repeats: If True, allow repeated elements to reach ``n_pairs``.
        :type allow_repeats: bool
        :return: List of (ct_scan, wsi_folder) pairs.
        :rtype: list
        """
        max_unique_pairs = min(len(ct_scans), len(wsi_folders))

        if not allow_repeats:
            n_pairs = min(n_pairs, max_unique_pairs)

        # Generate initial unique pairs
        shuffled_ct = ct_scans.copy()
        shuffled_wsi = wsi_folders.copy()
        random.shuffle(shuffled_ct)
        random.shuffle(shuffled_wsi)

        pairs = list(
            zip(shuffled_ct[:max_unique_pairs], shuffled_wsi[:max_unique_pairs])
        )

        if n_pairs <= len(pairs):
            # Downsample if needed
            random.shuffle(pairs)
            return pairs[:n_pairs]

        if not allow_repeats:
            return pairs

        # Need to generate additional pairs with repeats
        while len(pairs) < n_pairs:
            ct_scan = random.choice(ct_scans)
            wsi_folder = random.choice(wsi_folders)
            pairs.append((ct_scan, wsi_folder))

        return pairs

    def _load_split(self):
        """
        Load and organize CT and WSI data for the current split.

        Pairing modes:
        - ``all_combinations``: all CT-WSI pairs
        - ``one_to_one``: random 1:1 pairing
        - ``fixed_count``: fixed number of pairs per patient

        :return: None.
        :rtype: None
        """

        # First pass: count maximum possible pairs per patient
        max_pairs_possible = float("inf")
        if self.pairing_mode == "fixed_count":
            for patient_id in self.labels_splits["case_id"].values:

                ct_path = os.path.join(self.ct_path, patient_id)
                ct_features = []
                if os.path.exists(ct_path):
                    ct_features = [f for f in os.listdir(ct_path)]

                wsi_path = os.path.join(self.wsi_path)
                wsi_features = [
                    f
                    for f in os.listdir(wsi_path)
                    if patient_id in f and os.path.isdir(os.path.join(wsi_path, f))
                ]

                patient_max_pairs = self._get_max_pairs_for_patient(
                    ct_features, wsi_features, self.allow_repeats
                )

                if patient_max_pairs > 0:  # Only update if patient has both modalities

                    max_pairs_possible = min(max_pairs_possible, patient_max_pairs)

        # Use provided pairs_per_patient or calculated maximum
        n_pairs = (
            self.pairs_per_patient
            if self.pairs_per_patient is not None
            else max_pairs_possible
        )

        # Main loading loop
        for row in self.labels_splits["case_id"].values:
            patient_id = row.strip()
            # Find all CT scans for this patient
            ct_path = os.path.join(self.ct_path, patient_id)
            ct_features = []
            if os.path.exists(ct_path):
                ct_features = [f for f in os.listdir(ct_path)]

            # Find all WSI .h5 files for this patient
            wsi_path = self.wsi_path
            wsi_features = [f for f in os.listdir(wsi_path) if patient_id in f]

            # Skip patient if we require both modalities and they don't have them
            if self.require_both_modalities and (not ct_features or not wsi_features):
                continue

            # Store available data for this patient
            self.patient_data[patient_id] = {
                "ct_features": ct_features,
                "wsi_features": wsi_features,
            }

            # Update modality statistics
            if ct_features and wsi_features:
                self.modality_stats["both"] += 1
            elif ct_features:
                self.modality_stats["ct_only"] += 1
            else:
                self.modality_stats["wsi_only"] += 1

            cnt = 0
            # Generate samples based on available data and pairing mode
            if ct_features and wsi_features:
                if self.pairing_mode == "fixed_count":
                    # Generate fixed number of pairs
                    pairs = self._get_fixed_pairs(
                        ct_features, wsi_features, n_pairs, self.allow_repeats
                    )

                    for ct_feature, wsi_feature in pairs:
                        cnt += 1
                        self.samples.append(
                            {
                                "patient_id": patient_id,
                                "ct_path": os.path.join(
                                    self.ct_path, patient_id, ct_feature
                                ),
                                "wsi_folder": os.path.join(self.wsi_path, wsi_feature),
                                "base_modality_mask": [1, 1],
                            }
                        )

                elif self.pairing_mode == "one_to_one":
                    # Original one_to_one logic
                    num_pairs = min(len(ct_features), len(wsi_features))
                    shuffled_ct = ct_features.copy()
                    shuffled_wsi = wsi_features.copy()
                    random.shuffle(shuffled_ct)
                    random.shuffle(shuffled_wsi)

                    for ct_feature, wsi_feature in zip(
                        shuffled_ct[:num_pairs], shuffled_wsi[:num_pairs]
                    ):
                        cnt += 1
                        self.samples.append(
                            {
                                "patient_id": patient_id,
                                "ct_path": os.path.join(
                                    self.ct_path, patient_id, ct_feature
                                ),
                                "wsi_feature": os.path.join(self.wsi_path, wsi_feature),
                                "base_modality_mask": [1, 1],
                            }
                        )

                else:  # 'all_combinations' mode
                    for ct_feature, wsi_feature in product(ct_features, wsi_features):
                        cnt += 1
                        self.samples.append(
                            {
                                "patient_id": patient_id,
                                "ct_path": os.path.join(
                                    self.ct_path, patient_id, ct_feature
                                ),
                                "wsi_feature": os.path.join(self.wsi_path, wsi_feature),
                                "base_modality_mask": [1, 1],
                            }
                        )

            elif ct_features:
                for ct_feature in ct_features:
                    cnt += 1
                    self.samples.append(
                        {
                            "patient_id": patient_id,
                            "ct_path": os.path.join(
                                self.ct_path, patient_id, ct_feature
                            ),
                            "wsi_feature": None,
                            "base_modality_mask": [1, 0],
                        }
                    )

            elif wsi_features:
                for wsi_feature in wsi_features:
                    cnt += 1
                    self.samples.append(
                        {
                            "patient_id": patient_id,
                            "ct_path": None,
                            "wsi_feature": os.path.join(self.wsi_path, wsi_feature),
                            "base_modality_mask": [0, 1],
                        }
                    )

    def _load_ct_feature(self, ct_path):
        """
        Load a CT feature tensor from disk.

        :param ct_path: Path to a CT feature file (.npy or .pt).
        :type ct_path: str
        :return: CT feature array.
        :rtype: numpy.ndarray
        """
        if ct_path.endswith(".pt"):
            volume = np.array(torch.load(ct_path, weights_only=True))
        elif ct_path.endswith(".npy"):
            volume = np.load(ct_path)
        return volume

    def _load_wsi_feature(self, wsi_path):
        """
        Load a WSI feature tensor from an HDF5 file.

        :param wsi_path: Path to a WSI feature file (.h5).
        :type wsi_path: str
        :return: WSI feature array.
        :rtype: numpy.ndarray
        """
        feature = None
        with h5py.File(wsi_path, "r") as f:
            feature = np.array(f["features"][:])

        return feature

    def _get_empty_ct_feature(self):
        """
        Return a zero-filled CT feature with the expected shape.

        :return: Empty CT feature array.
        :rtype: numpy.ndarray
        """
        return np.zeros((self.n_patches, self.rad_dim))

    def _get_empty_wsi_feature(self):
        """
        Return a zero-filled WSI feature with the expected shape.

        :return: Empty WSI feature array.
        :rtype: numpy.ndarray
        """
        return np.zeros((self.histo_dim,))

    def __getitem__(self, index):
        """
        Fetch a sample by index.

        :param index: Sample index.
        :type index: int
        :return: Sample dict with features, labels, and modality masks.
        :rtype: dict
        """

        sample = self.samples[index]
        patient_id = sample["patient_id"]
        base_mask = sample["base_modality_mask"]

        # Apply additional random masking only to available modalities
        final_mask = base_mask.copy()
        if self.missing_modality_prob > 0:
            if self.missing_modality == "both":
                for i in range(2):
                    if (
                        base_mask[i] == 1
                        and random.random() < self.missing_modality_prob
                    ):
                        final_mask[i] = 0
            elif self.missing_modality == "ct":
                if base_mask[0] == 1 and random.random() < self.missing_modality_prob:
                    final_mask[0] = 0
            elif self.missing_modality == "wsi":
                if base_mask[1] == 1 and random.random() < self.missing_modality_prob:
                    final_mask[1] = 0

            # Ensure at least one modality remains if it was originally available
            if sum(final_mask) == 0 and sum(base_mask) > 0:
                # Randomly choose one of the originally available modalities
                available_indices = [i for i in range(2) if base_mask[i] == 1]
                chosen_idx = random.choice(available_indices)
                final_mask[chosen_idx] = 1

        # Load features based on final mask
        ct_feature = (
            self._load_ct_feature(sample["ct_path"])
            if final_mask[0] and sample["ct_path"]
            else self._get_empty_ct_feature()
        )

        wsi_feature = (
            self._load_wsi_feature(sample["wsi_feature"])
            if final_mask[1] and sample["wsi_feature"]
            else self._get_empty_wsi_feature()
        )

        return {
            "patient_id": patient_id,
            "ct_feature": torch.from_numpy(ct_feature).float(),
            "wsi_feature": torch.from_numpy(wsi_feature).float(),
            "survtime": torch.tensor(
                self.labels_splits[self.labels_splits["case_id"] == patient_id][
                    "OS_days"
                ].iloc[0],
                dtype=torch.long,
            ),
            "censor": ~torch.tensor(
                self.labels_splits[self.labels_splits["case_id"] == patient_id][
                    "OS_event"
                ].iloc[0],
                dtype=torch.bool,
            ),
            "modality_mask": torch.tensor(final_mask, dtype=torch.float32),
            "base_modality_mask": torch.tensor(base_mask, dtype=torch.float32),
        }

    def __len__(self):
        """
        Return the number of samples.

        :return: Dataset length.
        :rtype: int
        """
        return len(self.samples)

    def stats(self):
        """
        Return dataset statistics.

        :return: Summary statistics for the dataset.
        :rtype: dict
        """
        return {
            "total_samples": len(self.samples),
            "total_patients": len(self.patient_data),
            "modality_availability": self.modality_stats,
            "missing_modality_prob": self.missing_modality_prob,
        }

    @staticmethod
    def move_batch_to_device(batch, device):
        """
        Move all tensors in a batch to the target device (in-place).

        :param batch: Batch dict with tensor values.
        :type batch: dict
        :param device: Target device.
        :type device: torch.device
        :return: None.
        :rtype: None
        """
        batch["ct_feature"] = batch["ct_feature"].to(device)
        batch["wsi_feature"] = batch["wsi_feature"].to(device)
        batch["survtime"] = batch["survtime"].to(device)
        batch["censor"] = batch["censor"].to(device)
        batch["modality_mask"] = batch["modality_mask"].to(device)
