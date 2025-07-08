"""
Custom PyTorch Lightning DataModule and Dataset for the Jigsaw Toxic Comment Classification dataset.

This module provides:
- JigsawDataset: a torch.utils.data.Dataset for loading and accessing Jigsaw data.
- JigsawDataModule: a pl.LightningDataModule for managing splits and DataLoaders.

Notes:
    - https://lightning.ai/docs/pytorch/stable/data/datamodule.html
    - https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset
"""  # noqa: E501

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import lightning.pytorch as pl
import pandas as pd
import torch
from lightning_utilities.core.rank_zero import rank_zero_info
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split

from nm.config import DataModuleConfig


class JigsawDataset(Dataset):
    """
    PyTorch Dataset for the Jigsaw Toxic Comment Classification dataset.

    Args:
        data_dir: Path to the directory containing the dataset CSV files.
        split: Which split to load ('train' or 'test').
        classes: Tuple of class/label names.

    Notes:
        Expects the following files in data_dir:
            - train.csv
            - test.csv
            - test_labels.csv
        For 'test' split, merges test.csv and test_labels.csv on 'id' and filters out
        rows with toxic == -1.
    """

    def __init__(
        self,
        data_dir: str | Path,
        *,
        split: Literal["train", "test"],
        classes: tuple[str, ...],
    ) -> None:
        self.data_dir = Path(data_dir) if isinstance(data_dir, str) else data_dir
        self.split = split
        self.classes = classes
        self.data = self.load_data(self.split)

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[str, Tensor]:
        """
        Returns the comment text and label tensor for a given index.

        Args:
            index: Index of the sample to retrieve.

        Returns:
            Tuple of (comment_text, labels_tensor)
        """
        row = self.data.iloc[index]
        comment_text = str(row["comment_text"])
        labels = torch.tensor(
            row[list(self.classes)].astype(float).values,
            dtype=torch.float32,
        )
        return comment_text, labels

    def load_data(
        self,
        split: Literal["train", "test"],
        train_filename: str = "train.csv",
        test_text_filename: str = "test.csv",
        test_labels_filename: str = "test_labels.csv",
    ) -> pd.DataFrame:
        """
        Loads the data for the specified split.

        Args:
            split: 'train' or 'test'.
            train_filename: Filename for training data.
            test_text_filename: Filename for test comments.
            test_labels_filename: Filename for test labels.

        Returns:
            DataFrame containing the requested split.
        """
        train_path = self.data_dir / train_filename
        test_text_path = self.data_dir / test_text_filename
        test_labels_path = self.data_dir / test_labels_filename

        if split == "test":
            df1 = pd.read_csv(test_text_path)
            df2 = pd.read_csv(test_labels_path)
            df3 = df1.merge(df2, on="id", validate="1:1")
            return df3[df3["toxic"] != -1]

        return pd.read_csv(train_path)


class JigsawDataModule(pl.LightningDataModule):
    """
    PyTorch LightningDataModule for the Jigsaw Toxic Comment Classification dataset.

    Handles data preparation, setup, and DataLoader creation for training, validation,
    testing, and prediction.

    Args:
        data_dir: Path to the dataset directory.
        classes: Tuple of class/label names.
        val_size: Fraction of training data to use for validation.
        batch_size: Batch size for DataLoaders.
        num_workers: Number of workers for DataLoaders.

    Notes:
        https://lightning.ai/docs/pytorch/stable/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str | Path = DataModuleConfig.data_dir,
        classes: tuple[str, ...] = DataModuleConfig.classes,
        val_size: float = DataModuleConfig.val_size,
        batch_size: int = DataModuleConfig.batch_size,
        num_workers: int = DataModuleConfig.num_workers,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir) if isinstance(data_dir, str) else data_dir
        self.classes = classes
        self.val_size = val_size
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self) -> None:
        """
        Checks if the data directory exists and is not empty. Raises FileNotFoundError
        with instructions if missing.

        Notes:
            https://lightning.ai/docs/pytorch/stable/data/datamodule.html#prepare-data
        """
        data_dir_is_empty = (
            len([f for f in self.data_dir.iterdir() if f.is_file()]) == 0
        )
        if not self.data_dir.exists() or data_dir_is_empty:
            msg = (
                f"Data directory '{self.data_dir}' is missing or empty. "
                "Please download the dataset by running:\n"
                "   python -m nm.download --destination {destination} \n"
            )
            raise FileNotFoundError(msg)
        rank_zero_info(
            f"[{datetime.now(UTC)!s}] Data directory exists. Loading datasets."
        )

    def setup(self, stage: str) -> None:
        """
        Sets up datasets for different stages ('fit', 'validate', 'test', 'predict').

        Args:
            stage: One of 'fit', 'validate', 'test', or 'predict'.

        Notes:
            https://lightning.ai/docs/pytorch/stable/data/datamodule.html#setup
        """
        if stage in ("fit", "validate"):
            dataset = JigsawDataset(self.data_dir, split="train", classes=self.classes)
            self.train_dataset, self.val_dataset = random_split(
                dataset, [1 - self.val_size, self.val_size]
            )
            del dataset

        if stage == "test":
            self.test_dataset = JigsawDataset(
                self.data_dir, split="test", classes=self.classes
            )

        if stage == "predict":
            self.predict_dataset = JigsawDataset(
                self.data_dir, split="test", classes=self.classes
            )

    def train_dataloader(self) -> DataLoader:
        """
        Returns the DataLoader for the training set.

        Notes:
            https://lightning.ai/docs/pytorch/stable/data/datamodule.html#train-dataloader
        """
        return DataLoader(self.train_dataset)

    def val_dataloader(self) -> DataLoader:
        """
        Returns the DataLoader for the validation set.

        Notes:
            https://lightning.ai/docs/pytorch/stable/data/datamodule.html#val-dataloader
        """
        return DataLoader(self.val_dataset)

    def test_dataloader(self) -> DataLoader:
        """
        Returns the DataLoader for the test set.

        Notes:
            https://lightning.ai/docs/pytorch/stable/data/datamodule.html#test-dataloader
        """
        return DataLoader(self.test_dataset)

    def predict_dataloader(self) -> DataLoader:
        """
        Returns the DataLoader for the prediction set (uses test set).

        Notes:
            https://lightning.ai/docs/pytorch/stable/data/datamodule.html#predict-dataloader
        """
        return DataLoader(self.test_dataset)


if __name__ == "__main__":
    """
    Example usage for running and testing the JigsawDataModule from the command line.
    """
    dm = JigsawDataModule()
    dm.prepare_data()
    dm.setup(stage="fit")
    train_dl = dm.train_dataloader()
    val_dl = dm.val_dataloader()
    print(f"Num train batches: {len(train_dl)}")
    print(f"Num val batches: {len(val_dl)}")
    dm.setup(stage="test")
    test_dl = dm.test_dataloader()
    print(f"Num test batches: {len(test_dl)}")
