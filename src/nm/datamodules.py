from datetime import datetime
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
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[str, Tensor]:
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
    def __init__(
        self,
        data_dir: str = DataModuleConfig.data_dir,
        classes: tuple[str, ...] = DataModuleConfig.classes,
        val_size: float = DataModuleConfig.val_size,
        batch_size: int = DataModuleConfig.batch_size,
        num_workers: int = DataModuleConfig.num_workers,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.classes = classes
        self.val_size = val_size
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self) -> None:
        if not Path(self.data_dir).exists():
            msg = (
                f"{self.data_dir} does not exist. "
                "Make sure you download the Kaggle data here via "
                "`download_competition(save_dir=...)`"
            )
            raise FileNotFoundError(msg)

        if Path(self.data_dir).iterdir() == 0:
            msg = (
                f"{self.data_dir} is empty. "
                "Make sure you download the Kaggle data here via "
                "`download_competition(save_dir=...)`"
            )
            raise FileNotFoundError(msg)

        rank_zero_info(
            f"[{datetime.now()!s}] Data dir exsists. Loading datasets from here."
        )

    def setup(self, stage: str) -> None:
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
        return DataLoader(self.train_dataset)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset)


if __name__ == "__main__":
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
