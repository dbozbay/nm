from datetime import datetime
from pathlib import Path

import lightning.pytorch as pl
import torch
import torch.utils
import torch.utils.data
from lightning_utilities.core.rank_zero import rank_zero_info
from torch.utils.data import DataLoader

from nm.config import DataModuleConfig
from nm.datasets import JigsawDataset


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
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
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
