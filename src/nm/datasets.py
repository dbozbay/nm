import os
from pathlib import Path
from typing import Literal

import kagglehub
import pandas as pd
import torch
from dotenv import load_dotenv
from torch import Tensor
from torch.utils.data import Dataset

from nm.config import DataModuleConfig

load_dotenv()


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


def download_competition(
    competition: str,
    *,
    download_dir: str | None,
) -> str:
    if download_dir is not None:
        os.environ["KAGGLEHUB_CACHE"] = download_dir
    return kagglehub.competition_download(competition)


if __name__ == "__main__":
    classes = DataModuleConfig.classes
    data_dir = DataModuleConfig.data_dir
    train_ds = JigsawDataset(data_dir=data_dir, split="train", classes=classes)
    print(train_ds[0])
