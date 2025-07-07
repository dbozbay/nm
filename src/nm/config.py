import os
from dataclasses import dataclass
from multiprocessing import cpu_count
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

HF_TOKEN: str | None = os.environ.get("HF_TOKEN")

ROOT_PATH = Path(__file__).resolve().parents[2]
COMPETITION = "jigsaw-toxic-comment-classification-challenge"
CLASSES = ("toxic", "threat")


@dataclass
class DataModuleConfig:
    data_dir: str = str(ROOT_PATH / "data" / "competitions" / COMPETITION)
    classes: tuple[str, ...] = CLASSES
    val_size: float = 0.2
    batch_size: int = 64
    num_workers: int = cpu_count()


@dataclass
class ModelConfig:
    model_name: str = "prajjwal1/bert-tiny"
    num_classes: int = len(CLASSES)
    max_token_len: int = 128
    learning_rate: float = 3e-5
    cache_dir: str | None = str(ROOT_PATH / "data")


@dataclass
class TrainerConfig:
    accelerator: str = "auto"
    devices: int | str = "auto"
    strategy: str = "auto"
    precision: str | None = "16-mixed"
    max_epochs: int = 5
    log_dir: str | None = str(ROOT_PATH / "logs")
    ckpt_dir: str | None = str(ROOT_PATH / "checkpoints")


@dataclass
class AppConfig:
    finetuned: str = (
        "google-bert-uncased-L-2-H-128-A-2_LR3e-5_BS64_MSL512_20250430-161306.ckpt"
    )
    accelerator: str = "auto"
    devices: int | str = "auto"
    timeout: int = 30
    track_requests: bool = True
    generate_client_file: bool = False


if __name__ == "__main__":
    print(ROOT_PATH)
    data_dir = DataModuleConfig.data_dir
    print(data_dir)
