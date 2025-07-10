import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

HF_TOKEN: str | None = os.environ.get("HF_TOKEN")

COMPETITION = "jigsaw-toxic-comment-classification-challenge"
CLASSES = ("toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate")


@dataclass
class DataModuleConfig:
    data_dir: str | Path = "data/competitions/" + COMPETITION
    classes: tuple[str, ...] = CLASSES
    val_size: float = 0.2
    batch_size: int = 64
    num_workers: int | str | None = "auto"  # 'auto' means use all available CPU cores
    persistent_workers: bool = True


@dataclass
class ModelConfig:
    model_name: str = "prajjwal1/bert-tiny"
    num_classes: int = len(CLASSES)
    max_token_len: int = 16
    learning_rate: float = 3e-5
    cache_dir: str | None = "data"


@dataclass
class TrainerConfig:
    # precision: str | None = "16-mixed"
    max_epochs: int = 5
    seed: int = 18
    deterministic: bool = True  # For reproducability
    log_dir: str | None = None
    ckpt_dir: str | None = "checkpoints"


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
