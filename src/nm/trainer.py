"""
Trainer module for running the ToxicModel with PyTorch Lightning CLI.

Provides a CLI entry point for training and evaluating the ToxicModel using the JigsawDataModule, with support for early stopping and model checkpointing callbacks.

Notes:
    - PyTorch Lightning CLI: https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html
    - EarlyStopping: https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.EarlyStopping.html
    - ModelCheckpoint: https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html
"""

import os

import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.cli import LightningCLI

from nm.config import TrainerConfig
from nm.datamodule import JigsawDataModule
from nm.module import ToxicModel

# Disable tokenizer parallelism to avoid deadlocks due to forking
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# see https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
torch.set_float32_matmul_precision("medium")


class MyLightningCLI(LightningCLI):
    """Custom LightningCLI with additional arguments for callbacks.

    Adds support for configuring EarlyStopping and ModelCheckpoint via CLI arguments.

    Notes:
        - EarlyStopping and ModelCheckpoint are added as configurable forced callbacks.
        - https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_expert.html#configure-forced-callbacks
    """

    def add_arguments_to_parser(self, parser):
        parser.add_lightning_class_args(EarlyStopping, "early_stopping")
        parser.set_defaults(
            {"early_stopping.monitor": "val_loss", "early_stopping.patience": 3}
        )

        parser.add_lightning_class_args(ModelCheckpoint, "checkpoint")
        parser.set_defaults(
            {
                "checkpoint.monitor": "val_loss",
                "checkpoint.filename": "jigsaw-{epoch:02d}-{val_loss:.2f}",
                "checkpoint.verbose": True,
            }
        )


def cli_main():
    """Entry point for running training or evaluation via LightningCLI.

    Instantiates MyLightningCLI with ToxicModel and JigsawDataModule, using TrainerConfig defaults.
    """
    MyLightningCLI(
        ToxicModel,
        JigsawDataModule,
        trainer_defaults={
            "max_epochs": TrainerConfig.max_epochs,
            "deterministic": TrainerConfig.deterministic,
        },
        seed_everything_default=TrainerConfig.seed,
    )


if __name__ == "__main__":
    cli_main()
