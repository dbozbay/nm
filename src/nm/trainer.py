import torch

# see https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
torch.set_float32_matmul_precision("medium")

import os

from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.cli import LightningCLI

from nm.config import TrainerConfig
from nm.datamodule import JigsawDataModule
from nm.module import ToxicModel

# Disable tokenizer parallelism to avoid deadlocks due to forking
os.environ["TOKENIZERS_PARALLELISM"] = "false"

seed_everything(TrainerConfig.seed, workers=True)

# checkpoint_callback = ModelCheckpoint(
#     monitor="val_loss",
#     dirpath=TrainerConfig.ckpt_dir,
#     filename="jigsaw-{epoch:02d}-{val_loss:.2f}",
# )

# early_stopping = EarlyStopping(monitor="val_loss", patience=3, verbose=True)


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_lightning_class_args(EarlyStopping, "early_stopping")
        parser.add_lightning_class_args(ModelCheckpoint, "model_checkpoint")
        parser.set_defaults(
            {
                "early_stopping.monitor": "val_loss",
                "early_stopping.patience": 5,
                "early_stopping.verbose": True,
                "model_checkpoint.monitor": "val_loss",
                "model_checkpoint.dirpath": "my_checkpoints",
                "model_checkpoint.filename": "jigsaw-{epoch:02d}-{val_loss:.2f}",
            }
        )

        parser.set_defaults(
            {"trainer.callbacks": ["early_stopping", "model_checkpoint"]}
        )


def cli_main():
    cli = LightningCLI(
        ToxicModel,
        JigsawDataModule,
        trainer_defaults={
            "precision": TrainerConfig.precision,
            "max_epochs": TrainerConfig.max_epochs,
            "deterministic": TrainerConfig.deterministic,
            "callbacks": [EarlyStopping(monitor="val_loss")],
        },
    )


if __name__ == "__main__":
    cli_main()
