from lightning.pytorch.cli import LightningCLI

from nm.datamodule import JigsawDataModule
from nm.module import ToxicModel


def cli_main():
    cli = LightningCLI(ToxicModel, JigsawDataModule)


if __name__ == "__main__":
    cli_main()
