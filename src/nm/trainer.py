from lightning.pytorch.cli import LightningCLI

from nm.datamodules import JigsawDataModule
from nm.models import ToxicModel


def cli_main():
    cli = LightningCLI(ToxicModel, JigsawDataModule)


if __name__ == "__main__":
    cli_main()
