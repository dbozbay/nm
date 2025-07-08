import os
from pathlib import Path

import kagglehub
from jsonargparse import auto_cli

from nm.config import COMPETITION
from nm.utils import unzip_directory


def download_competition(
    destination: str | Path,
    competition: str = COMPETITION,
    *,
    unzip: bool = True,
) -> Path:
    """
    Download a Kaggle competition dataset using kagglehub, with optional extraction.

    Args:
        destination (str or Path):
            The directory where the competition data will be downloaded. This sets the
            KAGGLEHUB_CACHE environment variable, and the data will be placed under
            destination/competitions/<competition>/.
        competition (str, optional):
            The Kaggle competition identifier (slug). Defaults to the value of
            COMPETITION from `nm.config`.
        unzip (bool, optional):
            If True (default), automatically unzip the downloaded files and delete any
            existing extracted data in the target directory.

    Returns:
        Path: The path to the downloaded (and optionally unzipped) competition data
        directory as a pathlib.Path object.

    Example (CLI usage):
        python -m nm.download_data --destination ./data \
            --competition jigsaw-toxic-comment-classification-challenge --unzip True
    """
    os.environ["KAGGLEHUB_CACHE"] = str(destination)
    download_dir = kagglehub.competition_download(competition)
    if unzip:
        unzip_directory(download_dir, delete_existing=True)
    print(f"Data downloaded: {download_dir}")
    return Path(download_dir)


if __name__ == "__main__":
    auto_cli(download_competition, as_positional=False)
