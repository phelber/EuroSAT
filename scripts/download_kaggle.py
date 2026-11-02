"""Download Kaggle competition files for the project."""

from __future__ import annotations

import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KAGGLE_DATA_DIR = PROJECT_ROOT / "data" / "kaggle"
COMPETITION_HANDLE = "7-854-1-00-machine-learning-2026-coding-challenge"
PLACEHOLDER_FILE = ".gitkeep"


def relative(path: Path) -> Path:
    """Path relative to the project root, for readable output."""
    try:
        return Path(path).resolve().relative_to(PROJECT_ROOT)
    except ValueError:
        return Path(path)


def directory_contents(path: Path) -> list[Path]:
    """Return sorted directory contents, or an empty list if it does not exist."""
    if not path.exists():
        return []
    return sorted(path.iterdir())


def actual_data_files(path: Path) -> list[Path]:
    """Return directory contents excluding the Git placeholder file."""
    return [item for item in directory_contents(path) if item.name != PLACEHOLDER_FILE]


def remove_placeholder_if_alone(path: Path) -> None:
    """Remove .gitkeep when it is the only file blocking KaggleHub output."""
    contents = directory_contents(path)
    if len(contents) == 1 and contents[0].name == PLACEHOLDER_FILE:
        contents[0].unlink()


def list_downloaded_files(path: Path, limit: int = 20) -> list[Path]:
    """Return a small list of downloaded files for display."""
    return actual_data_files(path)[:limit]


def build_download_error(exc: Exception) -> RuntimeError:
    """Create a specific, actionable error for common Kaggle download failures."""
    message = str(exc)
    lower_message = message.lower()
    exception_name = type(exc).__name__.lower()

    if isinstance(exc, FileExistsError) or "output_dir is not empty" in lower_message:
        return RuntimeError(
            "Kaggle download failed because the output directory contains "
            f"existing files: {relative(KAGGLE_DATA_DIR)}\n"
            "Remove those files or rerun with --force.\n"
            f"Original error: {message}"
        )

    auth_markers = (
        "401",
        "unauthorized",
        "unauthenticated",
        "authentication",
        "credentials",
        "login",
        "token",
    )
    if "auth" in exception_name or any(marker in lower_message for marker in auth_markers):
        return RuntimeError(
            "Kaggle download failed because Kaggle authentication is missing "
            "or invalid. Run:\n"
            'python -c "import kagglehub; kagglehub.login()"\n'
            "Then try the download again.\n"
            f"Original error: {message}"
        )

    access_markers = (
        "403",
        "forbidden",
        "permission",
        "access denied",
        "competition",
        "rules",
    )
    if any(marker in lower_message for marker in access_markers):
        return RuntimeError(
            "Kaggle download failed because this account may not have access "
            "to the competition files. Join the competition and accept its "
            "rules on Kaggle, then try again.\n"
            f"Original error: {message}"
        )

    return RuntimeError(
        "Kaggle download failed with an unexpected error.\n"
        f"{type(exc).__name__}: {message}"
    )


def download_kaggle(force: bool = False) -> Path:
    """Download Kaggle competition files into data/kaggle/."""
    KAGGLE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if actual_data_files(KAGGLE_DATA_DIR) and not force:
        print("Kaggle competition data already exists, skipping.")
        return KAGGLE_DATA_DIR

    remove_placeholder_if_alone(KAGGLE_DATA_DIR)

    try:
        import kagglehub

        kagglehub.competition_download(
            COMPETITION_HANDLE,
            output_dir=str(KAGGLE_DATA_DIR),
            force_download=force,
        )
    except Exception as exc:
        raise build_download_error(exc) from exc

    print(f"Kaggle competition files are stored in: {relative(KAGGLE_DATA_DIR)}")
    files = list_downloaded_files(KAGGLE_DATA_DIR)
    if files:
        print("Downloaded files:")
        for file_path in files:
            print(f"- {file_path.name}")
    else:
        print("No files were listed after download. Check KaggleHub output above.")

    return KAGGLE_DATA_DIR


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Download Kaggle competition data into data/kaggle/."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if files already exist in data/kaggle/.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the Kaggle downloader from the command line."""
    args = parse_args()
    download_kaggle(force=args.force)


if __name__ == "__main__":
    main()
