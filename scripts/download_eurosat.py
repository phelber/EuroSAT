"""Download and extract the official EuroSAT datasets from Zenodo."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EUROSAT_DIR = PROJECT_ROOT / "data" / "eurosat"
DOWNLOAD_DIR = EUROSAT_DIR / ".downloads"


def relative(path: Path) -> Path:
    """Path relative to the project root, for readable output."""
    try:
        return Path(path).resolve().relative_to(PROJECT_ROOT)
    except ValueError:
        return Path(path)


@dataclass(frozen=True)
class EuroSATArchive:
    name: str
    label: str
    url: str
    md5: str
    extract_dir: Path


EUROSAT_MS = EuroSATArchive(
    name="EuroSAT_MS.zip",
    label="EuroSAT multispectral",
    url="https://zenodo.org/records/7711810/files/EuroSAT_MS.zip",
    md5="091174add3c8e680a49244acf185b9f0",
    extract_dir=EUROSAT_DIR / "ms",
)

EUROSAT_RGB = EuroSATArchive(
    name="EuroSAT_RGB.zip",
    label="EuroSAT RGB",
    url="https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip",
    md5="f46e308c4d50d4bf32fedad2d3d62f3b",
    extract_dir=EUROSAT_DIR / "rgb",
)


def directory_has_data(path: Path) -> bool:
    """Return True when the extraction directory appears populated."""
    return path.exists() and any(path.iterdir())


def calculate_md5(path: Path) -> str:
    """Calculate an MD5 checksum without loading the whole file into memory."""
    digest = hashlib.md5()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, destination: Path) -> None:
    """Download a large file with streaming and a progress bar."""
    import requests
    from tqdm import tqdm

    destination.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True, timeout=30) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))

        with destination.open("wb") as file:
            with tqdm(
                total=total or None,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=destination.name,
            ) as progress:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        file.write(chunk)
                        progress.update(len(chunk))


def verify_md5(path: Path, expected_md5: str) -> None:
    """Raise a useful error if the downloaded file checksum is unexpected."""
    actual_md5 = calculate_md5(path)
    if actual_md5 != expected_md5:
        raise RuntimeError(
            f"Checksum validation failed for {path.name}. "
            f"Expected {expected_md5}, got {actual_md5}. "
            "Delete the file and try downloading again."
        )


def extract_zip(zip_path: Path, destination: Path, force: bool) -> None:
    """Extract a ZIP archive into the destination directory."""
    if force and destination.exists():
        shutil.rmtree(destination)

    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(destination)


def download_archive(archive: EuroSATArchive, force: bool = False) -> Path:
    """Download, validate, and extract one EuroSAT archive."""
    if directory_has_data(archive.extract_dir) and not force:
        print(f"{archive.label} dataset already exists, skipping.")
        return archive.extract_dir

    zip_path = DOWNLOAD_DIR / archive.name

    if force and zip_path.exists():
        zip_path.unlink()

    if not zip_path.exists():
        print(f"Downloading {archive.label} dataset from Zenodo...")
        download_file(archive.url, zip_path)
    else:
        print(f"Using existing downloaded archive: {relative(zip_path)}")

    print(f"Verifying checksum for {archive.name}...")
    verify_md5(zip_path, archive.md5)

    print(f"Extracting {archive.name} to {relative(archive.extract_dir)}...")
    extract_zip(zip_path, archive.extract_dir, force=force)

    zip_path.unlink(missing_ok=True)
    print(f"Finished {archive.label}: {relative(archive.extract_dir)}")
    return archive.extract_dir


def download_eurosat(
    *,
    ms: bool = True,
    rgb: bool = True,
    force: bool = False,
) -> list[Path]:
    """Download the requested EuroSAT datasets and return their directories."""
    downloaded_dirs: list[Path] = []

    if ms:
        downloaded_dirs.append(download_archive(EUROSAT_MS, force=force))

    if rgb:
        downloaded_dirs.append(download_archive(EUROSAT_RGB, force=force))

    return downloaded_dirs


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Download and extract the official EuroSAT datasets."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--ms-only",
        action="store_true",
        help="Download only the multispectral EuroSAT archive.",
    )
    group.add_argument(
        "--rgb-only",
        action="store_true",
        help="Download only the RGB EuroSAT archive.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and re-extract even if dataset directories exist.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the EuroSAT downloader from the command line."""
    args = parse_args()
    ms = not args.rgb_only
    rgb = not args.ms_only

    final_dirs = download_eurosat(ms=ms, rgb=rgb, force=args.force)

    print("\nEuroSAT download summary")
    for path in final_dirs:
        print(f"- {relative(path)}")


if __name__ == "__main__":
    main()
