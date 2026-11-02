"""Download all datasets required for the EuroSAT Kaggle project."""

from __future__ import annotations

import argparse
from pathlib import Path

from download_eurosat import download_eurosat, relative
from download_kaggle import download_kaggle


def print_header(title: str) -> None:
    """Print a readable section header."""
    line = "=" * 40
    print(f"\n{line}\n{title}\n{line}")


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Download EuroSAT and Kaggle datasets for the project."
    )
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--eurosat-only",
        action="store_true",
        help="Download only EuroSAT data.",
    )
    source_group.add_argument(
        "--kaggle-only",
        action="store_true",
        help="Download only Kaggle competition data.",
    )

    eurosat_group = parser.add_mutually_exclusive_group()
    eurosat_group.add_argument(
        "--ms-only",
        action="store_true",
        help="Download only EuroSAT multispectral data.",
    )
    eurosat_group.add_argument(
        "--rgb-only",
        action="store_true",
        help="Download only EuroSAT RGB data.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and re-extract datasets even if they already exist.",
    )
    return parser.parse_args()


def main() -> None:
    """Download the requested project datasets."""
    args = parse_args()
    summary: list[Path] = []

    if args.kaggle_only and (args.ms_only or args.rgb_only):
        raise SystemExit("--ms-only and --rgb-only cannot be used with --kaggle-only.")

    download_any_eurosat = not args.kaggle_only
    download_any_kaggle = not args.eurosat_only and not args.ms_only and not args.rgb_only

    if download_any_eurosat:
        ms = not args.rgb_only
        rgb = not args.ms_only

        if ms:
            print_header("Downloading EuroSAT multispectral data")
            summary.extend(download_eurosat(ms=True, rgb=False, force=args.force))

        if rgb:
            print_header("Downloading EuroSAT RGB data")
            summary.extend(download_eurosat(ms=False, rgb=True, force=args.force))

    if download_any_kaggle:
        print_header("Downloading Kaggle competition data")
        summary.append(download_kaggle(force=args.force))

    print_header("Dataset summary")
    for path in summary:
        print(f"- {relative(path)}")


if __name__ == "__main__":
    main()
