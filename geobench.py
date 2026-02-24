#!/usr/bin/env python

import argparse
from pathlib import Path
import zipfile

from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm


DATASET_REPO = "recursix/geo-bench-1.0"


def decompress_zip_with_progress(zip_file_path: Path, extract_to_folder: Path | None = None, delete_zip: bool = True):
    if extract_to_folder is None:
        extract_to_folder = zip_file_path.parent

    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        file_names = zip_ref.namelist()
        with tqdm(total=len(file_names), unit="file", desc=f"Extracting {zip_file_path.name}") as pbar:
            for file in file_names:
                zip_ref.extract(file, extract_to_folder)
                pbar.update(1)

    if delete_zip:
        zip_file_path.unlink(missing_ok=True)


def is_mbigearthnet_file(path: str) -> bool:
    lowered = path.lower()
    return lowered.startswith("m-bigearthnet/") or "m-bigearthnet" in lowered


def download_mbigearthnet(local_directory: Path, extract: bool = True, delete_zip: bool = True, list_only: bool = False):
    local_directory.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    dataset_files = api.list_repo_files(repo_id=DATASET_REPO, repo_type="dataset")
    target_files = [f for f in dataset_files if is_mbigearthnet_file(f)]

    if not target_files:
        raise RuntimeError("No m-bigearthnet files found in repo listing.")

    print(f"Found {len(target_files)} m-bigearthnet files.")
    if list_only:
        for file in target_files:
            print(file)
        return

    for file in target_files:
        local_file_path = local_directory / file
        local_file_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Downloading {file}...")
        hf_hub_download(
            repo_id=DATASET_REPO,
            filename=file,
            cache_dir=local_directory,
            local_dir=local_directory,
            repo_type="dataset",
        )

    if extract:
        zip_files = [file for file in target_files if file.endswith(".zip")]
        for i, zip_file in enumerate(zip_files):
            zip_path = local_directory / zip_file
            if zip_path.exists():
                print(f"Decompressing {i + 1}/{len(zip_files)}: {zip_file} ...")
                decompress_zip_with_progress(zip_path, delete_zip=delete_zip)

    print("m-bigearthnet download completed.")


def main():
    parser = argparse.ArgumentParser(description="Download only m-bigearthnet from Geo-Bench (HF dataset repo).")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("m-bigearthnet"),
        help="Output directory for downloaded files (default: ./m-bigearthnet)",
    )
    parser.add_argument("--no-extract", action="store_true", help="Do not extract downloaded zip files.")
    parser.add_argument("--keep-zip", action="store_true", help="Keep zip files after extraction.")
    parser.add_argument("--list-only", action="store_true", help="Only list matching files, do not download.")
    args = parser.parse_args()

    download_mbigearthnet(
        local_directory=args.out,
        extract=not args.no_extract,
        delete_zip=not args.keep_zip,
        list_only=args.list_only,
    )


if __name__ == "__main__":
    main()
