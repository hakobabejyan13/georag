import argparse
import json
from pathlib import Path

from config import DATASET_DIR, EMBEDDING_IDS_PATH, PROJECT_ROOT
from dataset_loader import load_hdf5
from preprocess import extract_rgb, normalize_percentile, to_pil


def load_sample_id(sample_index: int) -> str:
    ids_path = Path(EMBEDDING_IDS_PATH)
    if ids_path.exists():
        with ids_path.open("r", encoding="utf-8") as f:
            ids = json.load(f)
        if sample_index < 0 or sample_index >= len(ids):
            raise IndexError(f"sample index {sample_index} out of range [0, {len(ids)-1}]")
        return ids[sample_index]

    # Fallback: sorted HDF5 files if ids file is not available.
    hdf5_files = sorted(DATASET_DIR.glob("*.hdf5"))
    if not hdf5_files:
        raise FileNotFoundError(f"No .hdf5 files found in {DATASET_DIR}")
    if sample_index < 0 or sample_index >= len(hdf5_files):
        raise IndexError(f"sample index {sample_index} out of range [0, {len(hdf5_files)-1}]")
    return hdf5_files[sample_index].stem


def main():
    parser = argparse.ArgumentParser(description="Create a demo RGB image from m-BigEarthNet HDF5.")
    parser.add_argument("--index", type=int, default=0, help="Sample index used in demo query (default: 0).")
    parser.add_argument(
        "--out",
        type=str,
        default=str(PROJECT_ROOT / "demo_query_image.png"),
        help="Output image path.",
    )
    parser.add_argument("--show", action="store_true", help="Open the image after saving.")
    args = parser.parse_args()

    sample_id = load_sample_id(args.index)
    hdf5_path = Path(DATASET_DIR) / f"{sample_id}.hdf5"
    if not hdf5_path.exists():
        raise FileNotFoundError(f"Sample file not found: {hdf5_path}")

    arr13 = load_hdf5(str(hdf5_path))
    rgb = extract_rgb(arr13)
    rgb_norm = normalize_percentile(rgb)
    img = to_pil(rgb_norm)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)

    print(f"Saved demo image: {out_path}")
    print(f"Sample ID: {sample_id}")

    if args.show:
        img.show()


if __name__ == "__main__":
    main()
