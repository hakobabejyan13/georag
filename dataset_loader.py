from pathlib import Path
from typing import List

import h5py
import numpy as np


def _sort_band_keys(keys: List[str]) -> List[str]:
    def key_fn(name: str):
        prefix = name.split(" ", 1)[0]
        try:
            return int(prefix)
        except ValueError:
            return 999

    return sorted(keys, key=key_fn)


def load_hdf5(path: str) -> np.ndarray:
    """
    Load Sentinel-2 HDF5 and return array (13, H, W), float32.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {file_path}")

    with h5py.File(file_path, "r") as f:
        band_keys = _sort_band_keys(list(f.keys()))
        bands = [np.array(f[key]) for key in band_keys]

    if not bands:
        raise ValueError(f"No bands found in: {file_path}")

    while len(bands) < 13:
        bands.append(np.zeros_like(bands[-1]))

    return np.stack(bands[:13]).astype(np.float32)
