import numpy as np

def normalize_vector(vec: np.ndarray) -> np.ndarray:
    """
    L2 normalize a vector for cosine similarity search
    """
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    L2 normalize all embeddings
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return embeddings / norms


def load_numpy(path: str, dtype=np.float32) -> np.ndarray:
    """
    Load numpy array and convert to float32
    """
    arr = np.load(path)
    if arr.dtype != dtype:
        arr = arr.astype(dtype)
    return arr
