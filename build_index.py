import os
import numpy as np
import faiss
from config import EMBEDDINGS_PATH, INDEX_PATH, USE_COSINE


def load_embeddings(path: str) -> np.ndarray:
    """
    Load precomputed embeddings (.npy file)
    """
    embeddings = np.load(path)

    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype("float32")

    return embeddings


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    L2 normalize embeddings (recommended for cosine similarity search)
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return embeddings / norms


def build_faiss_index(embeddings: np.ndarray, use_cosine: bool = True):
    """
    Build FAISS index.
    If use_cosine=True → uses Inner Product with normalized vectors.
    """
    dim = embeddings.shape[1]

    if use_cosine:
        index = faiss.IndexFlatIP(dim)
    else:
        index = faiss.IndexFlatL2(dim)

    index.add(embeddings)
    return index



def save_index(index, path: str):
    # make sure the target directory exists before writing
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    faiss.write_index(index, path)


if __name__ == "__main__":
    print("Loading embeddings...")
    embeddings = load_embeddings(str(EMBEDDINGS_PATH))

    if USE_COSINE:
        print("Normalizing embeddings for cosine similarity...")
        embeddings = normalize_embeddings(embeddings)

    print("Building FAISS index...")
    index = build_faiss_index(embeddings, use_cosine=USE_COSINE)

    print("Saving index...")
    save_index(index, str(INDEX_PATH))

    print("Index built and saved successfully.")
