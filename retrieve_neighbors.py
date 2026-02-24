import json
from pathlib import Path
from typing import List, Sequence, Union

import faiss
import numpy as np

from config import EMBEDDING_IDS_PATH, EMBEDDINGS_PATH, INDEX_PATH, LABELS_PATH, TOP_K, USE_LABEL_NAMES

LabelType = Union[str, List[str]]


def _load_embedding_ids(path: Union[str, Path]) -> List[str]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return [str(x) for x in data]


def _load_label_names(path: Union[str, Path]) -> List[str]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [str(x) for x in data]
    if isinstance(data, dict):
        out = []
        i = 0
        while str(i) in data:
            out.append(str(data[str(i)]))
            i += 1
        return out
    return []


def _decode_binary_label_vector(vec: Sequence[Union[int, float]], label_names: List[str]) -> List[str]:
    if not isinstance(vec, list) or not vec:
        return ["unlabeled"]

    decoded = []
    for i, value in enumerate(vec):
        if int(value) != 1:
            continue
        if USE_LABEL_NAMES and i < len(label_names):
            decoded.append(label_names[i])
        else:
            decoded.append(f"class_{i}")
    return decoded if decoded else ["unlabeled"]


def load_index(index_path: str):
    return faiss.read_index(index_path)


def load_embeddings(path: str):
    emb = np.load(path)
    if emb.dtype != np.float32:
        emb = emb.astype("float32")
    return emb


def load_labels(path: str):
    """
    Load labels in index order.

    Supported formats:
    - JSON list of labels per sample
    - m-bigearthnet label_stats.json dict: {sample_id: [0/1, ...]}
    """
    labels_path = Path(path)
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    with labels_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Already a list aligned by embedding order.
    if isinstance(data, list):
        return data

    if not isinstance(data, dict):
        raise ValueError("Unsupported labels format. Expected list or dict JSON.")

    label_names = _load_label_names(labels_path.parent.parent / "label_names.json")
    embedding_ids = _load_embedding_ids(EMBEDDING_IDS_PATH)

    if embedding_ids:
        labels = []
        for sample_id in embedding_ids:
            vec = data.get(sample_id)
            if vec is None:
                labels.append(["unlabeled"])
            else:
                labels.append(_decode_binary_label_vector(vec, label_names))
        return labels

    # Fallback: rely on dict insertion order when no embedding-id mapping is available.
    return [_decode_binary_label_vector(v, label_names) for v in data.values()]


def normalize_vector(vec: np.ndarray):
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def retrieve(
    query_embedding: np.ndarray,
    index,
    labels,
    k: int = 5,
    normalize_query: bool = True,
):
    """
    Returns:
        indices, scores, retrieved_labels
    """
    if query_embedding.dtype != np.float32:
        query_embedding = query_embedding.astype("float32")

    if normalize_query:
        query_embedding = normalize_vector(query_embedding)

    query_embedding = query_embedding.reshape(1, -1)
    top_k = min(k, index.ntotal)
    scores, indices = index.search(query_embedding, top_k)

    valid_indices = [int(i) for i in indices[0] if i >= 0 and i < len(labels)]
    valid_scores = [float(scores[0][j]) for j, i in enumerate(indices[0]) if i >= 0 and i < len(labels)]
    retrieved_labels = [labels[i] for i in valid_indices]

    return np.array(valid_indices), np.array(valid_scores), retrieved_labels


if __name__ == "__main__":
    print("Loading index...")
    index = load_index(str(INDEX_PATH))

    print("Loading embeddings...")
    embeddings = load_embeddings(str(EMBEDDINGS_PATH))

    print("Loading labels...")
    labels = load_labels(str(LABELS_PATH))

    query_embedding = embeddings[0]

    indices, scores, retrieved_labels = retrieve(
        query_embedding=query_embedding,
        index=index,
        labels=labels,
        k=TOP_K,
        normalize_query=True,
    )

    print("Top-K indices:", indices.tolist())
    print("Similarity scores:", scores.tolist())
    print("Retrieved labels:", retrieved_labels)
