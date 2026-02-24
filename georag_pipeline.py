import numpy as np

from retrieve_neighbors import load_index, load_labels, retrieve
from landcover_knowledge import build_context, summarize_labels
from generate_explanation import generate_explanation


class GeoRAG:
    def __init__(
        self,
        index_path: str,
        labels_path: str,
        normalize_query: bool = True,
        k: int = 5,
    ):
        self.index = load_index(index_path)
        self.labels = load_labels(labels_path)
        self.normalize_query = normalize_query
        self.k = k
        self.dim = self.index.d

    def query(self, query_embedding: np.ndarray):
        if query_embedding.ndim != 1:
            raise ValueError(f"query_embedding must be 1D, got shape={query_embedding.shape}")
        if query_embedding.shape[0] != self.dim:
            raise ValueError(
                f"query_embedding dimension mismatch: expected {self.dim}, got {query_embedding.shape[0]}"
            )

        indices, scores, retrieved_labels = retrieve(
            query_embedding=query_embedding,
            index=self.index,
            labels=self.labels,
            k=self.k,
            normalize_query=self.normalize_query,
        )

        label_summary = summarize_labels(retrieved_labels, scores=scores.tolist())
        context = build_context(retrieved_labels, scores=scores.tolist())

        explanation = generate_explanation(context, label_summary=label_summary)

        return {
            "indices": indices.tolist(),
            "scores": scores.tolist(),
            "retrieved_labels": retrieved_labels,
            "label_summary": label_summary,
            "context": context,
            "explanation": explanation,
        }
