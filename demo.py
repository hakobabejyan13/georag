import numpy as np

from config import EMBEDDINGS_PATH, INDEX_PATH, LABELS_PATH, TOP_K
from georag_pipeline import GeoRAG


def _validate_files():
    missing = [p for p in [EMBEDDINGS_PATH, INDEX_PATH, LABELS_PATH] if not p.exists()]
    if missing:
        joined = "\n".join(f"- {p}" for p in missing)
        raise FileNotFoundError(f"Missing required files:\n{joined}")


if __name__ == "__main__":
    _validate_files()

    # Load embeddings just for demo query.
    embeddings = np.load(EMBEDDINGS_PATH).astype("float32")

    georag = GeoRAG(
        index_path=str(INDEX_PATH),
        labels_path=str(LABELS_PATH),
        k=TOP_K,
    )

    # Example: use embedding[0] as query
    query_embedding = embeddings[0]

    result = georag.query(query_embedding)

    print("=== Retrieved Labels ===")
    print(result["retrieved_labels"])

    print("\n=== Label Summary ===")
    print("{")
    for item in result["label_summary"]:
        label = item["label"]
        desc = item["description"].replace('"', '\\"')
        print(f'  "{label}": "{desc}",')
    print("}")

    print("\n=== Generated Explanation ===")
    print(result["explanation"])
