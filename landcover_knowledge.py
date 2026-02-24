from collections import defaultdict
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from config import CLASS_DESCRIPTIONS_PATH


LANDCOVER_DESCRIPTIONS: Dict[str, str] = {
    "Coniferous forest": "Areas dominated by evergreen needle-leaf trees such as pine or spruce.",
    "Mixed forest": "Areas with a mixture of coniferous and broad-leaved trees.",
    "Broad-leaved forest": "Forests dominated by deciduous broad-leaf trees.",
    "Urban fabric": "Land covered by buildings, roads, and infrastructure.",
    "Industrial or commercial units": "Areas occupied by industrial facilities or commercial buildings.",
    "Arable land": "Land used for crop cultivation.",
    "Pastures": "Grassland areas used for grazing livestock.",
    "Water bodies": "Lakes, rivers, or reservoirs containing open water.",
    "Transitional woodland": "Areas transitioning between forest and shrubland.",
}


def _load_class_descriptions(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8-sig") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except Exception:
        return {}
    return {}


LANDCOVER_DESCRIPTIONS.update(_load_class_descriptions(Path(CLASS_DESCRIPTIONS_PATH)))


def get_description(label: str) -> str:
    return LANDCOVER_DESCRIPTIONS.get(
        label,
        "No detailed description available for this land cover type.",
    )


def summarize_labels(
    retrieved_labels: List[List[str]],
    scores: Optional[Iterable[float]] = None,
) -> List[Dict[str, float]]:
    """
    Build a ranked summary from retrieved labels and similarity scores.
    """
    stats = defaultdict(lambda: {"count": 0, "score_sum": 0.0})
    score_list = list(scores) if scores is not None else [1.0] * len(retrieved_labels)

    for label_group, score in zip(retrieved_labels, score_list):
        unique_group = set(label_group)
        for label in unique_group:
            stats[label]["count"] += 1
            stats[label]["score_sum"] += float(score)

    summary = []
    for label, vals in stats.items():
        summary.append(
            {
                "label": label,
                "count": vals["count"],
                "score_sum": vals["score_sum"],
                "description": get_description(label),
            }
        )

    summary.sort(key=lambda x: (x["score_sum"], x["count"]), reverse=True)
    return summary


def build_context(
    retrieved_labels: List[List[str]],
    scores: Optional[Iterable[float]] = None,
    max_labels: int = 10,
) -> str:
    """
    Return ranked context text for generation.
    """
    summary = summarize_labels(retrieved_labels, scores=scores)
    lines = []
    for item in summary[:max_labels]:
        lines.append(
            f"{item['label']} | count={item['count']} | score={item['score_sum']:.4f}: {item['description']}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    example_labels = [
        ["Coniferous forest"],
        ["Mixed forest", "Pastures"],
        ["Urban fabric"],
    ]
    example_scores = [0.92, 0.88, 0.74]
    print(build_context(example_labels, example_scores))
