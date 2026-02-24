from typing import Dict, List, Optional, Tuple

from config import DEVICE, ENABLE_HF_GENERATION, LLM_MODEL_NAME, MAX_NEW_TOKENS, TEMPERATURE


_GENERATOR = None
_GENERATOR_ERROR = None


def _parse_context(context: str) -> List[Tuple[str, str]]:
    items = []
    for raw in context.splitlines():
        line = raw.strip()
        if not line:
            continue
        if ":" in line:
            label, desc = line.split(":", 1)
            items.append((label.strip(), desc.strip()))
        else:
            items.append((line, ""))
    return items


def _heuristic_explanation(context: str, label_summary: Optional[List[Dict]] = None) -> str:
    if label_summary:
        top = label_summary[0]
        unique_labels = len(label_summary)
        confidence = "High" if unique_labels <= 2 else "Medium" if unique_labels <= 4 else "Low"
        alternatives = ", ".join(item["label"] for item in label_summary[1:4])
        return (
            f"Most likely land cover type: {top['label']}.\n"
            f"Reasoning: it has the strongest combined nearest-neighbor evidence "
            f"(count={top['count']}, cumulative_similarity={top['score_sum']:.4f}).\n"
            f"Supporting description: {top['description']}\n"
            f"Other plausible labels: {alternatives if alternatives else 'none'}.\n"
            f"Confidence: {confidence}."
        )

    items = _parse_context(context)
    if not items:
        return "No land-cover context was provided, so confidence is Low."

    top_label, top_desc = items[0]
    unique_labels = len({label for label, _ in items})
    confidence = "High" if unique_labels <= 2 else "Medium" if unique_labels <= 4 else "Low"

    lines = [
        f"Most likely land cover type: {top_label}.",
        "Reasoning: this label appears in the nearest retrieved examples and matches the provided context.",
    ]
    if top_desc:
        lines.append(f"Supporting description: {top_desc}")
    lines.append(f"Confidence: {confidence}.")
    return "\n".join(lines)


def _get_generator():
    global _GENERATOR, _GENERATOR_ERROR
    if not ENABLE_HF_GENERATION:
        _GENERATOR_ERROR = RuntimeError("HF generation disabled in config.")
        return None

    if _GENERATOR is not None or _GENERATOR_ERROR is not None:
        return _GENERATOR

    try:
        from transformers import pipeline

        _GENERATOR = pipeline(
            "text-generation",
            model=LLM_MODEL_NAME,
            device=DEVICE,
        )
    except Exception as exc:
        _GENERATOR_ERROR = exc
        _GENERATOR = None

    return _GENERATOR


def build_prompt(context: str) -> str:
    return (
        "You are a remote sensing expert.\n\n"
        "The following land cover types were retrieved as the most similar to a satellite image:\n\n"
        f"{context}\n\n"
        "Based ONLY on this retrieved information:\n"
        "1. Explain what land cover type the satellite image most likely represents.\n"
        "2. Justify your reasoning using the retrieved descriptions.\n"
        "3. Provide a confidence level (Low / Medium / High).\n"
    )


def generate_explanation(context: str, label_summary: Optional[List[Dict]] = None) -> str:
    generator = _get_generator()
    if generator is None:
        return _heuristic_explanation(context, label_summary=label_summary)

    prompt = build_prompt(context)
    try:
        out = generator(
            prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
        )
        text = out[0].get("generated_text", "").strip()
        if text.startswith(prompt):
            text = text[len(prompt) :].strip()
        if text:
            return text
    except Exception:
        pass

    return _heuristic_explanation(context, label_summary=label_summary)


if __name__ == "__main__":
    dummy_context = (
        "Coniferous forest: Areas dominated by evergreen needle-leaf trees.\n"
        "Mixed forest: Areas with a mixture of coniferous and broad-leaved trees.\n"
        "Pastures: Grassland areas used for grazing livestock."
    )
    print(generate_explanation(dummy_context))
