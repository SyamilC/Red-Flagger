"""Export the trained TF-IDF NLP classifier into browser-readable JSON.

This does not create a TensorFlow graph. It serializes sklearn TF-IDF metadata,
classifier coefficients, intercepts, labels, thresholds, and descriptions. The current
browser scorer consumes the top-level word TF-IDF block; feature_blocks preserves the
full word plus char FeatureUnion metadata for a richer browser scorer later.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "nlp_tfidf_logreg_model.pkl"
DEFAULT_LABELS = ROOT / "models" / "nlp_label_columns.pkl"
DEFAULT_THRESHOLDS = ROOT / "models" / "nlp_custom_thresholds.pkl"
DEFAULT_DESCRIPTIONS = ROOT / "models" / "nlp_label_descriptions.pkl"
DEFAULT_OUTPUT = ROOT / "chrome_extension" / "models" / "nlp_model.json"


def _to_jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _vectorizer_blocks(tfidf_step: Any) -> list[dict[str, Any]]:
    if hasattr(tfidf_step, "vocabulary_") and hasattr(tfidf_step, "idf_"):
        return [{"name": "tfidf", "vectorizer": tfidf_step, "offset": 0}]

    transformer_list = getattr(tfidf_step, "transformer_list", None)
    if transformer_list is None:
        raise TypeError("The tfidf step must be a TfidfVectorizer or FeatureUnion of TfidfVectorizers.")

    blocks = []
    offset = 0
    for name, vectorizer in transformer_list:
        if not hasattr(vectorizer, "vocabulary_") or not hasattr(vectorizer, "idf_"):
            raise TypeError(f"FeatureUnion block {name!r} does not expose vocabulary_ and idf_.")
        size = len(vectorizer.vocabulary_)
        blocks.append({"name": str(name), "vectorizer": vectorizer, "offset": offset})
        offset += size
    return blocks


def _block_payload(block: dict[str, Any]) -> dict[str, Any]:
    vectorizer = block["vectorizer"]
    analyzer = getattr(vectorizer, "analyzer", "word")
    ngram_range = getattr(vectorizer, "ngram_range", None)
    return {
        "name": block["name"],
        "offset": int(block["offset"]),
        "size": int(len(vectorizer.vocabulary_)),
        "analyzer": str(analyzer),
        "ngram_range": list(ngram_range) if ngram_range is not None else None,
        "vocab": {str(token): int(index) for token, index in vectorizer.vocabulary_.items()},
        "idf": vectorizer.idf_.tolist(),
    }


def _choose_browser_block(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    for block in blocks:
        analyzer = getattr(block["vectorizer"], "analyzer", "word")
        if analyzer == "word" or block["name"].lower().startswith("word"):
            return block
    return blocks[0]


def export_nlp_model(model_path: Path, labels_path: Path, thresholds_path: Path, descriptions_path: Path, output_path: Path) -> None:
    pipeline = joblib.load(model_path)
    labels = list(joblib.load(labels_path))
    thresholds = joblib.load(thresholds_path) if thresholds_path.exists() else {label: 0.5 for label in labels}
    descriptions = joblib.load(descriptions_path) if descriptions_path.exists() else {}

    if not hasattr(pipeline, "named_steps"):
        raise TypeError("Expected a sklearn Pipeline with named_steps.")

    tfidf_step = pipeline.named_steps.get("tfidf")
    classifier = pipeline.named_steps.get("classifier")
    if tfidf_step is None or classifier is None:
        raise KeyError("Expected pipeline steps named 'tfidf' and 'classifier'.")

    blocks = _vectorizer_blocks(tfidf_step)
    browser_block = _choose_browser_block(blocks)
    browser_vectorizer = browser_block["vectorizer"]
    browser_offset = int(browser_block["offset"])
    browser_size = len(browser_vectorizer.vocabulary_)

    estimators = getattr(classifier, "estimators_", None)
    if estimators is None:
        raise TypeError("Expected a one-vs-rest style classifier with estimators_.")

    coefs = []
    intercepts = []
    for estimator in estimators:
        if not hasattr(estimator, "coef_") or not hasattr(estimator, "intercept_"):
            raise TypeError("Each estimator must expose coef_ and intercept_.")
        coef = estimator.coef_.ravel()
        coefs.append(coef[browser_offset:browser_offset + browser_size].tolist())
        intercepts.append(float(estimator.intercept_.ravel()[0]))

    payload = {
        "format": "red-flagger-tfidf-linear-v1",
        "browser_feature_block": browser_block["name"],
        "browser_scoring_note": "Top-level vocab/idf/coefs are the word TF-IDF slice used by the current extension scorer. See feature_blocks for full FeatureUnion metadata.",
        "labels": labels,
        "thresholds": {label: float(thresholds.get(label, 0.5)) for label in labels},
        "descriptions": {label: str(descriptions.get(label, "")) for label in labels},
        "vocab": {str(token): int(index) for token, index in browser_vectorizer.vocabulary_.items()},
        "idf": browser_vectorizer.idf_.tolist(),
        "coefs": coefs,
        "intercepts": intercepts,
        "feature_blocks": [_block_payload(block) for block in blocks],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_to_jsonable(payload), indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Red-Flagger NLP model metadata for browser scoring.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--thresholds", type=Path, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--descriptions", type=Path, default=DEFAULT_DESCRIPTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    export_nlp_model(args.model, args.labels, args.thresholds, args.descriptions, args.output)


if __name__ == "__main__":
    main()