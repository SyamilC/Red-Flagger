from __future__ import annotations

import json
import math
import os
import re
import shutil
import sys
import uuid
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

try:
    import sklearn._loss.loss as _sklearn_loss

    sys.modules.setdefault("_loss", _sklearn_loss)
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "models").exists():
    ROOT = Path(r"C:\Users\LENOVO\anaconda3\Red-Flagger")

DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
API_DIR = ROOT / "api"
MOBILE_UI = API_DIR / "mobile_interface.html"
EXTENSION_TEST_UI = API_DIR / "extension_test_profile.html"
UPLOAD_DIR = API_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

COCO_PERSON = 0
COCO_CELL_PHONE = 67

LABEL_FALLBACK = [
    "aggressive_tone",
    "hookup_focus",
    "negativity",
    "sarcasm_cynicism",
    "substance_risk",
    "incomplete_profile",
    "controlling_behavior",
    "emotional_manipulation",
    "entitlement_superiority",
    "poor_conflict_resolution",
    "main_character_syndrome",
    "dismissive",
]

TABULAR_DEFAULTS = {
    "age": 27,
    "status": "single",
    "sex": "m",
    "orientation": "straight",
    "body_type": "average",
    "diet": "mostly anything",
    "drinks": "socially",
    "drugs": "never",
    "education": "graduated from college/university",
    "ethnicity": "asian",
    "height": 170,
    "income": -1,
    "job": "student",
    "offspring": "doesn't have kids",
    "pets": "likes dogs",
    "religion": "agnosticism",
    "sign": "leo",
    "smokes": "no",
}

PROFILE_DISPLAY_FIELDS = [
    "age",
    "status",
    "sex",
    "orientation",
    "body_type",
    "diet",
    "drinks",
    "drugs",
    "education",
    "ethnicity",
    "height",
    "income",
    "job",
    "offspring",
    "pets",
    "religion",
    "sign",
    "smokes",
]

VISUAL_RISK_LOOKUP = {
    "clean": 0.0,
    "Group_Photo_Ambiguity": 0.35,
    "Face_Obscured": 0.50,
    "No_Face_Present": 0.55,
    "Heavy_Filtering_or_Edited": 0.40,
}

VISUAL_FLAG_DESCRIPTIONS = {
    "clean": "No major visual presentation issue detected.",
    "Group_Photo_Ambiguity": "Multiple people were detected, which may make profile ownership ambiguous.",
    "Face_Obscured": "The image may contain an object overlap or presentation pattern that obscures the face.",
    "No_Face_Present": "YOLO did not detect a person in the uploaded or linked profile image.",
    "Heavy_Filtering_or_Edited": "The image may contain strong filtering, blur, or editing artifacts.",
}

SEVERE_TEXT_FLAGS = {
    "controlling_behavior",
    "emotional_manipulation",
    "entitlement_superiority",
    "poor_conflict_resolution",
}


class CustomAuditRequest(BaseModel):
    bio: str = Field(..., min_length=1)
    profile_id: str = "custom_mobile_demo"
    visual_flags: list[str] | str = "clean"
    tabular: dict[str, Any] = Field(default_factory=dict)
    use_openrouter: bool = False
    openrouter_api_key: str | None = None


app = FastAPI(
    title="Red-Flagger Profile Auditor API",
    description="Local API and mobile interface for the Dating Red Flag Detector prototype.",
    version="1.1.0",
)
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if value is None:
        return None
    try:
        if not isinstance(value, (list, tuple, dict)) and pd.isna(value):
            return None
    except Exception:
        pass
    return value


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@lru_cache(maxsize=1)
def artifacts() -> dict[str, Any]:
    labels = LABEL_FALLBACK
    if (MODELS_DIR / "nlp_label_columns.pkl").exists():
        labels = list(joblib.load(MODELS_DIR / "nlp_label_columns.pkl"))

    thresholds = {label: 0.5 for label in labels}
    if (MODELS_DIR / "nlp_custom_thresholds.pkl").exists():
        thresholds.update(joblib.load(MODELS_DIR / "nlp_custom_thresholds.pkl"))

    descriptions = {label: label.replace("_", " ").title() for label in labels}
    if (MODELS_DIR / "nlp_label_descriptions.pkl").exists():
        descriptions.update(joblib.load(MODELS_DIR / "nlp_label_descriptions.pkl"))

    keyword_rules = {}
    if (MODELS_DIR / "nlp_keyword_rules.pkl").exists():
        keyword_rules = joblib.load(MODELS_DIR / "nlp_keyword_rules.pkl")

    return {
        "profiles": _read_csv(DATA_DIR / "okcupid_cleaned_redflags.csv"),
        "visual": _read_csv(DATA_DIR / "visual_inference_features.csv"),
        "nlp_model": joblib.load(MODELS_DIR / "nlp_tfidf_logreg_model.pkl"),
        "label_columns": labels,
        "thresholds": thresholds,
        "descriptions": descriptions,
        "keyword_rules": keyword_rules,
        "tabular_preprocessor": joblib.load(MODELS_DIR / "tabular_preprocessor.pkl"),
        "tabular_models": joblib.load(MODELS_DIR / "tabular_rf_smote_models.pkl"),
        "metrics": load_metrics_summary(),
    }


@lru_cache(maxsize=1)
def transformer_components() -> dict[str, Any]:
    model_path = MODELS_DIR / "distilbert_redflag_model"
    if not model_path.exists():
        return {"available": False, "reason": "Saved DistilBERT folder was not found."}
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as exc:
        return {"available": False, "reason": f"transformers/torch is not available: {exc}"}
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval()
        return {"available": True, "tokenizer": tokenizer, "model": model, "torch": torch}
    except Exception as exc:
        return {"available": False, "reason": f"Transformer load failed: {exc}"}


@lru_cache(maxsize=1)
def yolo_model() -> dict[str, Any]:
    model_path = MODELS_DIR / "yolo_model.pt"
    if not model_path.exists() or model_path.stat().st_size == 0:
        return {"available": False, "reason": "YOLO weights were not found."}
    try:
        from ultralytics import YOLO

        return {"available": True, "model": YOLO(str(model_path))}
    except Exception as exc:
        return {"available": False, "reason": f"YOLO load failed: {exc}"}


def load_metrics_summary() -> dict[str, Any]:
    summary: dict[str, Any] = {}
    metric_files = {
        "nlp_tfidf_logreg_overall": REPORTS_DIR / "nlp_overall_metrics.csv",
        "tabular_smote_random_forest_overall": REPORTS_DIR / "tabular_overall_metrics.csv",
        "yolo_visual_rows": REPORTS_DIR / "yolo_visual_metrics.csv",
    }
    for key, path in metric_files.items():
        if path.exists():
            summary[key] = pd.read_csv(path).to_dict(orient="records")

    yolo_path = metric_files["yolo_visual_rows"]
    if yolo_path.exists():
        yolo = pd.read_csv(yolo_path)
        summary["yolo_visual_subset"] = {
            "mean_precision": round(float(yolo["precision"].mean()), 4) if "precision" in yolo else None,
            "mean_recall": round(float(yolo["recall"].mean()), 4) if "recall" in yolo else None,
            "mean_iou": round(float(yolo["mean_iou"].mean()), 4) if "mean_iou" in yolo else None,
            "rows": yolo.to_dict(orient="records"),
        }
    return summary


def _format_metric_rows(rows: list[dict[str, Any]]) -> str:
    formatted = []
    for row in rows:
        metric = row.get("Metric", row.get("metric", "Metric"))
        score = row.get("Score", row.get("score"))
        if isinstance(score, (int, float, np.floating)):
            formatted.append(f"{metric}: {float(score):.4f}")
        else:
            formatted.append(f"{metric}: {score}")
    return ", ".join(formatted)


def _extract_positive_probabilities(predict_proba_output: Any, labels: list[str]) -> dict[str, float]:
    probabilities: dict[str, float] = {}
    if isinstance(predict_proba_output, list):
        for label, arr in zip(labels, predict_proba_output):
            arr = np.asarray(arr)
            probabilities[label] = float(arr[0, 1]) if arr.ndim == 2 and arr.shape[1] > 1 else float(arr.ravel()[0])
        return probabilities
    arr = np.asarray(predict_proba_output)
    if arr.ndim == 2 and arr.shape[0] == 1 and arr.shape[1] == len(labels):
        for i, label in enumerate(labels):
            probabilities[label] = float(arr[0, i])
    return probabilities


def run_nlp_inference(profile_text: str, state: dict[str, Any]) -> dict[str, Any]:
    labels = state["label_columns"]
    probabilities = _extract_positive_probabilities(state["nlp_model"].predict_proba([str(profile_text)]), labels)
    results = {}
    detected = []
    for label in labels:
        probability = float(probabilities.get(label, 0.0))
        threshold = float(state["thresholds"].get(label, 0.5))
        predicted = int(probability >= threshold)
        results[label] = {
            "probability": round(probability, 4),
            "threshold": threshold,
            "predicted": predicted,
            "description": state["descriptions"].get(label, ""),
        }
        if predicted:
            detected.append(label)
    return {"available": True, "module": "TF-IDF Logistic Regression NLP model", "detected_flags": detected, "label_scores": results}


def _rule_matches(text: str, pattern: str) -> bool:
    pattern = str(pattern).strip()
    if not pattern:
        return False
    try:
        return re.search(pattern, text, flags=re.IGNORECASE) is not None
    except re.error:
        return pattern.lower() in text.lower()


def run_keyword_rule_inference(profile_text: str, state: dict[str, Any]) -> dict[str, Any]:
    rules = state.get("keyword_rules", {})
    labels = state["label_columns"]
    if not isinstance(rules, dict) or not rules:
        return {"available": False, "reason": "No keyword rules artifact found.", "detected_flags": [], "label_scores": {}}

    results = {}
    detected = []
    for label in labels:
        matches = [str(pattern) for pattern in rules.get(label, []) if _rule_matches(str(profile_text), str(pattern))]
        probability = min(0.95, 0.55 + 0.10 * len(matches)) if matches else 0.0
        predicted = int(bool(matches))
        results[label] = {
            "probability": round(float(probability), 4),
            "threshold": 0.55,
            "predicted": predicted,
            "description": state["descriptions"].get(label, ""),
            "matched_phrases": matches[:8],
        }
        if predicted:
            detected.append(label)
    return {"available": True, "module": "Transparent keyword/rule safety layer", "detected_flags": detected, "label_scores": results}


def run_transformer_inference(profile_text: str, state: dict[str, Any]) -> dict[str, Any]:
    components = transformer_components()
    labels = state["label_columns"]
    if not components.get("available"):
        return {"available": False, "module": "DistilBERT Transformer NLP model", "reason": components.get("reason"), "detected_flags": [], "label_scores": {}}
    try:
        inputs = components["tokenizer"](str(profile_text), return_tensors="pt", truncation=True, padding=True, max_length=256)
        with components["torch"].no_grad():
            outputs = components["model"](**inputs)
            probabilities = components["torch"].sigmoid(outputs.logits).numpy()[0]
    except Exception as exc:
        return {"available": False, "module": "DistilBERT Transformer NLP model", "reason": f"Transformer inference failed: {exc}", "detected_flags": [], "label_scores": {}}

    results = {}
    detected = []
    for i, label in enumerate(labels):
        probability = float(probabilities[i]) if i < len(probabilities) else 0.0
        predicted = int(probability >= 0.5)
        results[label] = {
            "probability": round(probability, 4),
            "threshold": 0.5,
            "predicted": predicted,
            "description": state["descriptions"].get(label, ""),
        }
        if predicted:
            detected.append(label)
    note = None
    if len(probabilities) != len(labels):
        note = f"Transformer produced {len(probabilities)} outputs for {len(labels)} labels; missing labels are treated as 0.0."
    return {"available": True, "module": "DistilBERT Transformer NLP model", "detected_flags": detected, "label_scores": results, "note": note}


def run_tabular_inference(profile_row: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    preprocessor = state["tabular_preprocessor"]
    features = list(getattr(preprocessor, "feature_names_in_", TABULAR_DEFAULTS.keys()))
    input_data = {feature: profile_row.get(feature, TABULAR_DEFAULTS.get(feature, np.nan)) for feature in features}
    processed = preprocessor.transform(pd.DataFrame([input_data]))
    results = {}
    detected = []
    for label, model in state["tabular_models"].items():
        probability = 0.0
        try:
            probs = model.predict_proba(processed)[0]
            probability = float(probs[1]) if len(probs) > 1 else 0.0
        except Exception:
            probability = 0.0
        predicted = int(probability >= 0.5)
        results[label] = {
            "probability": round(probability, 4),
            "threshold": 0.5,
            "predicted": predicted,
            "description": state["descriptions"].get(label, ""),
        }
        if predicted:
            detected.append(label)
    return {
        "available": True,
        "module": "SMOTE Random Forest tabular model",
        "detected_flags": detected,
        "label_scores": results,
        "feature_count": len(features),
        "features_used": features,
        "metadata_used": _json_safe(input_data),
    }


def normalize_visual_flags(flags: list[str] | str | None) -> list[str]:
    if flags is None:
        return ["clean"]
    if isinstance(flags, str):
        raw = [part.strip() for part in re.split(r"[,|]", flags) if part.strip()]
    else:
        raw = [str(part).strip() for part in flags if str(part).strip()]
    return raw or ["clean"]


def safe_json_loads(value: Any, fallback: Any) -> Any:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return fallback
    try:
        return json.loads(value) if isinstance(value, str) else value
    except Exception:
        return fallback


def compute_iou(box_a: list[float], box_b: list[float]) -> float:
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])
    inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)
    box_a_area = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    box_b_area = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union_area = box_a_area + box_b_area - inter_area
    return 0.0 if union_area == 0 else inter_area / union_area


def visual_from_profile(profile_id: int, state: dict[str, Any]) -> dict[str, Any]:
    visual_df = state["visual"]
    if visual_df.empty:
        return manual_visual("clean", image_path="no_visual_csv", source="missing_visual_csv")
    match = visual_df[visual_df["profile_id"].astype(int) == int(profile_id)]
    if match.empty:
        return manual_visual("clean", image_path="no_visual_row", source="missing_visual_row")
    row = match.iloc[0].to_dict()
    flags = normalize_visual_flags(row.get("detected_visual_flags", "clean"))
    return {
        "available": True,
        "module": "YOLOv8 visual heuristic module",
        "detected_visual_flags": flags,
        "visual_risk_score": round(float(max([VISUAL_RISK_LOOKUP.get(flag, 0.5) for flag in flags] or [0.0])), 4),
        "visual_flag_explanations": {flag: VISUAL_FLAG_DESCRIPTIONS.get(flag, "Custom visual flag.") for flag in flags},
        "bounding_boxes": safe_json_loads(row.get("bounding_boxes"), []),
        "confidence_scores": safe_json_loads(row.get("confidence_scores"), []),
        "num_persons_detected": int(float(row.get("num_persons_detected", 0) or 0)),
        "iou_metric": None if pd.isna(row.get("iou_metric", np.nan)) else round(float(row.get("iou_metric")), 4),
        "iou_explanation": "IoU comes from Notebook 04 ground-truth boxes for the prototype visual subset.",
        "visual_flag_source": row.get("visual_flag_source", "unknown"),
        "image_path": row.get("image_path", None),
    }


def manual_visual(flags: list[str] | str, image_path: str = "manual_visual_input", source: str = "manual_demo_input") -> dict[str, Any]:
    normalized = normalize_visual_flags(flags)
    return {
        "available": True,
        "module": "Manual visual demo input",
        "detected_visual_flags": normalized,
        "visual_risk_score": round(float(max([VISUAL_RISK_LOOKUP.get(flag, 0.5) for flag in normalized] or [0.0])), 4),
        "visual_flag_explanations": {flag: VISUAL_FLAG_DESCRIPTIONS.get(flag, "Custom visual flag.") for flag in normalized},
        "bounding_boxes": [],
        "confidence_scores": [],
        "num_persons_detected": 1,
        "iou_metric": None,
        "iou_explanation": "Manual demo visual input has no ground-truth box, so IoU is not computed.",
        "visual_flag_source": source,
        "image_path": image_path,
    }


def classify_yolo_result(result: Any, image_path: Path, image_url: str | None = None) -> dict[str, Any]:
    try:
        from PIL import Image

        with Image.open(image_path) as image:
            width, height = image.size
    except Exception:
        width, height = None, None

    boxes = result.boxes.xyxy.cpu().numpy().tolist() if result.boxes is not None else []
    scores = result.boxes.conf.cpu().numpy().tolist() if result.boxes is not None else []
    classes = result.boxes.cls.cpu().numpy().tolist() if result.boxes is not None else []
    person_boxes = []
    person_scores = []
    other_boxes = []
    other_classes = []
    for box, score, cls in zip(boxes, scores, classes):
        if int(cls) == COCO_PERSON:
            person_boxes.append([round(float(v), 2) for v in box])
            person_scores.append(round(float(score), 4))
        else:
            other_boxes.append([round(float(v), 2) for v in box])
            other_classes.append(int(cls))

    flags = []
    if len(person_boxes) == 0:
        flags.append("No_Face_Present")
    elif len(person_boxes) > 2:
        flags.append("Group_Photo_Ambiguity")
    else:
        obscured = False
        for other_box, other_cls in zip(other_boxes, other_classes):
            if other_cls == COCO_CELL_PHONE:
                obscured = any(compute_iou(other_box, person_box) > 0.02 for person_box in person_boxes)
        if obscured:
            flags.append("Face_Obscured")
    if not flags:
        flags.append("clean")

    return {
        "available": True,
        "module": "YOLOv8 live uploaded image module",
        "detected_visual_flags": flags,
        "visual_risk_score": round(float(max([VISUAL_RISK_LOOKUP.get(flag, 0.5) for flag in flags] or [0.0])), 4),
        "visual_flag_explanations": {flag: VISUAL_FLAG_DESCRIPTIONS.get(flag, "Custom visual flag.") for flag in flags},
        "bounding_boxes": person_boxes,
        "confidence_scores": person_scores,
        "num_persons_detected": len(person_boxes),
        "iou_metric": None,
        "iou_explanation": "Uploaded photos have no human ground-truth bounding box, so live-upload IoU is not computed.",
        "visual_flag_source": "yolo_live_upload",
        "image_path": str(image_path),
        "image_url": image_url,
        "image_width": width,
        "image_height": height,
    }


def analyze_uploaded_photo(image_path: Path, image_url: str | None = None) -> dict[str, Any]:
    loaded = yolo_model()
    if not loaded.get("available"):
        visual = manual_visual("clean", image_path=str(image_path), source="yolo_unavailable")
        visual["available"] = False
        visual["reason"] = loaded.get("reason", "YOLO unavailable")
        return visual
    try:
        result = loaded["model"](str(image_path), device="cpu", verbose=False)[0]
        return classify_yolo_result(result, image_path=image_path, image_url=image_url)
    except Exception as exc:
        visual = manual_visual("clean", image_path=str(image_path), source="yolo_failed")
        visual["available"] = False
        visual["reason"] = f"YOLO inference failed: {exc}"
        return visual


def combine_label_scores(*module_results: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    combined = {}
    labels = state["label_columns"]
    source_names = ["nlp", "keyword_rules", "transformer", "tabular"]
    for label in labels:
        scores = []
        predictions = []
        sources = []
        matched = []
        for source_name, result in zip(source_names, module_results):
            info = result.get("label_scores", {}).get(label)
            if not info:
                continue
            score = float(info.get("probability", 0.0))
            predicted = int(info.get("predicted", 0))
            scores.append(score)
            predictions.append(predicted)
            if predicted:
                sources.append(source_name)
            matched.extend(info.get("matched_phrases", []))
        max_probability = float(np.max(scores)) if scores else 0.0
        combined[label] = {
            "mean_probability": round(float(np.mean(scores)) if scores else 0.0, 4),
            "max_probability": round(max_probability, 4),
            "ensemble_predicted": int(max_probability >= 0.5 or any(predictions)),
            "sources": sources,
            "matched_phrases": sorted(set(matched)),
            "description": state["descriptions"].get(label, ""),
        }
    return combined


def risk_level(score: float) -> str:
    if score >= 70:
        return "High"
    if score >= 40:
        return "Moderate"
    if score >= 20:
        return "Low-to-Moderate"
    return "Low"


def calculate_overall_risk(combined: dict[str, Any], visual: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    label_probabilities = [float(info["max_probability"]) for info in combined.values()]
    top_label_risk = float(np.mean(sorted(label_probabilities, reverse=True)[:3])) if label_probabilities else 0.0
    visual_risk = float(visual.get("visual_risk_score", 0.0))
    severe = [label for label, info in combined.items() if label in SEVERE_TEXT_FLAGS and info["ensemble_predicted"] == 1]
    severe_boost = min(0.25, 0.06 * len(severe))
    score = (0.75 * top_label_risk + 0.25 * visual_risk + severe_boost) * 100
    if len(severe) >= 3:
        score = max(score, 70.0)
    elif len(severe) >= 2:
        score = max(score, 55.0)
    return round(min(score, 100.0), 2), {
        "top_label_risk": round(top_label_risk, 4),
        "visual_risk": round(visual_risk, 4),
        "severe_detected": severe,
        "severe_boost": round(severe_boost, 4),
    }


def generate_local_report(assessment: dict[str, Any]) -> str:
    score = assessment["overall_risk_score"]
    level = assessment["overall_risk_level"]
    visual = assessment["module_outputs"]["visual_yolo"]
    scoring = assessment["scoring_breakdown"]
    lines = [
        f"# Dating Profile Audit Report - Profile {assessment['profile_id']}",
        "",
        f"Overall result: {level} risk ({score}/100).",
        "",
        "This is an automated prototype assessment. It should support user decision-making, not judge the person as inherently unsafe.",
        "",
        "## 1. Text and bio red flags",
    ]
    if assessment["detected_text_tabular_flags"]:
        for flag in assessment["detected_text_tabular_flags"]:
            info = assessment["combined_label_scores"][flag]
            sources = f" Sources: {', '.join(info['sources'])}." if info.get("sources") else ""
            matched = f" Matched phrases: {', '.join(info['matched_phrases'][:4])}." if info.get("matched_phrases") else ""
            lines.append(f"- {flag}: {info['description']} Max probability: {info['max_probability']:.4f}; mean ensemble probability: {info['mean_probability']:.4f}.{sources}{matched}")
    else:
        lines.append("- No major text/tabular red flags were detected by the ensemble threshold.")

    lines.extend(["", "## 2. Tabular/profile metadata signals"])
    tabular = assessment["module_outputs"].get("tabular_smote_random_forest", {})
    if tabular.get("detected_flags"):
        for flag in tabular["detected_flags"]:
            prob = tabular["label_scores"][flag]["probability"]
            lines.append(f"- {flag} was triggered by the tabular SMOTE model with probability {prob:.4f}.")
    else:
        lines.append("- The tabular SMOTE model did not independently trigger any metadata flag.")

    lines.extend(["", "## 3. Visual profile image signals"])
    if visual.get("available"):
        for flag in visual.get("detected_visual_flags", []):
            lines.append(f"- {flag}: {visual.get('visual_flag_explanations', {}).get(flag, '')}")
        lines.append(f"- Detected persons: {visual.get('num_persons_detected')}")
        lines.append(f"- Bounding boxes: {visual.get('bounding_boxes')}")
        lines.append(f"- IoU for this row: {visual.get('iou_metric')}")
        lines.append(f"- IoU note: {visual.get('iou_explanation')}")
        lines.append(f"- Visual source: {visual.get('visual_flag_source')}")
    else:
        lines.append(f"- Visual module unavailable: {visual.get('reason')}")

    lines.extend(["", "## 4. Scoring explanation"])
    lines.append(f"- Text risk from strongest labels: {scoring.get('top_label_risk')}.")
    lines.append(f"- Visual risk contribution: {scoring.get('visual_risk')}.")
    lines.append(f"- Severe interpersonal toxicity labels detected: {', '.join(scoring.get('severe_detected', [])) if scoring.get('severe_detected') else 'None'}.")
    lines.append(f"- Severe-label caution boost: {scoring.get('severe_boost')}.")

    perf = assessment.get("previous_model_performance", {})
    lines.extend(["", "## 5. Model performance note"])
    if "nlp_tfidf_logreg_overall" in perf:
        lines.append(f"- NLP model overall metrics: {_format_metric_rows(perf['nlp_tfidf_logreg_overall'])}.")
    if "tabular_smote_random_forest_overall" in perf:
        lines.append(f"- Tabular SMOTE model overall metrics: {_format_metric_rows(perf['tabular_smote_random_forest_overall'])}.")
    if "yolo_visual_subset" in perf:
        yolo = perf["yolo_visual_subset"]
        lines.append(f"- YOLO visual subset metrics: Precision {yolo.get('mean_precision')}, Recall {yolo.get('mean_recall')}, Mean IoU {yolo.get('mean_iou')}.")

    lines.extend(["", "## 6. Final recommendation"])
    if score >= 70:
        lines.append("- The app should warn the user to review this profile carefully before engaging.")
    elif score >= 40:
        lines.append("- The app should show a caution note and highlight the specific evidence.")
    else:
        lines.append("- The app can mark the profile as relatively low-risk while still showing transparent model evidence.")
    return "\n".join(lines)


def clean_openrouter_key(api_key: str | None = None) -> str:
    return (api_key or os.getenv("OPENROUTER_API_KEY", "")).strip().strip('"').strip("'")


def call_openrouter_for_report(assessment: dict[str, Any], api_key_override: str | None = None) -> str | None:
    api_key = clean_openrouter_key(api_key_override)
    if not api_key:
        return None
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost",
                "X-OpenRouter-Title": "Dating Red Flag Detector Demo",
            },
            json={
                "model": os.getenv("OPENROUTER_MODEL", "openrouter/free"),
                "messages": [
                    {"role": "system", "content": "You convert model outputs into careful dating profile audit reports."},
                    {"role": "user", "content": "Rewrite this JSON into a careful profile audit report. Do not invent findings.\n\n" + json.dumps(assessment, indent=2)},
                ],
                "temperature": 0.3,
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception:
        return None


def openrouter_key_configured(api_key_override: str | None = None) -> bool:
    return bool(clean_openrouter_key(api_key_override))


def build_assessment(
    profile_row: dict[str, Any],
    profile_id: str | int,
    visual: dict[str, Any],
    state: dict[str, Any],
    use_openrouter: bool = False,
    openrouter_api_key: str | None = None,
) -> dict[str, Any]:
    text = str(profile_row.get("clean_bio") or profile_row.get("full_bio") or profile_row.get("bio") or "")
    nlp = run_nlp_inference(text, state)
    keyword = run_keyword_rule_inference(text, state)
    transformer = run_transformer_inference(text, state)
    tabular = run_tabular_inference(profile_row, state)
    combined = combine_label_scores(nlp, keyword, transformer, tabular, state=state)
    score, scoring = calculate_overall_risk(combined, visual)
    detected = [label for label, info in combined.items() if info["ensemble_predicted"] == 1]
    assessment = {
        "profile_id": profile_id,
        "profile_summary": {field: profile_row.get(field, None) for field in PROFILE_DISPLAY_FIELDS},
        "bio_excerpt": str(profile_row.get("full_bio", text))[:1200],
        "overall_risk_score": score,
        "overall_risk_level": risk_level(score),
        "scoring_breakdown": scoring,
        "detected_text_tabular_flags": detected,
        "detected_visual_flags": visual.get("detected_visual_flags", []),
        "combined_label_scores": combined,
        "module_outputs": {
            "nlp_tfidf_logreg": nlp,
            "nlp_keyword_rules": keyword,
            "nlp_transformer": transformer,
            "tabular_smote_random_forest": tabular,
            "visual_yolo": visual,
        },
        "previous_model_performance": state["metrics"],
    }
    local_report = generate_local_report(assessment)
    key_configured = openrouter_key_configured(openrouter_api_key)
    env_key_configured = openrouter_key_configured()
    request_key_configured = bool(clean_openrouter_key(openrouter_api_key))
    openrouter_report = call_openrouter_for_report(assessment, openrouter_api_key) if use_openrouter else None
    assessment["report_markdown"] = openrouter_report or local_report
    assessment["report_source"] = "openrouter" if openrouter_report else "local_fallback"
    assessment["llm_report_status"] = {
        "beautify_key_name": "OPENROUTER_API_KEY",
        "beautify_key_configured": key_configured,
        "beautify_key_source": "request_input" if request_key_configured else ("environment" if env_key_configured else "missing"),
        "beautify_requested": bool(use_openrouter),
        "beautify_used": bool(openrouter_report),
        "report_source": assessment["report_source"],
        "reason": "OpenRouter report returned successfully."
        if openrouter_report
        else (
            "OpenRouter was requested, but the key is missing or the request failed; local fallback report used."
            if use_openrouter
            else "OpenRouter beautifier was not requested; local fallback report used."
        ),
    }
    return _json_safe(assessment)


def profile_row_from_custom(bio: str, tabular: dict[str, Any] | None = None) -> dict[str, Any]:
    row = dict(TABULAR_DEFAULTS)
    row.update(tabular or {})
    row["full_bio"] = bio
    row["clean_bio"] = bio
    return row


def parse_tabular_json(tabular_json: str | None) -> dict[str, Any]:
    if not tabular_json or not tabular_json.strip():
        return {}
    try:
        value = json.loads(tabular_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid tabular_json: {exc}") from exc
    if not isinstance(value, dict):
        raise HTTPException(status_code=400, detail="tabular_json must decode to an object")
    return value


def save_upload(photo: UploadFile) -> tuple[Path, str]:
    suffix = Path(photo.filename or "upload.jpg").suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        suffix = ".jpg"
    filename = f"{uuid.uuid4().hex}{suffix}"
    path = UPLOAD_DIR / filename
    with path.open("wb") as handle:
        shutil.copyfileobj(photo.file, handle)
    return path, f"/uploads/{filename}"


@app.get("/", response_class=HTMLResponse)
def mobile_ui() -> HTMLResponse:
    if not MOBILE_UI.exists():
        raise HTTPException(status_code=404, detail="mobile_interface.html not found")
    return HTMLResponse(MOBILE_UI.read_text(encoding="utf-8"))


@app.get("/extension-test", response_class=HTMLResponse)
def extension_test_page() -> HTMLResponse:
    if not EXTENSION_TEST_UI.exists():
        raise HTTPException(status_code=404, detail="extension_test_profile.html not found")
    return HTMLResponse(EXTENSION_TEST_UI.read_text(encoding="utf-8"))


@app.get("/health")
def health() -> dict[str, Any]:
    state = artifacts()
    transformer_status = transformer_components()
    yolo_status = yolo_model()
    return {
        "status": "ok",
        "profiles": int(len(state["profiles"])),
        "labels": state["label_columns"],
        "has_visual_csv": not state["visual"].empty,
        "transformer_available": bool(transformer_status.get("available")),
        "transformer_reason": transformer_status.get("reason"),
        "yolo_available": bool(yolo_status.get("available")),
        "yolo_reason": yolo_status.get("reason"),
        "openrouter_key_configured": openrouter_key_configured(),
    }


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    return _json_safe(artifacts()["metrics"])


@app.get("/audit/sample/{profile_id}")
def audit_sample(profile_id: int, use_openrouter: bool = False) -> dict[str, Any]:
    state = artifacts()
    profiles = state["profiles"]
    if profile_id < 0 or profile_id >= len(profiles):
        raise HTTPException(status_code=404, detail=f"profile_id must be between 0 and {len(profiles) - 1}")
    row = profiles.iloc[int(profile_id)].to_dict()
    return build_assessment(row, profile_id, visual_from_profile(profile_id, state), state, use_openrouter=use_openrouter)


@app.post("/audit/custom")
def audit_custom(request: CustomAuditRequest) -> dict[str, Any]:
    state = artifacts()
    row = profile_row_from_custom(request.bio, request.tabular)
    return build_assessment(
        row,
        request.profile_id,
        manual_visual(request.visual_flags),
        state,
        use_openrouter=request.use_openrouter,
        openrouter_api_key=request.openrouter_api_key,
    )


@app.post("/audit/photo")
def audit_photo(
    bio: str = Form(...),
    profile_id: str = Form("photo_upload_demo"),
    tabular_json: str = Form("{}"),
    use_openrouter: bool = Form(False),
    openrouter_api_key: str = Form(""),
    photo: UploadFile = File(...),
) -> dict[str, Any]:
    state = artifacts()
    image_path, image_url = save_upload(photo)
    row = profile_row_from_custom(bio, parse_tabular_json(tabular_json))
    visual = analyze_uploaded_photo(image_path, image_url=image_url)
    return build_assessment(row, profile_id, visual, state, use_openrouter=use_openrouter, openrouter_api_key=openrouter_api_key)
