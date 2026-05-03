from __future__ import annotations

import csv
import io
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

MODEL1_ARTIFACT = Path(os.getenv("MODEL1_ARTIFACT", MODELS_DIR / "model1_requirement_detector.joblib"))

MODEL2_DIR = Path(os.getenv("MODEL2_DIR", MODELS_DIR / "FR_requirement_classifier_bert_tiny"))


MODEL3_TRANSFORMER_DIR = os.getenv("MODEL3_TRANSFORMER_DIR", "")
MODEL3_ARTIFACT = Path(os.getenv("MODEL3_ARTIFACT", MODELS_DIR / "model3_nfr_type_classifier.joblib"))


MODEL2_LABELS = {
    0: {"code": "FR", "name": "Functional", "tone": "blue"},
    1: {"code": "NFR", "name": "Non-functional", "tone": "purple"},
}

MODEL3_LABELS = {
    "A": {"name": "Availability", "description": "Uptime, access continuity, downtime, service reachability."},
    "PE": {"name": "Performance", "description": "Speed, response time, load, scale, throughput."},
    "SE": {"name": "Security", "description": "Authentication, authorization, encryption, privacy, threats."},
    "US": {"name": "Usability", "description": "Ease of use, navigation, clarity, learnability, errors."},
}


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def split_candidate_sentences(text: str) -> list[str]:
    text = re.sub(r"\r\n?", "\n", text or "")
    text = re.sub(r"[ \t]+", " ", text)
    chunks: list[str] = []
    for line in text.splitlines():
        line = line.strip(" -\t")
        if not line:
            continue
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])", line)
        chunks.extend(parts)
    cleaned = []
    for part in chunks:
        item = re.sub(r"\s+", " ", part).strip()
        if len(item) >= 18:
            cleaned.append(item)
    return cleaned


def extract_text_from_upload(filename: str, content: bytes) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(content))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as exc:  # pragma: no cover - depends on external files
            raise ValueError(f"Could not read PDF: {exc}") from exc
    if suffix == ".docx":
        try:
            from docx import Document

            doc = Document(io.BytesIO(content))
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception as exc:  # pragma: no cover - depends on external files
            raise ValueError(f"Could not read DOCX: {exc}") from exc
    return content.decode("utf-8", errors="ignore")


@dataclass
class Prediction:
    label: Any
    confidence: float
    probabilities: dict[str, float]


class SklearnClassifier:
    def __init__(self, path: Path, label_names: dict[Any, str] | None = None):
        import joblib

        self.path = path
        self.label_names = label_names or {}
        self.model = joblib.load(path)

    def predict_one(self, text: str) -> Prediction:
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba([text])[0]
            classes = list(self.model.classes_)
            idx = int(np.argmax(probs))
            label = classes[idx]
            mapped = {str(self.label_names.get(c, c)): float(probs[i]) for i, c in enumerate(classes)}
            return Prediction(label=label, confidence=float(probs[idx]), probabilities=mapped)
        label = self.model.predict([text])[0]
        return Prediction(label=label, confidence=0.65, probabilities={str(label): 0.65})


class Model1Detector:
    def __init__(self):
        self.status = "heuristic"
        self.model = None

        if MODEL1_ARTIFACT.exists():
            try:
                self.model = SklearnClassifier(
                    MODEL1_ARTIFACT,
                    {0: "Non-requirement", 1: "Requirement"}
                )
                self.status = "trained"
                print(f"[Model1] Loaded trained model from {MODEL1_ARTIFACT}")

            except Exception as e:
                raise RuntimeError(
                    f"[Model1] Failed to load model from {MODEL1_ARTIFACT}: {e}"
                ) from e

    def predict_one(self, text: str) -> Prediction:
        if self.model:
            return self.model.predict_one(text)
        score = heuristic_requirement_score(text)
        return Prediction(
            label=1 if score >= 0.48 else 0,
            confidence=float(max(score, 1 - score)),
            probabilities={"Non-requirement": float(1 - score), "Requirement": float(score)},
        )


class Model2FrNfr:
    def __init__(self):
        self.status = "unavailable"
        self.threshold = 0.5
        self.model: Any = None
        self.tokenizer: Any = None
        self.device: Any = None
        if (MODEL2_DIR / "best_threshold.pkl").exists():
            with open(MODEL2_DIR / "best_threshold.pkl", "rb") as fh:
                self.threshold = float(pickle.load(fh))
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch

            if MODEL2_DIR.exists():
                self.tokenizer = AutoTokenizer.from_pretrained(str(MODEL2_DIR), local_files_only=True)
                self.model = AutoModelForSequenceClassification.from_pretrained(str(MODEL2_DIR), local_files_only=True)
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.model.to(self.device)
                self.model.eval()
                self.status = "bert-tiny"
                return
        except Exception:
            self.model = None
        

    def predict_one(self, text: str) -> Prediction:
        if self.status == "bert-tiny":
            import torch

            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = self.model(**inputs).logits.detach().cpu().numpy()
            if logits.shape[-1] == 1:
                nfr_prob = float(1 / (1 + np.exp(-logits[0][0])))
            else:
                nfr_prob = float(_softmax(logits)[0][1])
            label = 1 if nfr_prob >= self.threshold else 0
            confidence = nfr_prob if label == 1 else 1 - nfr_prob
            return Prediction(
                label=label,
                confidence=confidence,
                probabilities={"Functional": float(1 - nfr_prob), "Non-functional": nfr_prob},
            )
        
        raise RuntimeError("[Model2] BERT model is not loaded.")


class Model3NfrType:
    def __init__(self):
        self.status = "heuristic"
        self.model: Any = None
        self.tokenizer: Any = None
        self.device: Any = None
        if MODEL3_TRANSFORMER_DIR:
            try:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                import torch

                path = Path(MODEL3_TRANSFORMER_DIR)
                self.tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
                self.model = AutoModelForSequenceClassification.from_pretrained(str(path), local_files_only=True)
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.model.to(self.device)
                self.model.eval()
                self.status = "transformer"
                return
            except Exception:
                self.model = None
        if MODEL3_ARTIFACT.exists():
            self.model = SklearnClassifier(MODEL3_ARTIFACT)
            self.status = "trained"

    def predict_one(self, text: str) -> Prediction:
        if self.status == "transformer":
            import torch

            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = self.model(**inputs).logits.detach().cpu().numpy()
            probs = _softmax(logits)[0]
            idx = int(np.argmax(probs))
            label = self.model.config.id2label.get(idx, str(idx))
            return Prediction(label=label, confidence=float(probs[idx]), probabilities={label: float(probs[idx])})
        if isinstance(self.model, SklearnClassifier):
            return self.model.predict_one(text)
        label, probs = heuristic_nfr_type(text)
        return Prediction(label=label, confidence=float(probs[label]), probabilities={k: float(v) for k, v in probs.items()})


def heuristic_requirement_score(text: str) -> float:
    t = text.lower()
    strong = ["shall", "must", "should", "will", "is required to", "be able to", "provide", "support", "allow"]
    weak = ["requirement", "req-", "system", "user", "application", "software"]
    score = 0.22
    score += min(sum(term in t for term in strong) * 0.24, 0.58)
    score += min(sum(term in t for term in weak) * 0.07, 0.22)
    if len(t.split()) >= 5:
        score += 0.08
    return min(score, 0.95)


def heuristic_nfr_score(text: str) -> float:
    t = text.lower()
    nfr_terms = [
        "security", "secure", "encrypt", "password", "authentication", "availability", "available", "uptime",
        "performance", "respond", "response", "latency", "fast", "load", "usable", "easy", "intuitive",
        "reliable", "scalable", "privacy", "unauthorized", "downtime", "simple", "clear", "user-friendly",
    ]
    functional_terms = ["create", "edit", "delete", "view", "print", "calculate", "submit", "store", "display"]
    score = 0.38 + min(sum(term in t for term in nfr_terms) * 0.15, 0.55)
    score -= min(sum(term in t for term in functional_terms) * 0.08, 0.22)
    return float(max(0.08, min(score, 0.94)))


def heuristic_nfr_type(text: str) -> tuple[str, dict[str, float]]:
    t = text.lower()
    patterns = {
        "A": ["available", "availability", "uptime", "downtime", "accessible", "interruption", "offline", "24/7"],
        "PE": ["response", "respond", "seconds", "fast", "quick", "load", "latency", "traffic", "scale", "throughput"],
        "SE": ["secure", "security", "encrypt", "password", "auth", "access control", "unauthorized", "privacy", "confidential"],
        "US": ["easy", "usable", "usability", "intuitive", "navigate", "simple", "clear", "user-friendly", "training"],
    }
    raw = {label: 1.0 + sum(term in t for term in terms) * 2.2 for label, terms in patterns.items()}
    total = sum(raw.values())
    probs = {label: value / total for label, value in raw.items()}
    label = max(probs, key=probs.get)
    return label, probs


class RequirementPipeline:
    def __init__(self):
        self.model1 = Model1Detector()
        self.model2 = Model2FrNfr()
        self.model3 = Model3NfrType()

    def status(self) -> dict[str, Any]:
        return {
            "model1": {"status": self.model1.status, "artifact": str(MODEL1_ARTIFACT)},
            "model2": {"status": self.model2.status, "artifact": str(MODEL2_DIR), "threshold": self.model2.threshold},
            "model3": {"status": self.model3.status, "artifact": str(MODEL3_ARTIFACT)},
        }

    def analyze(self, text: str) -> dict[str, Any]:
        candidates = split_candidate_sentences(text)
        rows = []
        for idx, sentence in enumerate(candidates, start=1):
            req = self.model1.predict_one(sentence)
            if int(req.label) != 1:
                continue
            frnfr = self.model2.predict_one(sentence)
            frnfr_info = MODEL2_LABELS[int(frnfr.label)]
            nfr_type = None
            if int(frnfr.label) == 1:
                pred3 = self.model3.predict_one(sentence)
                label = str(pred3.label)
                nfr_type = {
                    "code": label,
                    "name": MODEL3_LABELS.get(label, {}).get("name", label),
                    "description": MODEL3_LABELS.get(label, {}).get("description", ""),
                    "confidence": round(pred3.confidence, 4),
                    "probabilities": pred3.probabilities,
                }
            rows.append(
                {
                    "id": idx,
                    "text": sentence,
                    "requirementConfidence": round(req.confidence, 4),
                    "class": frnfr_info,
                    "classConfidence": round(frnfr.confidence, 4),
                    "classProbabilities": frnfr.probabilities,
                    "nfrType": nfr_type,
                }
            )
        summary = {
            "candidates": len(candidates),
            "requirements": len(rows),
            "functional": sum(1 for row in rows if row["class"]["code"] == "FR"),
            "nonFunctional": sum(1 for row in rows if row["class"]["code"] == "NFR"),
        }
        type_counts: dict[str, int] = {}
        for row in rows:
            if row["nfrType"]:
                key = row["nfrType"]["name"]
                type_counts[key] = type_counts.get(key, 0) + 1
        summary["nfrTypes"] = type_counts
        return {"summary": summary, "requirements": rows, "status": self.status()}


def read_csv_rows(path: Path) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            sentence = (row.get("sentence") or "").strip()
            label = (row.get("label") or "").strip()
            if sentence and label:
                texts.append(sentence)
                labels.append(label)
    return texts, labels
