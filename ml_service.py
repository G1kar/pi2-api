"""Общая логика ML: загрузка модели и предсказания для Flask и FastAPI."""

from __future__ import annotations

import os
import pickle
import re
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LEN = 40
MAX_WORDS = 20000
MODEL_PATH = "disaster_model.h5"
TOKENIZER_PATH = "tokenizer.pkl"
METRICS_PATH = "metrics.txt"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

_model = None
_tokenizer = None


def clean_text(s: str) -> str:
    """Та же функция очистки, что и в train.py."""
    s = s.lower()
    s = re.sub(r"http\S+|www\S+", " ", s)
    s = re.sub(r"@[A-Za-z0-9_]+", " ", s)
    s = re.sub(r"#", " ", s)
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _ensure_loaded() -> None:
    global _model, _tokenizer
    if _model is None:
        model_path = os.path.join(BASE_DIR, MODEL_PATH)
        tokenizer_path = os.path.join(BASE_DIR, TOKENIZER_PATH)
        _model = load_model(model_path)
        with open(tokenizer_path, "rb") as f:
            _tokenizer = pickle.load(f)


def _label_name(label: int) -> str:
    return "disaster" if label == 1 else "no_disaster"


def _predict_proba(cleaned_texts: list[str]) -> np.ndarray:
    _ensure_loaded()
    seq = _tokenizer.texts_to_sequences(cleaned_texts)
    pad = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    return _model.predict(pad, verbose=0).reshape(-1)


def get_model_info() -> dict[str, Any]:
    """Метаданные модели для API и веб-интерфейса."""
    _ensure_loaded()
    val_accuracy = None
    metrics_path = os.path.join(BASE_DIR, METRICS_PATH)
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            line = f.readline().strip()
            if line.startswith("val_accuracy="):
                val_accuracy = float(line.split("=", 1)[1])
    except FileNotFoundError:
        pass

    model_path = os.path.join(BASE_DIR, MODEL_PATH)
    trained_at = None
    try:
        trained_at = datetime.fromtimestamp(os.path.getmtime(model_path))
    except OSError:
        pass

    return {
        "architecture": "Embedding + LSTM",
        "max_words": MAX_WORDS,
        "max_len": MAX_LEN,
        "val_accuracy": val_accuracy,
        "trained_at": trained_at.isoformat() if trained_at else None,
        "threshold": 0.5,
    }


def predict_text(text: str) -> dict[str, Any]:
    """Предсказание для одного текста."""
    cleaned = clean_text(text)
    proba = float(_predict_proba([cleaned])[0])
    label = int(proba >= 0.5)
    return {
        "label": label,
        "label_name": _label_name(label),
        "probability": proba,
        "cleaned_text": cleaned,
    }


def predict_texts(texts: list[str]) -> list[dict[str, Any]]:
    """Пакетное предсказание для списка текстов."""
    if not texts:
        return []
    cleaned_list = [clean_text(t) for t in texts]
    proba = _predict_proba(cleaned_list)
    results = []
    for original, cleaned, p in zip(texts, cleaned_list, proba):
        label = int(p >= 0.5)
        results.append(
            {
                "text": original,
                "label": label,
                "label_name": _label_name(label),
                "probability": float(p),
                "cleaned_text": cleaned,
            }
        )
    return results


def predict_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Предсказания для DataFrame с колонкой text. Возвращает DF и сводку."""
    if "text" not in df.columns:
        raise ValueError("В DataFrame не найдена колонка 'text'.")

    texts = df["text"].astype(str).tolist()
    cleaned_list = [clean_text(t) for t in texts]
    proba = _predict_proba(cleaned_list)
    preds = (proba >= 0.5).astype(int)

    out = df.copy()
    out["prediction"] = preds
    out["probability"] = proba

    total = len(preds)
    disaster = int(preds.sum())
    safe = total - disaster
    disaster_pct = round(disaster / total * 100, 1) if total else 0.0
    safe_pct = round(safe / total * 100, 1) if total else 0.0

    summary = {
        "total": total,
        "disaster": disaster,
        "safe": safe,
        "disaster_pct": disaster_pct,
        "safe_pct": safe_pct,
    }
    return out, summary
