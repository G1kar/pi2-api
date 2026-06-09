"""REST API на FastAPI для модели классификации твитов о бедствиях."""

from __future__ import annotations

import io
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import ml_service

app = FastAPI(
    title="Twitter Disaster Classifier API",
    description="API для модели бинарной классификации твитов (бедствие / нет).",
    version="1.0.0",
)


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Текст твита")


class PredictResponse(BaseModel):
    label: int
    label_name: str
    probability: float
    cleaned_text: str


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, description="Список текстов")


class BatchPredictItem(BaseModel):
    text: str
    label: int
    label_name: str
    probability: float
    cleaned_text: str


class BatchPredictResponse(BaseModel):
    count: int
    items: list[BatchPredictItem]


class BatchSummary(BaseModel):
    total: int
    disaster: int
    safe: int
    disaster_pct: float
    safe_pct: float


class CsvPredictResponse(BaseModel):
    summary: BatchSummary
    items: list[dict[str, Any]]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


@app.get("/health", response_model=HealthResponse, tags=["service"])
def health() -> HealthResponse:
    """Проверка доступности сервиса."""
    try:
        ml_service.get_model_info()
        return HealthResponse(status="ok", model_loaded=True)
    except Exception:
        return HealthResponse(status="degraded", model_loaded=False)


@app.get("/model/info", tags=["model"])
def model_info() -> dict[str, Any]:
    """Информация о модели: архитектура, метрики, дата обучения."""
    try:
        return ml_service.get_model_info()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Модель недоступна: {e}") from e


@app.post("/predict", response_model=PredictResponse, tags=["predict"])
def predict_one(body: PredictRequest) -> PredictResponse:
    """Предсказание для одного твита."""
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Поле text не может быть пустым.")
    try:
        result = ml_service.predict_text(text)
        return PredictResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["predict"])
def predict_batch(body: BatchPredictRequest) -> BatchPredictResponse:
    """Пакетное предсказание для списка текстов."""
    texts = [t.strip() for t in body.texts if t.strip()]
    if not texts:
        raise HTTPException(status_code=400, detail="Список texts пуст или содержит только пробелы.")
    try:
        items = ml_service.predict_texts(texts)
        return BatchPredictResponse(
            count=len(items),
            items=[BatchPredictItem(**item) for item in items],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/predict/csv", tags=["predict"])
async def predict_csv(
    file: UploadFile = File(..., description="CSV-файл с колонкой text"),
    download: bool = False,
):
    """
    Предсказания для CSV-файла.

    По умолчанию возвращает JSON со сводкой и записями.
    При `download=true` — CSV-файл с колонками prediction и probability.
    """
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Ожидается файл с расширением .csv")

    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        result_df, summary = ml_service.predict_dataframe(df)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки CSV: {e}") from e

    if download:
        buf = io.StringIO()
        result_df.to_csv(buf, index=False)
        buf.seek(0)
        out_name = file.filename.replace(".csv", "_predictions.csv")
        return StreamingResponse(
            iter([buf.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{out_name}"'},
        )

    records = result_df.where(pd.notnull(result_df), None).to_dict(orient="records")
    return CsvPredictResponse(summary=BatchSummary(**summary), items=records)
