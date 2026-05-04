from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response

from nlp_pipeline.inference import InferenceService
from nlp_pipeline.logging_utils import configure_logging
from nlp_pipeline.monitoring import RequestTimer, metrics_payload, record_prediction, record_request
from nlp_pipeline.schemas import (
    ArticleAnalysisRequest,
    ArticleAnalysisResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ConversationRequest,
    ConversationResponse,
    PredictionRequest,
    PredictionResponse,
    SearchRequest,
    SearchResponse,
)

configure_logging()
LOGGER = logging.getLogger(__name__)
app = FastAPI(title="NewsBot Intelligence API", version="1.0.0")


@lru_cache
def get_service() -> InferenceService:
    model_path = Path("artifacts/model.joblib")
    corpus_path = Path("artifacts/news_corpus.json")
    if not model_path.exists():
        raise RuntimeError("Model artifact not found. Run training first.")
    return InferenceService(model_path=model_path, corpus_path=corpus_path)


@app.get("/health")
def health() -> dict:
    try:
        get_service()
        return {"status": "ok"}
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/metrics")
def metrics() -> Response:
    payload, content_type = metrics_payload()
    return Response(content=payload, media_type=content_type)


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    with RequestTimer("/predict"):
        try:
            prediction = get_service().predict_one(request.text)
            record_prediction(prediction["label"])
            record_request("/predict", "success")
            return PredictionResponse(**prediction)
        except Exception as exc:  # pragma: no cover
            LOGGER.exception("Prediction failed")
            record_request("/predict", "error")
            raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    with RequestTimer("/predict/batch"):
        try:
            predictions = get_service().predict_many([item.text for item in request.items])
            for prediction in predictions:
                record_prediction(prediction["label"])
            record_request("/predict/batch", "success")
            return BatchPredictionResponse(
                predictions=[PredictionResponse(**prediction) for prediction in predictions]
            )
        except Exception as exc:  # pragma: no cover
            LOGGER.exception("Batch prediction failed")
            record_request("/predict/batch", "error")
            raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/analyze", response_model=ArticleAnalysisResponse)
def analyze_article(request: ArticleAnalysisRequest) -> ArticleAnalysisResponse:
    with RequestTimer("/analyze"):
        try:
            analysis = get_service().analyze_article(request.text, request.metadata)
            record_prediction(analysis["classification"]["label"])
            record_request("/analyze", "success")
            return ArticleAnalysisResponse(**analysis)
        except Exception as exc:  # pragma: no cover
            LOGGER.exception("Analysis failed")
            record_request("/analyze", "error")
            raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    with RequestTimer("/search"):
        try:
            results = get_service().search(request.query, top_k=request.top_k)
            record_request("/search", "success")
            return SearchResponse(query=request.query, results=results)
        except Exception as exc:  # pragma: no cover
            LOGGER.exception("Search failed")
            record_request("/search", "error")
            raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/chat", response_model=ConversationResponse)
def chat(request: ConversationRequest) -> ConversationResponse:
    with RequestTimer("/chat"):
        try:
            response = get_service().chat(request.query, context=request.context)
            record_request("/chat", "success")
            return ConversationResponse(**response)
        except Exception as exc:  # pragma: no cover
            LOGGER.exception("Conversation failed")
            record_request("/chat", "error")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
