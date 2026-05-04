from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BatchPredictionRequest(BaseModel):
    items: list[PredictionRequest] = Field(..., min_length=1)


class PredictionResponse(BaseModel):
    label: str
    confidence: float
    probabilities: dict[str, float]


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


class ArticleAnalysisRequest(BaseModel):
    text: str = Field(..., min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class ConversationRequest(BaseModel):
    query: str = Field(..., min_length=1)
    context: dict[str, Any] = Field(default_factory=dict)


class ArticleAnalysisResponse(BaseModel):
    text: str
    classification: dict[str, Any]
    sentiment: dict[str, Any]
    entities: list[dict[str, Any]]
    relationships: list[dict[str, Any]]
    topics: list[dict[str, Any]]
    summary: str
    keywords: list[str]
    language: dict[str, Any]
    translation: dict[str, Any]
    related_articles: list[dict[str, Any]]


class SearchResponse(BaseModel):
    query: str
    results: list[dict[str, Any]]


class ConversationResponse(BaseModel):
    intent: str
    answer: str
    entities: dict[str, Any]
    results: dict[str, Any]
