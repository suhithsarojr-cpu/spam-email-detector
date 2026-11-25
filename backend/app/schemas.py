"""Pydantic request/response schemas."""
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from . import config


class PredictRequest(BaseModel):
    subject: str = Field("", max_length=config.MAX_SUBJECT_CHARS)
    body: str = Field("", max_length=config.MAX_BODY_CHARS)
    save_message: bool = Field(True, description="If false, the email text is not stored in history.")

    @field_validator("subject", "body")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip()

    @model_validator(mode="after")
    def _not_empty(self):
        if not self.subject and not self.body:
            raise ValueError("Provide an email subject, a message body, or both.")
        return self


class TermContribution(BaseModel):
    term: str
    weight: float


class Explanation(BaseModel):
    summary: str
    top_terms: list[TermContribution]
    disclaimer: str


class Finding(BaseModel):
    id: str
    label: str
    severity: str  # ok | warn | bad
    detail: str
    points: float


class PredictResponse(BaseModel):
    risk_score: int
    verdict: str
    findings: list[Finding]
    id: Optional[int]
    label: str
    is_spam: bool
    confidence: float
    probabilities: dict[str, float]
    explanation: Explanation
    saved: bool
    created_at: datetime


class HistoryItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    subject: str
    body: str
    label: str
    confidence: float
    content_saved: bool
    created_at: datetime


class HistoryPage(BaseModel):
    items: list[HistoryItem]
    total: int
    page: int
    page_size: int
    pages: int


class StatsResponse(BaseModel):
    total_predictions: int
    spam_count: int
    ham_count: int
    spam_rate: float
    model_available: bool
    metrics: Optional[dict[str, Any]] = None


class ModelInfo(BaseModel):
    model_available: bool
    model_type: Optional[str] = None
    trained_at: Optional[str] = None
    dataset: Optional[dict[str, Any]] = None
    metrics: Optional[dict[str, Any]] = None
    notes: list[str]
