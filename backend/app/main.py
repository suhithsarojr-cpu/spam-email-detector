"""FastAPI application: routes only; DB access in crud.py, inference in ml_service.py."""
import logging
from contextlib import asynccontextmanager
from datetime import timezone
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from . import config, content_checks, crud, schemas
from .auth import require_api_key
from .models import ApiUser
from .database import get_db, init_db
from .ml_service import ModelNotAvailable, SpamModel, get_model_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("spam_detector")  # note: email text is never logged

DEMO_NOTICE = ("Educational demonstration only. Not a production-grade email security system; "
               "predictions can be wrong.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    get_model_service().load()
    yield


app = FastAPI(
    title="Spam Email Detection API",
    description=DEMO_NOTICE,
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,  # local development origins; restrict before any deployment
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "X-API-Key"],
)


@app.get("/health")
def health(model: SpamModel = Depends(get_model_service)):
    return {"status": "ok", "model_loaded": model.available}


@app.post("/api/predict", response_model=schemas.PredictResponse)
def predict(req: schemas.PredictRequest, db: Session = Depends(get_db),
            model: SpamModel = Depends(get_model_service), user: ApiUser = Depends(require_api_key)):
    try:
        result = model.predict(req.subject, req.body)
    except ModelNotAvailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    checks = content_checks.analyze(req.subject, req.body)
    p_spam = result["probabilities"].get("spam", 0.0)
    risk, verdict = content_checks.combine(p_spam, checks["rule_score"])
    save = req.save_message and config.STORE_EMAIL_TEXT
    row = crud.create_prediction(
        db, user_id=user.id,
        subject=req.subject if save else "(not stored)",
        body=req.body if save else "",
        label=result["label"],
        confidence=result["confidence"],
        content_saved=save,
    )
    created = row.created_at if row.created_at.tzinfo else row.created_at.replace(tzinfo=timezone.utc)
    return schemas.PredictResponse(
        risk_score=risk, verdict=verdict, findings=checks["findings"],
        id=row.id,
        label=result["label"],
        is_spam=result["label"] == "spam",
        confidence=round(result["confidence"], 4),
        probabilities={k: round(v, 4) for k, v in result["probabilities"].items()},
        explanation=schemas.Explanation(**result["explanation"]),
        saved=save,
        created_at=created,
    )


@app.get("/api/history", response_model=schemas.HistoryPage)
def history(page: int = Query(1, ge=1), page_size: int = Query(10, ge=1, le=100),
            q: Optional[str] = Query(None, max_length=200),
            label: Optional[str] = Query(None, pattern="^(spam|ham)$"),
            db: Session = Depends(get_db), user: ApiUser = Depends(require_api_key)):
    items, total, pages = crud.list_predictions(db, user_id=user.id, page=page, page_size=page_size, q=q, label=label)
    return schemas.HistoryPage(items=items, total=total, page=page, page_size=page_size, pages=pages)


@app.delete("/api/history/{prediction_id}", status_code=204)
def delete_one(prediction_id: int, db: Session = Depends(get_db), user: ApiUser = Depends(require_api_key)):
    if not crud.delete_prediction(db, user.id, prediction_id):
        raise HTTPException(status_code=404, detail="Prediction not found.")


@app.delete("/api/history")
def delete_history(db: Session = Depends(get_db), user: ApiUser = Depends(require_api_key)):
    return {"deleted": crud.delete_all(db, user.id)}


@app.get("/api/stats", response_model=schemas.StatsResponse)
def stats(db: Session = Depends(get_db), model: SpamModel = Depends(get_model_service),
          user: ApiUser = Depends(require_api_key)):
    c = crud.get_counts(db, user.id)
    return schemas.StatsResponse(
        total_predictions=c["total"], spam_count=c["spam"], ham_count=c["ham"],
        spam_rate=c["spam_rate"], model_available=model.available,
        metrics=model.metadata.get("metrics"),
    )


@app.get("/api/model-info", response_model=schemas.ModelInfo)
def model_info(model: SpamModel = Depends(get_model_service)):
    notes = [
        DEMO_NOTICE,
        "Metrics were computed on a held-out test set during the actual training run.",
        "Metrics from the bundled synthetic sample dataset are NOT indicative of real-world performance.",
    ]
    meta = model.metadata
    return schemas.ModelInfo(
        model_available=model.available,
        model_type=meta.get("model_type"),
        trained_at=meta.get("trained_at"),
        dataset=meta.get("dataset"),
        metrics=meta.get("metrics"),
        notes=notes,
    )


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/app/")


if config.FRONTEND_DIR.exists():
    app.mount("/app", StaticFiles(directory=config.FRONTEND_DIR, html=True), name="frontend")
