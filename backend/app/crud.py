"""Database access functions. Every history query is scoped to one user."""
import math
from typing import Optional

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from .models import Prediction


def create_prediction(db: Session, *, user_id: int, subject: str, body: str, label: str,
                      confidence: float, content_saved: bool) -> Prediction:
    row = Prediction(user_id=user_id, subject=subject, body=body, label=label,
                     confidence=confidence, content_saved=content_saved)
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def list_predictions(db: Session, *, user_id: int, page: int, page_size: int,
                     q: Optional[str] = None, label: Optional[str] = None):
    conds = [Prediction.user_id == user_id]
    if label in ("spam", "ham"):
        conds.append(Prediction.label == label)
    if q:
        like = f"%{q.strip()}%"
        conds.append(or_(Prediction.subject.ilike(like), Prediction.body.ilike(like)))
    total = db.scalar(select(func.count()).select_from(Prediction).where(*conds)) or 0
    items = db.scalars(select(Prediction).where(*conds)
                       .order_by(Prediction.created_at.desc(), Prediction.id.desc())
                       .offset((page - 1) * page_size).limit(page_size)).all()
    return items, total, max(1, math.ceil(total / page_size))


def get_counts(db: Session, user_id: int) -> dict:
    rows = dict(db.execute(select(Prediction.label, func.count())
                           .where(Prediction.user_id == user_id).group_by(Prediction.label)).all())
    spam, ham = rows.get("spam", 0), rows.get("ham", 0)
    total = spam + ham
    return {"total": total, "spam": spam, "ham": ham, "spam_rate": round(spam / total, 4) if total else 0.0}


def delete_prediction(db: Session, user_id: int, prediction_id: int) -> bool:
    row = db.get(Prediction, prediction_id)
    if not row or row.user_id != user_id:
        return False
    db.delete(row)
    db.commit()
    return True


def delete_all(db: Session, user_id: int) -> int:
    n = db.query(Prediction).filter(Prediction.user_id == user_id).delete()
    db.commit()
    return n
