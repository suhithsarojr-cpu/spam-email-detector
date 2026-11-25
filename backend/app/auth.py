"""API-key authentication for up to MAX_USERS people, plus a simple per-key rate limit."""
import hashlib
import secrets
import time
from collections import defaultdict, deque

from fastapi import Depends, Header, HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from . import config
from .database import get_db
from .models import ApiUser

_hits: dict[int, deque] = defaultdict(deque)  # in-memory; resets on restart (fine for a demo)


class UserLimitReached(Exception):
    pass


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def create_user(db: Session, name: str) -> tuple[ApiUser, str]:
    active = db.scalar(select(func.count()).select_from(ApiUser).where(ApiUser.is_active.is_(True))) or 0
    if active >= config.MAX_USERS:
        raise UserLimitReached(f"Limit of {config.MAX_USERS} active users reached. Revoke one first.")
    key = "spk_" + secrets.token_urlsafe(32)
    user = ApiUser(name=name.strip()[:100], key_hash=hash_key(key), key_prefix=key[:8])
    db.add(user)
    db.commit()
    db.refresh(user)
    return user, key  # the plain key is returned ONCE and never stored


def require_api_key(x_api_key: str | None = Header(None), db: Session = Depends(get_db)) -> ApiUser:
    if not x_api_key:
        raise HTTPException(401, "Missing API key. Send it in the X-API-Key header.")
    user = db.scalar(select(ApiUser).where(ApiUser.key_hash == hash_key(x_api_key),
                                           ApiUser.is_active.is_(True)))
    if not user:
        raise HTTPException(401, "Invalid or revoked API key.")
    now, q = time.monotonic(), _hits[user.id]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= config.RATE_LIMIT_PER_MIN:
        raise HTTPException(429, f"Rate limit: {config.RATE_LIMIT_PER_MIN} requests per minute.")
    q.append(now)
    return user
