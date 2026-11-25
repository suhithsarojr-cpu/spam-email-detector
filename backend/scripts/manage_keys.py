"""Manage API keys (max 50 active users).
  python -m scripts.manage_keys create "Alice"
  python -m scripts.manage_keys list
  python -m scripts.manage_keys revoke 3
  python -m scripts.manage_keys bulk 50          # creates user-01 ... user-50, prints keys once
"""
import sys

from sqlalchemy import select

from app.auth import UserLimitReached, create_user
from app.database import SessionLocal, init_db
from app.models import ApiUser


def main(argv):
    init_db()
    cmd = argv[1] if len(argv) > 1 else ""
    with SessionLocal() as db:
        if cmd == "create" and len(argv) > 2:
            u, key = create_user(db, argv[2])
            print(f"User #{u.id} '{u.name}'\nAPI key (shown once, store it safely): {key}")
        elif cmd == "bulk" and len(argv) > 2:
            for i in range(1, int(argv[2]) + 1):
                try:
                    u, key = create_user(db, f"user-{i:02d}")
                    print(f"{u.id},{u.name},{key}")
                except UserLimitReached as e:
                    print(e); break
        elif cmd == "list":
            for u in db.scalars(select(ApiUser).order_by(ApiUser.id)):
                print(f"{u.id:>3}  {u.name:<20} {u.key_prefix}...  {'active' if u.is_active else 'revoked'}")
        elif cmd == "revoke" and len(argv) > 2:
            u = db.get(ApiUser, int(argv[2]))
            if not u: sys.exit("No such user.")
            u.is_active = False; db.commit(); print(f"Revoked #{u.id}")
        else:
            print(__doc__)


if __name__ == "__main__":
    main(sys.argv)
