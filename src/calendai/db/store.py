"""SQLite persistence layer.

One Store per process (or per eval scenario). Connections are opened eagerly
and must be closed explicitly - the eval runner relies on close() so temp DB
files can be deleted on Windows.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from typing import Any

from calendai.auth.tokens import TokenCipher
from calendai.core.clock import Clock, SystemClock
from calendai.core.models import FactType, MemoryFact, User


def _iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat()


def _parse_dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


class Store:
    def __init__(
        self,
        db_path: str | Path,
        clock: Clock | None = None,
        *,
        check_same_thread: bool = True,
    ) -> None:
        self.db_path = Path(db_path)
        self._clock = clock or SystemClock()
        if str(self.db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False lets the FastAPI threadpool share one
        # connection; the web app serializes writes with a process lock.
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=check_same_thread)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()

    def _init_schema(self) -> None:
        schema = resources.files("calendai.db").joinpath("schema.sql").read_text(encoding="utf-8")
        self._conn.executescript(schema)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @property
    def conn(self) -> sqlite3.Connection:
        return self._conn

    # -- users ----------------------------------------------------------

    def upsert_user(self, user: User) -> User:
        now = _iso(self._clock.now())
        self._conn.execute(
            """INSERT INTO users (id, email, display_name, timezone, created_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                   email = excluded.email,
                   display_name = excluded.display_name,
                   timezone = excluded.timezone""",
            (user.id, user.email, user.display_name, user.timezone, now),
        )
        self._conn.commit()
        return self.get_user(user.id)  # type: ignore[return-value]

    def get_user(self, user_id: str) -> User | None:
        row = self._conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return self._row_to_user(row) if row else None

    def get_user_by_email(self, email: str) -> User | None:
        row = self._conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        return self._row_to_user(row) if row else None

    def list_users(self) -> list[User]:
        rows = self._conn.execute("SELECT * FROM users ORDER BY created_at").fetchall()
        return [self._row_to_user(r) for r in rows]

    @staticmethod
    def _row_to_user(row: sqlite3.Row) -> User:
        return User(
            id=row["id"],
            email=row["email"],
            display_name=row["display_name"],
            timezone=row["timezone"],
            created_at=_parse_dt(row["created_at"]),
        )

    # -- memory facts ---------------------------------------------------

    def upsert_fact(self, fact: MemoryFact) -> MemoryFact:
        """Insert a fact; if an active fact with the same (user, key) exists,
        the old one is deactivated and linked via superseded_by.

        Atomic: serialization happens before any mutation, and the
        deactivate -> insert -> link sequence runs in one transaction, so a
        failure can never leave the user without an active fact.
        """
        value_json = json.dumps(fact.value, sort_keys=True)  # fail BEFORE mutating
        now = _iso(self._clock.now())
        with self._conn:  # transaction: commits on success, rolls back on error
            old = self._conn.execute(
                "SELECT id FROM memory_facts WHERE user_id = ? AND key = ? AND active = 1",
                (fact.user_id, fact.key),
            ).fetchone()
            if old:
                self._conn.execute(
                    "UPDATE memory_facts SET active = 0, updated_at = ? WHERE id = ?",
                    (now, old["id"]),
                )
            cur = self._conn.execute(
                """INSERT INTO memory_facts
                   (user_id, fact_type, key, value, statement, provenance, active,
                    created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)""",
                (
                    fact.user_id,
                    fact.fact_type.value,
                    fact.key,
                    value_json,
                    fact.statement,
                    fact.provenance,
                    now,
                    now,
                ),
            )
            new_id = cur.lastrowid
            if old:
                self._conn.execute(
                    "UPDATE memory_facts SET superseded_by = ? WHERE id = ?", (new_id, old["id"])
                )
        return self.get_fact(new_id)  # type: ignore[arg-type,return-value]

    def get_fact(self, fact_id: int) -> MemoryFact | None:
        row = self._conn.execute("SELECT * FROM memory_facts WHERE id = ?", (fact_id,)).fetchone()
        return self._row_to_fact(row) if row else None

    def list_facts(
        self, user_id: str, fact_type: FactType | None = None, active_only: bool = True
    ) -> list[MemoryFact]:
        query = "SELECT * FROM memory_facts WHERE user_id = ?"
        params: list[object] = [user_id]
        if fact_type is not None:
            query += " AND fact_type = ?"
            params.append(fact_type.value)
        if active_only:
            query += " AND active = 1"
        query += " ORDER BY created_at, id"  # id tie-breaker: deterministic under FrozenClock
        rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_fact(r) for r in rows]

    def deactivate_fact(self, fact_id: int) -> None:
        now = _iso(self._clock.now())
        self._conn.execute(
            "UPDATE memory_facts SET active = 0, updated_at = ? WHERE id = ?", (now, fact_id)
        )
        self._conn.commit()

    @staticmethod
    def _row_to_fact(row: sqlite3.Row) -> MemoryFact:
        return MemoryFact(
            id=row["id"],
            user_id=row["user_id"],
            fact_type=FactType(row["fact_type"]),
            key=row["key"],
            value=json.loads(row["value"]),
            statement=row["statement"],
            provenance=row["provenance"],
            active=bool(row["active"]),
            created_at=_parse_dt(row["created_at"]),
            updated_at=_parse_dt(row["updated_at"]),
        )

    # -- sessions -------------------------------------------------------

    def create_session(self, token: str, user_id: str, expires_at: datetime | None = None) -> None:
        self._conn.execute(
            "INSERT INTO sessions (token, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
            (token, user_id, _iso(self._clock.now()), _iso(expires_at) if expires_at else None),
        )
        self._conn.commit()

    def get_session_user(self, token: str) -> User | None:
        row = self._conn.execute(
            "SELECT user_id, expires_at FROM sessions WHERE token = ?", (token,)
        ).fetchone()
        if not row:
            return None
        expires_at = _parse_dt(row["expires_at"])
        if expires_at and expires_at < self._clock.now():
            return None
        return self.get_user(row["user_id"])

    # -- messages -------------------------------------------------------

    def add_message(
        self, user_id: str, role: str, content: str, session_token: str | None = None
    ) -> None:
        self._conn.execute(
            """INSERT INTO messages (user_id, session_token, role, content, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (user_id, session_token, role, content, _iso(self._clock.now())),
        )
        self._conn.commit()

    def recent_messages(self, user_id: str, limit: int = 20) -> list[dict[str, str]]:
        rows = self._conn.execute(
            """SELECT role, content FROM messages WHERE user_id = ?
               ORDER BY id DESC LIMIT ?""",
            (user_id, limit),
        ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    # -- oauth tokens (encryption enforced by API shape) ----------------

    def save_oauth_token(
        self, user_id: str, payload: dict[str, Any], cipher: TokenCipher, provider: str = "google"
    ) -> None:
        """Persist an OAuth token payload, Fernet-encrypted. There is no
        public path that stores a plaintext token."""
        blob = cipher.encrypt(payload)
        self._conn.execute(
            """INSERT INTO oauth_tokens (user_id, provider, token_blob, updated_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(user_id) DO UPDATE SET
                   token_blob = excluded.token_blob,
                   provider = excluded.provider,
                   updated_at = excluded.updated_at""",
            (user_id, provider, blob, _iso(self._clock.now())),
        )
        self._conn.commit()

    def load_oauth_token(self, user_id: str, cipher: TokenCipher) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT token_blob FROM oauth_tokens WHERE user_id = ?", (user_id,)
        ).fetchone()
        return cipher.decrypt(row["token_blob"]) if row else None
