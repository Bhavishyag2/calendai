from __future__ import annotations

from calendai.core.models import FactType, MemoryFact, User
from calendai.db.store import Store

ALICE = User(id="u_alice", email="alice@example.com", display_name="Alice")


def rule_fact(statement: str = "Never schedule meetings before 10:00 IST") -> MemoryFact:
    return MemoryFact(
        user_id=ALICE.id,
        fact_type=FactType.RULE,
        key="rule:no_meetings_before",
        value={"time": "10:00", "timezone": "Asia/Kolkata"},
        statement=statement,
        provenance="user said: I never take meetings before 10 AM",
    )


def test_user_roundtrip(store: Store):
    store.upsert_user(ALICE)
    fetched = store.get_user(ALICE.id)
    assert fetched is not None and fetched.email == ALICE.email
    assert store.get_user_by_email(ALICE.email).id == ALICE.id


def test_fact_upsert_supersedes_same_key(store: Store):
    store.upsert_user(ALICE)
    old = store.upsert_fact(rule_fact("Never before 10:00"))
    new = store.upsert_fact(rule_fact("Never before 09:00"))

    active = store.list_facts(ALICE.id)
    assert len(active) == 1
    assert active[0].statement == "Never before 09:00"

    all_facts = store.list_facts(ALICE.id, active_only=False)
    assert len(all_facts) == 2
    old_row = next(f for f in all_facts if f.id == old.id)
    assert old_row.active is False
    # supersession chain is recorded for provenance/audit
    row = store.conn.execute(
        "SELECT superseded_by FROM memory_facts WHERE id = ?", (old.id,)
    ).fetchone()
    assert row["superseded_by"] == new.id


def test_facts_filter_by_type(store: Store):
    store.upsert_user(ALICE)
    store.upsert_fact(rule_fact())
    store.upsert_fact(
        MemoryFact(
            user_id=ALICE.id,
            fact_type=FactType.CONTACT,
            key="contact:alex",
            value={"email": "alex@example.com"},
            statement="Alex is alex@example.com",
            provenance="user introduced Alex",
        )
    )
    assert len(store.list_facts(ALICE.id, FactType.RULE)) == 1
    assert len(store.list_facts(ALICE.id, FactType.CONTACT)) == 1


def test_persistence_across_restart(tmp_path, clock):
    """The 'stateful' requirement: facts must survive a process restart.

    Simulated by closing the Store (connection) and opening a fresh one on
    the same file — not just reusing in-process state.
    """
    db = tmp_path / "restart.db"
    first = Store(db, clock=clock)
    first.upsert_user(ALICE)
    first.upsert_fact(rule_fact())
    first.close()

    second = Store(db, clock=clock)
    try:
        facts = second.list_facts(ALICE.id)
        assert len(facts) == 1
        assert facts[0].value == {"time": "10:00", "timezone": "Asia/Kolkata"}
    finally:
        second.close()


def test_messages_recent_order_and_limit(store: Store):
    store.upsert_user(ALICE)
    for i in range(5):
        store.add_message(ALICE.id, "user" if i % 2 == 0 else "assistant", f"msg {i}")
    recent = store.recent_messages(ALICE.id, limit=3)
    assert [m["content"] for m in recent] == ["msg 2", "msg 3", "msg 4"]


def test_sessions(store: Store):
    store.upsert_user(ALICE)
    store.create_session("tok123", ALICE.id)
    assert store.get_session_user("tok123").id == ALICE.id
    assert store.get_session_user("nope") is None


def test_token_blob_roundtrip(store: Store):
    store.upsert_user(ALICE)
    store.save_token_blob(ALICE.id, b"encrypted-bytes")
    assert store.get_token_blob(ALICE.id) == b"encrypted-bytes"
    store.save_token_blob(ALICE.id, b"rotated")
    assert store.get_token_blob(ALICE.id) == b"rotated"
