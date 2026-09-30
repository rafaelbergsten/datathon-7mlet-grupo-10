"""SQLite-backed recommendation and feedback service."""

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from banditlab.bandit import ACTIONS, CONTEXTS, MESSAGES, ThompsonSampling
from banditlab.data import ROOT


class FeedbackError(ValueError):
    def __init__(self, message, status):
        super().__init__(message)
        self.status = status


class DecisionService:
    def __init__(self, database: Path, model: Path = ROOT / "models/policy.json", seed=None):
        if not model.exists():
            raise FileNotFoundError("Missing policy artifact. Run python -m banditlab.experiment")
        self.model = json.loads(model.read_text())
        if self.model["contexts"] != list(CONTEXTS) or self.model["actions"] != list(ACTIONS):
            raise ValueError("Model schema does not match the service")
        ThompsonSampling(alpha=self.model["alpha"], beta=self.model["beta"])
        self.database = Path(database)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self.rng = np.random.default_rng(seed)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS posterior (
                    context INTEGER, action INTEGER, alpha REAL NOT NULL, beta REAL NOT NULL,
                    PRIMARY KEY (context, action));
                CREATE TABLE IF NOT EXISTS decisions (
                    id TEXT PRIMARY KEY, context INTEGER NOT NULL, action INTEGER NOT NULL,
                    version TEXT NOT NULL, mode TEXT NOT NULL, created_at REAL NOT NULL,
                    reward INTEGER CHECK(reward IN (0, 1)));
            """)
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT value FROM metadata WHERE key='model_version'").fetchone()
            if row is not None and row[0] != self.model["version"]:
                raise ValueError("Model version changed. Use a new state database for explicit rollout.")
            if row is None:
                db.execute("INSERT INTO metadata VALUES ('model_version', ?)", (self.model["version"],))
                for c in range(2):
                    for a in range(3):
                        db.execute(
                            "INSERT INTO posterior VALUES (?, ?, ?, ?)",
                            (c, a, self.model["alpha"][c][a], self.model["beta"][c][a]),
                        )

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.database, timeout=15)
        try:
            with db:
                yield db
        finally:
            db.close()

    def recommend(self, contact, eligible, consent, mode="sample"):
        if type(eligible) is not bool or type(consent) is not bool:
            raise ValueError("Eligibility and consent must be booleans")
        if contact not in (*CONTEXTS, "unknown"):
            raise ValueError("Unsupported contact context")
        if mode not in ("sample", "greedy"):
            raise ValueError("Unsupported decision mode")
        blocked = (
            "ineligible"
            if not eligible
            else "no_consent"
            if not consent
            else "unknown_context"
            if contact == "unknown"
            else None
        )
        if blocked:
            return {
                "status": "withheld",
                "reason": blocked,
                "action": None,
                "message": None,
                "decision_id": None,
                "model_version": self.model["version"],
                "training_kind": "semi_synthetic",
            }
        c = CONTEXTS.index(contact)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            rows = db.execute(
                "SELECT alpha, beta FROM posterior WHERE context=? ORDER BY action", (c,)
            ).fetchall()
            alpha, beta = np.array(rows).T
            means = alpha / (alpha + beta)
            a = int(np.argmax(means if mode == "greedy" else self.rng.beta(alpha, beta)))
            decision_id = str(uuid.uuid4())
            db.execute(
                "INSERT INTO decisions VALUES (?, ?, ?, ?, ?, ?, NULL)",
                (decision_id, c, a, self.model["version"], mode, time.time()),
            )
        return {
            "status": "recommended",
            "decision_id": decision_id,
            "action": ACTIONS[a],
            "message": MESSAGES[ACTIONS[a]],
            "contact": contact,
            "mode": mode,
            "posterior_mean": float(means[a]),
            "model_version": self.model["version"],
            "training_kind": "semi_synthetic",
        }

    def feedback(self, decision_id, reward):
        if type(reward) is not int or reward not in (0, 1):
            raise ValueError("Reward must be integer zero or one")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT context, action, reward FROM decisions WHERE id=?", (str(decision_id),)
            ).fetchone()
            if row is None:
                raise FeedbackError("Unknown or expired decision", 404)
            c, a, old_reward = row
            if old_reward is not None:
                if old_reward != reward:
                    raise FeedbackError("Conflicting feedback for an already finalized decision", 409)
                return {"status": "already_recorded"}
            db.execute("UPDATE decisions SET reward=? WHERE id=?", (reward, str(decision_id)))
            db.execute(
                "UPDATE posterior SET alpha=alpha+?, beta=beta+? WHERE context=? AND action=?",
                (reward, 1 - reward, c, a),
            )
        return {"status": "recorded"}

    def metrics(self):
        with self.connect() as db:
            row = db.execute(
                "SELECT COUNT(*), COUNT(reward), COALESCE(SUM(reward), 0) FROM decisions"
            ).fetchone()
            by_action = db.execute("SELECT action, COUNT(*) FROM decisions GROUP BY action").fetchall()
        total, feedback, conversions = row
        return {
            "decisions": total,
            "feedback_received": feedback,
            "conversions": conversions,
            "pending_feedback": total - feedback,
            "conversion_among_finalized": conversions / feedback if feedback else None,
            "allocation": {ACTIONS[a]: n for a, n in by_action},
            "window": "retained_events_only",
            "model_version": self.model["version"],
        }

    def purge_old_events(self, days=30):
        """Delete decision events older than the retention window; posterior counts are kept."""
        if days < 1:
            raise ValueError("Retention must be at least one day")
        with self.connect() as db:
            result = db.execute("DELETE FROM decisions WHERE created_at < ?", (time.time() - days * 86400,))
            return result.rowcount
