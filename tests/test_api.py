import json
import sqlite3
import time

import pytest
from fastapi.testclient import TestClient

from banditlab.api import create_app
from banditlab.bandit import ACTIONS, CONTEXTS
from banditlab.service import DecisionService


@pytest.fixture
def model(tmp_path):
    path = tmp_path / "policy.json"
    path.write_text(
        json.dumps(
            {
                "contexts": list(CONTEXTS),
                "actions": list(ACTIONS),
                "alpha": [[90, 1, 1], [1, 90, 1]],
                "beta": [[10, 50, 50], [50, 10, 50]],
                "version": "test-v1",
                "training_kind": "semi_synthetic",
            }
        )
    )
    return path


@pytest.fixture
def client(tmp_path, model):
    with TestClient(create_app(tmp_path / "state.sqlite", model)) as api:
        yield api


def eligible(**overrides):
    return {"contact": "cellular", "eligible": True, "consent": True, "mode": "greedy", **overrides}


@pytest.mark.parametrize(
    "payload,reason",
    [
        (eligible(eligible=False), "ineligible"),
        (eligible(consent=False), "no_consent"),
        (eligible(contact="unknown"), "unknown_context"),
    ],
)
def test_withheld_recommendations_are_not_stored(client, payload, reason):
    response = client.post("/recommend", json=payload)
    assert response.status_code == 200
    assert response.json()["reason"] == reason
    assert response.json()["action"] is None
    assert client.get("/metrics").json()["decisions"] == 0


def test_request_validation_does_not_assume_consent(client):
    for payload in [
        eligible(consent="yes"),
        eligible(age=50),
        eligible(contact="sms"),
        {"contact": "cellular", "eligible": True},
    ]:
        assert client.post("/recommend", json=payload).status_code == 422


def test_feedback_is_linked_and_idempotent(client):
    result = client.post("/recommend", json=eligible()).json()
    assert result["action"] == "short_message"
    event = {"decision_id": result["decision_id"], "reward": 1}
    assert client.post("/feedback", json=event).json()["status"] == "recorded"
    assert client.post("/feedback", json=event).json()["status"] == "already_recorded"
    assert client.post("/feedback", json={**event, "reward": 0}).status_code == 409
    assert client.post("/feedback", json={**event, "reward": True}).status_code == 422
    assert client.post("/feedback", json={**event, "reward": 2}).status_code == 422
    assert (
        client.post(
            "/feedback", json={**event, "decision_id": "00000000-0000-0000-0000-000000000000"}
        ).status_code
        == 404
    )
    metrics = client.get("/metrics").json()
    assert metrics["conversions"] == metrics["feedback_received"] == 1
    assert metrics["pending_feedback"] == 0


def test_restart_preserves_feedback_and_version_guard(tmp_path, model):
    path = tmp_path / "state.sqlite"
    service = DecisionService(path, model)
    result = service.recommend(**eligible())
    service.feedback(result["decision_id"], 1)
    restarted = DecisionService(path, model)
    assert restarted.metrics()["conversions"] == 1
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT alpha FROM posterior WHERE context=0 AND action=0").fetchone()[0] == 91
    changed = json.loads(model.read_text())
    changed["version"] = "test-v2"
    model.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="version changed"):
        DecisionService(path, model)


def test_retention_removes_event_but_preserves_aggregate_learning(tmp_path, model):
    path = tmp_path / "state.sqlite"
    service = DecisionService(path, model)
    result = service.recommend(**eligible())
    service.feedback(result["decision_id"], 1)
    with sqlite3.connect(path) as db:
        db.execute("UPDATE decisions SET created_at=?", (time.time() - 31 * 86400,))
    assert service.purge_old_events(days=30) == 1
    assert service.metrics()["decisions"] == 0
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT alpha FROM posterior WHERE context=0 AND action=0").fetchone()[0] == 91
