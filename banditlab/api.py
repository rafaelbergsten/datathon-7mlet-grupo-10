"""FastAPI app that serves message recommendations and collects feedback."""

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from uuid import UUID

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, StrictBool

from banditlab.data import ROOT
from banditlab.service import DecisionService, FeedbackError


class CustomerContext(BaseModel):
    model_config = ConfigDict(extra="forbid")
    contact: Literal["cellular", "telephone", "unknown"] = "cellular"
    eligible: StrictBool
    consent: StrictBool
    mode: Literal["sample", "greedy"] = "sample"


class Feedback(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision_id: UUID
    reward: int = Field(strict=True, ge=0, le=1)


def create_app(database=None, model=None):
    @asynccontextmanager
    async def lifespan(app):
        options = {} if model is None else {"model": Path(model)}
        path = Path(database or os.environ.get("DATATHON_STATE_DB", ROOT / "runtime/decisions.sqlite"))
        app.state.service = DecisionService(path, **options)
        app.state.service.purge_old_events()
        app.state.requests = 0
        app.state.errors = 0
        app.state.total_latency_ms = 0.0
        yield

    app = FastAPI(
        title="Datathon MLET | Adaptive Messages",
        version="1.0.0",
        description="Chooses one of three messages per contact context with Thompson Sampling.",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def observe(request, call_next):
        start = time.perf_counter()
        try:
            response = await call_next(request)
            app.state.errors += int(response.status_code >= 400)
            return response
        except Exception:
            app.state.errors += 1
            raise
        finally:
            app.state.requests += 1
            app.state.total_latency_ms += (time.perf_counter() - start) * 1000

    @app.get("/health")
    def health():
        return {"status": "ok", "model_version": app.state.service.model["version"]}

    @app.post("/recommend")
    def recommend(customer: CustomerContext):
        return app.state.service.recommend(**customer.model_dump())

    @app.post("/feedback")
    def feedback(event: Feedback):
        try:
            return app.state.service.feedback(str(event.decision_id), event.reward)
        except FeedbackError as error:
            raise HTTPException(status_code=error.status, detail=str(error)) from error

    @app.get("/metrics")
    def metrics():
        requests = app.state.requests
        return {
            **app.state.service.metrics(),
            "http_requests_since_start": requests,
            "http_errors_since_start": app.state.errors,
            "mean_latency_ms_since_start": app.state.total_latency_ms / requests if requests else 0.0,
        }

    return app


app = create_app()
