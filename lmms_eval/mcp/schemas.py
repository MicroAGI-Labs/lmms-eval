from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TaskInfo(BaseModel):
    """Information about a single evaluation task."""

    name: str
    type: str  # "task", "group", "tag"
    yaml_path: str | None = None
    output_type: str | None = None  # "generate_until", "loglikelihood", etc.


class TaskListResponse(BaseModel):
    """Response for list_tasks tool."""

    tasks: list[TaskInfo]
    total: int
    query: str | None = None


class ModelInfo(BaseModel):
    """Information about a single model backend."""

    model_id: str
    has_chat: bool
    has_simple: bool
    aliases: list[str] = Field(default_factory=list)


class ModelListResponse(BaseModel):
    """Response for list_models tool."""

    models: list[ModelInfo]
    total: int


class EvalRunSubmitted(BaseModel):
    """Response when an evaluation run is submitted."""

    run_id: str
    status: str  # "running" or "queued"
    position_in_queue: int | None = None
    message: str


class EvalRunStatus(BaseModel):
    """Response for get_run_status tool."""

    run_id: str
    status: str  # "queued", "running", "completed", "failed", "cancelled"
    created_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    position_in_queue: int | None = None
    error: str | None = None


class EvalRunResult(BaseModel):
    """Response for get_run_result tool."""

    run_id: str
    status: str
    results: dict[str, Any] | None = None
    error: str | None = None
