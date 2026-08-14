"""
API Protocol definitions for LMMS-Eval Server.

This module contains all Pydantic models used for request/response
validation across the HTTP server and client.
"""

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Status of an evaluation job."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EvaluateRequest(BaseModel):
    """Request model for evaluation endpoint."""

    model: str = Field(..., description="Model name or path")
    tasks: list[str] = Field(..., description="List of task names to evaluate")
    model_args: dict[str, Any] | None = Field(default=None, description="Model arguments")
    num_fewshot: int | None = Field(default=None, description="Number of few-shot examples")
    batch_size: int | str | None = Field(default=None, description="Batch size")
    device: str | None = Field(default=None, description="Device to run on")
    limit: int | float | None = Field(default=None, description="Limit number of examples")
    gen_kwargs: str | None = Field(default=None, description="Generation kwargs")
    log_samples: bool = Field(default=True, description="Whether to log samples")
    predict_only: bool = Field(default=False, description="Only generate predictions")
    num_gpus: int = Field(default=1, description="Number of GPUs to use")
    output_dir: str | None = Field(default=None, description="Output directory for results")


class JobInfo(BaseModel):
    """Information about a job."""

    job_id: str
    status: JobStatus
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    request: EvaluateRequest
    result: dict[str, Any] | None = None
    error: str | None = None
    position_in_queue: int | None = None


class JobSubmitResponse(BaseModel):
    """Response when submitting a job."""

    job_id: str
    status: JobStatus
    position_in_queue: int
    message: str


class QueueStatusResponse(BaseModel):
    """Response for queue status."""

    queue_size: int
    running_job: str | None = None
    queued_jobs: list[str]
    completed_jobs: int
    failed_jobs: int


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    timestamp: str
    queue_size: int


class MergeRequest(BaseModel):
    """Request model for merging FSDP2 sharded checkpoints."""

    checkpoint_path: str = Field(..., description="Path to sharded checkpoint directory")
    output_path: str | None = Field(default=None, description="Output path for merged checkpoint")
    checkpoint_type: Literal["regular", "ema"] = Field(
        default="regular",
        description="Type of checkpoint to merge: 'regular' for main model weights, 'ema' for EMA weights",
    )


class MergeResponse(BaseModel):
    """Response model for checkpoint merge operations."""

    success: bool = Field(..., description="Whether the merge operation succeeded")
    message: str = Field(..., description="Detailed message about the merge operation")
    merged_path: str | None = Field(default=None, description="Path to the merged checkpoint if successful")
