from dataclasses import dataclass, field
from typing import Any

# Configuration for retry logic
DEFAULT_NUM_RETRIES = 5
DEFAULT_RETRY_DELAY = 10  # seconds


@dataclass
class ServerConfig:
    """Configuration for judge models"""

    model_name: str
    temperature: float = 0.0
    max_tokens: int = 1024
    top_p: float | None = None
    timeout: int = 60
    num_retries: int = DEFAULT_NUM_RETRIES
    retry_delay: float = DEFAULT_RETRY_DELAY
    max_concurrent: int = 10  # Maximum concurrent requests

    # Additional config for specific judge tasks
    system_prompt: str | None = None
    response_format: str | None = None  # 'json' or 'text'

    # Judge-specific parameters
    judge_type: str = "general"  # 'general', 'binary', 'score', 'comparative'
    output_format: str | None = None  # For binary: '0/1' or 'yes/no'
    score_range: tuple[float, float] | None = None  # For scoring judges
    evaluation_criteria: dict[str, Any] | None = None  # Custom evaluation criteria


@dataclass
class Request:
    """Standard request format for judge evaluation"""

    messages: list[dict[str, Any]]
    images: list[str | bytes] | None = None  # Image paths or base64 encoded
    config: ServerConfig | None = None

    # Structured input for specific judge types
    question: str | None = None
    answer: str | None = None  # Ground truth
    prediction: str | None = None  # Model prediction
    context: str | None = None  # Additional context
    options: list[str] | None = None  # For multiple choice

    # For comparative evaluation
    response1: str | None = None
    response2: str | None = None

    # Custom evaluation prompt
    custom_prompt: str | None = None
    prompt_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class Response:
    """Standard response format from judge evaluation"""

    content: str
    model_used: str
    usage: dict[str, int] | None = None
    raw_response: Any | None = None

    # Parsed results for specific judge types
    parsed_result: int | float | bool | tuple[float, float] | dict[str, Any] | None = None
    success: bool = True
    error_message: str | None = None
