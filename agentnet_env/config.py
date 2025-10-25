from __future__ import annotations

from enum import Enum
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field, PositiveInt, field_validator


class LoggingLevel(str, Enum):
    minimal = "minimal"
    standard = "standard"
    verbose = "verbose"


class RedactionBox(BaseModel):
    x0: float = Field(ge=0.0, le=1.0)
    y0: float = Field(ge=0.0, le=1.0)
    x1: float = Field(ge=0.0, le=1.0)
    y1: float = Field(ge=0.0, le=1.0)

    @field_validator("x1")
    @classmethod
    def _validate_x(cls, v: float, info):  # type: ignore[override]
        # Pydantic v2 validator signature
        return v

    @field_validator("y1")
    @classmethod
    def _validate_y(cls, v: float, info):  # type: ignore[override]
        return v


class EnvConfig(BaseModel):
    monitor_index: int = 0
    capture_region: Optional[Tuple[int, int, int, int]] = (
        None  # left, top, width, height
    )
    screenshot_size: Optional[Tuple[PositiveInt, PositiveInt]] = None  # w, h (resized)

    # Timing & determinism
    pause_seconds: float = 0.05
    action_delay_seconds: float = 0.0
    drag_default_duration: float = 0.2
    rate_limit_hz: Optional[float] = None
    seed: Optional[int] = None

    # Execution behavior
    strict_dataset_mapping: bool = True
    dry_run: bool = False

    # Logging
    log_dir: str = "runs"
    logging_level: LoggingLevel = LoggingLevel.standard
    save_video: bool = False
    enable_hashes: bool = True
    redaction_boxes: List[RedactionBox] = Field(default_factory=list)

    # Platform-specific toggles (may be ignored on some OSes)
    fail_fast_on_permissions: bool = True

    model_config = {
        "extra": "ignore",
    }
