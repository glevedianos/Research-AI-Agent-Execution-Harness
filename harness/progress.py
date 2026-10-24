
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional


@dataclass
class ProgressEvent:
    """Represents one step in an agent execution."""

    stage: str
    message: str
    status: str = "running"
    details: dict = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(
            timezone.utc
        ).isoformat()
    )


ProgressCallback = Callable[[ProgressEvent], None]


def emit_progress(
    callback: Optional[ProgressCallback],
    stage: str,
    message: str,
    status: str = "running",
    **details,
):
    """Send an event to the UI, if a callback exists."""

    event = ProgressEvent(
        stage=stage,
        message=message,
        status=status,
        details=details,
    )

    if callback is not None:
        callback(event)

    return event
