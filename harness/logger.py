
import json
import logging
from pathlib import Path


LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "agent.jsonl"


logger = logging.getLogger("research_agent")
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    file_handler = logging.FileHandler(
        LOG_FILE,
        encoding="utf-8"
    )

    file_handler.setFormatter(
        logging.Formatter("%(message)s")
    )

    logger.addHandler(file_handler)


def log_event(event: str, **kwargs):
    """
    Save a structured event to the JSONL log file.
    """
    record = {
        "event": event,
        **kwargs
    }

    logger.info(
        json.dumps(record, ensure_ascii=False)
    )
