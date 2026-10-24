
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime


class APILimitError(Exception):
    """A provider rate or quota limit was reached."""

    def __init__(
        self,
        message,
        limit_type="unknown",
        retry_after=None,
    ):
        super().__init__(message)
        self.limit_type = limit_type
        self.retry_after = retry_after


def get_retry_delay(error):
    """Extract a retry delay, in seconds, when available."""

    response = getattr(error, "response", None)

    if response is not None:
        headers = getattr(response, "headers", {})

        raw = headers.get("retry-after")

        if raw:
            try:
                return max(0.0, float(raw))
            except (ValueError, TypeError):
                try:
                    retry_date = parsedate_to_datetime(raw)
                    if retry_date.tzinfo is None:
                        retry_date = retry_date.replace(
                            tzinfo=timezone.utc
                        )
                    return max(
                        0.0,
                        (retry_date - datetime.now(timezone.utc))
                        .total_seconds()
                    )
                except (ValueError, TypeError, OverflowError):
                    pass

    message = str(error)

    match = re.search(
        r"please try again in\s+([\d.]+)\s*s",
        message,
        flags=re.IGNORECASE,
    )

    if match:
        return float(match.group(1))

    return None


def classify_api_error(error):
    """Classify Groq/OpenRouter rate and quota errors."""

    if isinstance(error, APILimitError):
        return error

    status = getattr(error, "status_code", None)
    response = getattr(error, "response", None)

    if status is None and response is not None:
        status = getattr(response, "status_code", None)

    message = str(error).lower()

    indicators = (
        "rate limit",
        "rate_limit",
        "ratelimit",
        "quota",
        "too many requests",
        "tokens per minute",
        "requests per minute",
        "tpm",
        "rpm",
        "tpd",
        "rpd",
    )

    if status != 429 and not any(
        term in message for term in indicators
    ):
        return None

    daily_indicators = (
        "free-models-per-day",
        "openrouter_free_tier_daily",
        "daily limit",
        "daily quota",
        "daily token",
        "daily request",
        "per day",
        "per-day",
        "tokens per day",
        "requests per day",
        "tpd",
        "rpd",
    )

    temporary_indicators = (
        "tokens per minute",
        "requests per minute",
        "per minute",
        "tpm",
        "rpm",
        "temporarily rate limited",
    )

    if any(term in message for term in daily_indicators):
        limit_type = "daily"
    elif any(term in message for term in temporary_indicators):
        limit_type = "temporary"
    else:
        limit_type = "unknown"

    return APILimitError(
        "The AI provider has reached a request limit.",
        limit_type=limit_type,
        retry_after=get_retry_delay(error),
    )
