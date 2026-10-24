
from api_errors import classify_api_error, get_retry_delay


class FakeAPIError(Exception):
    def __init__(self, message, status_code=429, headers=None):
        super().__init__(message)
        self.status_code = status_code
        self.response = type(
            "FakeResponse",
            (),
            {"headers": headers or {}}
        )()


tests = [
    ("Groq TPM", "Rate limit reached on tokens per minute (TPM)", "temporary"),
    ("Groq TPD", "Rate limit reached on tokens per day (TPD)", "daily"),
    ("Groq RPM", "Rate limit reached on requests per minute (RPM)", "temporary"),
    ("OpenRouter daily", "openrouter_free_tier_daily", "daily"),
    ("Unknown 429", "Too many requests", "unknown"),
]

for name, message, expected in tests:
    error = FakeAPIError(message)
    result = classify_api_error(error)

    assert result is not None
    assert result.limit_type == expected
    print(f"PASS: {name}")


error = FakeAPIError(
    "Rate limit reached",
    headers={"retry-after": "15"}
)

assert get_retry_delay(error) == 15.0
print("PASS: Retry-After header")


error = FakeAPIError(
    "Please try again in 2.5s"
)

assert get_retry_delay(error) == 2.5
print("PASS: Provider retry message")


error = FakeAPIError(
    "Invalid API key",
    status_code=401
)

assert classify_api_error(error) is None
print("PASS: Unrelated error")
