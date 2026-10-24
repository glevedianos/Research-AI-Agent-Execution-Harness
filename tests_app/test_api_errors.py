
from api_errors import classify_api_error


class FakeAPIError(Exception):
    def __init__(self, status_code, message):
        super().__init__(message)
        self.status_code = status_code


# Test 1: Daily quota
daily_error = FakeAPIError(
    429,
    "Free model daily token limit reached"
)

result = classify_api_error(daily_error)

assert result is not None
assert result.limit_type == "daily"

print("PASS: Daily quota detected")


# Test 2: Temporary rate limit
temporary_error = FakeAPIError(
    429,
    "Tokens per minute rate limit exceeded"
)

result = classify_api_error(temporary_error)

assert result is not None
assert result.limit_type == "temporary"

print("PASS: Temporary rate limit detected")


# Test 3: Unknown 429
unknown_error = FakeAPIError(
    429,
    "Too many requests"
)

result = classify_api_error(unknown_error)

assert result is not None
assert result.limit_type == "unknown"

print("PASS: Unknown rate limit detected")


# Test 4: Unrelated error
other_error = FakeAPIError(
    401,
    "Invalid API key"
)

result = classify_api_error(other_error)

assert result is None

print("PASS: Unrelated error ignored")
