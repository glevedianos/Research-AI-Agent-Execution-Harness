
from harness.code_validator import (
    validate_python_syntax,
    validate_python_blocks,
    validate_api_methods,
)

def test_valid_python_code():
    code = """
from concurrent import interpreters

interp = interpreters.create()
interp.exec('print("Hello!")')
interp.close()
"""

    result = validate_python_syntax(code)

    assert result["valid"] is True
    assert result["error"] is None


def test_invalid_python_code():
    code = """
def hello(
    print("Hello")
"""

    result = validate_python_syntax(code)

    assert result["valid"] is False
    assert result["error"] is not None
    assert result["line"] is not None



def test_extracts_valid_python_block():
    answer = '''
Here is an example:

```python
print("Hello")
```
'''

    results = validate_python_blocks(answer)

    assert len(results) == 1
    assert results[0]["valid"] is True


def test_detects_invalid_python_block():
    answer = '''
Here is an example:

```python
def hello(
    print("Hello")
```
'''

    results = validate_python_blocks(answer)

    assert len(results) == 1
    assert results[0]["valid"] is False


def test_ignores_non_python_blocks():
    answer = '''
```json
{"name": "example"}
```
'''

    results = validate_python_blocks(answer)

    assert results == []


def test_accepts_documented_api_methods():
    code = """
from concurrent import interpreters

interp = interpreters.create()
interp.exec("print('Hello')")
interp.close()
"""

    result = validate_api_methods(
        code,
        object_name="interp",
        allowed_methods={"exec", "call", "close"}
    )

    assert result["valid"] is True
    assert result["unknown_methods"] == []


def test_detects_unknown_api_method():
    code = """
from concurrent import interpreters

interp = interpreters.create()
interp.run("print('Hello')")
"""

    result = validate_api_methods(
        code,
        object_name="interp",
        allowed_methods={"exec", "call", "close"}
    )

    assert result["valid"] is False
    assert result["unknown_methods"] == ["run"]
