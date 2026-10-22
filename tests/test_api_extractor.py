
from harness.api_extractor import (
    extract_documented_methods,
    extract_called_methods,
    verify_methods_against_docs,
    extract_object_names,
    documentation_mentions_object,
    extract_imports,
    extract_object_origins,
    documentation_matches_origin,
)

def test_extracts_documented_methods():
    documentation = """
    Interpreter objects
    close ( ) Finalize and destroy the interpreter.
    exec ( code , / ) Run the given source code.
    call ( callable , / , * args , ** kwargs )
    """

    result = extract_documented_methods(
        documentation,
        {"close", "exec", "call", "run"}
    )

    assert result == {"close", "exec", "call"}


def test_returns_empty_set_when_methods_are_missing():
    documentation = "This module manages interpreters."

    result = extract_documented_methods(
        documentation,
        {"exec", "call", "run"}
    )

    assert result == set()


def verify_methods_against_docs(
    code: str,
    documentation: str,
    object_name: str
) -> dict:
    """
    Check whether methods called on an object
    appear in the retrieved documentation.

    Missing methods are unverified, not necessarily invalid.
    """

    try:
        called_methods = extract_called_methods(
            code,
            object_name
        )

    except SyntaxError as error:
        return {
            "valid_syntax": False,
            "verified_methods": [],
            "unverified_methods": [],
            "error": str(error)
        }

    documented_methods = extract_documented_methods(
        documentation,
        called_methods
    )

    unverified_methods = called_methods - documented_methods

    return {
        "valid_syntax": True,
        "verified_methods": sorted(documented_methods),
        "unverified_methods": sorted(unverified_methods),
        "error": None
    }


def test_extracts_called_methods():
    code = """
interp.exec("print('Hello')")
interp.close()
"""

    result = extract_called_methods(
        code,
        object_name="interp"
    )

    assert result == {"exec", "close"}


def test_verifies_methods_against_documentation():
    code = """
interp.exec("print('Hello')")
interp.run("print('Hello')")
"""

    documentation = """
Interpreter methods:
exec ( code , / ) Run source code.
close ( ) Destroy the interpreter.
"""

    result = verify_methods_against_docs(
        code,
        documentation,
        object_name="interp"
    )

    assert result["valid_syntax"] is True
    assert result["verified_methods"] == ["exec"]
    assert result["unverified_methods"] == ["run"]


def test_verification_handles_invalid_syntax():
    result = verify_methods_against_docs(
        code="def broken(",
        documentation="exec ( code )",
        object_name="interp"
    )

    assert result["valid_syntax"] is False
    assert result["error"] is not None


def test_extracts_object_names():
    code = """
interp.exec("print('Hello')")
interp.close()
client.chat("Hello")
"""

    result = extract_object_names(code)

    assert result == {"interp", "client"}



def test_documentation_mentions_object():
    documentation = """
    Create an interpreter using:
    interp = interpreters.create()

    Then call interp.exec().
    """

    assert documentation_mentions_object(
        documentation,
        "interp"
    ) is True


def test_documentation_does_not_mention_object():
    documentation = """
    Interpreter methods:
    exec ( code , / )
    close ( )
    """

    assert documentation_mentions_object(
        documentation,
        "df"
    ) is False


def test_documentation_object_match_uses_word_boundaries():
    documentation = "The variable interpreter is available."

    assert documentation_mentions_object(
        documentation,
        "interp"
    ) is False



def test_extracts_import_aliases():
    code = """
import pandas as pd
import numpy as np
"""

    result = extract_imports(code)

    assert result == {
        "pd": "pandas",
        "np": "numpy"
    }


def test_extracts_from_imports():
    code = """
from concurrent import interpreters
from pathlib import Path
"""

    result = extract_imports(code)

    assert result == {
        "interpreters": "concurrent.interpreters",
        "Path": "pathlib.Path"
    }


def test_extracts_regular_imports():
    code = """
import os
import json
"""

    result = extract_imports(code)

    assert result == {
        "os": "os",
        "json": "json"
    }



def test_traces_interpreter_creation():
    code = """
from concurrent import interpreters
interp = interpreters.create()
interp.exec("print('Hello')")
"""

    result = extract_object_origins(code)

    assert result == {
        "interp": "concurrent.interpreters.create"
    }


def test_traces_pandas_dataframe_creation():
    code = """
import pandas as pd
df = pd.DataFrame({"name": ["Anna"]})
"""

    result = extract_object_origins(code)

    assert result == {
        "df": "pandas.DataFrame"
    }


def test_ignores_unknown_object_creation():
    code = """
obj = unknown_factory()
"""

    result = extract_object_origins(code)

    assert result == {}


def test_documentation_matches_object_origin():
    documentation = """
    Python concurrent.interpreters documentation.

    Interpreter methods:
    exec ( code , / )
    close ( )
    """

    result = documentation_matches_origin(
        documentation,
        "concurrent.interpreters.create"
    )

    assert result is True


def test_documentation_does_not_match_unrelated_origin():
    documentation = """
    Python concurrent.interpreters documentation.
    """

    result = documentation_matches_origin(
        documentation,
        "pandas.DataFrame"
    )

    assert result is False


def test_documentation_match_requires_origin():
    assert documentation_matches_origin(
        "Some documentation",
        ""
    ) is False
