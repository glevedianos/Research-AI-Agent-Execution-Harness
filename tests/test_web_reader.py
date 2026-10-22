
from tools.web_reader import (
    extract_readable_text,
    select_query_context,
)


HTML = b'''<html>
<head><title>Python 3.14 documentation</title></head>
<body>
<nav>Navigation</nav>
<main>
<h1>Interpreter API</h1>
<p>Introduction mentioning create.</p>
<dl>
<dt>interpreters.create()</dt>
<dd>Create an interpreter.</dd>
<dt>Interpreter.exec(code)</dt>
<dd>Execute code in the interpreter.</dd>
</dl>
</main>
</body>
</html>'''


def test_extracts_api_signatures_and_omits_navigation():
    title, text = extract_readable_text(HTML)

    assert title == "Python 3.14 documentation"
    assert "Navigation" not in text
    assert "interpreters.create()" in text
    assert "Interpreter.exec(code)" in text


def test_finds_api_definition():
    _, text = extract_readable_text(HTML)

    found, content = select_query_context(
        text,
        "Interpreter.exec",
        1000
    )

    assert found
    assert "Interpreter.exec(code)" in content


def test_unmatched_query():
    _, text = extract_readable_text(HTML)

    found, content = select_query_context(
        text,
        "run_string",
        1000
    )

    assert not found
    assert content == ""


def test_searches_beyond_initial_excerpt():
    text = (
        "A" * 12000
        + "\ninterpreters.create()\n"
    )

    found, content = select_query_context(
        text,
        "interpreters.create",
        1000
    )

    assert found
    assert "interpreters.create()" in content
