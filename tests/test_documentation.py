
from harness.documentation import (
    create_documentation_record,
    combine_documentation_content,
)


def test_creates_documentation_record():
    record = create_documentation_record(
        url="https://docs.python.org/example",
        title="Python Documentation",
        content="Interpreter methods: exec(), close()"
    )

    assert record == {
        "url": "https://docs.python.org/example",
        "title": "Python Documentation",
        "content": "Interpreter methods: exec(), close()"
    }


def test_combines_documentation_content():
    records = [
        {
            "url": "https://example.com/one",
            "title": "First",
            "content": "Method one()"
        },
        {
            "url": "https://example.com/two",
            "title": "Second",
            "content": "Method two()"
        }
    ]

    result = combine_documentation_content(records)

    assert result == "Method one()\n\nMethod two()"


def test_ignores_empty_documentation_content():
    records = [
        {
            "url": "https://example.com/empty",
            "title": "Empty",
            "content": ""
        },
        {
            "url": "https://example.com/valid",
            "title": "Valid",
            "content": "Useful documentation"
        }
    ]

    result = combine_documentation_content(records)

    assert result == "Useful documentation"



from harness.documentation import find_relevant_documentation


def test_finds_relevant_documentation():
    records = [
        {
            "url": "https://docs.python.org/3/library/concurrent.interpreters.html",
            "title": "Python interpreters",
            "content": "Interpreter methods: exec(), close()"
        },
        {
            "url": "https://pandas.pydata.org/docs/",
            "title": "Pandas documentation",
            "content": "DataFrame methods: head(), tail()"
        }
    ]

    result = find_relevant_documentation(
        records,
        "concurrent.interpreters.create"
    )

    assert result == [records[0]]


def test_ignores_unrelated_documentation():
    records = [
        {
            "url": "https://pandas.pydata.org/docs/",
            "title": "Pandas documentation",
            "content": "DataFrame methods: head(), tail()"
        }
    ]

    result = find_relevant_documentation(
        records,
        "concurrent.interpreters.create"
    )

    assert result == []


def test_returns_empty_for_unknown_origin():
    records = [
        {
            "url": "https://example.com",
            "title": "Documentation",
            "content": "Some methods"
        }
    ]

    assert find_relevant_documentation(records, "") == []
