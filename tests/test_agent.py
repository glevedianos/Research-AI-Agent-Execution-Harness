
import json
from types import SimpleNamespace
from unittest.mock import MagicMock
from agent.brain import ResearchAgent
import agent.brain as brain
from harness.documentation import combine_documentation_content

def fake_response(content="", tool_calls=None):
    """
    Simulate a response from the LLM.
    """
    response = MagicMock()

    response.content = content
    response.tool_calls = tool_calls or []

    response.model_dump.return_value = {
        "role": "assistant",
        "content": content,
    }

    return response


def fake_tool_call(name, arguments, call_id="call_1"):
    """
    Simulate a tool call requested by the LLM.
    """
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(
            name=name,
            arguments=json.dumps(arguments),
        ),
    )


def create_agent(monkeypatch, fake_llm):
    """
    Replace the real Groq client with a fake LLM.
    """
    monkeypatch.setattr(
        brain,
        "LLMClient",
        lambda: fake_llm,
    )

    return brain.ResearchAgent()


def test_agent_answers_without_tools(monkeypatch):
    """
    The agent should answer directly when
    no tool calls are requested.
    """
    fake_llm = MagicMock()

    fake_llm.chat_with_tools.return_value = fake_response(
        content="A decorator modifies a function."
    )

    agent = create_agent(monkeypatch, fake_llm)

    answer = agent.run("What is a decorator?")

    assert answer == "A decorator modifies a function."

    fake_llm.chat_with_tools.assert_called_once()


def test_agent_remembers_conversation(monkeypatch):
    """
    The agent should remember completed exchanges.
    """
    fake_llm = MagicMock()

    fake_llm.chat_with_tools.side_effect = [
        fake_response(content="Verleumdung means defamation."),
        fake_response(content="The word was Verleumdung."),
    ]

    agent = create_agent(monkeypatch, fake_llm)

    agent.run("Translate Verleumdung.")
    answer = agent.run("What word did I ask about?")

    assert answer == "The word was Verleumdung."

    second_request_messages = (
        fake_llm.chat_with_tools.call_args_list[1].args[0]
    )

    assert any(
        message.get("content") == "Translate Verleumdung."
        for message in second_request_messages
    )

    assert any(
        message.get("content")
        == "Verleumdung means defamation."
        for message in second_request_messages
    )


def test_agent_executes_tool(monkeypatch):
    """
    The agent should execute a requested tool.
    """
    fake_llm = MagicMock()

    tool_call = fake_tool_call(
        "web_search",
        {"query": "Python 3.14"},
    )

    fake_llm.chat_with_tools.side_effect = [
        fake_response(tool_calls=[tool_call]),
        fake_response(content="Python 3.14 has new features."),
    ]

    executed_tools = []

    def fake_execute_tool(name, arguments, run_id=None):
        executed_tools.append(name)

        return json.dumps({
            "tool": name,
            "success": True,
            "result": [{"title": "Python documentation"}],
            "error": None,
        })

    monkeypatch.setattr(
        brain,
        "execute_tool",
        fake_execute_tool,
    )

    agent = create_agent(monkeypatch, fake_llm)

    answer = agent.run("Research Python 3.14.")

    assert answer == "Python 3.14 has new features."
    assert executed_tools == ["web_search"]


def test_agent_enforces_tool_budget(monkeypatch):
    """
    The agent must not exceed its tool-call budget.
    """
    fake_llm = MagicMock()

    fake_llm.chat_with_tools.side_effect = [
        fake_response(
            tool_calls=[
                fake_tool_call(
                    "web_search",
                    {"query": "Python"},
                    "call_1",
                )
            ]
        ),
        fake_response(
            tool_calls=[
                fake_tool_call(
                    "read_webpage",
                    {"url": "https://example.com"},
                    "call_2",
                )
            ]
        ),
    ]

    fake_llm.chat.return_value = "Research limit reached."

    executed_tools = []

    def fake_execute_tool(name, arguments, run_id=None):
        executed_tools.append(name)
        return json.dumps({"success": True})

    monkeypatch.setattr(
        brain,
        "execute_tool",
        fake_execute_tool,
    )

    agent = create_agent(monkeypatch, fake_llm)
    agent.max_tool_calls = 1

    answer = agent.run("Research Python.")

    assert executed_tools == ["web_search"]
    assert answer == "Research limit reached."

    fake_llm.chat.assert_called_once()


def test_agent_handles_llm_error(monkeypatch):
    """
    LLM failures should propagate to the caller.
    """
    fake_llm = MagicMock()

    fake_llm.chat_with_tools.side_effect = RuntimeError(
        "Simulated LLM failure"
    )

    agent = create_agent(monkeypatch, fake_llm)

    try:
        agent.run("Hello")
    except RuntimeError as error:
        assert "Simulated LLM failure" in str(error)
    else:
        raise AssertionError(
            "Expected RuntimeError was not raised"
        )


def test_agent_enforces_deadline(monkeypatch):
    """
    The agent should stop research when its
    execution deadline has been reached.
    """
    fake_llm = MagicMock()
    fake_llm.chat.return_value = (
        "Research stopped because of the deadline."
    )

    agent = create_agent(monkeypatch, fake_llm)

    # Force the deadline to be reached immediately.
    agent.max_duration_seconds = 0

    answer = agent.run("Research AI agents.")

    assert answer == (
        "Research stopped because of the deadline."
    )

    # The agent must not enter the normal tool-calling loop.
    fake_llm.chat_with_tools.assert_not_called()

    # It should generate a final response without tools.
    fake_llm.chat.assert_called_once()



def test_agent_trims_conversation_history(monkeypatch):
    """
    The agent should retain only the most recent
    completed conversation turns.
    """
    fake_llm = MagicMock()

    fake_llm.chat_with_tools.side_effect = [
        fake_response(content=f"Answer {i}")
        for i in range(7)
    ]

    agent = create_agent(monkeypatch, fake_llm)

    agent.max_history_turns = 5

    for i in range(7):
        agent.run(f"Question {i}")

    # 1 system message + 5 user/assistant pairs.
    assert len(agent.messages) == 11

    # The oldest two conversations were removed.
    assert not any(
        message.get("content") == "Question 0"
        for message in agent.messages
    )

    assert not any(
        message.get("content") == "Question 1"
        for message in agent.messages
    )

    # The newest conversation is still available.
    assert any(
        message.get("content") == "Question 6"
        for message in agent.messages
    )


def test_agent_accepts_valid_python_code():
    agent = ResearchAgent()

    answer = (
        "Example:\n"
        "```python\n"
        "print('Hello')\n"
        "```"
    )

    result = agent.validate_answer(answer)

    assert result == answer


def test_agent_warns_about_invalid_python_code():
    agent = ResearchAgent()

    answer = (
        "Example:\n"
        "```python\n"
        "def broken(\n"
        "```"
    )

    result = agent.validate_answer(answer)

    assert "Python syntax validation failed" in result
    assert "(block 1):" in result


def test_agent_validates_generated_answer(monkeypatch):
    agent = ResearchAgent()

    invalid_answer = (
        "Here is the code:\n"
        "```python\n"
        "def broken(\n"
        "```"
    )

    class FakeResponse:
        tool_calls = None
        content = invalid_answer

        def model_dump(self, exclude_none=True):
            return {
                "role": "assistant",
                "content": self.content
            }

    monkeypatch.setattr(
        agent.llm,
        "chat_with_tools",
        lambda messages, tools: FakeResponse()
    )

    result = agent.run("Write a Python function.")

    assert "Python syntax validation failed" in result
    assert "(block 1):" in result

def test_agent_still_answers_without_documentation():
    agent = ResearchAgent()

    answer = "Python is a programming language."

    result = agent.validate_answer(answer)

    assert result == answer

def test_agent_flags_unverified_api_method():
    agent = ResearchAgent()

    answer = (
        "Example:\n"
        "```python\n"
        "interp.run('hello')\n"
        "```"
    )

    documentation = """
    Interpreter methods:
    exec ( code , / )
    close ( )
    """

    result = agent.validate_answer(
        answer,
        documentation=documentation,
        object_name="interp"
    )

    assert "Unverified API methods" in result
    assert "run" in result


def test_agent_accepts_documented_api_method():
    agent = ResearchAgent()

    answer = (
        "Example:\n"
        "```python\n"
        "interp.exec('hello')\n"
        "```"
    )

    documentation = """
    Interpreter methods:
    exec ( code , / )
    close ( )
    """

    result = agent.validate_answer(
        answer,
        documentation=documentation,
        object_name="interp"
    )

    assert "Unverified API methods" not in result


def test_agent_detects_unverified_methods_automatically():
    agent = ResearchAgent()

    answer = (
        "Example:\n"
        "```python\n"
        "from concurrent import interpreters\n"
        "interp = interpreters.create()\n"
        "interp.run('hello')\n"
        "interp.exec('hello')\n"
        "```"
    )

    documentation = """
    Python concurrent.interpreters documentation.

    Interpreter methods:
    create ( )
    exec ( code , / )
    close ( )
    """

    result = agent.validate_answer(
        answer,
        documentation=documentation
    )

    assert "Unverified API methods on 'interp': run" in result




def test_agent_verifies_api_using_retrieved_documentation(
    monkeypatch
):
    import json
    from types import SimpleNamespace

    agent = ResearchAgent()

    # First LLM response: request documentation.
    tool_call = SimpleNamespace(
        id="call_123",
        function=SimpleNamespace(
            name="read_webpage",
            arguments=json.dumps({
                "url": "https://docs.python.org/example"
            })
        )
    )

    class ToolResponse:
        tool_calls = [tool_call]
        content = None

        def model_dump(self, exclude_none=True):
            return {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "read_webpage",
                            "arguments": tool_call.function.arguments
                        }
                    }
                ]
            }


    class FinalResponse:
        tool_calls = None
        content = (
            "Example:\n"
            "```python\n"
            "from concurrent import interpreters\n"
            "interp = interpreters.create()\n"
            "interp.run('hello')\n"
            "```"
        )

        def model_dump(self, exclude_none=True):
            return {
                "role": "assistant",
                "content": self.content
            }

    responses = iter([
        ToolResponse(),
        FinalResponse()
    ])

    monkeypatch.setattr(
        agent.llm,
        "chat_with_tools",
        lambda messages, tools: next(responses)
    )

    # Mock the webpage reader result.
    def fake_execute_tool(name, arguments, run_id=None):
        assert name == "read_webpage"

        return json.dumps({
            "tool": "read_webpage",
            "success": True,
            "result": {
                "url": "https://docs.python.org/example",
                "title": "Interpreter documentation",
                "content": (
                    "Python concurrent.interpreters documentation.\n"
                    "Interpreter methods:\n"
                    "create ( )\n"
                    "exec ( code , / )\n"
                    "close ( )"
                )
            },
            "error": None
        })

    monkeypatch.setattr(
        "agent.brain.execute_tool",
        fake_execute_tool
    )

    result = agent.run(
        "Show me how to use Python interpreters."
    )

    assert "Unverified API methods" in result
    assert "'interp': run" in result



def test_agent_skips_unrelated_documentation():
    agent = ResearchAgent()

    answer = (
        "```python\n"
        "import pandas as pd\n"
        "df = pd.DataFrame({'name': ['Anna']})\n"
        "df.head()\n"
        "```"
    )

    documentation = """
    Python concurrent.interpreters documentation.
    Interpreter methods:
    exec ( code , / )
    close ( )
    """

    result = agent.validate_answer(
        answer,
        documentation=documentation
    )

    assert "Unverified API methods" not in result


def test_agent_checks_methods_with_matching_origin():
    agent = ResearchAgent()

    answer = (
        "```python\n"
        "from concurrent import interpreters\n"
        "interp = interpreters.create()\n"
        "interp.run('hello')\n"
        "```"
    )

    documentation = """
    Python concurrent.interpreters documentation.
    Interpreter methods:
    exec ( code , / )
    close ( )
    """

    result = agent.validate_answer(
        answer,
        documentation=documentation
    )

    assert "Unverified API methods on 'interp': run" in result



def test_agent_skips_unknown_object_origin():
    agent = ResearchAgent()

    answer = (
        "```python\n"
        "client = get_client()\n"
        "client.connect()\n"
        "```"
    )

    documentation = """
    pandas.DataFrame documentation.
    head()
    tail()
    """

    result = agent.validate_answer(
        answer,
        documentation=documentation
    )

    assert "Unverified API methods" not in result


def test_agent_explicitly_verifies_unknown_object():
    agent = ResearchAgent()

    answer = (
        "Example:\n"
        "```python\n"
        "client = get_client()\n"
        "client.connect()\n"
        "```"
    )

    documentation = """
    Client API documentation.

    Available methods:
    disconnect()
    close()
    """

    result = agent.validate_answer(
        answer,
        documentation=documentation,
        object_name="client"
    )

    assert "Unverified API methods on 'client': connect" in result



def test_agent_preserves_documentation_metadata(monkeypatch):
    import json
    from types import SimpleNamespace

    agent = ResearchAgent()

    tool_call = SimpleNamespace(
        id="call_docs",
        function=SimpleNamespace(
            name="read_webpage",
            arguments=json.dumps({
                "url": "https://docs.python.org/example"
            })
        )
    )

    class ToolResponse:
        tool_calls = [tool_call]
        content = None

        def model_dump(self, exclude_none=True):
            return {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_docs",
                    "type": "function",
                    "function": {
                        "name": "read_webpage",
                        "arguments": tool_call.function.arguments
                    }
                }]
            }

    class FinalResponse:
        tool_calls = None
        content = "Research complete."

        def model_dump(self, exclude_none=True):
            return {
                "role": "assistant",
                "content": self.content
            }

    responses = iter([ToolResponse(), FinalResponse()])

    monkeypatch.setattr(
        agent.llm,
        "chat_with_tools",
        lambda messages, tools: next(responses)
    )

    def fake_execute_tool(name, arguments, run_id=None):
        return json.dumps({
            "tool": "read_webpage",
            "success": True,
            "result": {
                "url": "https://docs.python.org/example",
                "title": "Python Documentation",
                "content": "Interpreter methods: exec(), close()"
            },
            "error": None
        })

    monkeypatch.setattr(
        "agent.brain.execute_tool",
        fake_execute_tool
    )

    captured = {}

    def fake_combine(records):
        captured["records"] = records.copy()
        return "\n\n".join(
            record["content"] for record in records
        )

    monkeypatch.setattr(
        "agent.brain.combine_documentation_content",
        fake_combine
    )

    result = agent.run("Research Python interpreters.")

    assert result == "Research complete."
    assert captured["records"] == [{
        "url": "https://docs.python.org/example",
        "title": "Python Documentation",
        "content": "Interpreter methods: exec(), close()"
    }]



def test_agent_uses_relevant_documentation_only():
    agent = ResearchAgent()

    answer = (
        "```python\n"
        "from concurrent import interpreters\n"
        "interp = interpreters.create()\n"
        "interp.run('hello')\n"
        "```"
    )

    records = [
        {
            "url": "https://docs.python.org/3/library/concurrent.interpreters.html",
            "title": "Python interpreters",
            "content": "Interpreter methods: exec(), close()"
        },
        {
            "url": "https://example.com/unrelated",
            "title": "Unrelated documentation",
            "content": "run()"
        }
    ]

    result = agent.validate_answer(
        answer,
        documentation=combine_documentation_content(records),
        documentation_records=records
    )

    assert "Unverified API methods on 'interp': run" in result



def test_agent_flags_unverified_interpreter_method():
    agent = ResearchAgent()

    answer = (
        "```python\n"
        "from concurrent import interpreters\n"
        "interp = interpreters.create()\n"
        "interp.run('hello')\n"
        "```"
    )

    records = [
        {
            "url": (
                "https://docs.python.org/3.14/library/"
                "concurrent.interpreters.html"
            ),
            "title": "Python 3.14 concurrent.interpreters",
            "content": (
                "Python concurrent.interpreters documentation.\n"
                "Interpreter methods: exec(), close()"
            )
        }
    ]

    result = agent.validate_answer(
        answer,
        documentation=combine_documentation_content(records),
        documentation_records=records
    )

    assert "Unverified API methods on 'interp': run" in result
