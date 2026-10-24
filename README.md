# Research AI Agent & Execution Harness

A Python-based AI research assistant that combines LLM-driven tool use with a custom execution harness for reliable, controlled research workflows.

The agent can search the web, retrieve documentation, and answer technical questions using external sources. The harness manages tool execution, resource limits, validation, and structured logging.

## Features

- **Autonomous research:** Uses an LLM to decide when to search the web, retrieve webpages, and generate an answer.
- **Custom execution harness:** Controls tool calls, execution time, and maximum research iterations.
- **Tool registry:** Restricts execution to explicitly registered tools with validated arguments.
- **Documentation verification:** Checks generated Python code against retrieved API documentation and flags potentially unverified methods.
- **Python syntax validation:** Detects syntax errors in generated Python code without executing it.
- **Context management:** Reduces tool-result size to limit unnecessary token consumption.
- **Tool caching:** Reuses results from identical tool calls within a research run.
- **Error handling:** Handles tool failures and recognized LLM token-limit errors.
- **Structured logging:** Records research execution events for debugging and inspection.
- **Streamlit GUI:** Provides chat history, configurable progress displays, and execution details.
- **Mock Mode:** Supports testing without consuming API tokens.
-	**Automated Tests:** Covers retry handling, progress events, tool execution, caching, and execution budgets.


## Tech Stack

Python, LLM API integration, tool calling, Requests, Beautiful Soup, pytest, JSONL logging.

## Project Structure

- `agent/` — Agent reasoning and research orchestration.
- `harness/` — Tool execution, validation, documentation checks, and logging.
- `tools/` — Web search and webpage retrieval.
- `tests/` — Automated regression tests.
- `main_agent.py` — Command-line entry point.
- `llm_client.py` — LLM provider integration.

## Getting Started

Clone the repository and navigate to the project directory.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Configure your API credentials in a local `.env` file, using `.env.example` as a reference.

Run the agent:

```bash
python main_agent.py
```

If you want to launch the Streamlit interface:

```bash
python -m streamlit run app.py
```
## Example Research Task

> Research the Python 3.14 `concurrent.interpreters` API using only official documentation. Show how to create an interpreter and execute code. Do not guess API names or signatures.

The agent searches for documentation, retrieves relevant API definitions, and generates an answer with source references.

## Testing

Run the test suite:

```bash
python -m pytest -q
```

Latest local result: **70 tests passed**.

## Limitations

- Research quality depends on retrieved sources and LLM behavior.
- API verification is heuristic and does not guarantee runtime correctness.
- LLM provider rate limits can interrupt research.
- Generated Python code is syntax-checked but not automatically executed.
- Documentation retrieval and tool selection may require multiple iterations.

## Project Purpose

Built as a hands-on exploration of AI agents, tool calling, execution harnesses, and practical reliability techniques for LLM-powered applications.
