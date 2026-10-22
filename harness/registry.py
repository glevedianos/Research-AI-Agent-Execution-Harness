from tools.web_search import web_search
from tools.web_reader import read_webpage


TOOL_FUNCTIONS = {
    "web_search": web_search,
    "read_webpage": read_webpage,
}


TOOL_SCHEMAS = [

    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for relevant information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query."
                    }
                },
                "required": ["query"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_webpage",
            "description": "Read a public webpage, optionally searching for a specific term.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Public webpage URL."
                    },
                    "query": {
                        "type": "string",
                        "description": "Optional keyword or API name to locate."
                    }
                },
                "required": ["url"],
                "additionalProperties": False
            }
        }
    }

]
