from autogen import AssistantAgent, UserProxyAgent, register_function
import json
from dotenv import load_dotenv
import traceback

# Local imports
from config import LLM_CONFIG
from tools.websearch_tool import ResearchPaperSearchTool
from tools.evaluation_tool import evaluate_response

load_dotenv() # load config

# ======================
# Agent Setup
# ======================
search_tool = ResearchPaperSearchTool()

search_tool_spec = {
    "type": "function",
    "function": {
        "name": "search_research_papers",
        "description": "Search academic papers on arXiv based on topic, year, and comparison type.",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "The research topic to search for."},
                "year": {"type": "integer", "description": "The publication year to use as a reference point."},
                "comparison": {
                    "type": "string",
                    "enum": ["after", "before", "in"],
                    "description": "Specify relation to the year: 'after' (published after year), 'before' (published before year), 'in' (published in that exact year)."
                },
                "min_citations": {"type": "integer",
                                  "description": "Minimum number of citations."}
            },
            "required": ["topic", "year", "comparison"]
        }
    }
}

ASSISTANT_SYSTEM_MESSAGE = """You are an AI research assistant. Your sole task is to use the 'search_research_papers' tool to find academic papers and return ONLY the findings as a valid JSON list of objects.
Each object in the list MUST represent a paper and contain the following keys: 'title' (string), 'authors' (list of strings), 'year' (integer), and 'link' (string).
Example of a valid response with one paper:
[{"title": "Example Paper Title", "authors": ["Author A", "Author B"], "year": 2023, "link": "http://example.com/paper_link"}]
If no papers are found for the query, you MUST return an empty JSON list: [].
Do NOT include any introductory text, concluding text, explanations, apologies, summaries, or any other conversational filler in your response. Your entire output must be the JSON list itself, starting with '[' and ending with ']'.
Adhere strictly to the 'comparison' parameter values ('after', 'before', 'in') when searching by year."""

assistant = AssistantAgent(
    name="research_assistant",
    system_message=ASSISTANT_SYSTEM_MESSAGE,
    llm_config={
        **LLM_CONFIG,
        "tools": [search_tool_spec]
    }
)

def is_final_json_list(message_dict) -> bool:
    """
    Checks if the message content is a string that starts with '[' and ends with ']',
    and is valid JSON.
    """
    content = message_dict.get("content", "")
    if isinstance(content, str):
        content_stripped = content.strip()
        if content_stripped.startswith("[") and content_stripped.endswith("]"):
            try:
                json.loads(content_stripped)
                return True
            except json.JSONDecodeError:
                return False
    return False

def search_wrapper(topic: str, year: int, comparison: str, min_citations: int) -> str:
    try:
        return search_tool.search(topic, year, comparison, min_citations)
    except Exception as e:
        print(f"Error during search_tool.search: {e}")
        return "[]"

user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=3,
    code_execution_config=False,
    is_termination_msg=is_final_json_list
)

register_function(
    search_wrapper,
    caller=assistant,
    executor=user_proxy,
    name="search_research_papers",
    description="Search arXiv for academic papers based on topic, year, and comparison type."
)

if __name__ == "__main__":
    query = "Find papers about horses published after 2020 and has 10 citations"

    final_response_content = None

    try:
        print(f"User Query: {query}\n")
        user_proxy.initiate_chat(
            assistant,
            message=query,
            clear_history=True
        )

        all_messages = user_proxy.chat_messages.get(assistant, [])
        for msg in reversed(all_messages):
            if msg.get("role") == "assistant" and is_final_json_list(msg):
                final_response_content = msg["content"].strip()
                break

        if final_response_content is None:
            last_assistant_message = next((m for m in reversed(all_messages) if m.get("role") == "assistant"), None)
            if last_assistant_message and last_assistant_message.get("content"):
                final_response_content = str(last_assistant_message["content"]).strip()
            else:
                last_user_proxy_message = user_proxy.last_message(assistant)
                if last_user_proxy_message and last_user_proxy_message.get("content"):
                    final_response_content = str(last_user_proxy_message["content"]).strip()
                else:
                    final_response_content = "[]"
                    print("Warning: No substantive response captured from the assistant. Defaulting to empty list.")

        print("=== Formatted Agent Response ===")
        parsed_successfully = False
        if final_response_content:
                if isinstance(final_response_content, str):
                    parsed_json = json.loads(final_response_content)
                    print(json.dumps(parsed_json, indent=2))
                    parsed_successfully = True
                else:
                    print("Error: Final response content captured was not a string.")
                    print("Raw content received:")
                    print(final_response_content)
                    final_response_content = str(final_response_content)

        if not isinstance(final_response_content, str):
            final_response_content = str(final_response_content if final_response_content is not None else "[]")

        print("\n=== Evaluation ===")
        evaluation = evaluate_response(query, final_response_content)
        print(json.dumps(evaluation, indent=2))

    except Exception as e:
        print(f"\nMain execution failed: {str(e)}")
        traceback.print_exc()