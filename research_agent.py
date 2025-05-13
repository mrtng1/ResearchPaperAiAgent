from autogen import AssistantAgent, UserProxyAgent, register_function
import json
from dotenv import load_dotenv

# Local imports
from config import LLM_CONFIG
from tools.websearch_tool import ResearchPaperSearchTool
from tools.evaluation_tool import evaluate_response

load_dotenv()

# ======================
# Agent Setup
# ======================
search_tool = ResearchPaperSearchTool()


def search_wrapper(topic: str, year: int, comparison: str, min_citations: int) -> str:
    return search_tool.search(topic, year, comparison, min_citations)


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
                                  "description": "Minimum number of citations (Note: This feature is not effectively used by the current tool version)."}
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


# More robust termination message checker
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
    query = "Find papers about fruit published after 2021 and has 1 citations"

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

        if final_response_content is None and all_messages:
            last_assistant_message = next((m for m in reversed(all_messages) if m.get("role") == "assistant"), None)
            if last_assistant_message:
                final_response_content = last_assistant_message["content"]
            else:
                final_response_content = user_proxy.last_message(assistant)["content"] if user_proxy.last_message(
                    assistant) else "No response captured."

        print("\n=== Raw Agent Response (Captured for Evaluation) ===")
        print(final_response_content)

        print("\n=== Formatted Results (Attempting to Parse as JSON) ===")
        try:
            if isinstance(final_response_content, str):
                parsed_json = json.loads(final_response_content)
                print(json.dumps(parsed_json, indent=2))
            else:
                print("Final response content was not a string, cannot parse as JSON.")
                print(final_response_content)
        except json.JSONDecodeError:
            print("Failed to parse the final response as JSON. Raw response was printed above.")
        except Exception as e:
            print(f"An unexpected error occurred during JSON parsing: {e}")

        print("\n=== Evaluation ===")
        evaluation_input = final_response_content if isinstance(final_response_content, str) else str(
            final_response_content)
        evaluation = evaluate_response(query, evaluation_input)
        print(json.dumps(evaluation, indent=2))

    except Exception as e:
        print(f"\nMain execution failed: {str(e)}")
        import traceback

        traceback.print_exc()