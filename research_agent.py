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
        "description": "Search academic papers on arXiv",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "Research topic"},
                "year": {"type": "integer", "description": "Publication year reference"},
                "comparison": {
                    "type": "string",
                    "enum": ["after", "before", "in"],
                    "description": "Use 'after' for papers published after the year, 'before' for before, 'in' for exact year"
                },
                "min_citations": {"type": "integer", "description": "Not available in this version"}
            },
            "required": ["topic", "year", "comparison"]
        }
    }
}

assistant = AssistantAgent(
    name="research_assistant",
    system_message="You are an AI research assistant. Use the search tool to find academic papers. "
                   "Always use 'after' for papers published after a year, 'before' for before, and 'in' for exact year matches. "
                   "Return results as JSON with title, authors, year, and link.",
    llm_config={
        **LLM_CONFIG,
        "tools": [search_tool_spec]
    }
)

user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=3,
    code_execution_config=False,
    is_termination_msg=lambda x: x.get("content", "").strip().endswith("]")
)

register_function(
    search_wrapper,
    caller=assistant,
    executor=user_proxy,
    name="search_research_papers",
    description="Search arXiv for academic papers"
)



if __name__ == "__main__":
    query = "Find papers about fruit published after 2021 and has 1 citations"

    try:
        user_proxy.initiate_chat(
            assistant,
            message=query,
            clear_history=True
        )

        final_response = user_proxy.chat_messages[assistant][-1]["content"]
        print("\n=== Results ===")
        try:
            print(json.dumps(json.loads(final_response), indent=2))
        except:
            print(final_response)

        print("\n=== Evaluation ===")
        evaluation = evaluate_response(query, final_response)
        print(json.dumps(evaluation, indent=2))

    except Exception as e:
        print(f"Main execution failed: {str(e)}")