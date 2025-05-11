from typing import Dict
from autogen import AssistantAgent, UserProxyAgent
import json
from config import LLM_CONFIG

# ======================
# Evaluation System
# ======================
def evaluate_response(query: str, response: str) -> Dict:
    evaluator = AssistantAgent(
        name="evaluator",
        llm_config=LLM_CONFIG,
        system_message="Evaluate if the JSON response matches the query criteria. "
                       "Check format, year constraints, and topic relevance. "
                       "Return JSON with 'score' (1-10) and 'explanation'."
    )

    eval_proxy = UserProxyAgent(
        name="eval_proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=1,
        code_execution_config=False
    )

    prompt = f"Query: {query}\nResponse: {response}\nEvaluate and return JSON score:"
    eval_proxy.initiate_chat(evaluator, message=prompt)
    return _parse_evaluation(eval_proxy.last_message()["content"])


def _parse_evaluation(content: str) -> Dict:
    try:
        content = content.replace("```json", "").replace("```", "").strip()
        if content.startswith("{"):
            return json.loads(content)

        json_start = content.find("{")
        json_end = content.rfind("}") + 1
        return json.loads(content[json_start:json_end])
    except Exception as e:
        print(f"Evaluation parsing failed: {str(e)}")
        return {"score": 0, "explanation": "Evaluation parsing failed"}