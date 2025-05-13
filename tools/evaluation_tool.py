from typing import Dict
import autogen
import json
import os
from config import LLM_CONFIG


def _parse_critic_evaluation(content: str, user_query_for_debug: str, agent_response_for_debug: str) -> Dict:
    default_error_response = {
        "completeness": 0, "quality": 0, "robustness": 0,
        "consistency": 0, "specificity": 0,
        "feedback": "Evaluation parsing failed or critic returned non-JSON/malformed content."
    }
    json_str_for_error_reporting = content

    try:
        cleaned_content = content.strip()
        if cleaned_content.startswith("```json"):
            cleaned_content = cleaned_content[len("```json"):]
        if cleaned_content.endswith("```"):
            cleaned_content = cleaned_content[:-len("```")]
        cleaned_content = cleaned_content.strip()

        json_start = cleaned_content.find("{")
        json_end = cleaned_content.rfind("}") + 1

        if json_start != -1 and json_end != 0 and json_start < json_end:
            json_str = cleaned_content[json_start:json_end]
            json_str_for_error_reporting = json_str
            parsed_json = json.loads(json_str)

            expected_keys = ["completeness", "quality", "robustness", "consistency", "specificity", "feedback"]
            missing_keys = [key for key in expected_keys if key not in parsed_json]
            if missing_keys:
                raise ValueError(f"Parsed JSON is missing expected keys: {missing_keys}")

            for key in expected_keys:
                if key == "feedback":
                    if not isinstance(parsed_json[key], str):
                        raise ValueError(f"Field '{key}' must be a string, got {type(parsed_json[key])}")
                else:
                    if not isinstance(parsed_json[key], int):
                        try:
                            parsed_json[key] = int(str(parsed_json[key]))
                        except (ValueError, TypeError):
                             raise ValueError(f"Field '{key}' must be an integer, got {type(parsed_json[key])} with value '{parsed_json[key]}'")
                    if not (0 <= parsed_json[key] <= 5):
                        if parsed_json[key] == 0 and "parsing failed" in parsed_json.get("feedback","").lower():
                            pass
                        else:
                            raise ValueError(f"Score for '{key}' ({parsed_json[key]}) is out of valid range (0-5, typically 1-5 from critic).")
            return parsed_json
        else:
            error_msg = "Could not find valid JSON object in critic's response."
            default_error_response["feedback"] = f"{error_msg} Raw response snippet: {content[:200]}..."
            return default_error_response

    except json.JSONDecodeError as e:
        error_msg = f"JSON decoding failed: {str(e)}."
        default_error_response["feedback"] = f"{error_msg} Problematic content snippet: {json_str_for_error_reporting[:200]}..."
        return default_error_response
    except ValueError as e:
        error_msg = f"Validation error in parsed JSON: {str(e)}."
        default_error_response["feedback"] = f"{error_msg} Full content snippet: {content[:200]}..."
        print(f"[Critic Evaluation Error] Validation error: {str(e)}. Enable verbose logs for details.")
        return default_error_response
    except Exception as e:
        error_msg = f"An unexpected error occurred during critic evaluation parsing: {str(e)}."
        default_error_response["feedback"] = f"{error_msg} Full content snippet: {content[:200]}..."
        print(f"[Critic Evaluation Error] Unexpected parsing error: {str(e)}. Enable verbose logs for details.")
        return default_error_response

def is_valid_json_object_message(message_dict) -> bool:
    """
    Checks if the message content is a string that represents a valid JSON object.
    Handles potential markdown code blocks around the JSON.
    """
    content = message_dict.get("content", "")
    if not isinstance(content, str):
        return False

    content_stripped = content.strip()
    if content_stripped.startswith("```json"):
        content_stripped = content_stripped[len("```json"):]
    if content_stripped.endswith("```"):
        content_stripped = content_stripped[:-len("```")]
    content_stripped = content_stripped.strip()

    if content_stripped.startswith("{") and content_stripped.endswith("}"):
        try:
            json.loads(content_stripped)
            return True
        except json.JSONDecodeError:
            return False
    return False

def evaluate_response(user_query: str, agent_response: str) -> Dict:
    agent_type_description = "AI research assistant"
    critic_system_message = (
        f"You are an AI Critic. Your task is to meticulously evaluate the response of an {agent_type_description} "
        "based on a user's query and a defined set of criteria. "
        "Provide your evaluation strictly in the specified JSON format. "
        "Do not add any explanatory text before or after the JSON object. "
        "Ensure all score fields (completeness, quality, robustness, consistency, specificity) are integers between 1 and 5."
    )

    critic_agent = autogen.AssistantAgent(
        name="critic_agent",
        llm_config=LLM_CONFIG,
        system_message=critic_system_message,
        is_termination_msg=is_valid_json_object_message
    )

    critic_prompt = f"""
You are evaluating the response from an {agent_type_description}.

**User Query:**
{user_query}

**Agent's Response:**
{agent_response}

**Evaluation Criteria & Instructions:**
Please evaluate the agent's response based on the following criteria. For each criterion, provide a score from 1 to 5 (1=Poor, 5=Excellent).
Provide your evaluation as a single JSON object.

1.  **Completeness (1-5):**
    * Did the agent address all aspects of the user's query? (e.g., topic, year constraints, number of citations if applicable).
    * If the query had multiple parts, were all parts covered in the response?

2.  **Quality (1-5):**
    * Was the information provided accurate and correct? (e.g., are paper details plausible, year matching?).
    * Was the response clear, well-organized, and easy to understand?
    * If the response is expected to be JSON (as requested for the research assistant), is the JSON well-formed and does it contain the required fields (title, authors, year, link)?

3.  **Robustness (1-5):**
    * How well did the agent handle the specific query? (e.g., handling of year parameters like 'after', 'before', 'in').
    * If the query had potentially tricky aspects (e.g., non-existent topics, impossible year constraints), how did it respond? (Score N/A or 3 if the query is straightforward and handled well. If the agent simply fails or errors out on tricky inputs that it *should* handle gracefully, score lower).
    * Note: The 'min_citations' parameter might be 'Not available in this version' for the tool; assess how the agent handles this if queried.

4.  **Consistency (1-5):**
    * Is the response internally consistent?
    * Does the response align with the requirements and constraints mentioned in the user query? (e.g., if the query asked for papers 'after' a year, are all results indeed after that year?).
    * Is the agent's use of its tools (e.g. search_research_papers function) consistent with its instructions?

5.  **Specificity (1-5):**
    * Does the agent provide specific details for each paper (title, authors, year, link as requested)?
    * Is the level of detail appropriate for a research assistant's findings?

**Additional Checks (to be covered in feedback):**
* **Context and Justifications:** Did the agent provide clear context for its response? (e.g. number of papers found, any issues encountered).
* **Interpretation of Query:** Did the agent correctly interpret all parts of the query, including topic, year, and comparison type?
* **Feasibility of Results:** Are the results (e.g. paper titles, authors) plausible for the given topic and year? (The critic cannot verify external links).
* **Format Adherence:** Did the agent return results in the specified JSON format with all required fields?

**Output Format:**
Return your evaluation STRICTLY as a JSON object with the following fields:
-   `completeness` (integer, 1-5)
-   `quality` (integer, 1-5)
-   `robustness` (integer, 1-5)
-   `consistency` (integer, 1-5)
-   `specificity` (integer, 1-5)
-   `feedback` (string, a detailed descriptive explanation for the scores, incorporating the additional checks. Provide specific examples from the agent's response to support your evaluation.)

Example JSON output:
{{
  "completeness": 4,
  "quality": 5,
  "robustness": 3,
  "consistency": 5,
  "specificity": 4,
  "feedback": "The agent correctly identified the topic and year constraints. All papers listed were published after the specified year. The JSON format was correct. Robustness was fair, as it handled a standard query well. Specificity was good, providing necessary details for each paper. One aspect of a multi-part query might have been overlooked, leading to a completeness of 4."
}}

Begin evaluation. Provide ONLY the JSON object.
"""

    evaluation_request_proxy = autogen.UserProxyAgent(
        name="evaluation_request_proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=1,
        code_execution_config=False,
    )

    evaluation_request_proxy.initiate_chat(
        recipient=critic_agent,
        message=critic_prompt,
        clear_history=True,
    )

    critic_json_response = ""
    chat_history = evaluation_request_proxy.chat_messages.get(critic_agent, [])

    if chat_history:
        critic_json_response = chat_history[0].get('content', "")
        if len(chat_history) > 1:
            print(f"Warning: Critic agent sent {len(chat_history)} messages. Expected 1. Using the first.")
            for i, msg_data in enumerate(chat_history):
                print(f"Critic Message {i+1}: {msg_data.get('content','')[:200]}...")
    else:
        last_msg_obj = evaluation_request_proxy.last_message()
        if last_msg_obj and last_msg_obj.get("role") != "user":
             critic_json_response = last_msg_obj.get("content","")

    if not critic_json_response:
        print("Error: No response content retrieved from the critic agent. Evaluation cannot proceed.")

    return _parse_critic_evaluation(critic_json_response, user_query, agent_response)