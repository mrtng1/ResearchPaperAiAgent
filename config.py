import os

# ======================
# Configuration
# ======================
LLM_CONFIG = {
    "config_list": [
        {
            "model": "mistral-large-latest",
            "api_key": os.getenv("MISTRAL_API_KEY"),
            "base_url": "https://api.mistral.ai/v1",
            "api_type": "mistral",
            "temperature": 0.3,
            "timeout": 120,
            "max_retries": 3,
            "retry_wait_time": 5
        }
    ]
}