import os

# ======================
# Configuration
# ======================
LLM_CONFIG_CACHE = {
    "config_list": [
        {
            "model": "open-mistral-nemo",
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

LLM_CONFIG = {
    "config_list": [
        {
            "model": "open-mistral-nemo",
            "api_key": os.getenv("MISTRAL_API_KEY"), # You correctly use os.getenv("MISTRAL_API_KEY")
            "api_type": "mistral",
            "api_rate_limit": 0.25,     # <<< CRITICALLY MISSING
            "repeat_penalty": 1.1,      # <<< MISSING
            "temperature": 0.0,         # <<< Your is 0.3
            "seed": 42,                 # <<< MISSING
            "stream": False,            # <<< MISSING
            "native_tool_calls": False, # <<< MISSING
            "cache_seed": None,         # <<< MISSING
        }
    ]
}