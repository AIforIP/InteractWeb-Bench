import os
import sys
import time
import random
import httpx
from openai import OpenAI, APIConnectionError, RateLimitError, APITimeoutError
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("OPENAILIKE_API_KEY") or os.environ.get("OPENAI_API_KEY")
base_url = os.environ.get("OPENAILIKE_BASE_URL") or os.environ.get("OPENAI_BASE_URL")

if not api_key:
    raise ValueError("API Key not found! Please set OPENAILIKE_API_KEY or OPENAI_API_KEY in .env")


robust_timeout = 1800
robust_transport = httpx.HTTPTransport(retries=3)

http_client = httpx.Client(
    base_url=base_url,
    follow_redirects=True,
    timeout=robust_timeout,
    transport=robust_transport
)

client = OpenAI(
    api_key=api_key,
    base_url=base_url,
    http_client=http_client
)


def get_local_client(model_name: str):
    # 1. Get the mapping string from environment variables
    # Example: "Qwen3.5-9B=http://localhost:8024/v1,llama3=http://localhost:8025/v1"
    mapping_str = os.environ.get("LOCAL_MODELS_MAP", "")
    if not mapping_str.strip():
        return None


    local_route_dict = {}
    for item in mapping_str.split(","):
        if "=" in item:
            m_name, m_url = item.split("=", 1)
            local_route_dict[m_name.strip().lower()] = m_url.strip()


    model_lower = model_name.lower()
    if model_lower in local_route_dict:
        local_base_url = local_route_dict[model_lower]

        return OpenAI(
            api_key="EMPTY",
            base_url=local_base_url,
            http_client=httpx.Client(
                base_url=local_base_url,
                follow_redirects=True,
                timeout=robust_timeout,
                transport=robust_transport
            )
        )


    return None

STREAMING_MODELS = {
    "kimi-k2.5",
    "qwen3.5-397b-a17b",
    "gpt-5.2"
    "claude-sonnet-4-6"
}


def _is_streaming_model(model: str) -> bool:

    return model.lower() in STREAMING_MODELS


def _call_with_stream(api_client, params: dict) -> str:

    full_content = []
    with api_client.chat.completions.create(**params, stream=True) as stream:
        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                print(delta.content, end="", flush=True)
                full_content.append(delta.content)
    return "".join(full_content)


def llm_generation(messages, model, max_tokens=-1, max_completion_tokens=-1, temperature=0.5, base_url=None,
                   api_key=None, **kwargs):
    """
    Generate text using LLM with robust retry logic and dynamic isolated credentials.
    For kimi-k2.5 and qwen3.5-397b-a17b, streaming is used automatically.
    """


    max_retries = 5
    base_delay = 2
    use_stream = _is_streaming_model(model)

    for attempt in range(max_retries):
        try:

            params = {
                "model": model,
                "messages": messages,
                "temperature": temperature
            }
            params.update(kwargs)

            if max_completion_tokens > 0:
                params["max_completion_tokens"] = max_completion_tokens
            elif max_tokens > 0:
                params["max_tokens"] = max_tokens

            local_client = get_local_client(model)

            if local_client:
                active_client = local_client
            elif base_url and api_key:

                active_client = OpenAI(
                    api_key=api_key,
                    base_url=base_url,
                    http_client=httpx.Client(
                        base_url=base_url,
                        follow_redirects=True,
                        timeout=robust_timeout,
                        transport=robust_transport
                    )
                )
            else:
                active_client = client

            if use_stream:

                return _call_with_stream(active_client, params)
            else:
                chat_response = active_client.chat.completions.create(**params)
                return chat_response.choices[0].message.content

        except (APIConnectionError, APITimeoutError, RateLimitError, httpx.ConnectError) as e:

            print(f"\033[93m[Warn] LLM Network Error (Attempt {attempt + 1}/{max_retries}): {e}\033[0m")

            if attempt == max_retries - 1:

                print(f"\033[91m[Error] Max retries reached. Failing.\033[0m")
                raise e


            sleep_time = (base_delay * (2 ** attempt)) + random.uniform(0, 1)
            print(f"Retrying in {sleep_time:.2f} seconds...")
            time.sleep(sleep_time)

        except Exception as e:
           
            print(f"\033[91m[Fatal Error] LLM Generation failed: {e}\033[0m")
            raise e