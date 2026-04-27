import os
import sys
import base64
import binascii
import time
import random
import httpx
from dotenv import load_dotenv

# ── Third-party SDKs ───────────────────────────────────────────────────────────
from openai import OpenAI, APIConnectionError, RateLimitError, APITimeoutError
from anthropic import Anthropic, APIConnectionError as AnthropicConnectionError

load_dotenv()

robust_transport = httpx.HTTPTransport(retries=3)
robust_timeout = 1800.0

openai_base_url = os.environ.get("OPENAILIKE_VLM_BASE_URL") or os.environ.get("OPENAILIKE_BASE_URL") or os.environ.get(
    "OPENAI_BASE_URL")
openai_api_key = os.environ.get("OPENAILIKE_VLM_API_KEY") or os.environ.get("OPENAILIKE_API_KEY") or os.environ.get(
    "OPENAI_API_KEY")

openai_client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_base_url,
    http_client=httpx.Client(
        base_url=openai_base_url,
        follow_redirects=True,
        timeout=robust_timeout,
        transport=robust_transport
    )
) if openai_api_key else None

anthropic_api_key = os.environ.get("ANTHROPIC_VLM_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
anthropic_base_url = os.environ.get("ANTHROPIC_VLM_BASE_URL") or "https://api.anthropic.com/v1"

anthropic_client = None
if anthropic_api_key:
    anthropic_client = Anthropic(
        api_key=anthropic_api_key,
        base_url=anthropic_base_url,
        http_client=httpx.Client(
            timeout=robust_timeout,
            transport=robust_transport
        )
    )


def get_local_client(model_name: str):

    raw_map = os.environ.get("LOCAL_MODELS_MAP", "")
    if not raw_map.strip():
        return None

    mapping_str = os.path.expandvars(raw_map)

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

def _call_with_stream(api_client, params: dict) -> str:
    full_content = []
    with api_client.chat.completions.create(**params, stream=True) as stream:
        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                print(delta.content, end="", flush=True)
                full_content.append(delta.content)
    print()
    return "".join(full_content)


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def _detect_media_type(b64: str, fallback: str = "image/jpeg") -> str:
    try:
        hdr = base64.b64decode(b64[:64], validate=False)
    except binascii.Error:
        return fallback

    if hdr.startswith(b"\xFF\xD8\xFF"): return "image/jpeg"
    if hdr.startswith(b"\x89PNG\r\n\x1A\n"): return "image/png"
    if hdr.startswith(b"GIF87a") or hdr.startswith(b"GIF89a"): return "image/gif"
    if hdr[0:4] == b"RIFF" and hdr[8:12] == b"WEBP": return "image/webp"
    return fallback



def _convert_to_anthropic(messages: list[dict]) -> tuple[str | None, list[dict]]:
    converted: list[dict] = []
    system_text: str | None = None

    for msg in messages:
        role = msg["role"]
        if role == "system":
            system_text = msg["content"]
            continue

        new_msg: dict = {"role": role, "content": []}
        if isinstance(msg["content"], str):
            new_msg["content"].append({"type": "text", "text": msg["content"]})
            converted.append(new_msg)
            continue

        for block in msg["content"]:
            if block["type"] == "text":
                new_msg["content"].append({"type": "text", "text": block["text"]})
            elif block["type"] == "image_url":
                url = block["image_url"]["url"]
                if ";base64," in url:
                    media_prefix, b64_data = url.split(";base64,", maxsplit=1)
                    media_type = _detect_media_type(b64_data, fallback=media_prefix.replace("data:", ""))
                else:
                    media_type = "image/jpeg"
                    b64_data = ""

                new_msg["content"].append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": b64_data,
                    },
                })
        converted.append(new_msg)
    return system_text, converted



def vlm_generation(messages: list[dict], model: str, max_tokens=-1, max_completion_tokens=-1, temperature=0.5,
                   base_url=None, api_key=None, stream=False, use_anthropic_sdk=False, **kwargs) -> str:
    """
    Route the request to Anthropic or OpenAI depending on `model`.
    Includes robust retry logic and supports dynamic isolated credentials.
    """
    max_retries = 5
    base_delay = 2

    for attempt in range(max_retries):
        try:

            params = {"model": model, "temperature": temperature}
            params.update(kwargs)


            if use_anthropic_sdk:

                if base_url and api_key:
                    active_anthropic_client = Anthropic(
                        api_key=api_key,
                        base_url=base_url,
                        http_client=httpx.Client(timeout=robust_timeout, transport=robust_transport)
                    )
                else:
                    active_anthropic_client = anthropic_client

                if not active_anthropic_client:
                    raise ValueError("Anthropic API Key not set but Anthropic SDK requested.")

                system_text, claude_msgs = _convert_to_anthropic(messages)

                final_max_tokens = max_completion_tokens if max_completion_tokens > 0 else (
                    max_tokens if max_tokens > 0 else 4096)


                response = active_anthropic_client.messages.create(
                    model=model,
                    system=system_text,
                    messages=claude_msgs,
                    max_tokens=final_max_tokens,
                    temperature=temperature,
                )
                return response.content[0].text


            if max_completion_tokens > 0:
                params["max_completion_tokens"] = max_completion_tokens
            elif max_tokens > 0:
                params["max_tokens"] = max_tokens
            else:
                params["max_tokens"] = 1000

            params["messages"] = messages


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
                active_client = openai_client
                if not active_client:
                    raise ValueError("OpenAI client not initialized. Missing API Key.")

            if stream:
                return _call_with_stream(active_client, params)
            else:
                chat_response = active_client.chat.completions.create(**params)
                return chat_response.choices[0].message.content

        except (APIConnectionError, APITimeoutError, RateLimitError, httpx.ConnectError, AnthropicConnectionError) as e:
            print(f"\033[93m[Warn] VLM Network Error (Attempt {attempt + 1}/{max_retries}): {e}\033[0m")

            if attempt == max_retries - 1:
                print(f"\033[91m[Error] Max retries reached for VLM. Failing.\033[0m")
                raise e

            sleep_time = (base_delay * (2 ** attempt)) + random.uniform(0, 1)
            print(f"Retrying in {sleep_time:.2f} seconds...")
            time.sleep(sleep_time)

        except Exception as e:
            print(f"\033[91m[Fatal Error] VLM Generation failed: {e}\033[0m")
            raise e


from PIL import Image
import io


def compress_and_encode_image(image_path, max_size=(800, 800), quality=70):

    try:
        with Image.open(image_path) as img:
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=quality)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Image compression failed: {e}. Falling back to raw encoding.")
        return encode_image(image_path)