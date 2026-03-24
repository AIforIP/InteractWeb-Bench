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

# ── 配置更加健壮的 HTTP 客户端 (核心修复点) ──────────────────────────────────────
# 1. 增加 timeout 和 底层 transport 重试
robust_transport = httpx.HTTPTransport(retries=3)
robust_timeout = 60.0

# 2. 初始化 OpenAI 客户端 (带重试)
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
)

# 3. 初始化 Anthropic 客户端 (带重试)
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


# ── 【新增】动态本地模型路由 ──────────────────────────────────────────────────
def get_local_client(model_name: str):
    """
    根据模型名称判断是否为本地部署的模型，并分配对应的端口。
    如果你未来部署了多个本地模型，可以在这里直接添加端口映射。
    """
    model_lower = model_name.lower()
    port = None

    # 将 Qwen 指向 8024 端口
    if "qwen" in model_lower:
        port = 8024
    # 为未来可能部署的 Llama 预留 8025 端口
    elif "llama" in model_lower:
        port = 8025

    if port:
        return OpenAI(
            api_key="EMPTY",  # 本地 vLLM 服务通常不需要真实的 API Key
            base_url=f"http://localhost:{port}/v1",
            http_client=httpx.Client(
                base_url=f"http://localhost:{port}/v1",
                follow_redirects=True,
                timeout=robust_timeout,
                transport=robust_transport
            )
        )
    return None


# ── Helper: 图片编码 ──────────────────────────────────────────────────────────
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


# ── Helper: MIME type 侦测 ────────────────────────────────────────────────────
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


# ── Helper: OpenAI 格式转 Anthropic 格式 ──────────────────────────────────────
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
                    # 处理普通 URL 或其他情况
                    media_type = "image/jpeg"
                    b64_data = ""  # Anthropic API 不直接支持 image_url URL，通常需要 base64

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


# ── Unified generation function (带自动重试) ───────────────────────────────────
def vlm_generation(messages: list[dict], model: str, **kwargs) -> str:
    """
    Route the request to Anthropic or OpenAI depending on `model`.
    Includes robust retry logic for network stability.
    """
    max_retries = 5
    base_delay = 2

    # 判断是否使用 Anthropic
    is_anthropic = any(tag in model.lower() for tag in ("anthropic", "claude"))

    for attempt in range(max_retries):
        try:
            # ── Branch A: Anthropic ──
            if is_anthropic:
                if not anthropic_client:
                    raise ValueError("Anthropic API Key not set but Anthropic model requested.")

                system_text, claude_msgs = _convert_to_anthropic(messages)
                response = anthropic_client.messages.create(
                    model=model,
                    system=system_text,
                    messages=claude_msgs,
                    max_tokens=kwargs.pop("max_tokens", 4096),
                    **kwargs,
                )
                return response.content[0].text

            # ── Branch B: OpenAI-Compatible (默认) ──
            # VLM 调用通常需要较大的 max_tokens
            if "max_tokens" not in kwargs:
                kwargs["max_tokens"] = 1000

            # 检查是否为配置在本地白名单中的模型
            local_client = get_local_client(model)

            if local_client:
                # 走本地 vLLM 服务
                chat_response = local_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    **kwargs,
                )
            else:
                # 走云端 API
                chat_response = openai_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    **kwargs,
                )

            return chat_response.choices[0].message.content

        # ── 错误捕获与重试逻辑 ──
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


# [新增] 带压缩的图片编码函数
def compress_and_encode_image(image_path, max_size=(800, 800), quality=70):
    """
    压缩图片并转为 Base64。
    - max_size: 最大宽高，超过会等比缩放。
    - quality: JPEG 压缩质量 (1-100)，越低越模糊但 Token 越少。
    """
    try:
        with Image.open(image_path) as img:
            # 1. 转换格式 (PNG -> RGB for JPEG)
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')

            # 2. 调整尺寸 (保持比例)
            img.thumbnail(max_size, Image.Resampling.LANCZOS)

            # 3. 压缩并转为 Bytes
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=quality)

            # 4. 转 Base64
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Image compression failed: {e}. Falling back to raw encoding.")
        return encode_image(image_path)