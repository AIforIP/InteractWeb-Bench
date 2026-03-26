import os
import sys
import time
import random
import httpx
from openai import OpenAI, APIConnectionError, RateLimitError, APITimeoutError
from dotenv import load_dotenv

load_dotenv()

# --- 1. 配置更加健壮的 HTTP 客户端 ---
# 自动检测环境变量，优先使用 OPENAILIKE 系列（如果存在），否则使用标准 OPENAI
api_key = os.environ.get("OPENAILIKE_API_KEY") or os.environ.get("OPENAI_API_KEY")
base_url = os.environ.get("OPENAILIKE_BASE_URL") or os.environ.get("OPENAI_BASE_URL")

if not api_key:
    raise ValueError("API Key not found! Please set OPENAILIKE_API_KEY or OPENAI_API_KEY in .env")

# 使用 httpx 自定义客户端，增加超时时间和底层重试
robust_timeout = 300
robust_transport = httpx.HTTPTransport(retries=3)

http_client = httpx.Client(
    base_url=base_url,
    follow_redirects=True,
    timeout=robust_timeout,  # 增加超时时间到 300 秒，防止大模型生成长代码时断开
    transport=robust_transport  # 在 TCP/SSL 协议层自动重试 3 次
)

client = OpenAI(
    api_key=api_key,
    base_url=base_url,
    http_client=http_client
)


def get_local_client(model_name: str):
    """
    根据 .env 中的 LOCAL_MODELS_MAP 配置，动态判断当前模型是否为本地部署。
    支持本地模型与 API 模型的完美混用。
    """
    # 1. 从环境变量获取映射字符串
    # 例: "Qwen3.5-9B=http://localhost:8024/v1,llama3=http://localhost:8025/v1"
    mapping_str = os.environ.get("LOCAL_MODELS_MAP", "")
    if not mapping_str.strip():
        return None

    # 2. 解析为字典 (将模型名转为小写以实现忽略大小写的匹配)
    local_route_dict = {}
    for item in mapping_str.split(","):
        if "=" in item:
            m_name, m_url = item.split("=", 1)
            local_route_dict[m_name.strip().lower()] = m_url.strip()

    # 3. 检查当前模型是否在本地白名单中
    model_lower = model_name.lower()
    if model_lower in local_route_dict:
        local_base_url = local_route_dict[model_lower]
        # 如果命中，返回一个指向本地端口的独立 Client
        return OpenAI(
            api_key="EMPTY",  # 本地 vLLM 不需要真实 Key
            base_url=local_base_url,
            http_client=httpx.Client(
                base_url=local_base_url,
                follow_redirects=True,
                timeout=robust_timeout,  # 使用文件顶部定义好的健壮 timeout
                transport=robust_transport  # 使用文件顶部定义好的健壮 transport
            )
        )

    # 4. 如果不在列表中，返回 None，外层代码会自动使用配置了云端 API_KEY 的主 Client
    return None


def llm_generation(messages, model, max_tokens=-1, max_completion_tokens=-1, temperature=0.5):
    """
    Generate text using LLM with robust retry logic.
    """

    # 定义重试参数
    max_retries = 5
    base_delay = 2  # 基础等待时间 2 秒

    for attempt in range(max_retries):
        try:
            # -----------------------------------------------------------------
            # 2. 构造请求参数 (使用字典解包简化代码)
            # -----------------------------------------------------------------
            params = {
                "model": model,
                "messages": messages,
                "temperature": temperature
            }

            # 兼容不同的 token 参数名 (O1 模型 vs 标准模型)
            if max_completion_tokens > 0:
                params["max_completion_tokens"] = max_completion_tokens
            elif max_tokens > 0:
                params["max_tokens"] = max_tokens

            # -----------------------------------------------------------------
            # 3. 发起调用 (动态路由)
            # -----------------------------------------------------------------
            local_client = get_local_client(model)

            if local_client:
                # 命中本地白名单，调用本地 vLLM 接口
                chat_response = local_client.chat.completions.create(**params)
            else:
                # 走正常的云端 API
                chat_response = client.chat.completions.create(**params)

            return chat_response.choices[0].message.content

        except (APIConnectionError, APITimeoutError, RateLimitError, httpx.ConnectError) as e:
            # -----------------------------------------------------------------
            # 4. 捕获网络/限流错误并重试
            # -----------------------------------------------------------------
            print(f"\033[93m[Warn] LLM Network Error (Attempt {attempt + 1}/{max_retries}): {e}\033[0m")

            if attempt == max_retries - 1:
                # 如果是最后一次尝试依然失败，则抛出异常，让上层知道失败了
                print(f"\033[91m[Error] Max retries reached. Failing.\033[0m")
                raise e

            # 指数退避: 2s, 4s, 8s... 加上一点随机抖动防止并发冲突
            sleep_time = (base_delay * (2 ** attempt)) + random.uniform(0, 1)
            print(f"Retrying in {sleep_time:.2f} seconds...")
            time.sleep(sleep_time)

        except Exception as e:
            # 其他未知错误（如参数错误、认证错误）通常重试没用，直接抛出
            print(f"\033[91m[Fatal Error] LLM Generation failed: {e}\033[0m")
            raise e