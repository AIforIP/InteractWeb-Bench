import os
import openai
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

import sys
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from prompts.system import system_prompt

# --- 修复从这里开始 ---
import httpx  # 1. 导入 httpx

# 2. 使用 httpx.Client 初始化
client = OpenAI(
    api_key=os.environ["OPENAILIKE_FB_API_KEY"],
    base_url=os.environ["OPENAILIKE_FB_BASE_URL"],
    http_client=httpx.Client(
        base_url=os.environ["OPENAILIKE_FB_BASE_URL"],
        follow_redirects=True,
    )
)
# --- 修复到这里结束 ---


def llm_fb_generation(messages, model, max_tokens=-1, max_completion_tokens=-1):
    try:
        # -----------------------------------------------------------------
        # 1. 这是您正在使用的 API 调用逻辑
        # -----------------------------------------------------------------
        if max_tokens > 0:
            chat_response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
            )
        elif max_completion_tokens > 0:
            chat_response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=max_completion_tokens,
            )
        else:
            chat_response = client.chat.completions.create(
                model=model,
                messages=messages,
            )

        # -----------------------------------------------------------------
        # 2. 正确的返回方式是 .choices[0].message.content
        # -----------------------------------------------------------------
        return chat_response.choices[0].message.content

    # -----------------------------------------------------------------
    # 3. 必须返回错误字符串，以防止返回 None。
    # -----------------------------------------------------------------
    except Exception as e:
        print(f"Error in LLM FB generation: {e}")
        return str(e) #