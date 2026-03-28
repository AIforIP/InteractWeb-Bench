import platform
import argparse
import time
import json
import re
import os
import shutil
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any, List
import tempfile
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

# =========================================================
# 引入 Playwright (替换掉所有的 Selenium 引入)
# =========================================================
from playwright.sync_api import sync_playwright

from .webvoyager_prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_TEXT_ONLY
from openai import OpenAI
from .webvoyager_utils import (
    get_web_element_rect,
    encode_image,
    extract_information,
    print_message,
    get_webarena_accessibility_tree,
    get_pdf_retrieval_ans_from_assistant,
    clip_message_and_obs,
    clip_message_and_obs_text_only,
)

import sys
import base64
import binascii

from anthropic import Anthropic
from types import SimpleNamespace
from typing import Tuple, Optional


# -----------------------------------------
# Helper: convert Claude response ➜ OpenAI-like
# -----------------------------------------
def _anthropic_to_openai(claude_resp) -> SimpleNamespace:
    prompt_tokens = claude_resp.usage.input_tokens
    completion_tokens = claude_resp.usage.output_tokens
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )

    if isinstance(claude_resp.content, list):
        text_blocks = [c.text for c in claude_resp.content if hasattr(c, "text")]
        content = "".join(text_blocks)
    else:
        content = claude_resp.content

    message = SimpleNamespace(role="assistant", content=content)
    choice = SimpleNamespace(index=0,
                             finish_reason=getattr(claude_resp, "stop_reason", None),
                             message=message)

    openai_like = SimpleNamespace(
        id=claude_resp.id,
        object="chat.completion",
        created=int(time.time()),
        model=claude_resp.model,
        usage=usage,
        choices=[choice],
    )
    return openai_like


def _detect_media_type(b64: str, fallback: str = "image/jpeg") -> str:
    try:
        hdr = base64.b64decode(b64[:64], validate=False)
    except binascii.Error:
        return fallback

    if hdr.startswith(b"\xFF\xD8\xFF"):               return "image/jpeg"
    if hdr.startswith(b"\x89PNG\r\n\x1A\n"):          return "image/png"
    if hdr.startswith(b"GIF87a") or hdr.startswith(b"GIF89a"): return "image/gif"
    if hdr[0:4] == b"RIFF" and hdr[8:12] == b"WEBP":  return "image/webp"
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
                media_prefix, b64_data = url.split(";base64,", maxsplit=1)
                media_type = _detect_media_type(b64_data, fallback=media_prefix.replace("data:", ""))

                new_msg["content"].append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": media_type, "data": b64_data},
                })
        converted.append(new_msg)
    return system_text, converted


def setup_logger(folder_path: str):
    """Initialise a fresh logger that writes to *folder_path*/agent.log."""
    os.makedirs(folder_path, exist_ok=True)
    log_file_path = os.path.join(folder_path, "agent.log")

    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    # 1. 写入日志文件，并强制指定 utf-8 编码，防止 Windows GBK 报错
    handler = logging.FileHandler(log_file_path, encoding='utf-8')
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.setLevel(logging.INFO)

    # 2. 注释或删除掉控制台输出处理器 (StreamHandler)
    # 这样命令行就不会再被大量 Prompt Tokens 和大模型的 Base64 字典刷屏了
    # console_handler = logging.StreamHandler()
    # console_handler.setFormatter(formatter)
    # logger.addHandler(console_handler)


# ────────────────────────────────────────────────────────────────────────────────
# Prompt‑formatting helpers
# ────────────────────────────────────────────────────────────────────────────────

def format_msg(it, init_msg, pdf_obs, warn_obs, web_img_b64, web_text):
    if it == 1:
        init_msg += (
                "I've provided the tag name of each element and the text it contains (if text exists). "
                "Note that <textarea> or <input> may be textbox, but not exactly. "
                "Please focus more on the screenshot and then refer to the textual information.\n" + web_text
        )
        init_msg_format = {"role": "user", "content": [{"type": "text", "text": init_msg}]}
        init_msg_format["content"].append(
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{web_img_b64}"}})
        return init_msg_format
    else:
        if not pdf_obs:
            curr_msg = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Observation:{warn_obs} please analyze the attached screenshot and give the Thought and Action. I've provided the tag name of each element and the text it contains (if text exists). Note that <textarea> or <input> may be textbox, but not exactly. Please focus more on the screenshot and then refer to the textual information.\n" + web_text,
                    },
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{web_img_b64}"}},
                ],
            }
        else:
            curr_msg = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Observation: {pdf_obs} Please analyze the response given by Assistant, then consider whether to continue iterating or not. The screenshot of the current page is also attached, give the Thought and Action. I've provided the tag name of each element and the text it contains (if text exists). Note that <textarea> or <input> may be textbox, but not exactly. Please focus more on the screenshot and then refer to the textual information.\n" + web_text,
                    },
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{web_img_b64}"}},
                ],
            }
        return curr_msg

def format_msg_text_only(it, init_msg, pdf_obs, warn_obs, ac_tree):
    if it == 1:
        init_msg_format = {"role": "user", "content": init_msg + "\n" + ac_tree}
        return init_msg_format
    else:
        if not pdf_obs:
            curr_msg = {"role": "user",
                        "content": f"Observation:{warn_obs} please analyze the accessibility tree and give the Thought and Action.\n" + ac_tree}
        else:
            curr_msg = {"role": "user",
                        "content": f"Observation: {pdf_obs} Please analyze the response given by Assistant, then consider whether to continue iterating or not. The accessibility tree of the current page is also given, give the Thought and Action.\n" + ac_tree}
        return curr_msg


def call_gpt4v_api(args, openai_client, anthropic_client, messages):
    retry_times = 0
    while True:
        try:
            logging.info("Calling %s API…", args.api_model)
            is_anthropic = any(tag in args.api_model.lower() for tag in ("anthropic", "claude"))
            if is_anthropic:
                system_text, claude_msgs = _convert_to_anthropic(messages)
                claude_response = anthropic_client.messages.create(
                    model=args.api_model, system=system_text, messages=claude_msgs, max_tokens=2048, timeout=60
                )
                openai_response = _anthropic_to_openai(claude_response)
            else:
                openai_response = openai_client.chat.completions.create(
                    model=args.api_model, messages=messages, max_tokens=2048, seed=args.seed, timeout=60
                )

            prompt_tokens = openai_response.usage.prompt_tokens
            completion_tokens = openai_response.usage.completion_tokens
            logging.info("Prompt Tokens: %s; Completion Tokens: %s", prompt_tokens, completion_tokens)
            return prompt_tokens, completion_tokens, False, openai_response

        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            logging.warning("Error %s: %s, retrying…", error_type, error_msg)

            if "RateLimit" in error_type or "429" in error_msg:
                time.sleep(10)
            elif "APIError" in error_type:
                time.sleep(15)
            elif any(k in error_type for k in ["Timeout", "Connection"]) or any(k in error_msg for k in ["502", "503"]):
                time.sleep(10)
            elif "InvalidRequest" in error_type or "400" in error_msg:
                logging.error("Fatal API Error (e.g. 400 Bad Request). Cannot recover.")
                return None, None, True, None
            else:
                time.sleep(5)

        retry_times += 1
        if retry_times >= 10:
            logging.error("Retry limit reached while calling API")
            return None, None, True, None


# ────────────────────────────────────────────────────────────────────────────────
# 🚀 Playwright Action 统一控制层
# ────────────────────────────────────────────────────────────────────────────────

def exec_action_click(info, web_ele, page):
    """Playwright 坐标点击法，无视物理遮挡"""
    page.mouse.click(web_ele["x"], web_ele["y"])
    time.sleep(3)


def exec_action_type(info, web_ele, page):
    """Playwright 强制覆盖输入法"""
    warn_obs = ""
    type_content = info["content"]

    # 获取焦点
    page.mouse.click(web_ele["x"], web_ele["y"])
    time.sleep(0.5)

    # 操作系统级全选清除
    mod_key = "Meta" if platform.system() == "Darwin" else "Control"
    page.keyboard.press(f"{mod_key}+A")
    page.keyboard.press("Backspace")
    time.sleep(0.5)

    # 键入新内容并提交
    page.keyboard.type(type_content)
    time.sleep(1)
    page.keyboard.press("Enter")
    time.sleep(6)  # 留给网络加载的时间
    return warn_obs


def exec_action_scroll(info, web_eles, page, args):
    """Playwright 页面或元素范围滚动"""
    scroll_ele_number = info["number"]
    scroll_content = info["content"]

    # 计算滚动的像素步长
    delta_y = args.window_height * 2 // 3 if scroll_content == 'down' else -(args.window_height * 2 // 3)

    if scroll_ele_number == "WINDOW":
        page.mouse.wheel(0, delta_y)
    else:
        web_ele = web_eles[int(scroll_ele_number)]
        # 先将鼠标移动到该元素的中心区域
        page.mouse.move(web_ele["x"], web_ele["y"])
        # 然后滚动鼠标滚轮
        page.mouse.wheel(0, delta_y)
    time.sleep(3)


# =========================================================================
ui_limit_prompt_template = (
    "You have reached the maximum number of allowed interactions with the website.\n\n"
    "Please evaluate the outcome of your attempts based on the Testing Checklist provided at the beginning of our conversation.\n"
    "Now, please stop exploring and output your final report using the 'ANSWER' action.\n"
    "You MUST strictly follow the format: [FAILED ID: X] Reason: ... or [PASSED ID: X] Reason: ... for every ID requested earlier. Do NOT use XML. Keep your reasons concise.\n"
    "Example format:\n"
    "Thought: Evaluation complete.\n"
    "Action: ANSWER;\n"
    "[FAILED ID: 0] Reason: The feature could not be tested because clicking the button had no response.\n"
    "[PASSED ID: 1] Reason: The navigation link was visible and correct."
)


# ────────────────────────────────────────────────────────────────────────────────
# 🚀 核心流程引擎 (全面迁移至 Playwright)
# ────────────────────────────────────────────────────────────────────────────────

def run_single_task(task: Dict[str, Any], args_dict: Dict[str, Any]):
    """Run one task in an isolated Playwright process."""
    args = argparse.Namespace(**args_dict)
    task_dir = os.path.join(args.output_dir, f"task{task['id']}")
    if os.path.isfile(os.path.join(task_dir, "interact_messages.json")):
        return

    setup_logger(task_dir)
    logging.info("########## TASK%s ##########", task["id"])

    dynamic_openai_url = getattr(args, "vlm_base_url", os.environ.get("OPENAILIKE_VLM_BASE_URL", ""))
    dynamic_openai_key = getattr(args, "vlm_api_key", os.environ.get("OPENAILIKE_VLM_API_KEY", "sk-xxx"))

    client = OpenAI(
        api_key=dynamic_openai_key,
        base_url=dynamic_openai_url
    )

    dynamic_anthropic_url = getattr(args, "anthropic_base_url",
                                    os.environ.get("ANTHROPIC_VLM_BASE_URL", "https://api.anthropic.com/v1"))
    dynamic_anthropic_key = getattr(args, "anthropic_api_key", os.environ.get("ANTHROPIC_VLM_API_KEY", ""))

    anthropic_client = Anthropic(
        api_key=dynamic_anthropic_key,
        base_url=dynamic_anthropic_url
    )
    os.makedirs(args.download_dir, exist_ok=True)
    for f in os.listdir(args.download_dir):
        fp = os.path.join(args.download_dir, f)
        if os.path.isfile(fp): os.remove(fp)

    # =======================================================
    # Playwright 启动并创建干净上下文
    # =======================================================
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=args.headless,
            args=["--no-sandbox", "--disable-dev-shm-usage"]
        )
        context = browser.new_context(
            viewport={"width": args.window_width, "height": args.window_height},
            device_scale_factor=1,
            accept_downloads=True,
            user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        )
        page = context.new_page()

        # [自动拦截原生弹窗，并呼叫子智能体动态推断输入内容]
        def handle_dialog(dialog):
            logging.info(f"[SYSTEM DIALOG] Type: {dialog.type}, Message: {dialog.message}")
            if dialog.type == "prompt":
                try:
                    logging.info("   -> 触发原生输入框，正在请求 LLM 动态推断输入内容...")

                    # 组装给子智能体的提示词
                    sys_msg = "You are an automated web testing sub-agent. Your job is to decide what text to input into a website's prompt dialog based on the main task."
                    user_msg = (
                        f"Main Task Instruction: {task['ques']}\n\n"
                        f"The website popup currently asks: '{dialog.message}'.\n"
                        "What short text should be inputted to satisfy the task? "
                        "Reply strictly with ONLY the exact text value to input, no quotes, no punctuation, no explanations."
                    )

                    # 临时调用一个纯文本接口进行快速推断 (利用外层已实例化的 client)
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",  # 使用小模型保证速度和低成本
                        messages=[
                            {"role": "system", "content": sys_msg},
                            {"role": "user", "content": user_msg}
                        ],
                        max_tokens=30,
                        temperature=0,
                        timeout=10  # 防止网络抖动卡死
                    )

                    # 提取动态生成的输入值，并清除多余的引号
                    dynamic_input = response.choices[0].message.content.strip().strip("'\"")
                    logging.info(f"   -> 子智能体决定填入: 【{dynamic_input}】")

                    # 将动态生成的内容填入弹窗并确认
                    dialog.accept(dynamic_input)

                except Exception as e:
                    logging.error(f"   -> 动态推理失败: {e}。使用安全回退方案。")
                    dialog.accept("https://example.com")  # 兜底方案
            else:
                # 普通的 alert 或 confirm 直接确认
                dialog.accept()

        page.on("dialog", handle_dialog)

        try:
            # 开启网络加载保护，防止卡死
            page.goto(task["web"], timeout=45000, wait_until="domcontentloaded")
            time.sleep(3)
            # 点击下 body 让页面获取焦点
            page.mouse.click(10, 10)
        except Exception as e:
            logging.error("Error: Cannot access the website %s. %s", task["web"], str(e))
            browser.close()
            return

        download_files: List[str] = []
        fail_obs = ""
        pdf_obs = ""
        warn_obs = ""
        pattern = r"Thought:|Action:|Observation:"

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT if not args.text_only else SYSTEM_PROMPT_TEXT_ONLY}
        ]

        obs_prompt = (
            "Observation: please analyze the attached screenshot and give the Thought and Action. "
            if not args.text_only else "Observation: please analyze the accessibility tree and give the Thought and Action."
        )

        init_msg = f"Now given a task: {task['ques']}  Please interact with https://www.example.com and get the answer. \n"
        init_msg = init_msg.replace("https://www.example.com", task["web"])
        init_msg += obs_prompt

        it = 0
        accumulate_prompt_token = 0
        accumulate_completion_token = 0

        while it < args.max_iter:
            logging.info("Iter: %s", it)
            it += 1

            if it == args.max_iter:
                curr_msg = {"role": "user", "content": ui_limit_prompt_template.format(instruction=task['ques'])}
                messages.append(curr_msg)
            elif not fail_obs:
                try:
                    if not args.text_only:
                        # 从 Utils 接收解析后的坐标字典 web_eles
                        web_eles, web_eles_text = get_web_element_rect(page, fix_color=args.fix_box_color)
                    else:
                        accessibility_tree_path = os.path.join(task_dir, f"accessibility_tree{it}")
                        # 兼容处理：如果 text_only 模式必须用，这里向旧版方法传递 page 对象可能需要修改 utils_webarena
                        ac_tree, obs_info = get_webarena_accessibility_tree(page, accessibility_tree_path)
                except Exception as e:
                    logging.error("Browser error when capturing page: %s", e)
                    break

                img_path = os.path.join(task_dir, f"screenshot{it}.png")
                page.screenshot(path=img_path)

                if (not args.text_only) and args.save_accessibility_tree:
                    accessibility_tree_path = os.path.join(task_dir, f"accessibility_tree{it}")
                    get_webarena_accessibility_tree(page, accessibility_tree_path)

                b64_img = encode_image(img_path)

                if not args.text_only:
                    curr_msg = format_msg(it, init_msg, pdf_obs, warn_obs, b64_img, web_eles_text)
                else:
                    curr_msg = format_msg_text_only(it, init_msg, pdf_obs, warn_obs, ac_tree)
                messages.append(curr_msg)
            else:
                messages.append({"role": "user", "content": fail_obs})

            messages = clip_message_and_obs(messages, args.max_attached_imgs)
            prompt_tokens, completion_tokens, gpt_call_error, openai_response = call_gpt4v_api(args, client,
                                                                                               anthropic_client,
                                                                                               messages)

            if gpt_call_error: break
            accumulate_prompt_token += prompt_tokens
            accumulate_completion_token += completion_tokens

            gpt_4v_res = openai_response.choices[0].message.content
            messages.append({"role": "assistant", "content": gpt_4v_res})

            # 清理标注红框 (无需操作 Python 的 DOM 句柄，执行一行 JS 即可全清)
            if not args.text_only:
                page.evaluate("document.querySelectorAll('.webvoyager-mark').forEach(e => e.remove())")

            try:
                assert "Thought:" in gpt_4v_res and "Action:" in gpt_4v_res
            except AssertionError:
                fail_obs = "Format ERROR: Both 'Thought' and 'Action' should be included in your reply."
                continue

            chosen_action = re.split(pattern, gpt_4v_res)[2].strip()
            action_key, info = extract_information(chosen_action)

            fail_obs = ""
            pdf_obs = ""
            warn_obs = ""

            try:
                if action_key == "click":
                    if not args.text_only:
                        click_ele_number = int(info[0])
                        web_ele = web_eles[click_ele_number]
                    else:
                        click_ele_number = info[0]
                        web_ele = {"x": obs_info[click_ele_number]["union_bound"][0] +
                                        obs_info[click_ele_number]["union_bound"][2] // 2,
                                   "y": obs_info[click_ele_number]["union_bound"][1] +
                                        obs_info[click_ele_number]["union_bound"][3] // 2}
                    exec_action_click(info, web_ele, page)

                    # Check for PDF downloads (沿用轮询检测机制)
                    current_files = sorted(os.listdir(args.download_dir))
                    if current_files != download_files:
                        time.sleep(5)
                        current_files = sorted(os.listdir(args.download_dir))
                        new_pdfs = [pdf for pdf in current_files if pdf not in download_files and pdf.endswith(".pdf")]
                        if new_pdfs:
                            pdf_file = new_pdfs[0]
                            pdf_obs = get_pdf_retrieval_ans_from_assistant(client,
                                                                           os.path.join(args.download_dir, pdf_file),
                                                                           task["ques"])
                            shutil.copy(os.path.join(args.download_dir, pdf_file), task_dir)
                            pdf_obs = "You downloaded a PDF file, I ask the Assistant API to answer the task based on the PDF file and get the following response: " + pdf_obs
                        download_files[:] = current_files

                elif action_key == "wait":
                    time.sleep(5)

                elif action_key == "type":
                    if not args.text_only:
                        type_ele_number = int(info["number"])
                        web_ele = web_eles[type_ele_number]
                    else:
                        type_ele_number = info["number"]
                        web_ele = {
                            "x": obs_info[type_ele_number]["union_bound"][0] + obs_info[type_ele_number]["union_bound"][
                                2] // 2,
                            "y": obs_info[type_ele_number]["union_bound"][1] + obs_info[type_ele_number]["union_bound"][
                                3] // 2}
                    warn_obs = exec_action_type(info, web_ele, page)
                    if "wolfram" in task["web"]: time.sleep(5)

                elif action_key == "scroll":
                    if not args.text_only:
                        exec_action_scroll(info, web_eles, page, args)
                    else:
                        exec_action_scroll(info, None, page, args)

                elif action_key == "goback":
                    page.go_back()
                    time.sleep(2)

                elif action_key == "google":
                    page.goto("https://www.google.com/")
                    time.sleep(2)

                elif action_key == "answer":
                    logging.info(info["content"])
                    break

                else:
                    raise NotImplementedError(f"Unknown action {action_key}")

            except Exception as e:
                logging.error("Driver error info: %s", e)
                fail_obs = "The action you have chosen cannot be executed. Please double-check if you have selected the wrong Numerical Label or Action or Action format. Then provide the revised Thought and Action."
                time.sleep(2)

        print_message(messages, task_dir)
        logging.info("Total cost: %.4f",
                     accumulate_prompt_token / 1000 * 0.01 + accumulate_completion_token / 1000 * 0.03)
        browser.close()

    return messages


if __name__ == "__main__":
    example_task = {
        "id": 1,
        "web": "http://localhost:36419/",
        "ques": "search for APPL",
    }
    args = {
        "output_dir": "experiment_results/debug_output",
        "download_dir": "experiment_results/debug_downloads",
        "window_width": 1200,
        "window_height": 800,
        "headless": True,
        "text_only": False,
        "fix_box_color": False,
        "save_accessibility_tree": False,
        "max_attached_imgs": 3,
        "max_iter": 10,
        "api_model": "claude-3-5-sonnet-20241022",
        "seed": 42,
    }
    run_single_task(example_task, args)