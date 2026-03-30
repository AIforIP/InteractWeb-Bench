import os
import time
import json
import re
from playwright.sync_api import sync_playwright

from utils.vlm_generation import vlm_generation, encode_image
# 引入新架构的动态嗅探和清理工具，移除 is_port_open
from utils.execute_for_feedback import (
    BrowserEnv,
    start_background_service,
    wait_for_url_in_log,
    stop_process_tree
)
from agent.webgen_agent import INTERNAL_TEST_PROMPT


def extract_instruction_from_jsonl(filepath, index):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        if index < len(lines):
            data = json.loads(lines[index])
            return data.get("task_id", "unknown"), data.get("instruction", "")
    return "unknown", ""


def run_standalone_visual_test(ground_truth_instruction, project_dir=None, start_cmd=None,
                               vlm_model="gpt-4o-mini", max_steps=5):
    print(f"启动纯视觉 Agent 单测 (同步动态弹窗处理)...")
    print(f"测试指令: {ground_truth_instruction}")

    log_dir = "./standalone_visual_logs"
    os.makedirs(log_dir, exist_ok=True)

    env = BrowserEnv(project_dir=".", log_dir=log_dir, start_cmd="")
    local_server_process = None
    actual_target_url = None

    # 🌟 核心修改 1：为独立测试脚本也准备一个“系统信箱”
    system_dialog_records = []

    def custom_start(dummy_url=None):
        nonlocal local_server_process, actual_target_url

        if project_dir and start_cmd:
            print(f"正在目录 {project_dir} 自动执行 '{start_cmd}' (动态嗅探模式)...")
            log_file = os.path.join(log_dir, "standalone_service.log")
            local_server_process, log_path = start_background_service(start_cmd, project_dir, log_file)

            try:
                actual_target_url = wait_for_url_in_log(log_path, timeout=30)
                print(f"本地测试服务已成功就绪，嗅探到运行地址: {actual_target_url}")
            except TimeoutError:
                print(f"警告: 等待服务启动超时。请检查 {log_file}")
                return

        print(f"正在访问目标网址: {actual_target_url}")
        env.playwright = sync_playwright().start()
        env.browser = env.playwright.chromium.launch(headless=False)

        env.context = env.browser.new_context(
            viewport={'width': 1280, 'height': 800},
            device_scale_factor=1
        )
        env.page = env.context.new_page()

        # 🌟 核心修改 2：升级这里的 handle_dialog，把动作记录到信箱里
        def handle_dialog(dialog):
            print(f"[系统弹窗监测] 类型: {dialog.type}, 消息: {dialog.message}")
            if dialog.type == "prompt":
                print("[子智能体] 正在根据任务指令推断弹窗输入内容...")
                sub_agent_sys = "You are an automated web testing helper. Reply with ONLY the exact string to input into the prompt dialog to satisfy the user's goal."
                sub_agent_user = f"Task: {ground_truth_instruction}\nPopup Message: {dialog.message}\nWhat should I input?"

                try:
                    dynamic_input = vlm_generation(
                        model=vlm_model,
                        messages=[
                            {"role": "system", "content": sub_agent_sys},
                            {"role": "user", "content": sub_agent_user}
                        ]
                    ).strip().strip("'\"")
                    print(f"[子智能体] 决定填入: {dynamic_input}")

                    system_dialog_records.append(
                        f"Auto-filled input prompt ('{dialog.message}') with text: '{dynamic_input}'")
                    dialog.accept(dynamic_input)
                except Exception as e:
                    print(f"[子智能体] 推理失败: {e}，使用默认回退值")
                    system_dialog_records.append(
                        f"Auto-filled input prompt ('{dialog.message}') with fallback text: 'Test Value'")
                    dialog.accept("Test Value")
            else:
                # 处理纯展示的 Alert/Confirm 弹窗
                system_dialog_records.append(
                    f"A browser popup appeared saying: '{dialog.message}'. The system automatically clicked OK.")
                dialog.accept()

        env.page.on("dialog", handle_dialog)
        env.page.goto(actual_target_url, wait_until="domcontentloaded", timeout=15000)
        env.console_logs = []

    env.start = custom_start

    try:
        env.start(None)

        if not actual_target_url:
            raise Exception("无法获取目标网址，退出测试。")

        history_text = ""

        for step_idx in range(max_steps):
            print(f"\n--- 测试步骤 {step_idx + 1}/{max_steps} ---")

            img_path = env.capture_observation(step_idx, draw_som=False)
            b64_img = encode_image(img_path)

            # 🌟 核心修改 3：提取系统信箱内容，准备向模型汇报
            sys_feedback = ""
            if system_dialog_records:
                sys_feedback = "\n [SYSTEM NOTIFICATIONS (DO NOT FAIL TEST BASED ON THIS)]:\n"
                for note in system_dialog_records:
                    sys_feedback += f"- {note}\n"
                system_dialog_records.clear()  # 汇报完清空信箱

            strict_criteria = ""
            sys_msg = INTERNAL_TEST_PROMPT.format(
                instruction=ground_truth_instruction,
                criteria=strict_criteria,
                context_summary="No previous context.",
                step_idx=step_idx + 1,
                max_steps=max_steps
            )

            # 🌟 核心修改 4：将通知拼接进上下文
            step_context = f"Action History:\n{history_text}\n"
            if sys_feedback:
                step_context += f"{sys_feedback}\n"

            print("正在等待 VLM 思考决策...")

            response = vlm_generation(
                model=vlm_model,
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": [
                        {"type": "text", "text": step_context},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}}
                    ]}
                ]
            )

            action = "Wait"
            for line in response.split('\n'):
                if "Action:" in line:
                    action = line.split("Action:", 1)[1].strip()
                    break

            print(f"Agent 的内心独白:\n{response}")
            print(f"Agent 决定执行: {action}")

            if "Finish" in action or "Fail" in action:
                print(f"测试结束")
                break

            result = env.execute_action(action)
            print(f"底层执行反馈: {result}")

            history_text += f"Step {step_idx}: {action} (Result: {result})\n"
            time.sleep(2)

    except Exception as e:
        print(f"测试发生错误: {e}")
    finally:
        env.close()
        if local_server_process:
            stop_process_tree(local_server_process)


if __name__ == "__main__":
    PROJECT_DIR = r"E:\Agent_work\src\experiment_results\workspaces\000002_P-RAM"
    START_CMD = "npm run dev"
    JSONL_FILE_PATH = r"E:\Agent_work\src\data_generation\test_mini.jsonl"
    TEST_DATA_INDEX = 0

    try:
        task_id, instruction = extract_instruction_from_jsonl(JSONL_FILE_PATH, TEST_DATA_INDEX)
        print(f"=====================================")
        print(f"成功加载任务 ID: {task_id}")
        print(f"即将使用动态嗅探启动服务")
        print(f"=====================================")

        run_standalone_visual_test(
            ground_truth_instruction=instruction,
            project_dir=PROJECT_DIR,
            start_cmd=START_CMD
        )
    except Exception as err:
        print(f"初始化失败: {err}")