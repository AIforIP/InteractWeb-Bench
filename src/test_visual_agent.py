import os
import time
import json
import re
from playwright.sync_api import sync_playwright

from utils.vlm_generation import vlm_generation, encode_image
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
    print(f"Starting pure visual Agent unit test (synchronous dynamic dialog handling)...")
    print(f"Test instruction: {ground_truth_instruction}")

    log_dir = "./standalone_visual_logs"
    os.makedirs(log_dir, exist_ok=True)

    env = BrowserEnv(project_dir=".", log_dir=log_dir, start_cmd="")
    local_server_process = None
    actual_target_url = None

    system_dialog_records = []

    def custom_start(dummy_url=None):
        nonlocal local_server_process, actual_target_url

        if project_dir and start_cmd:
            print(f"Automatically executing '{start_cmd}' in directory {project_dir} (dynamic sniffing mode)...")
            log_file = os.path.join(log_dir, "standalone_service.log")
            local_server_process, log_path = start_background_service(start_cmd, project_dir, log_file)

            try:
                actual_target_url = wait_for_url_in_log(log_path, timeout=30)
                print(f"Local test service is ready, detected running address: {actual_target_url}")
            except TimeoutError:
                print(f"Warning: Timeout while waiting for service to start. Please check {log_file}")
                return

        print(f"Accessing target URL: {actual_target_url}")
        env.playwright = sync_playwright().start()
        env.browser = env.playwright.chromium.launch(headless=False)

        env.context = env.browser.new_context(
            viewport={'width': 1280, 'height': 800},
            device_scale_factor=1
        )
        env.page = env.context.new_page()

        def handle_dialog(dialog):
            print(f"[System Dialog Detected] Type: {dialog.type}, Message: {dialog.message}")
            if dialog.type == "prompt":
                print("[Sub-agent] Inferring dialog input content based on task instruction...")
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
                    print(f"[Sub-agent] Decides to fill in: {dynamic_input}")

                    system_dialog_records.append(
                        f"Auto-filled input prompt ('{dialog.message}') with text: '{dynamic_input}'")
                    dialog.accept(dynamic_input)
                except Exception as e:
                    print(f"[Sub-agent] Inference failed: {e}, using default fallback value")
                    system_dialog_records.append(
                        f"Auto-filled input prompt ('{dialog.message}') with fallback text: 'Test Value'")
                    dialog.accept("Test Value")
            else:

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
            raise Exception("Failed to obtain the target URL, exiting the test.")

        history_text = ""

        for step_idx in range(max_steps):
            print(f"\n--- Test Step {step_idx + 1}/{max_steps} ---")

            img_path = env.capture_observation(step_idx, draw_som=False)
            b64_img = encode_image(img_path)

            sys_feedback = ""
            if system_dialog_records:
                sys_feedback = "\n [SYSTEM NOTIFICATIONS (DO NOT FAIL TEST BASED ON THIS)]:\n"
                for note in system_dialog_records:
                    sys_feedback += f"- {note}\n"
                system_dialog_records.clear()

            strict_criteria = ""
            sys_msg = INTERNAL_TEST_PROMPT.format(
                instruction=ground_truth_instruction,
                criteria=strict_criteria,
                context_summary="No previous context.",
                step_idx=step_idx + 1,
                max_steps=max_steps
            )


            step_context = f"Action History:\n{history_text}\n"
            if sys_feedback:
                step_context += f"{sys_feedback}\n"

            print("Waiting for VLM to make a decision...")

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

            print(f"Agent's internal monologue:\n{response}")
            print(f"Agent decides to execute: {action}")

            if "Finish" in action or "Fail" in action:
                print(f"Test finished")
                break

            result = env.execute_action(action)
            print(f"Underlying execution feedback: {result}")

            history_text += f"Step {step_idx}: {action} (Result: {result})\n"
            time.sleep(2)

    except Exception as e:
        print(f"Test encountered an error: {e}")
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
        print(f"Successfully loaded task ID: {task_id}")
        print(f"Service will be started using dynamic sniffing")
        print(f"=====================================")

        run_standalone_visual_test(
            ground_truth_instruction=instruction,
            project_dir=PROJECT_DIR,
            start_cmd=START_CMD
        )
    except Exception as err:
        print(f"Initialization failed: {err}")