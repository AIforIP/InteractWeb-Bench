import os
import sys
import re
import json
import time
import argparse
import pandas as pd
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from webvoyager.run import run_single_task
from utils.execute_for_feedback import start_background_service, wait_for_url_in_log, stop_process_tree

try:
    from utils import llm_generation
except ImportError:
    pass


def evaluate_with_webvoyager(target_url: str, user_instruction: str, oracle_slots: list, task_id: str,
                             args_dict: dict, app_port: int = None) -> dict:

    print(f"   [System] Received evaluation task. Target URL: {target_url}")
    checklist_str = ""
    for idx, slot in enumerate(oracle_slots):
        checklist_str += f"Checklist ID [{idx}]:\n"
        checklist_str += f"- Task: {slot.get('original_task')}\n"
        checklist_str += f"- Expected Result: {slot.get('expected_result')}\n\n"

    eval_ques = (
        f"You are a strict QA engineer verifying if the website satisfies: '{user_instruction}'.\n\n"
        f"Testing Checklist:\n{checklist_str}\n"
        "INSTRUCTIONS:\n"
        "1. Explore the website purposefully to test EVERY item in the checklist.\n"
        "2. **ANTI-STUCK RULE (CRITICAL)**: If a feature is broken, missing, or an action does not work after 1 attempt, DO NOT get stuck retrying. DO NOT try workarounds. Immediately move on to test the NEXT checklist item.\n"
        "3. **STEP-WISE VERIFICATION**: Document your findings in your 'Thought' immediately using EXACTLY these formats:\n"
        "   - If an item works: `[PASSED ID: X] Reason: your brief proof`\n"
        "   - If an item is broken/missing: `[FAILED ID: X] Reason: why it failed`\n"
        "   - **FATAL RULE: You MUST start your response with the exact word 'Thought: ' followed by your reasoning in a single paragraph. Then, output 'Action: ' on a new line!**\n"
        "   Example format:\n"
        "   Thought: Clicking the date picker did nothing. [FAILED ID: 1] Reason: Date picker is unresponsive. I will stop trying this and check the background color next. [PASSED ID: 4] Reason: Background is papaya whip.\n"
        "   Action: Click [2]\n"
        "4. When you have tested or skipped all items, output your final action as: `Action: ANSWER; Exploration complete`.\n"
        "5. **IGNORE TESTING ARTIFACTS (CRITICAL)**: The numerical labels (e.g., [0], [1]) and colored bounding/dashed boxes on the screenshot are injected by our automated testing framework. You MUST IGNORE them.\n"
        "6. **AUTO-FILLED DIALOGS (CRITICAL)**: Our framework instantly auto-fills native browser popups in the background. If you click an 'Add' or 'Create' button and immediately see a new item (e.g., 'Test Input Value') appear on the page, consider the addition function PASSED. Do NOT fail the item just because you didn't see an input form."
    )

    task = {
        "id": task_id,
        "web": target_url,
        "ques": eval_ques
    }

    eval_limit_prompt = (
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
    args_dict["limit_prompt_template"] = eval_limit_prompt

    messages = run_single_task(task, args_dict)

    print("\n   [System] WebVoyager exploration completed. Extracting step-by-step evaluation results from the interaction trajectory...")

    model_results = {
        i: {"passed": False, "reason": "Not observed or tested during exploration"}
        for i in range(len(oracle_slots))
    }

    if messages:
        for msg in messages:
            if msg.get("role") == "assistant":
                content = msg.get("content", "")

                pattern = r'\[(PASSED|FAILED)\s+ID:\s*(\d+)\]\s*(?:Reason:)?\s*(.*?)(?=\[(?:PASSED|FAILED)|Action:|$)'
                matches = re.finditer(pattern, content, re.IGNORECASE | re.DOTALL)

                for m in matches:
                    status_str = m.group(1).upper()
                    idx_str = m.group(2).strip()
                    reason_text = m.group(3).strip().replace('\n', ' ')

                    if idx_str.isdigit():
                        idx = int(idx_str)
                        if idx in model_results:
                            if status_str == "PASSED" and not model_results[idx]["passed"]:
                                model_results[idx]["passed"] = True
                                model_results[idx]["reason"] = f"[Verified] {reason_text}"
                            elif status_str == "FAILED" and not model_results[idx]["passed"]:
                                model_results[idx]["passed"] = False
                                model_results[idx]["reason"] = f"[Failed] {reason_text}"

    passed_weight = 0.0
    total_weight = 0.0
    all_passed = True
    details = []

    for i, slot in enumerate(oracle_slots):
        weight = float(slot.get("final_weight", 1.0))
        total_weight += weight

        res = model_results.get(i)
        is_item_passed = res["passed"]
        reason = res["reason"]

        if is_item_passed:
            passed_weight += weight
        else:
            all_passed = False

        details.append({
            "task": slot.get("original_task"),
            "passed": is_item_passed,
            "weight": weight,
            "reason": reason
        })

    tcr = round((passed_weight / total_weight), 4) if total_weight > 0 else 0.0

    print(f"   [System] Extracted {sum(1 for d in details if d['passed'])} passed criteria.")

    return {
        "Success_Rate_SR": 1 if all_passed and total_weight > 0 else 0,
        "Task_Completion_Rate_TCR": tcr,
        "Details": details,
        "Raw_Trajectory": messages
    }

if __name__ == "__main__":
    import shutil

    print("\n Starting standalone test tasks (screen visualization mode / results not written to trajectory)...")

    RETEST_JSONL_PATH = r"/app/data/test_mini.jsonl"
    LOGS_ROOT_DIR = r"/app/experiment_results/Qwen3.5-9B/logs"
    WORKSPACES_ROOT_DIR = r"/app/experiment_results/Qwen3.5-9B/workspaces"

    if not os.path.exists(RETEST_JSONL_PATH):
        print(f" Cannot find file list to be repaired: {RETEST_JSONL_PATH}")
        sys.exit(1)


    tasks_to_retest = []
    with open(RETEST_JSONL_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                data = json.loads(line)
                t_id = str(data.get("task_id") or data.get("id") or data.get("original_id"))
                if t_id and t_id != "None":
                    tasks_to_retest.append(t_id)
            except Exception as e:
                print(f" Failed to parse jsonl, skipping this line: {e}")

    tasks_to_retest = list(set(tasks_to_retest))
    print(f"\n A total of {len(tasks_to_retest)} test tasks were loaded.")


    for current_idx, task_folder_name in enumerate(tasks_to_retest, 1):
        print(f"\n" + "=" * 60)
        print(f" [{current_idx}/{len(tasks_to_retest)}] Running visualization test task: {task_folder_name}")
        print("=" * 60)

        workspace_dir = os.path.join(WORKSPACES_ROOT_DIR, task_folder_name)
        history_json_path = os.path.join(LOGS_ROOT_DIR, task_folder_name, "interaction_history.json")
        server_process = None

        try:
            if not os.path.exists(history_json_path):
                print(f" Skipped: Cannot find history file {history_json_path}")
                continue
            if not os.path.exists(workspace_dir):
                print(f" Skipped: Cannot find workspace {workspace_dir}")
                continue

            with open(history_json_path, "r", encoding="utf-8") as f:
                history_data = json.load(f)

            test_user_instruction = history_data["trajectory"][1]["content"]
            last_turn = history_data["trajectory"][-1]
            test_oracle_slots = last_turn["debug_info"].get("oracle_slots_used_for_grading", [])

            if not test_oracle_slots:
                print(f" Skipped: oracle_slots grading criteria not found in logs.")
                continue

            print(f"   [System] Successfully loaded {len(test_oracle_slots)} test criteria.")
            print(f"   [System] Starting npm run dev (dynamic sniffing mode)...")
            log_file = os.path.join(workspace_dir, "standalone_debug_server.log")

            server_process, log_path_str = start_background_service("npm run dev", workspace_dir, log_file)
            target_url = wait_for_url_in_log(log_path_str, timeout=30)
            print(f"   [System] Sniffing successful! Local test service is running at: {target_url}")
            debug_log_dir = os.path.join(project_root, "experiment_results", "standalone_test_log", task_folder_name)

            if os.path.exists(debug_log_dir):
                shutil.rmtree(debug_log_dir, ignore_errors=True)
            os.makedirs(debug_log_dir, exist_ok=True)

            test_args_dict = {
                "output_dir": debug_log_dir,
                "download_dir": os.path.join(debug_log_dir, "downloads"),
                "window_width": 1200,
                "window_height": 800,
                "headless": False,
                "text_only": False,
                "fix_box_color": False,
                "save_accessibility_tree": False,
                "max_attached_imgs": 3,
                "max_iter": 10,
                "api_model": "gpt-5-mini",
                "seed": 42
            }

            print("   [System] Running visual WebVoyager evaluation...")
            result = evaluate_with_webvoyager(
                target_url=target_url,
                user_instruction=test_user_instruction,
                oracle_slots=test_oracle_slots,
                task_id=task_folder_name,
                args_dict=test_args_dict,
                app_port=None
            )


            print("\n   [System] Evaluation completed! (Debug mode: results not written to interaction_history.json)")

            final_status = "SUCCESS" if result["Success_Rate_SR"] == 1 else "FAIL"

            print(
                f"    {task_folder_name} Evaluation Result: {final_status} | SR: {result['Success_Rate_SR']} | TCR: {result['Task_Completion_Rate_TCR']}")
            print("    [Metrics Details]:")
            for detail in result['Details']:
                print(
                    f"      - {detail['task']}: {'PASSED' if detail['passed'] else 'FAILED'} (Reason: {detail['reason']})")

        except Exception as e:
            print(f"\n    A critical error occurred while processing {task_folder_name}: {e}")

        finally:
            print("   [System] Cleaning up dedicated server resources for this round...")
            if server_process:
                              stop_process_tree(server_process)

    print("\n All visualization test tasks completed!")