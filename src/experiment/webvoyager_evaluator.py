import os
import sys
import re
import json
import time
import argparse
import time
# 在脚本顶部添加：
from utils.execute_for_feedback import is_port_open
# ==============================================================================
# 关键修正：将项目的 src 根目录强制加入环境变量，必须放在所有自定义模块导入之前
# ==============================================================================
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from webvoyager.run import run_single_task

# 假设项目中已存在 llm_generation 工具，用于发起纯文本对话
try:
    from utils import llm_generation
except ImportError:
    pass  # 如果导入失败，请确保路径正确或使用你自己的 API 调用封装


def evaluate_with_webvoyager(target_url: str, user_instruction: str, oracle_slots: list, task_id: str,
                             args_dict: dict, app_port: int) -> dict:  # 👈 新增 app_port

    # =================================================================
    # 0. 动态等待指定的应用端口就绪
    # =================================================================
    print(f"   [系统] 等待 localhost:{app_port} 端口就绪 (最长 30 秒)...")
    for _ in range(30):
        # 请确保引入了 is_port_open 函数
        if is_port_open(app_port):
            print(f"   [系统] 端口 {app_port} 已连通！开始测试。")
            break
        time.sleep(1)
    else:
        print(f"   [警告] 端口 {app_port} 未能就绪，测试可能会报错。")

    # 1. 构造检查清单
    checklist_str = ""
    for idx, slot in enumerate(oracle_slots):
        checklist_str += f"Checklist ID [{idx}]:\n"
        checklist_str += f"- Task: {slot.get('original_task')}\n"
        checklist_str += f"- Expected Result: {slot.get('expected_result')}\n\n"

    # =================================================================
    # 2. 提示词设计 (步进式验证 / Step-wise Verification)
    # =================================================================
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

    # 🌟 将目标 URL 动态指向分配的 app_port
    task = {
        "id": task_id,
        "web": f"http://localhost:{app_port}/",
        "ques": eval_ques
    }

    # =================================================================
    # 🌟 核心注入点：定义打分专用的方括号交卷模板，并动态注入到 args_dict 中
    # =================================================================
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

    # 3. 运行 WebVoyager
    messages = run_single_task(task, args_dict)

    # =================================================================
    # 4. 历遍对话历史，提取步进式打分结果
    # =================================================================
    print("\n   [系统] WebVoyager 探索结束，开始从对话轨迹中提取步进式打分结果...")

    # 初始化所有得分为 False (如果没被标记，就认为未通过)
    model_results = {
        i: {"passed": False, "reason": "Not observed or tested during exploration"}
        for i in range(len(oracle_slots))
    }

    if messages:
        for msg in messages:
            if msg.get("role") == "assistant":
                content = msg.get("content", "")

                # 正则捕捉 [PASSED ID: X] 或 [FAILED ID: X] 及其理由
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

    # =================================================================
    # 5. 计算最终得分
    # =================================================================
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

    print(f"   [系统] 共提取到 {sum(1 for d in details if d['passed'])} 项已通过指标。")

    return {
        "Success_Rate_SR": 1 if all_passed and total_weight > 0 else 0,
        "Task_Completion_Rate_TCR": tcr,
        "Details": details,
        "Raw_Trajectory": messages
    }


# ==============================================================================
# 独立调试与测试模块 (可视化有头模式，不污染轨迹文件)
# ==============================================================================
# ==============================================================================
# 独立调试与测试模块 (可视化有头模式，不污染轨迹文件)
# ==============================================================================
if __name__ == "__main__":
    import os
    import sys
    import json
    import shutil
    import subprocess
    import time
    import socket
    from contextlib import closing


    # 动态获取可用空闲端口的辅助函数
    def find_free_port():
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(('', 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return s.getsockname()[1]


    # 确保能找到项目根目录下的 utils 和 webvoyager 模块
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    print("\n 开始独立测试任务 (屏幕可视化模式 / 结果不写入轨迹)...")

    RETEST_JSONL_PATH = r"/app/data/test_mini.jsonl"
    LOGS_ROOT_DIR = r"/app/experiment_results/Qwen3.5-9B/logs"
    WORKSPACES_ROOT_DIR = r"/app/experiment_results/Qwen3.5-9B/workspaces"

    if not os.path.exists(RETEST_JSONL_PATH):
        print(f" 找不到待修复文件列表: {RETEST_JSONL_PATH}")
        sys.exit(1)

    # 1. 读取需要重测的任务列表
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
                print(f" 解析 jsonl 失败，跳过该行: {e}")

    tasks_to_retest = list(set(tasks_to_retest))
    print(f"\n 共读取到 {len(tasks_to_retest)} 个测试任务。")

    # 2. 遍历测试
    for current_idx, task_folder_name in enumerate(tasks_to_retest, 1):
        print(f"\n" + "=" * 60)
        print(f" [{current_idx}/{len(tasks_to_retest)}] 正在可视化测试任务: {task_folder_name}")
        print("=" * 60)

        workspace_dir = os.path.join(WORKSPACES_ROOT_DIR, task_folder_name)
        history_json_path = os.path.join(LOGS_ROOT_DIR, task_folder_name, "interaction_history.json")
        server_process = None

        try:
            if not os.path.exists(history_json_path):
                print(f" 跳过: 找不到历史文件 {history_json_path}")
                continue
            if not os.path.exists(workspace_dir):
                print(f" 跳过: 找不到工作区 {workspace_dir}")
                continue

            with open(history_json_path, "r", encoding="utf-8") as f:
                history_data = json.load(f)

            test_user_instruction = history_data["trajectory"][1]["content"]
            last_turn = history_data["trajectory"][-1]
            test_oracle_slots = last_turn["debug_info"].get("oracle_slots_used_for_grading", [])

            if not test_oracle_slots:
                print(f" 跳过: 日志中未找到 oracle_slots 打分标准。")
                continue

            print(f"   [系统] 成功加载 {len(test_oracle_slots)} 条测试标准。")

            # ========================================================
            # 💡 核心修改：动态分配端口，摒弃 3000
            # ========================================================
            app_port = find_free_port()
            target_url = f"http://localhost:{app_port}"

            print(f"   [系统] 启动 npm run dev (分配随机端口: {app_port})...")

            server_process = subprocess.Popen(
                f"npm run dev -- --port {app_port}",
                cwd=workspace_dir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                shell=True
            )

            # 备注：探活逻辑已被转移到 evaluate_with_webvoyager 内部
            # 这里不需要再写冗长的 for 循环探测端口了，直接交给评测函数处理
            time.sleep(2)

            # ======================================================
            # 配置 WebVoyager 测试参数
            debug_log_dir = os.path.join(project_root, "experiment_results", "standalone_test_log", task_folder_name)

            if os.path.exists(debug_log_dir):
                shutil.rmtree(debug_log_dir, ignore_errors=True)
            os.makedirs(debug_log_dir, exist_ok=True)

            test_args_dict = {
                "output_dir": debug_log_dir,
                "download_dir": os.path.join(debug_log_dir, "downloads"),
                "window_width": 1200,
                "window_height": 800,
                "headless": False,  # 确保这里是 False，从而在屏幕上显示浏览器
                "text_only": False,
                "fix_box_color": False,
                "save_accessibility_tree": False,
                "max_attached_imgs": 3,  # 强烈建议改成1，省钱省token
                "max_iter": 10,
                "api_model": "gpt-5-mini",  # 修正为 gpt-4o-mini 或你 .env 里的模型
                "seed": 42
                # 由于这是独立测试，没有 run_simulation 的动态传参，底层会自动 fallback 读 .env 里的 KEY 和 URL
            }

            print("   [系统] 正在进行可视化的 WebVoyager 打分...")
            result = evaluate_with_webvoyager(
                target_url=target_url,
                user_instruction=test_user_instruction,
                oracle_slots=test_oracle_slots,
                task_id=task_folder_name,
                args_dict=test_args_dict,
                app_port=app_port  # 👈 必须把端口传进去，否则报 TypeError!
            )

            # 仅在终端打印结果，不再覆写 interaction_history.json
            print("\n   [系统] 评估完成！(调试模式：结果不写入 interaction_history.json)")

            final_status = "SUCCESS" if result["Success_Rate_SR"] == 1 else "FAIL"

            print(
                f"    {task_folder_name} 评估结果: {final_status} | SR: {result['Success_Rate_SR']} | TCR: {result['Task_Completion_Rate_TCR']}")
            print("    [指标详情]:")
            for detail in result['Details']:
                print(
                    f"      - {detail['task']}: {'PASSED' if detail['passed'] else 'FAILED'} (Reason: {detail['reason']})")

        except Exception as e:
            print(f"\n    处理 {task_folder_name} 时发生严重错误: {e}")

        finally:
            print("   [系统] 清理本轮专属服务器占用...")
            try:
                if server_process:
                    server_process.terminate()
                    server_process.wait(timeout=3)
            except:
                pass

    print("\n 所有任务可视化测试结束！")