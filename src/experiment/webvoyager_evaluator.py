import os
import re
import json
import argparse
from webvoyager.run import run_single_task

# 假设项目中已存在 llm_generation 工具，用于发起纯文本对话
try:
    from utils import llm_generation
except ImportError:
    pass  # 如果导入失败，请确保路径正确或使用你自己的 API 调用封装


def evaluate_with_webvoyager(target_url: str, user_instruction: str, oracle_slots: list, task_id: str,
                             args_dict: dict) -> dict:
    # 1. 构造检查清单
    checklist_str = ""
    for idx, slot in enumerate(oracle_slots):
        checklist_str += f"Checklist ID [{idx}]:\n"
        checklist_str += f"- Task: {slot.get('original_task')}\n"
        checklist_str += f"- Expected Result: {slot.get('expected_result')}\n\n"

    # =================================================================
    # 2. 提示词设计 (步进式验证 / Step-wise Verification)
    # 核心魔法：要求大模型边探索边在 Thought 里面记录得分点
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
        "   - **FATAL RULE: You MUST write ALL your findings inside a SINGLE contiguous paragraph. DO NOT use line breaks (\\n). DO NOT output the word 'Thought:' more than once!**\n"
        "   Example: 'Clicking the date picker did nothing. [FAILED ID: 1] Reason: Date picker is unresponsive. I will stop trying this and check the background color next. [PASSED ID: 4] Reason: Background is papaya whip.'\n"
        "4. When you have tested or skipped all items, output: `Action: ANSWER; Exploration complete`.\n"
        "5. **IGNORE TESTING ARTIFACTS (CRITICAL)**: The numerical labels (e.g., [0], [1]) and colored bounding/dashed boxes on the screenshot are injected by our automated testing framework. You MUST IGNORE them. Do NOT treat them as 'unrequested UI elements' and do NOT let them cause any checklist item to fail."
    )

    task = {
        "id": task_id,
        "web": target_url,
        "ques": eval_ques
    }

    # =================================================================
    # 🌟 核心注入点：定义打分专用的方括号交卷模板，并动态注入到 args_dict 中
    # 这样 run.py 发生死循环触发交卷时，就会乖乖吐出 [FAILED ID: X] 格式
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
                    status_str = m.group(1).upper()  # 提取是 PASSED 还是 FAILED
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

    # 打印最终提取结果供核对
    print(f"   [系统] 共提取到 {sum(1 for d in details if d['passed'])} 项已通过指标。")

    return {
        "Success_Rate_SR": 1 if all_passed and total_weight > 0 else 0,
        "Task_Completion_Rate_TCR": tcr,
        "Details": details,
        "Raw_Trajectory": messages
    }


# ==============================================================================
# 🛠️ 独立调试与修复模块 (针对 000010_P-INT 等崩溃修复)
# ==============================================================================
if __name__ == "__main__":
    import os
    import sys
    import json
    import shutil
    import subprocess
    import time
    import re  # 确保正则库已导入

    # 确保能找到项目根目录下的 utils 和 webvoyager 模块
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    print("\n🚀 开始批量自动化恢复 ERROR 任务...")

    RETEST_JSONL_PATH = r"E:\Agent_work\src\data_generation\re_test1.jsonl"
    LOGS_ROOT_DIR = r"E:\Agent_work\src\experiment_results\gpt-4o-mini\logs"
    WORKSPACES_ROOT_DIR = r"E:\Agent_work\src\experiment_results\gpt-4o-mini\workspaces"

    if not os.path.exists(RETEST_JSONL_PATH):
        print(f"❌ 找不到待修复文件列表: {RETEST_JSONL_PATH}")
        sys.exit(1)

    # 1. 读取需要重测的任务列表
    tasks_to_retest = []
    with open(RETEST_JSONL_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                data = json.loads(line)
                # 提取完整 task_id (例如 "000010_P-INT")
                t_id = str(data.get("task_id") or data.get("id") or data.get("original_id"))
                if t_id and t_id != "None":
                    tasks_to_retest.append(t_id)
            except Exception as e:
                print(f"⚠️ 解析 jsonl 失败，跳过该行: {e}")

    # 去重处理
    tasks_to_retest = list(set(tasks_to_retest))
    print(f"\n📋 共读取到 {len(tasks_to_retest)} 个需要重测的独立任务。")

    # 2. 遍历重测
    for current_idx, task_folder_name in enumerate(tasks_to_retest, 1):
        print(f"\n" + "=" * 60)
        print(f"▶️ [{current_idx}/{len(tasks_to_retest)}] 正在处理任务: {task_folder_name}")
        print("=" * 60)

        workspace_dir = os.path.join(WORKSPACES_ROOT_DIR, task_folder_name)
        history_json_path = os.path.join(LOGS_ROOT_DIR, task_folder_name, "interaction_history.json")
        target_url = "http://localhost:3000"
        server_process = None

        try:
            # 校验文件是否存在
            if not os.path.exists(history_json_path):
                print(f"❌ 跳过: 找不到历史文件 {history_json_path}")
                continue
            if not os.path.exists(workspace_dir):
                print(f"❌ 跳过: 找不到工作区 {workspace_dir}")
                continue

            # 从 history.json 中提取要求
            with open(history_json_path, "r", encoding="utf-8") as f:
                history_data = json.load(f)

            test_user_instruction = history_data["trajectory"][1]["content"]
            last_turn = history_data["trajectory"][-1]
            test_oracle_slots = last_turn["debug_info"].get("oracle_slots_used_for_grading", [])

            if not test_oracle_slots:
                print(f"❌ 跳过: 日志中未找到 oracle_slots 打分标准。")
                continue

            print(f"   [系统] 成功加载 {len(test_oracle_slots)} 条测试标准。")

            import platform

            if platform.system() == "Windows":
                os.system("taskkill /f /im chromedriver.exe /t >nul 2>&1")
                os.system("taskkill /f /im chrome.exe /t >nul 2>&1")
            else:
                os.system("pkill -f chromedriver >/dev/null 2>&1")
                os.system("pkill -f chrome >/dev/null 2>&1")
            time.sleep(1)

            # =========================================================
            # 修改核心区域：增强版拦截脚本注入
            # =========================================================
            index_html_path = os.path.join(workspace_dir, "index.html")
            if os.path.exists(index_html_path):
                with open(index_html_path, "r", encoding="utf-8") as f:
                    html_content = f.read()

                # 检查是否已经注入了包含 prompt 的完整拦截脚本
                if "window.prompt = function" not in html_content:
                    print("   [系统] 正在清理旧脚本并注入 Alert/Confirm/Prompt 全面拦截脚本...")
                    injection = """
<script>
window.alert = function(msg) {
    var d = document.createElement('div');
    d.style.cssText = 'position:fixed;top:20px;left:50%;transform:translateX(-50%);background:#fff;border:2px solid #333;padding:15px;z-index:99999;box-shadow:0 4px 6px rgba(0,0,0,0.1);font-size:16px;color:#000;';
    d.innerHTML = '<b>System Popup:</b> ' + msg + '<br><button style="margin-top:10px;" onclick="this.parentElement.remove()">OK</button>';
    document.body.appendChild(d);
};
window.confirm = function(msg) {
    window.alert("Confirm requested: " + msg);
    return true; 
};
window.prompt = function(msg, defaultText) {
    window.alert("Prompt requested: " + msg);
    return "Test Input Value"; 
};
</script>
"""
                    # 使用正则清理掉之前可能残留的、不完整的 window.alert 拦截脚本
                    html_content = re.sub(r'<script>\s*window\.alert = function.*?<\/script>', '', html_content,
                                          flags=re.DOTALL)

                    # 重新将增强版的拦截脚本注入到 <head> 中
                    html_content = html_content.replace("<head>", "<head>\n" + injection)
                    with open(index_html_path, "w", encoding="utf-8") as f:
                        f.write(html_content)
            # =========================================================

            # 启动前端服务器
            print(f"   [系统] 启动 npm run dev...")
            try:
                from utils.execute_for_feedback import force_kill_port_3000

                force_kill_port_3000()
            except ImportError:
                pass

            server_process = subprocess.Popen(
                ["npm", "run", "dev"],
                cwd=workspace_dir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                shell=True
            )
            time.sleep(5)  # 等待冷启动

            # 配置 WebVoyager 测试参数
            debug_log_dir = os.path.join(project_root, "experiment_results", "standalone_test_log", task_folder_name)

            # 🧹 防止独立测试脚本的缓存干扰，先清空历史打分记录
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
                # ✅ 关键修正：确保使用支持视觉功能的评估模型！防止 Base64 引起 Token 爆炸死循环。
                "api_model": "gpt-5-mini",
                "seed": 42
            }

            print("   [系统] 正在进行 WebVoyager 打分...")
            result = evaluate_with_webvoyager(
                target_url=target_url,
                user_instruction=test_user_instruction,
                oracle_slots=test_oracle_slots,
                task_id=task_folder_name,
                args_dict=test_args_dict
            )

            # 动态判断并修复 JSON 状态
            print("\n   [系统] 将打分结果写回 interaction_history.json...")

            # ✅ 修复之前强行设置为 SUCCESS 的 Bug
            final_status = "SUCCESS" if result["Success_Rate_SR"] == 1 else "FAIL"

            last_turn["debug_info"]["evaluation_detail"] = {
                "status": final_status,
                "sr": result["Success_Rate_SR"],
                "tcr": result["Task_Completion_Rate_TCR"],
                "text": "Evaluation recovered via batch retest script.",
                "raw_metrics": result["Details"]
            }

            with open(history_json_path, "w", encoding="utf-8") as f:
                json.dump(history_data, f, indent=2, ensure_ascii=False)

            print(
                f"   ✅ {task_folder_name} 修复完毕！当前状态已更新为: {final_status} | TCR: {result['Task_Completion_Rate_TCR']}")

        except Exception as e:
            print(f"\n   ❌ 处理 {task_folder_name} 时发生严重错误: {e}")

        finally:
            # 确保每轮结束强制关闭服务器，防止堆积
            print("   [系统] 清理本轮服务器占用...")
            try:
                if server_process:
                    server_process.terminate()
                from utils.execute_for_feedback import force_kill_port_3000

                force_kill_port_3000()
            except:
                pass

    print("\n🎉 所有任务批量处理结束！请重新运行全局分析脚本查看最新战报。")