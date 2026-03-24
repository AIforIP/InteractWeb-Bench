import argparse
import os
import json
import sys
import subprocess
import time
import shutil
import yaml
# 将 src 加入路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.webgen_agent import WebGenAgent
from experiment.simulation_agents import UserSimulator
# 【修改 1】引入 force_kill_port_3000
from utils.execute_for_feedback import execute_for_feedback, force_kill_port_3000

# 【新增】导入 WebVoyager 评估器
# 确保 webvoyager_evaluator.py 位于 src/experiment/ 目录下
try:
    from experiment.webvoyager_evaluator import evaluate_with_webvoyager
except ImportError:
    # 兼容性导入，以防在不同目录下运行
    from webvoyager_evaluator import evaluate_with_webvoyager

DEFAULT_DATA_PATH = r"/home/hhr/home/hhr/src/data_generation/test_mini.jsonl"
DEFAULT_OUTPUT_DIR = r"experiment_results"

MAX_TURNS_MAPPING = {"easy": 15, "middle": 20, "hard": 25}
ERROR_LIMIT_MAPPING = {"easy": 6, "middle": 8, "hard": 10}
MAX_SIMULATION_STEPS = 8


# ==============================================================================
#  数据持久化逻辑
# ==============================================================================
def save_interaction_history(messages, output_file, violating_chitchat_count):
    history = []
    stats = {
        "PATH_A_CLARIFY": 0, "PATH_B_IMPLEMENT": 0,
        "PATH_C_VERIFY": 0, "PATH_D_SUBMIT": 0,
        "VIOLATING_CHITCHAT_COUNT": violating_chitchat_count
    }

    for i, msg in enumerate(messages):
        entry = {"turn": i, "role": msg["role"], "content": msg["content"]}
        if "info" in msg:
            entry["debug_info"] = msg["info"]
            info = msg["info"]
            if info.get("is_question"):
                stats["PATH_A_CLARIFY"] += 1
            elif "internal_test_trace" in info:
                stats["PATH_C_VERIFY"] += 1
            elif info.get("is_final"):
                stats["PATH_D_SUBMIT"] += 1
            elif msg["role"] == "assistant":
                stats["PATH_B_IMPLEMENT"] += 1
        history.append(entry)

    final_output = {"path_distribution_stats": stats, "trajectory": history}
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)
    return stats


# ==============================================================================
#  评估逻辑 (已替换为 WebVoyager Agent-as-a-Judge)
# ==============================================================================
def perform_final_evaluation(builder, user_sim, workspace_dir, log_dir, oracle_slots, user_instruction, task_id, args,
                             stop_reason="submitted"):
    print(f"\n⚡ 正在执行最终评估 (原因: {stop_reason})...")

    # 强制验证打分标准是否存在
    if not oracle_slots:
        print("\033[91m[警告] 传入的 oracle_slots 为空！本次评估注定为 0 分。\033[0m")
    else:
        print(f"   [系统] 成功加载 {len(oracle_slots)} 项打分标准，准备启动 WebVoyager 验收。")

    # 1. 生成最终截图 (此步骤结束后，execute_for_feedback 会把服务器关闭)
    #    我们需要截图作为最后的留档，但 WebVoyager 会自己重新截图
    env_info = execute_for_feedback(workspace_dir, log_dir, step_idx="final_eval")

    if env_info.get("start_error"):
        # 如果代码本身就有语法错误导致起不来，直接 0 分
        eval_result = {
            "status": "CRASHED", "sr": 0, "tcr": 0.0,
            "text": f"Evaluation failed: Server Crash.\n{env_info.get('start_results')}",
            "raw_metrics": {"Total_Weight": 0.0, "Details": []}
        }
    else:
        # ==========================================
        #  启动 WebVoyager 评估流程
        # ==========================================
        print("   [系统] 正在为 WebVoyager 打分程序唤醒前端服务器...")

        # 【关键修复 1】：在启动 WebVoyager 专属服务器前，无情剿灭 3000 端口幽灵！
        os.system("fuser -k 3000/tcp >/dev/null 2>&1")
        import platform
        if platform.system() == "Windows":
            os.system("taskkill /f /im chromedriver.exe /t >nul 2>&1")
            os.system("taskkill /f /im chrome.exe /t >nul 2>&1")
        else:
            os.system("pkill -f chromedriver >/dev/null 2>&1")
            os.system("pkill -f chrome >/dev/null 2>&1")
        # 3. 【你的策略】：冷却期。等待 5~8 秒，让操作系统彻底回收文件锁，并避开网络限流
        print("   [系统] 进入冷却期，等待系统释放底层驱动资源...")
        time.sleep(8)

        # 启动前端开发服务器
        server_process = subprocess.Popen(
            "npm run dev",
            cwd=workspace_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=True
        )

        # 必须给服务器 5 秒钟的冷启动时间
        time.sleep(5)

        try:
            # 2. 配置 WebVoyager 参数
            # 将评估日志与生成日志分开，避免混淆
            eval_log_dir = os.path.join(log_dir, "eval_webvoyager")
            eval_download_dir = os.path.join(log_dir, "eval_downloads")

            import shutil
            if os.path.exists(eval_log_dir):
                shutil.rmtree(eval_log_dir, ignore_errors=True)
            os.makedirs(eval_log_dir, exist_ok=True)

            if os.path.exists(eval_download_dir):
                shutil.rmtree(eval_download_dir, ignore_errors=True)
            os.makedirs(eval_download_dir, exist_ok=True)

            wv_args_dict = {
                "output_dir": eval_log_dir,
                "download_dir": eval_download_dir,
                "window_width": 1200,
                "window_height": 800,
                "headless": True,  # 设为 False 可在本地看到浏览器弹窗
                "text_only": False,
                "fix_box_color": False,
                "save_accessibility_tree": False,
                "max_attached_imgs": 3,
                "max_iter": 16,  # 限制评估步数，防止死循环
                "api_model": args.webvoyager_model,
                "seed": 42
            }

            print(f"   [系统] WebVoyager 评估代理已切入网页 (Model: {args.webvoyager_model})...")

            # 3. 调用 WebVoyager 进行打分 (支持 oracle_slots 加权)
            raw_eval_result = evaluate_with_webvoyager(
                target_url="http://localhost:3000",
                user_instruction=user_instruction,
                oracle_slots=oracle_slots,
                task_id=f"{task_id}_eval",
                args_dict=wv_args_dict
            )

            # 4. 格式化结果
            eval_result = {
                "status": "PASS" if raw_eval_result["Success_Rate_SR"] == 1 else "FAIL",
                "sr": raw_eval_result["Success_Rate_SR"],
                "tcr": raw_eval_result["Task_Completion_Rate_TCR"],
                "text": "WebVoyager evaluation complete. Check details in raw_metrics.",
                "raw_metrics": raw_eval_result
            }

        except Exception as e:
            print(f"\033[91m[错误] WebVoyager 评估过程中发生异常: {e}\033[0m")
            eval_result = {
                "status": "ERROR", "sr": 0, "tcr": 0.0,
                "text": f"Evaluation Error: {str(e)}",
                "raw_metrics": {}
            }
        finally:
            # ==========================================
            #  打分结束，强制关闭服务器
            # ==========================================
            force_kill_port_3000()
            os.system("taskkill /f /im chromedriver.exe /t >nul 2>&1")
            os.system("taskkill /f /im chrome.exe /t >nul 2>&1")
            print("   [系统] 打分完毕，正在清理端口占用...")



    print(f"   => Final TCR: {eval_result.get('tcr', 0.0) * 100:.1f}% | Status: {eval_result.get('status')}")

    # 3. 将打分详情写入日志
    builder.messages.append({
        "role": "user",
        "content": f"[SYSTEM]: Task Stopped ({stop_reason}).\nEvaluation Report:\n{eval_result.get('text', '')}\nDetails: {json.dumps(eval_result.get('raw_metrics', {}).get('Details', []), indent=2)}",
        "info": {
            "evaluation_detail": eval_result,
            "final_env_state": env_info,
            "is_final": True,
            "stop_reason": stop_reason,
            "oracle_slots_used_for_grading": oracle_slots
        }
    })
    return eval_result


# ==============================================================================
#  任务运行引擎
# ==============================================================================
def run_single_task(task, args):
    # 1. 提取 task_id 并统一转为字符串，防止路径拼接报错
    task_id = str(task.get("id", "unknown"))

    # 2. 提前计算路径
    safe_model_name = args.builder_model.replace("/", "-").replace(":", "-")
    workspace_dir = os.path.join(args.output_dir, safe_model_name, "workspaces", task_id)
    log_dir = os.path.join(args.output_dir, safe_model_name, "logs", task_id)
    history_file = os.path.join(log_dir, "interaction_history.json")

    # 3. 断点续传检查：如果没开启覆盖模式，且最终的历史记录文件已存在，跳过当前任务
    if not args.overwrite and os.path.exists(history_file):
        print(f"\n==== Task {task_id} ==== [已完成，触发断点续传，跳过当前任务]")
        return

    # 4. 脏数据清理：如果发现残留的中断数据（有文件夹但无最终 json），进行清理
    if not args.overwrite and (os.path.exists(workspace_dir) or os.path.exists(log_dir)):
        print(f"\n==== Task {task_id} ==== [发现残留的中断数据，正在清理并重新开始]")
        import shutil
        shutil.rmtree(workspace_dir, ignore_errors=True)
        shutil.rmtree(log_dir, ignore_errors=True)

    # 1. 把提取难度的默认值改为字典中不存在的 "test"
    difficulty = task.get("difficulty", "test")
    persona = task.get("persona", "P-MIN")
    ground_truth = task.get("ground_truth_instruction", task.get("instruction"))
    user_instruction = task.get("instruction")
    current_oracle_slots = task.get("oracle_slots", [])

    # 2. 此时字典找不到 "test"，就会生效后备值 3 和 2
    max_turns = MAX_TURNS_MAPPING.get(difficulty.lower(), 3)
    error_limit = ERROR_LIMIT_MAPPING.get(difficulty.lower(), 2)
    print(f"\n==== Task {task_id} [{difficulty.upper()}] 开始运行 ====")

    # 初始化 WebGenAgent 时，传入修改后的目录
    builder = WebGenAgent(
        model=args.builder_model,
        vlm_model=args.visual_copilot_model,
        fb_model=args.user_model,
        workspace_dir=workspace_dir,
        log_dir=log_dir,
        instruction=user_instruction,
        max_iter=max_turns,
        overwrite=args.overwrite,
        error_limit=error_limit,
        difficulty=difficulty,
        max_simulation_steps=MAX_SIMULATION_STEPS
    )
    # UserSimulator 通常用于模拟用户行为，继续使用原本的参数即可
    user_sim = UserSimulator(
        ground_truth_instruction=task.get("ground_truth_instruction", user_instruction),
        initial_instruction=user_instruction,
        evaluation_checklist=task.get("evaluation_checklist", []),
        persona=persona, model=args.user_model, vlm_model=args.webvoyager_model
    )

    turn_counter = 0
    loop_idx = 0
    is_graded = False
    consecutive_no_tag_chitchat = 0
    total_violating_chitchat = 0

    while turn_counter < max_turns:
        print(f"\n--- Turn {turn_counter + 1}/{max_turns} ---")

        action, is_failed = builder.step(loop_idx, simulation_mode=True)
        builder.save_history(loop_idx)
        loop_idx += 1
        
        raw_output = builder.messages[-1]["content"] if builder.messages else ""

        if action["type"] == "question":
            has_valid_tag = any(tag in raw_output for tag in ["<boltAction", "<boltArtifact"])

            #  1. 触发零容忍格式审查
            if not has_valid_tag:
                total_violating_chitchat += 1
                print(
                    f"\033[93m   [拦截] Agent 输出了无效格式的纯文本 (累计违规: {total_violating_chitchat}次)，正在强制要求重试...\033[0m")

                # 【安全熔断机制】防止劣质模型无限死循环卡死程序并消耗巨额 API 费用
                if total_violating_chitchat >= 10:
                    print("\033[91m   [致命错误] Agent 累计格式违规达到 10 次，触发安全熔断，强制终止当前任务！\033[0m")
                    is_graded = False
                    break  # 跳出 while 循环，进入兜底评估阶段

                # 注入系统强警告
                system_feedback = "SYSTEM ALERT: Invalid format. You MUST use `<boltArtifact>` or `<boltAction>` tags to interact. Plain text chitchat is strictly prohibited. Please regenerate your response using the correct XML tags."
                builder.messages.append({"role": "user", "content": system_feedback})

                # 注意：这里不增加 turn_counter！
                # 这意味着重试操作不算作消耗用户的交互轮数
                continue

            #  2. 格式合规的正常提问 (<boltAction type="ask_user">)
            answer = user_sim.answer_question(action["content"])
            builder.messages.append({"role": "user", "content": answer})
            turn_counter += 1  # 只有正常的业务交互才消耗一轮
            continue

        consecutive_no_tag_chitchat = 0

        if action["type"] == "coding":
            turn_counter += 1
            continue
        elif action["type"] == "internal_test":
            if builder.is_finished: break
            turn_counter += 1
            continue
        elif action["type"] == "submitted":
            # 调用 perform_final_evaluation
            perform_final_evaluation(
                builder, user_sim, workspace_dir, log_dir,
                oracle_slots=current_oracle_slots,
                user_instruction=ground_truth,
                task_id=task_id,
                args=args,
                stop_reason="submitted"
            )
            is_graded = True
            break

    # 兜底评估：处理超时或 Deadlock 熔断的情况
    if not is_graded:
        reason = "max_turns_reached" if turn_counter >= max_turns else "verification_deadlock"
        # 兜底评估
        perform_final_evaluation(
            builder, user_sim, workspace_dir, log_dir,
            oracle_slots=current_oracle_slots,
            user_instruction=user_instruction,
            task_id=task_id,
            args=args,
            stop_reason=reason
        )

    save_interaction_history(
        builder.messages,
        os.path.join(log_dir, "interaction_history.json"),
        total_violating_chitchat
    )


def main():
    # ==========================================
    # 1. 读取 YAML 配置文件
    # ==========================================
    # 动态定位到项目根目录下的 config.yaml
    current_file_path = os.path.abspath(__file__)
    # src/experiment/run_simulation.py -> 向上退三级到项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
    config_path = os.path.join(project_root, "config.yaml")

    config = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
            print(f"[系统] 成功加载配置文件: {config_path}")
        except Exception as e:
            print(f"\033[91m[错误] 读取 config.yaml 失败: {e}\033[0m")
    else:
        print(f"\033[93m[警告] 未找到 {config_path}，将使用系统默认硬编码值。\033[0m")

    # ==========================================
    # 2. 提取配置项 (带有后备默认值)
    # ==========================================
    yaml_data_path = config.get("data_path", DEFAULT_DATA_PATH)
    models_cfg = config.get("models", {})

    # ==========================================
    # 3. 设置命令行参数解析
    # ==========================================
    parser = argparse.ArgumentParser()
    # 使用 yaml_data_path 作为 data_path 的默认值
    parser.add_argument("--data_path", type=str, default=yaml_data_path)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)

    # 优先使用 yaml 中的模型配置，如果 yaml 没写，则回退到原来的默认值
    parser.add_argument("--builder_model", type=str, default=models_cfg.get("builder_model", "gpt-5-mini"))
    parser.add_argument("--visual_copilot_model", type=str,
                        default=models_cfg.get("visual_copilot_model", "gpt-5-mini"))
    parser.add_argument("--webvoyager_model", type=str, default=models_cfg.get("webvoyager_model", "gpt-5-mini"))
    parser.add_argument("--user_model", type=str, default=models_cfg.get("user_model", "deepseek-v3.2"))
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    # 检查输入文件是否存在
    if not os.path.exists(args.data_path):
        print(f"\033[91m[错误] 数据文件不存在: {args.data_path}\033[0m")
        return

    with open(args.data_path, "r", encoding="utf-8") as f:
        tasks = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(tasks)} tasks.")
    for task in tasks:
        run_single_task(task, args)


if __name__ == "__main__":
    main()