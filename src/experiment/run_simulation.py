import argparse
import os
import json
import sys
import subprocess
import time
import shutil
import socket
import concurrent.futures
from contextlib import closing
from dotenv import load_dotenv
from tqdm import tqdm  # 引入进度条库

# 将 src 加入路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 加载环境变量
load_dotenv()

from agent.webgen_agent import WebGenAgent
from experiment.simulation_agents import UserSimulator

# 移除了写死的 force_kill_port_3000，引入 is_port_open
from utils.execute_for_feedback import execute_for_feedback, is_port_open

# 导入 WebVoyager 评估器
try:
    from experiment.webvoyager_evaluator import evaluate_with_webvoyager
except ImportError:
    from webvoyager_evaluator import evaluate_with_webvoyager

DEFAULT_DATA_PATH = r"/home/hhr/home/InteractWeb-Bench/data/test_mini.jsonl"
DEFAULT_OUTPUT_DIR = r"/home/hhr/home/experiment_results"

MAX_TURNS_MAPPING = {"easy": 15, "middle": 20, "hard": 25}
ERROR_LIMIT_MAPPING = {"easy": 6, "middle": 8, "hard": 10}
MAX_SIMULATION_STEPS = 8


# ==============================================================================
#  核心工具函数
# ==============================================================================
def find_free_port():
    """向操作系统申请一个绝对安全的随机空闲端口"""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def get_vlm_endpoint(model_name: str) -> str:
    """
    解析 .env 中的 LOCAL_MODELS_MAP，根据模型名动态返回对应的 API URL。
    如果本地路由表中没有该模型，则回退到 OPENAILIKE_VLM_BASE_URL。
    """
    raw_map = os.environ.get("LOCAL_MODELS_MAP", "")
    expanded_map = os.path.expandvars(raw_map)

    routes = {}
    if expanded_map:
        for pair in expanded_map.split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                routes[k.strip()] = v.strip()

    if model_name in routes:
        tqdm.write(f"   [路由] 模型 {model_name} 匹配到本地节点: {routes[model_name]}")
        return routes[model_name]
    else:
        fallback_url = os.environ.get("OPENAILIKE_VLM_BASE_URL", "https://api.chatanywhere.tech/v1")
        tqdm.write(f"   [路由] 模型 {model_name} 走云端默认节点: {fallback_url}")
        return fallback_url


# ==============================================================================
#  数据持久化逻辑
# ==============================================================================
def save_interaction_history(messages, output_file, format_error_count):
    history = []
    stats = {
        "PATH_A_CLARIFY": 0, "PATH_B_IMPLEMENT": 0,
        "PATH_C_VERIFY": 0, "PATH_D_SUBMIT": 0,
        "FORMAT_ERROR_COUNT": format_error_count
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
#  评估逻辑 (并发端口隔离 + 动态模型路由)
# ==============================================================================
def perform_final_evaluation(builder, user_sim, workspace_dir, log_dir, oracle_slots, user_instruction, task_id, args,
                             app_port, stop_reason="submitted"):
    tqdm.write(f"\n[系统] 正在执行最终评估 (任务: {task_id}, 端口: {app_port}, 原因: {stop_reason})...")

    if not oracle_slots:
        tqdm.write("\033[91m[警告] 传入的 oracle_slots 为空！本次评估注定为 0 分。\033[0m")
    else:
        tqdm.write(f"   [系统] 成功加载 {len(oracle_slots)} 项打分标准，准备启动 WebVoyager 验收。")

    # 动态构建启动命令并将端口传给探活机制
    dynamic_start_cmd = f"npm run dev -- --port {app_port}"
    env_info = execute_for_feedback(
        project_dir=workspace_dir,
        log_dir=log_dir,
        start_cmd=dynamic_start_cmd,
        step_idx="final_eval",
        app_port=app_port
    )

    if env_info.get("start_error"):
        eval_result = {
            "status": "CRASHED", "sr": 0, "tcr": 0.0,
            "text": f"Evaluation failed: Server Crash.\n{env_info.get('start_results')}",
            "raw_metrics": {"Total_Weight": 0.0, "Details": []}
        }
    else:
        tqdm.write(f"   [系统] 正在启动前端服务器 (指定端口: {app_port})...")

        # 动态绑定端口启动前端 (适应 Vite/CRA 等框架)
        server_process = subprocess.Popen(
            dynamic_start_cmd,
            cwd=workspace_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=True
        )

        # 智能探活
        tqdm.write(f"   [系统] 等待 localhost:{app_port} 端口就绪...")
        for _ in range(30):
            if is_port_open(app_port):
                tqdm.write(f"   [系统] 端口 {app_port} 已连通！")
                break
            time.sleep(1)
        else:
            tqdm.write(f"\033[93m   [警告] 端口 {app_port} 未能就绪，测试可能会失败。\033[0m")

        try:
            eval_log_dir = os.path.join(log_dir, "eval_webvoyager")
            eval_download_dir = os.path.join(log_dir, "eval_downloads")

            shutil.rmtree(eval_log_dir, ignore_errors=True)
            os.makedirs(eval_log_dir, exist_ok=True)
            shutil.rmtree(eval_download_dir, ignore_errors=True)
            os.makedirs(eval_download_dir, exist_ok=True)

            # 获取动态 VLM 路由和 API Key
            target_model = args.webvoyager_model
            assigned_vlm_url = get_vlm_endpoint(target_model)
            api_key = os.environ.get(
                "OPENAILIKE_VLM_API_KEY") if "chatanywhere" in assigned_vlm_url else "sk-local-test"

            wv_args_dict = {
                "output_dir": eval_log_dir,
                "download_dir": eval_download_dir,
                "window_width": 1280,
                "window_height": 800,
                "headless": True,  # 并发时强制无头模式
                "text_only": False,
                "fix_box_color": False,
                "save_accessibility_tree": False,
                "max_attached_imgs": 3,  # 极限控制 Token
                "max_iter": 16,
                "api_model": target_model,
                "vlm_base_url": assigned_vlm_url,  # 动态路由注入
                "vlm_api_key": api_key,  # 动态密钥注入
                "seed": 42
            }

            tqdm.write(f"   [系统] WebVoyager 代理已切入 (模型: {target_model} | 节点: {assigned_vlm_url})...")

            # 动态拼接 target_url 传给 WebVoyager
            target_url = f"http://localhost:{app_port}/"
            raw_eval_result = evaluate_with_webvoyager(
                target_url=target_url,
                user_instruction=user_instruction,
                oracle_slots=oracle_slots,
                task_id=f"{task_id}_eval",
                args_dict=wv_args_dict,
                app_port=app_port  # 传入以便底层也能验证
            )

            eval_result = {
                "status": "PASS" if raw_eval_result.get("Success_Rate_SR", 0) == 1 else "FAIL",
                "sr": raw_eval_result.get("Success_Rate_SR", 0),
                "tcr": raw_eval_result.get("Task_Completion_Rate_TCR", 0.0),
                "text": "WebVoyager evaluation complete. Check details in raw_metrics.",
                "raw_metrics": raw_eval_result
            }

        except Exception as e:
            tqdm.write(f"\033[91m[错误] WebVoyager 评估过程中发生异常: {e}\033[0m")
            eval_result = {
                "status": "ERROR", "sr": 0, "tcr": 0.0,
                "text": f"Evaluation Error: {str(e)}",
                "raw_metrics": {"Total_Weight": 0.0, "Details": []}
            }
        finally:
            # 安全清理专属的 Node.js 进程
            if 'server_process' in locals() and server_process.poll() is None:
                try:
                    server_process.terminate()
                    server_process.wait(timeout=3)
                except Exception:
                    server_process.kill()

    tqdm.write(f"   => Final TCR: {eval_result.get('tcr', 0.0) * 100:.1f}% | Status: {eval_result.get('status')}")

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
    task_id = task.get("id", "unknown")
    difficulty = task.get("difficulty", "middle")
    persona = task.get("persona", "P-MIN")
    ground_truth = task.get("ground_truth_instruction", task.get("instruction"))
    user_instruction = task.get("instruction")
    current_oracle_slots = task.get("oracle_slots", [])

    max_turns = MAX_TURNS_MAPPING.get(difficulty.lower(), 20)
    error_limit = ERROR_LIMIT_MAPPING.get(difficulty.lower(), 5)

    safe_model_name = args.builder_model.replace("/", "-").replace(":", "-")
    workspace_dir = os.path.join(args.output_dir, safe_model_name, "workspaces", task_id)
    log_dir = os.path.join(args.output_dir, safe_model_name, "logs", task_id)

    # =========================================================================
    # 2. 样例级断点续跑：深度校验是否真实完成了打分 (以 is_final: True 为唯一准则)
    # =========================================================================
    interaction_history_path = os.path.join(log_dir, "interaction_history.json")
    if not args.overwrite and os.path.exists(interaction_history_path):
        try:
            with open(interaction_history_path, "r", encoding="utf-8") as f:
                history_data = json.load(f)

            trajectory = history_data.get("trajectory", [])
            if trajectory:
                last_turn = trajectory[-1]
                debug_info = last_turn.get("debug_info", {})

                if debug_info.get("is_final") is True:
                    eval_detail = debug_info.get("evaluation_detail", {})
                    status = eval_detail.get("status", "UNKNOWN")
                    tqdm.write(f"[断点续跑] 任务 {task_id} 已有完整打分记录 (Status: {status})，自动跳过。")
                    return task_id
                else:
                    tqdm.write(f"[断点续跑] 任务 {task_id} 历史记录未包含最终打分阶段 (中途异常中断)，将重新执行。")
            else:
                tqdm.write(f"[断点续跑] 任务 {task_id} 历史轨迹为空，将重新执行。")
        except Exception as e:
            tqdm.write(f"[断点续跑] 读取任务 {task_id} 历史记录失败 ({e})，将重新执行。")

    # 3. 动态分配端口并打印启动信息
    app_port = find_free_port()
    tqdm.write(f"\n==== Task {task_id} [{difficulty.upper()}] 分配网页端口: {app_port} ====")

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
        max_simulation_steps=MAX_SIMULATION_STEPS,
        app_port=app_port
    )

    user_sim = UserSimulator(
        ground_truth_instruction=task.get("ground_truth_instruction", user_instruction),
        initial_instruction=user_instruction,
        evaluation_checklist=task.get("evaluation_checklist", []),
        persona=persona, model=args.user_model, vlm_model=args.webvoyager_model
    )

    turn_counter = 0
    loop_idx = 0
    is_graded = False

    while turn_counter < max_turns:
        tqdm.write(f"\n--- Task {task_id} | Turn {turn_counter + 1}/{max_turns} ---")

        action, is_failed = builder.step(loop_idx, simulation_mode=True)
        builder.save_history(loop_idx)
        loop_idx += 1

        # ==== 修复后的路由分发逻辑 ====
        if action["type"] == "question":
            # 正常提问，交给用户模拟器回答
            answer = user_sim.answer_question(action["content"])
            builder.messages.append({"role": "user", "content": answer})
            turn_counter += 1
            continue

        elif action["type"] in ["coding", "internal_test", "shell", "format_error"]:
            if builder.is_finished: break

            turn_counter += 1
            continue

        elif action["type"] == "submitted":
            perform_final_evaluation(
                builder, user_sim, workspace_dir, log_dir,
                oracle_slots=current_oracle_slots,
                user_instruction=ground_truth,
                task_id=task_id,
                args=args,
                app_port=app_port,
                stop_reason="submitted"
            )
            is_graded = True
            break

    if not is_graded:
        reason = "max_turns_reached" if turn_counter >= max_turns else "verification_deadlock"
        perform_final_evaluation(
            builder, user_sim, workspace_dir, log_dir,
            oracle_slots=current_oracle_slots,
            user_instruction=user_instruction,
            task_id=task_id,
            args=args,
            app_port=app_port,
            stop_reason=reason
        )

    # 提取在底层统计到的格式错误次数保存
    format_errors = getattr(builder, 'format_error_count', 0)
    save_interaction_history(
        builder.messages,
        os.path.join(log_dir, "interaction_history.json"),
        format_errors
    )
    return task_id


# ==============================================================================
#  主入口：并发控制
# ==============================================================================
def main():
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="YAML配置文件的路径")
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--builder_model", type=str, default="gpt-4.1")
    parser.add_argument("--visual_copilot_model", type=str, default="gpt-4.1")
    parser.add_argument("--webvoyager_model", type=str, default="gpt-5-mini")
    parser.add_argument("--user_model", type=str, default="deepseek-v3.2")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_workers", type=int, default=1, help="同时并发执行的任务数量")

    args = parser.parse_args()

    if args.config:
        if not os.path.exists(args.config):
            print(f"❌ 错误: 找不到配置文件 {args.config}")
            return

        with open(args.config, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)

        if config_data:
            if "data_path" in config_data:
                args.data_path = config_data["data_path"]
            if "output_dir" in config_data:
                args.output_dir = config_data["output_dir"]
            if "max_workers" in config_data:
                args.max_workers = config_data["max_workers"]

            if "models" in config_data:
                models_cfg = config_data["models"]
                if "builder_model" in models_cfg:
                    args.builder_model = models_cfg["builder_model"]
                if "visual_copilot_model" in models_cfg:
                    args.visual_copilot_model = models_cfg["visual_copilot_model"]
                if "webvoyager_model" in models_cfg:
                    args.webvoyager_model = models_cfg["webvoyager_model"]
                if "user_model" in models_cfg:
                    args.user_model = models_cfg["user_model"]

    if not os.path.exists(args.data_path):
        print(f"❌ 错误: 找不到数据文件 {args.data_path}")
        return

    with open(args.data_path, "r", encoding="utf-8") as f:
        tasks = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(tasks)} tasks.")
    print(f" 准备启动并发测试 | 并发数: {args.max_workers}")
    print(f" Builder 模型: {args.builder_model}")
    print(f" Copilot 模型: {args.visual_copilot_model}")
    print(f" User 模型: {args.user_model}")
    print(f" WebVoyager 模型: {args.webvoyager_model}")

    # 使用 tqdm 包装 concurrent.futures 的执行过程
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = []
        for task in tasks:
            futures.append(executor.submit(run_single_task, task, args))

        # 进度条展示，total 是任务总数
        with tqdm(total=len(tasks), desc="Processing Tasks", unit="task") as pbar:
            for future in concurrent.futures.as_completed(futures):
                try:
                    finished_task_id = future.result()
                    # 使用 tqdm.write 替代 print，防止多线程打印冲刷掉进度条
                    tqdm.write(f"✅ 任务 {finished_task_id} 整体流程执行完毕。")
                except Exception as e:
                    tqdm.write(f"❌ 任务执行期间发生未捕获异常: {e}")
                finally:
                    pbar.update(1)  # 任务完成（无论成功失败），进度条 +1


if __name__ == "__main__":
    main()