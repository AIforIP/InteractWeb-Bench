import argparse
import os
import json
import sys
import shutil
import socket
import concurrent.futures
from contextlib import closing
from urllib.parse import urlparse
from dotenv import load_dotenv
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

load_dotenv()

from agent.webgen_agent import WebGenAgent
from experiment.simulation_agents import UserSimulator

from utils.execute_for_feedback import (
    execute_for_feedback,
    start_background_service,
    wait_for_url_in_log,
    stop_process_tree
)


try:
    from experiment.webvoyager_evaluator import evaluate_with_webvoyager
except ImportError:
    from webvoyager_evaluator import evaluate_with_webvoyager

DEFAULT_DATA_PATH = r"/your_path/InteractWeb-Bench/data/test_mini.jsonl"
DEFAULT_OUTPUT_DIR = r"/your_path/experiment_results"

MAX_TURNS_MAPPING = {"easy": 15, "middle": 20, "hard": 25}
ERROR_LIMIT_MAPPING = {"easy": 6, "middle": 8, "hard": 10}
MAX_SIMULATION_STEPS = 8

def find_free_port():

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def get_vlm_endpoint(model_name: str) -> str:

    raw_map = os.environ.get("LOCAL_MODELS_MAP", "")
    expanded_map = os.path.expandvars(raw_map)

    routes = {}
    if expanded_map:
        for pair in expanded_map.split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                routes[k.strip()] = v.strip()

    if model_name in routes:
        tqdm.write(f"   [Routing] Model {model_name} matched to local endpoint: {routes[model_name]}")
        return routes[model_name]
    else:
        fallback_url = os.environ.get("OPENAILIKE_VLM_BASE_URL", "https://api.chatanywhere.tech/v1")
        tqdm.write(f"   [Routing] Model {model_name} using default cloud endpoint: {fallback_url}")
        return fallback_url

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

def perform_final_evaluation(builder, user_sim, workspace_dir, log_dir, oracle_slots, user_instruction, task_id, args,
                             stop_reason="submitted"):
    tqdm.write(f"\n[System] Performing final evaluation (Task: {task_id}, Reason: {stop_reason})...")

    if not oracle_slots:
        tqdm.write("\033[91m[Warning] oracle_slots is empty! This evaluation will inevitably result in a score of 0.\033[0m")
    else:
        tqdm.write(f"   [System] Successfully loaded {len(oracle_slots)} evaluation criteria. Preparing to launch WebVoyager validation.")

    preferred_port = find_free_port()
    dynamic_start_cmd = f"npm run dev -- --port {preferred_port}"

    env_info = execute_for_feedback(
        project_dir=workspace_dir,
        log_dir=log_dir,
        start_cmd=dynamic_start_cmd,
        step_idx="final_eval"
    )

    if env_info.get("start_error"):
        eval_result = {
            "status": "CRASHED",
            "sr": 0,
            "tcr": 0.0,
            "text": f"Evaluation failed: Server crash.\n{env_info.get('start_results')}",
            "raw_metrics": {"Total_Weight": 0.0, "Details": []}
        }
    else:
        tqdm.write(f"   [System] Launching frontend server for continuous WebVoyager evaluation...")

        server_log_path = os.path.join(log_dir, "frontend_eval_server.log")
        server_process, log_path_str = start_background_service(
            start_cmd=dynamic_start_cmd,
            cwd=workspace_dir,
            log_file=server_log_path
        )

        try:
            target_url = wait_for_url_in_log(log_path_str, timeout=30)
            tqdm.write(f"   [System] Detection successful! Frontend is running at: {target_url}")

            extracted_port = int(urlparse(target_url).port or 80)

            eval_log_dir = os.path.join(log_dir, "eval_webvoyager")
            eval_download_dir = os.path.join(log_dir, "eval_downloads")
            shutil.rmtree(eval_log_dir, ignore_errors=True)
            os.makedirs(eval_log_dir, exist_ok=True)
            shutil.rmtree(eval_download_dir, ignore_errors=True)
            os.makedirs(eval_download_dir, exist_ok=True)

            target_model = args.webvoyager_model
            assigned_vlm_url = os.environ.get("WEBVOYAGER_BASE_URL") or get_vlm_endpoint(target_model)

            api_key = os.environ.get("WEBVOYAGER_API_KEY")
            if not api_key:
                api_key = os.environ.get("OPENAILIKE_VLM_API_KEY") if "chatanywhere" in assigned_vlm_url else "sk-local-test"

            wv_args_dict = {
                "output_dir": eval_log_dir,
                "download_dir": eval_download_dir,
                "window_width": 1280,
                "window_height": 800,
                "headless": True,
                "text_only": False,
                "fix_box_color": False,
                "save_accessibility_tree": False,
                "max_attached_imgs": 3,
                "max_iter": 16,
                "api_model": target_model,
                "vlm_base_url": assigned_vlm_url,
                "vlm_api_key": api_key,
                "seed": 42
            }

            tqdm.write(f"   [System] WebVoyager agent initialized (Model: {target_model} | Endpoint: {assigned_vlm_url})...")

            raw_eval_result = evaluate_with_webvoyager(
                target_url=target_url,
                user_instruction=user_instruction,
                oracle_slots=oracle_slots,
                task_id=f"{task_id}_eval",
                args_dict=wv_args_dict,
                app_port=extracted_port
            )

            eval_result = {
                "status": "PASS" if raw_eval_result.get("Success_Rate_SR", 0) == 1 else "FAIL",
                "sr": raw_eval_result.get("Success_Rate_SR", 0),
                "tcr": raw_eval_result.get("Task_Completion_Rate_TCR", 0.0),
                "text": "WebVoyager evaluation completed. See raw_metrics for details.",
                "raw_metrics": raw_eval_result
            }

        except Exception as e:
            tqdm.write(f"\033[91m[Error] WebVoyager evaluation failed (or server detection timeout): {e}\033[0m")
            eval_result = {
                "status": "ERROR",
                "sr": 0,
                "tcr": 0.0,
                "text": f"Evaluation Error: {str(e)}",
                "raw_metrics": {"Total_Weight": 0.0, "Details": []}
            }
        finally:
            if 'server_process' in locals():
                stop_process_tree(server_process)

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

    interaction_history_path = os.path.join(log_dir, "interaction_history.json")
    needs_reeval_only = False
    old_history_data = None

    if not args.overwrite and os.path.exists(interaction_history_path):
        try:
            with open(interaction_history_path, "r", encoding="utf-8") as f:
                old_history_data = json.load(f)

            trajectory = old_history_data.get("trajectory", [])
            if trajectory:
                last_turn = trajectory[-1]
                debug_info = last_turn.get("debug_info", {})
                if debug_info.get("is_final") is True:
                    eval_detail = debug_info.get("evaluation_detail", {})
                    status = eval_detail.get("status", "UNKNOWN")

                    raw_metrics = eval_detail.get("raw_metrics", {})
                    details = raw_metrics.get("Details", [])
                    not_observed_count = sum(
                        1 for d in details
                        if isinstance(d, dict) and "Not observed or tested" in str(d.get("reason", ""))
                    )
                    total_slots = len(details)

                    has_index = os.path.exists(os.path.join(workspace_dir, "index.html"))
                    has_src = os.path.exists(os.path.join(workspace_dir, "src"))
                    workspace_has_code = has_index or has_src

                    if status in ["CRASHED", "ERROR"]:
                        tqdm.write(
                            f"\033[91m[Resume] Task {task_id} history status is {status}. Fatal error occurred during code generation. Cleaning workspace and logs for full rerun.\033[0m")
                        if os.path.exists(log_dir):
                            shutil.rmtree(log_dir, ignore_errors=True)

                    elif total_slots > 0 and not_observed_count == total_slots:
                        if workspace_has_code:
                            tqdm.write(
                                f"\033[93m[Resume] Task {task_id} detected full 'Not observed' anomaly. Code exists → entering safe mode: re-evaluation only.\033[0m")
                            needs_reeval_only = True
                        else:
                            tqdm.write(
                                f"\033[91m[Resume] Task {task_id} has invalid zero score and missing code. Forcing full rerun.\033[0m")
                            if os.path.exists(log_dir):
                                shutil.rmtree(log_dir, ignore_errors=True)
                    else:
                        tqdm.write(f"[Resume] Task {task_id} already has valid evaluation (Status: {status}), skipping.")
                        return task_id
                else:
                    tqdm.write(f"[Resume] Task {task_id} missing final evaluation stage. Cleaning and rerunning.")
                    if os.path.exists(log_dir):
                        shutil.rmtree(log_dir, ignore_errors=True)
            else:
                tqdm.write(f"[Resume] Task {task_id} has empty trajectory. Cleaning and rerunning.")
                if os.path.exists(log_dir):
                    shutil.rmtree(log_dir, ignore_errors=True)

        except Exception as e:
            tqdm.write(f"[Resume] Failed to read history of task {task_id} ({e}). Cleaning and rerunning.")
            if os.path.exists(log_dir):
                shutil.rmtree(log_dir, ignore_errors=True)

    # === Re-evaluation only mode ===
    if needs_reeval_only and old_history_data:
        tqdm.write(f"\n==== Task {task_id} [{difficulty.upper()}] entering standalone visual re-evaluation mode ====")

        user_model_url = os.environ.get("USER_MODEL_BASE_URL") or get_vlm_endpoint(args.user_model)
        user_model_key = os.environ.get("USER_MODEL_API_KEY") or os.environ.get(
            "OPENAILIKE_VLM_API_KEY", "sk-local-test"
        )

        user_sim = UserSimulator(
            ground_truth_instruction=task.get("ground_truth_instruction", user_instruction),
            initial_instruction=user_instruction,
            evaluation_checklist=task.get("evaluation_checklist", []),
            persona=persona,
            model=args.user_model,
            vlm_model=args.webvoyager_model,
            base_url=user_model_url,
            api_key=user_model_key
        )

        class MockBuilder:
            def __init__(self):
                self.messages = []
                self.format_error_count = 0

        mock_builder = MockBuilder()
        trajectory = old_history_data.get("trajectory", [])

        for item in trajectory[:-1]:
            msg = {"role": item["role"], "content": item["content"]}
            if "debug_info" in item:
                msg["info"] = item["debug_info"]
            mock_builder.messages.append(msg)

        mock_builder.format_error_count = old_history_data.get(
            "path_distribution_stats", {}
        ).get("FORMAT_ERROR_COUNT", 0)

        old_stop_reason = trajectory[-1].get("debug_info", {}).get("stop_reason", "submitted")
        eval_instruction = ground_truth if old_stop_reason == "submitted" else user_instruction

        perform_final_evaluation(
            builder=mock_builder,
            user_sim=user_sim,
            workspace_dir=workspace_dir,
            log_dir=log_dir,
            oracle_slots=current_oracle_slots,
            user_instruction=eval_instruction,
            task_id=task_id,
            args=args,
            stop_reason=old_stop_reason
        )

        save_interaction_history(
            mock_builder.messages,
            os.path.join(log_dir, "interaction_history.json"),
            mock_builder.format_error_count
        )
        return task_id

    # === Normal execution ===
    tqdm.write(f"\n==== Task {task_id} [{difficulty.upper()}] starting execution ====")

    builder_url = os.environ.get("BUILDER_BASE_URL") or get_vlm_endpoint(args.builder_model)
    builder_key = os.environ.get("BUILDER_API_KEY") or os.environ.get("OPENAILIKE_API_KEY")

    copilot_url = os.environ.get("COPILOT_BASE_URL") or get_vlm_endpoint(args.visual_copilot_model)
    copilot_key = os.environ.get("COPILOT_API_KEY") or os.environ.get(
        "OPENAILIKE_VLM_API_KEY"
    ) or os.environ.get("OPENAILIKE_API_KEY")

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
        max_tokens=16000,
        builder_url=builder_url,
        builder_key=builder_key,
        vlm_url=copilot_url,
        vlm_key=copilot_key
    )

    user_model_url = os.environ.get("USER_MODEL_BASE_URL") or get_vlm_endpoint(args.user_model)
    user_model_key = os.environ.get("USER_MODEL_API_KEY") or os.environ.get(
        "OPENAILIKE_VLM_API_KEY", "sk-local-test"
    )

    user_sim = UserSimulator(
        ground_truth_instruction=task.get("ground_truth_instruction", user_instruction),
        initial_instruction=user_instruction,
        evaluation_checklist=task.get("evaluation_checklist", []),
        persona=persona,
        model=args.user_model,
        vlm_model=args.webvoyager_model,
        base_url=user_model_url,
        api_key=user_model_key
    )

    turn_counter = 0
    loop_idx = 0
    is_graded = False

    while turn_counter < max_turns:
        tqdm.write(f"\n--- Task {task_id} | Turn {turn_counter + 1}/{max_turns} ---")

        action, is_failed = builder.step(loop_idx, simulation_mode=True)
        builder.save_history(loop_idx)
        loop_idx += 1

        if action["type"] == "question":
            answer = user_sim.answer_question(action["content"])
            builder.messages.append({"role": "user", "content": answer})
            turn_counter += 1
            continue

        elif action["type"] in ["coding", "internal_test", "format_error"]:
            if builder.is_finished:
                break
            turn_counter += 1
            continue

        elif action["type"] == "submitted":
            perform_final_evaluation(
                builder,
                user_sim,
                workspace_dir,
                log_dir,
                oracle_slots=current_oracle_slots,
                user_instruction=ground_truth,
                task_id=task_id,
                args=args,
                stop_reason="submitted"
            )
            is_graded = True
            break

    if not is_graded:
        reason = "max_turns_reached" if turn_counter >= max_turns else "verification_deadlock"
        perform_final_evaluation(
            builder,
            user_sim,
            workspace_dir,
            log_dir,
            oracle_slots=current_oracle_slots,
            user_instruction=user_instruction,
            task_id=task_id,
            args=args,
            stop_reason=reason
        )

    format_errors = getattr(builder, 'format_error_count', 0)

    save_interaction_history(
        builder.messages,
        os.path.join(log_dir, "interaction_history.json"),
        format_errors
    )

    return task_id

def main():
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to the YAML configuration file")
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--builder_model", type=str, default="gpt-4.1")
    parser.add_argument("--visual_copilot_model", type=str, default="gpt-4.1")
    parser.add_argument("--webvoyager_model", type=str, default="gpt-5-mini")
    parser.add_argument("--user_model", type=str, default="deepseek-v3.2")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_workers", type=int, default=1, help="Number of concurrent tasks")

    args = parser.parse_args()

    if args.config:
        if not os.path.exists(args.config):
            print(f" Error: Config file not found: {args.config}")
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
        print(f" Error: Data file not found: {args.data_path}")
        return

    with open(args.data_path, "r", encoding="utf-8") as f:
        tasks = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(tasks)} tasks.")
    print(f" Starting concurrent evaluation | Workers: {args.max_workers}")
    print(f" Builder model: {args.builder_model}")
    print(f" Copilot model: {args.visual_copilot_model}")
    print(f" User model: {args.user_model}")
    print(f" WebVoyager model: {args.webvoyager_model}")

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = []
        for task in tasks:
            futures.append(executor.submit(run_single_task, task, args))

        with tqdm(total=len(tasks), desc="Processing Tasks", unit="task") as pbar:
            for future in concurrent.futures.as_completed(futures):
                try:
                    finished_task_id = future.result()
                    tqdm.write(f" Task {finished_task_id} completed successfully.")
                except Exception as e:
                    tqdm.write(f" Uncaught exception during task execution: {e}")
                finally:
                    pbar.update(1)


if __name__ == "__main__":
    main()
