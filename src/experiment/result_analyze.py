import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import re

DIFFICULTY_ORDER = {"Easy": 1, "Middle": 2, "Hard": 3, "Unknown": 99}


def load_difficulty_map(dataset_paths):
    diff_map = {}
    if not dataset_paths:
        return diff_map

    if isinstance(dataset_paths, str):
        dataset_paths = [dataset_paths]

    for path in dataset_paths:
        if not os.path.exists(path):
            print(f"Dataset file not found: {path}, skipping.")
            continue

        try:
            with open(path, 'r', encoding='utf-8') as f:
                if str(path).endswith('.jsonl'):
                    for line in f:
                        if not line.strip(): continue
                        data = json.loads(line)
                        task_id = str(data.get("original_id") or data.get("id") or data.get("task_id"))
                        diff = data.get("difficulty") or data.get("level") or data.get("task_level")

                        if task_id and task_id != "None":
                            base_id = task_id.split('_')[0] if '_' in task_id else task_id
                            if diff and str(diff).strip().lower() != "unknown":
                                diff_map[base_id] = str(diff).strip().capitalize()
                            elif base_id not in diff_map:
                                diff_map[base_id] = "Unknown"

                elif str(path).endswith('.json'):
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            task_id = str(item.get("original_id") or item.get("id") or item.get("task_id"))
                            diff = item.get("difficulty") or item.get("level") or item.get("task_level")
                            if task_id and task_id != "None":
                                base_id = task_id.split('_')[0] if '_' in task_id else task_id
                                if diff and str(diff).strip().lower() != "unknown":
                                    diff_map[base_id] = str(diff).strip().capitalize()
                                elif base_id not in diff_map:
                                    diff_map[base_id] = "Unknown"
                    elif isinstance(data, dict):
                        for task_id, item in data.items():
                            diff = item.get("difficulty") or item.get("level") or item.get("task_level")
                            base_id = str(task_id).split('_')[0] if '_' in str(task_id) else str(task_id)
                            if diff and str(diff).strip().lower() != "unknown":
                                diff_map[base_id] = str(diff).strip().capitalize()
                            elif base_id not in diff_map:
                                diff_map[base_id] = "Unknown"
        except Exception as e:
            print(f"Failed to parse dataset {path}: {e}")

    return diff_map


def analyze_batch_trajectories(root_dir, dataset_paths=None):
    results_dict = {}
    root_path = Path(root_dir)

    overlap_output_dir = root_path.parent / "overlap_logs"
    overlap_tasks = []
    fixed_count = 0

    difficulty_map = load_difficulty_map(dataset_paths)

    print(f"Recursively scanning log directory: {root_path.absolute()}")

    for filepath in root_path.rglob('*'):
        if filepath.suffix not in ['.json', '.jsonl']:
            continue

        try:
            rel_path = filepath.relative_to(root_path)
            task_id = rel_path.parts[0]
        except Exception:
            task_id = filepath.parent.name

        base_task_id = task_id.split('_')[0] if '_' in task_id else task_id
        role = task_id.split('_')[-1] if '_' in task_id else 'Unknown_Role'

        if task_id not in results_dict:
            results_dict[task_id] = {
                "task_id": task_id,
                "base_task_id": base_task_id,
                "role": role,
                "difficulty": difficulty_map.get(base_task_id, "Unknown"),
                "final_state": "CRASHED/UNFINISHED",
                "termination_type": "(Unknown)",
                "tcr_score": 0.0,
                "tcr_score_no_hallu": 0.0,
                "hallucination_triggered": np.nan,
                "total_steps": 0,
                "path_a_count": 0,
                "path_b_count": 0,
                "path_c_count": 0,
                "path_d_count": 0,
                "ignored_a_count": 0,
                "ignored_b_count": 0,
                "ignored_c_count": 0,
                "ignored_d_count": 0,
                "conflict_2_paths": 0,
                "conflict_3_paths": 0,
                "conflict_4_paths": 0,
                "steal_step_count": 0,
                "blind_execution_count": 0,
                "ask_before_code_count": 0,
                "ask_after_code_count": 0,
                "violating_chitchat_count": 0,
                "overlap_actions_count": 0,
                "is_abnormal": True
            }

        try:
            mode = 'r+' if filepath.suffix == '.json' else 'r'
            with open(filepath, mode, encoding='utf-8') as f:
                if filepath.suffix == '.jsonl':
                    pass
                else:
                    data = json.load(f)

                    if isinstance(data, dict):
                        if results_dict[task_id]["difficulty"] == "Unknown":
                            diff = data.get("difficulty") or data.get("task_level") or data.get("level")
                            if diff:
                                results_dict[task_id]["difficulty"] = str(diff).capitalize()

                        old_stats = data.get("path_distribution_stats", {})
                        trajectory = data.get("trajectory", [])

                        if isinstance(trajectory, list) and len(trajectory) > 0:
                            new_stats = {
                                "PATH_A_CLARIFY": 0,
                                "PATH_B_IMPLEMENT": 0,
                                "PATH_C_VERIFY": 0,
                                "PATH_D_SUBMIT": 0,
                                "IGNORED_PATH_A": 0,
                                "IGNORED_PATH_B": 0,
                                "IGNORED_PATH_C": 0,
                                "IGNORED_PATH_D": 0,
                                "CONFLICT_2_PATHS": 0,
                                "CONFLICT_3_PATHS": 0,
                                "CONFLICT_4_PATHS": 0,
                                "STEAL_STEP_COUNT": 0,
                                "BLIND_EXECUTION_COUNT": 0,
                                "FORMAT_ERROR_COUNT": old_stats.get("FORMAT_ERROR_COUNT", 0),
                                "OVERLAP_ACTIONS_COUNT": 0
                            }

                            first_code_turn = -1
                            ask_before = 0
                            ask_after = 0
                            has_submit = False
                            action_sequence = []
                            has_overlap_in_task = False

                            for turn in trajectory:
                                if turn.get("role") == "assistant":
                                    content = turn.get("content", "")
                                    turn_num = turn.get("turn", -1)

                                    pos_ask = content.find('<boltAction type="ask_user"')
                                    pos_verify = content.find('<boltAction type="screenshot_validated"')
                                    pos_finish = content.find('<boltAction type="finish"')
                                    code_match = re.search(
                                        r'<boltArtifact|<boltAction\s+type\s*=\s*["\'](file|shell|start)["\']', content,
                                        re.IGNORECASE)
                                    pos_code = code_match.start() if code_match else -1

                                    has_ask = pos_ask != -1
                                    has_verify = pos_verify != -1
                                    has_finish = pos_finish != -1
                                    has_code = pos_code != -1

                                    action_sum = int(has_ask) + int(has_verify) + int(has_finish) + int(has_code)
                                    if action_sum > 1:
                                        new_stats["OVERLAP_ACTIONS_COUNT"] += 1
                                        new_stats[f"CONFLICT_{action_sum}_PATHS"] += 1
                                        has_overlap_in_task = True

                                    if has_code:
                                        new_stats["PATH_B_IMPLEMENT"] += 1
                                        action_sequence.append('B')

                                    dominant_action = None
                                    if has_finish:
                                        dominant_action = 'D'
                                        new_stats["PATH_D_SUBMIT"] += 1
                                        has_submit = True
                                        action_sequence.append('D')
                                    elif has_ask:
                                        dominant_action = 'A'
                                        new_stats["PATH_A_CLARIFY"] += 1
                                        action_sequence.append('A')
                                    elif has_verify:
                                        dominant_action = 'C'
                                        new_stats["PATH_C_VERIFY"] += 1
                                        action_sequence.append('C')
                                    elif not has_code:
                                        dominant_action = 'ERR'
                                        new_stats["FORMAT_ERROR_COUNT"] = new_stats.get("FORMAT_ERROR_COUNT", 0) + 1
                                        action_sequence.append('ERR')

                                    if has_ask and dominant_action != 'A':
                                        new_stats["IGNORED_PATH_A"] += 1
                                    if has_verify and dominant_action != 'C':
                                        new_stats["IGNORED_PATH_C"] += 1
                                    if has_finish and dominant_action != 'D':
                                        new_stats["IGNORED_PATH_D"] += 1

                                    if has_code and dominant_action is not None and dominant_action != 'B':
                                        if action_sum == 2:
                                            dom_pos = {'D': pos_finish, 'A': pos_ask, 'C': pos_verify}[dominant_action]
                                            if pos_code < dom_pos:
                                                new_stats["STEAL_STEP_COUNT"] += 1
                                            else:
                                                new_stats["BLIND_EXECUTION_COUNT"] += 1
                                        else:
                                            new_stats["IGNORED_PATH_B"] += 1

                                    if has_code and first_code_turn == -1:
                                        first_code_turn = turn_num

                                    if has_ask:
                                        if first_code_turn == -1 or turn_num < first_code_turn:
                                            ask_before += 1
                                        else:
                                            ask_after += 1

                            if (old_stats.get("PATH_B_IMPLEMENT") != new_stats["PATH_B_IMPLEMENT"] or
                                    old_stats.get("PATH_C_VERIFY") != new_stats["PATH_C_VERIFY"] or
                                    old_stats.get("PATH_D_SUBMIT") != new_stats["PATH_D_SUBMIT"] or
                                    old_stats.get("FORMAT_ERROR_COUNT") != new_stats["FORMAT_ERROR_COUNT"] or
                                    old_stats.get("OVERLAP_ACTIONS_COUNT", -1) != new_stats["OVERLAP_ACTIONS_COUNT"] or
                                    "STEAL_STEP_COUNT" not in old_stats):
                                data["path_distribution_stats"] = new_stats
                                f.seek(0)
                                json.dump(data, f, ensure_ascii=False, indent=2)
                                f.truncate()
                                fixed_count += 1

                            if has_overlap_in_task and task_id not in overlap_tasks:
                                overlap_tasks.append(task_id)

                            results_dict[task_id]["total_steps"] = len(action_sequence)
                            results_dict[task_id]["path_a_count"] = new_stats["PATH_A_CLARIFY"]
                            results_dict[task_id]["path_b_count"] = new_stats["PATH_B_IMPLEMENT"]
                            results_dict[task_id]["path_c_count"] = new_stats["PATH_C_VERIFY"]
                            results_dict[task_id]["path_d_count"] = new_stats["PATH_D_SUBMIT"]

                            results_dict[task_id]["ignored_a_count"] = new_stats["IGNORED_PATH_A"]
                            results_dict[task_id]["ignored_b_count"] = new_stats["IGNORED_PATH_B"]
                            results_dict[task_id]["ignored_c_count"] = new_stats["IGNORED_PATH_C"]
                            results_dict[task_id]["ignored_d_count"] = new_stats["IGNORED_PATH_D"]

                            results_dict[task_id]["conflict_2_paths"] = new_stats["CONFLICT_2_PATHS"]
                            results_dict[task_id]["conflict_3_paths"] = new_stats["CONFLICT_3_PATHS"]
                            results_dict[task_id]["conflict_4_paths"] = new_stats["CONFLICT_4_PATHS"]
                            results_dict[task_id]["steal_step_count"] = new_stats["STEAL_STEP_COUNT"]
                            results_dict[task_id]["blind_execution_count"] = new_stats["BLIND_EXECUTION_COUNT"]

                            results_dict[task_id]["ask_before_code_count"] = max(
                                results_dict[task_id]["ask_before_code_count"], ask_before)
                            results_dict[task_id]["ask_after_code_count"] = max(
                                results_dict[task_id]["ask_after_code_count"], ask_after)
                            results_dict[task_id]["violating_chitchat_count"] = max(
                                results_dict[task_id]["violating_chitchat_count"], new_stats["FORMAT_ERROR_COUNT"])
                            results_dict[task_id]["overlap_actions_count"] = new_stats["OVERLAP_ACTIONS_COUNT"]

                            stop_reason = ""
                            for item in reversed(trajectory):
                                if isinstance(item, dict) and "debug_info" in item:
                                    debug_info = item["debug_info"]
                                    if not isinstance(debug_info, dict): continue

                                    if "stop_reason" in debug_info and not stop_reason:
                                        stop_reason = debug_info["stop_reason"]

                                    eval_detail = debug_info.get("evaluation_detail", {})
                                    if "tcr" in eval_detail:
                                        results_dict[task_id]["tcr_score"] = float(eval_detail["tcr"])
                                        results_dict[task_id]["is_abnormal"] = False
                                        if "status" in eval_detail:
                                            results_dict[task_id]["final_state"] = str(eval_detail["status"]).upper()

                                        raw_metrics = eval_detail.get("raw_metrics", {})
                                        oracle_slots = debug_info.get("oracle_slots_used_for_grading", [])

                                        if raw_metrics and oracle_slots:
                                            pos_weight_total = 0.0
                                            pos_weight_passed = 0.0
                                            hallu_triggered = False

                                            details_list = []
                                            if isinstance(raw_metrics, dict) and "Details" in raw_metrics:
                                                details_list = raw_metrics["Details"]
                                            elif isinstance(raw_metrics, list):
                                                details_list = raw_metrics

                                            for idx, slot in enumerate(oracle_slots):
                                                passed = False

                                                if details_list and idx < len(details_list):
                                                    detail_item = details_list[idx]
                                                    if "passed" in detail_item:
                                                        passed = bool(detail_item["passed"])
                                                    elif "status" in detail_item:
                                                        passed = str(detail_item["status"]).lower() == "pass"
                                                else:
                                                    if isinstance(raw_metrics, dict):
                                                        v = raw_metrics.get(str(idx)) or raw_metrics.get(
                                                            f"Checklist ID [{idx}]")
                                                        if v:
                                                            status_str = v.get("status", "fail") if isinstance(v,
                                                                                                               dict) else v
                                                            passed = str(status_str).lower() == "pass"

                                                weight = float(slot.get("final_weight", 1.0))
                                                is_negative = slot.get("assertion_type", "").upper() == "NEGATIVE"

                                                if is_negative:
                                                    if not passed:
                                                        hallu_triggered = True
                                                else:
                                                    pos_weight_total += weight
                                                    if passed:
                                                        pos_weight_passed += weight

                                            if pos_weight_total > 0:
                                                results_dict[task_id][
                                                    "tcr_score_no_hallu"] = pos_weight_passed / pos_weight_total
                                            else:
                                                results_dict[task_id]["tcr_score_no_hallu"] = results_dict[task_id][
                                                    "tcr_score"]

                                            results_dict[task_id][
                                                "hallucination_triggered"] = 1 if hallu_triggered else 0
                                        break

                            final_state = results_dict[task_id]["final_state"]

                            if has_submit:
                                results_dict[task_id]["termination_type"] = "(Submit)"
                            elif len(action_sequence) >= 15:
                                is_loop = False
                                if len(action_sequence) >= 4:
                                    last_4_actions = action_sequence[-4:]
                                    if all(act in ['B', 'C', 'ERR'] for act in last_4_actions):
                                        is_loop = True
                                results_dict[task_id][
                                    "termination_type"] = "(Infinite Loop)" if is_loop else "(Max Turns)"
                            elif final_state in ["CRASHED", "ERROR"]:
                                results_dict[task_id]["termination_type"] = "(Crashed/Error)"
                            else:
                                is_loop = False
                                if len(action_sequence) >= 4:
                                    last_4_actions = action_sequence[-4:]
                                    if all(act in ['B', 'C', 'ERR'] for act in last_4_actions):
                                        is_loop = True
                                results_dict[task_id][
                                    "termination_type"] = "(Infinite Loop)" if is_loop else "(Interrupted)"

        except json.JSONDecodeError:
            pass
        except Exception as e:
            pass

    if fixed_count > 0:
        print(f"\nAutomatic repair completed successfully! A total of {fixed_count} task files had their header records precisely overwritten and corrected, while all original trajectory data was safely preserved.")

    df = pd.DataFrame(list(results_dict.values()))
    if df.empty:
        print("No valid data found!")
        return

    normal_df = df[df['is_abnormal'] == False]
    abnormal_df = df[df['is_abnormal'] == True]

    print("\n" + "=" * 65)
    print(f" PEBench [{root_path.name}] Global Batch Statistics Report")
    print("=" * 65)
    print(
        f"Total aggregated tasks: {len(df)} | Normal: {len(normal_df)} | Abnormal (forced to 0 score): {len(abnormal_df)}")
    print("\n" + "=" * 65)
    print("[Key Finding 1] Comparison by Role")
    print("=" * 65)

    roles = sorted(df['role'].unique())
    for r in roles:
        r_df = df[df['role'] == r]
        print(f"\nRole: [{r}] (Total tested tasks: {len(r_df)})")

        avg_tcr = r_df['tcr_score'].mean()
        avg_tcr_no_hallu = r_df['tcr_score_no_hallu'].mean()

        valid_hallu_df = r_df.dropna(subset=['hallucination_triggered'])
        hallu_mean = valid_hallu_df['hallucination_triggered'].mean() if not valid_hallu_df.empty else np.nan
        hallu_rate_str = f"{hallu_mean * 100:.1f}%" if pd.notna(hallu_mean) else "N/A (all samples in this group crashed, no scores available)"
        role_overall_ask_rate = (r_df['path_a_count'] > 0).mean() * 100
        role_avg_ask_count = r_df['path_a_count'].mean()

        print(f"   Average TCR Score: {avg_tcr:.4f} (Clean TCR without hallucinations: {avg_tcr_no_hallu:.4f})")
        print(f"   Hallucination Rate: {hallu_rate_str}")

        status_str = " | ".join([f"{k}: {v}" for k, v in r_df['final_state'].value_counts().items()])
        print(f"   Status Distribution: {status_str}")

        term_counts = r_df['termination_type'].value_counts()
        term_str = " | ".join([f"{k}: {v}" for k, v in term_counts.items()])
        print(f"   Termination Reasons: {term_str}")

        print(f"   Average Steps: {r_df['total_steps'].mean():.1f} steps")

        print(f"   Overall Ask Rate: {role_overall_ask_rate:.1f}%")
        print(f"   Average Ask Count: {role_avg_ask_count:.2f}")

        avg_a = r_df['path_a_count'].mean()
        avg_b = r_df['path_b_count'].mean()
        avg_c = r_df['path_c_count'].mean()
        avg_d = r_df['path_d_count'].mean()
        print(
            f"   Average PATH Execution Counts: [A Clarify]: {avg_a:.1f} | [B Implement (committed)]: {avg_b:.1f} | [C Verify]: {avg_c:.1f} | [D Submit]: {avg_d:.1f}"
        )

        avg_steal = r_df['steal_step_count'].mean()
        avg_blind = r_df['blind_execution_count'].mean()
        avg_ig_b_multi = r_df['ignored_b_count'].mean()
        print(
            f"   Code Conflicts (B only): [2-Path Step Stealing]: {avg_steal:.2f} | [2-Path Blind Execution]: {avg_blind:.2f} | [Multi-Path Loss]: {avg_ig_b_multi:.2f}"
        )

        c2, c3, c4 = r_df['conflict_2_paths'].sum(), r_df['conflict_3_paths'].sum(), r_df['conflict_4_paths'].sum()
        print(f"   Conflict Diversity: [2-Path]: {c2} | [3-Path]: {c3} | [4-Path]: {c4}")

        avg_ig_a = r_df['ignored_a_count'].mean()
        avg_ig_c = r_df['ignored_c_count'].mean()
        avg_ig_d = r_df['ignored_d_count'].mean()
        print(
            f"   Other Missed Conflicts: [A Clarify (not asked)]: {avg_ig_a:.2f} | [C Verify (no screenshot)]: {avg_ig_c:.2f} | [D Submit (not submitted)]: {avg_ig_d:.2f}"
        )

        avg_violating = r_df['violating_chitchat_count'].mean()
        avg_overlap = r_df['overlap_actions_count'].mean()
        print(
            f"   Avg Format Violations (text-only): {avg_violating:.2f} | Avg Premature Actions (multi-path): {avg_overlap:.2f}")

    print("\n" + "=" * 65)
    print("[Key Finding 2] Comparison by Task Difficulty")
    print("=" * 65)

    difficulties = sorted(df['difficulty'].unique(), key=lambda x: DIFFICULTY_ORDER.get(x, 99))

    for d in difficulties:
        d_df = df[df['difficulty'] == d]
        print(f"\nTask Difficulty: [{d}] (Total: {len(d_df)} tasks)")

        if d == "Unknown" and len(d_df) > 0:
            print(
                "   Note: The following tasks were not found in the provided dataset with a difficulty label and are categorized as Unknown:")
            for _, row in d_df.iterrows():
                print(f"      - {row['task_id']}")

        avg_tcr = d_df['tcr_score'].mean()
        avg_tcr_no_hallu = d_df['tcr_score_no_hallu'].mean()

        valid_hallu_df = d_df.dropna(subset=['hallucination_triggered'])
        hallu_mean = valid_hallu_df['hallucination_triggered'].mean() if not valid_hallu_df.empty else np.nan
        hallu_rate_str = f"{hallu_mean * 100:.1f}%" if pd.notna(hallu_mean) else "N/A (no scoring data)"

        print(f"   Average TCR Score: {avg_tcr:.4f} (Clean TCR without hallucinations: {avg_tcr_no_hallu:.4f})")
        print(f"   Hallucination Rate: {hallu_rate_str}")

        status_str = " | ".join([f"{k}: {v}" for k, v in d_df['final_state'].value_counts().items()])
        print(f"   Status Distribution: {status_str}")

        term_counts = d_df['termination_type'].value_counts()
        term_str = " | ".join([f"{k}: {v}" for k, v in term_counts.items()])
        print(f"   Termination Reasons: {term_str}")

        print(f"   Average Steps: {d_df['total_steps'].mean():.1f} steps")

        avg_a = d_df['path_a_count'].mean()
        avg_b = d_df['path_b_count'].mean()
        avg_c = d_df['path_c_count'].mean()
        avg_d = d_df['path_d_count'].mean()
        print(
            f"   Average PATH Execution Counts: [A Clarify]: {avg_a:.1f} | [B Implement (committed)]: {avg_b:.1f} | [C Verify]: {avg_c:.1f} | [D Submit]: {avg_d:.1f}"
        )
    print("\n" + "=" * 65)
    print(" [Special Analysis] Quality Comparison: Active Submit vs Non-Active Termination")
    print("=" * 65)

    submit_df = df[df['termination_type'] == 'Active Submit']
    non_submit_df = df[df['termination_type'] != 'Active Submit']

    print(f"[Active Submit] (Total: {len(submit_df)} tasks)")
    if not submit_df.empty:
        print(
            f"   Average TCR Score: {submit_df['tcr_score'].mean():.4f} (Clean TCR: {submit_df['tcr_score_no_hallu'].mean():.4f})"
        )
        valid_submit_hallu = submit_df.dropna(subset=['hallucination_triggered'])
        submit_hallu = valid_submit_hallu['hallucination_triggered'].mean() if not valid_submit_hallu.empty else np.nan
        print(f"   Hallucination Rate: {submit_hallu * 100:.1f}%" if pd.notna(
            submit_hallu) else "   Hallucination Rate: N/A")
    else:
        print("   (No data)")

    print(f"\n[Non-Active Submit (Other Passive Terminations)] (Total: {len(non_submit_df)} tasks)")
    if not non_submit_df.empty:
        print(
            f"   Average TCR Score: {non_submit_df['tcr_score'].mean():.4f} (Clean TCR: {non_submit_df['tcr_score_no_hallu'].mean():.4f})"
        )
        valid_non_submit_hallu = non_submit_df.dropna(subset=['hallucination_triggered'])
        non_submit_hallu = valid_non_submit_hallu[
            'hallucination_triggered'].mean() if not valid_non_submit_hallu.empty else np.nan
        print(f"   Hallucination Rate: {non_submit_hallu * 100:.1f}%" if pd.notna(
            non_submit_hallu) else "   Hallucination Rate: N/A")
    else:
        print("   (No data)")

    error_df = df[df['final_state'] == 'ERROR']
    crashed_df = df[df['final_state'] == 'CRASHED']

    if not error_df.empty or not crashed_df.empty:
        print("\n" + "=" * 65)
        print(
            "Anomaly Traceback: The following tasks experienced execution crashes or scoring failures. Please investigate!")
        print("=" * 65)

        if not error_df.empty:
            print("\n[ERROR Status] (Scoring system or browser crash):")
            for _, row in error_df.iterrows():
                print(f"   - Task ID: {row['task_id']:<18} | Role: {row['role']:<6} | Diff: {row['difficulty']:<6}")

        if not crashed_df.empty:
            print(
                "\n[CRASHED Status] (Model produced faulty code causing infinite loops or self-crash, not evaluated):")
            for _, row in crashed_df.iterrows():
                print(f"   - Task ID: {row['task_id']:<18} | Role: {row['role']:<6} | Diff: {row['difficulty']:<6}")
        abnormal_ids = set(error_df['task_id'].tolist()) | set(crashed_df['task_id'].tolist())
        retest_path = os.path.join(str(root_path.parent), "retest_tasks.jsonl")
        error_logs_dir = os.path.join(str(root_path.parent), "error_logs")
        extracted_count = 0

        copied_count = 0
        error_workspaces_dir = os.path.join(str(root_path.parent), "error_workspaces")
        workspaces_root = os.path.join(str(root_path.parent), "workspaces")
        if abnormal_ids:
            os.makedirs(error_logs_dir, exist_ok=True)
            os.makedirs(error_workspaces_dir, exist_ok=True)
            for task_id in sorted(abnormal_ids):
                src_log = os.path.join(str(root_path), task_id)
                dst_log = os.path.join(error_logs_dir, task_id)
                if os.path.exists(src_log):
                    if os.path.exists(dst_log):
                        shutil.rmtree(dst_log)
                    shutil.copytree(src_log, dst_log)
                    copied_count += 1

                src_ws = os.path.join(workspaces_root, task_id)
                dst_ws = os.path.join(error_workspaces_dir, task_id)
                if os.path.exists(src_ws):
                    if os.path.exists(dst_ws):
                        shutil.rmtree(dst_ws)
                    try:
                        shutil.copytree(
                            src_ws,
                            dst_ws,
                            symlinks=True,
                            ignore=shutil.ignore_patterns('node_modules', 'dist', '.git', '.next')
                        )
                    except Exception as e:
                        print(f"Failed to fully copy workspace {task_id}: {e}")

            print(f"\nCopied {copied_count} abnormal task logs to: {error_logs_dir}")
            print(f"Copied corresponding workspaces to: {error_workspaces_dir}")

        if dataset_paths and abnormal_ids:
            try:
                with open(retest_path, 'w', encoding='utf-8') as out_f:
                    extracted_ids = set()
                    for d_path in dataset_paths:
                        if not os.path.exists(d_path): continue
                        with open(d_path, 'r', encoding='utf-8') as in_f:
                            if str(d_path).endswith('.jsonl'):
                                for line in in_f:
                                    if not line.strip(): continue
                                    item_data = json.loads(line)
                                    t_id = str(
                                        item_data.get("id") or item_data.get("task_id") or item_data.get("original_id"))
                                    if t_id in abnormal_ids and t_id not in extracted_ids:
                                        out_f.write(line.strip() + '\n')
                                        extracted_ids.add(t_id)
                                        extracted_count += 1
                            elif str(d_path).endswith('.json'):
                                json_data = json.load(in_f)
                                items = json_data if isinstance(json_data, list) else list(
                                    json_data.values()) if isinstance(json_data, dict) else []
                                for item in items:
                                    t_id = str(item.get("id") or item.get("task_id") or item.get("original_id"))
                                    if t_id in abnormal_ids and t_id not in extracted_ids:
                                        out_f.write(json.dumps(item, ensure_ascii=False) + '\n')
                                        extracted_ids.add(t_id)
                                        extracted_count += 1
                print(f"\nExtracted {extracted_count} abnormal tasks (ERROR + CRASHED) to: {retest_path}")
            except Exception as e:
                print(f"Error occurred while extracting abnormal tasks: {e}")
        elif abnormal_ids and not dataset_paths:
            try:
                with open(retest_path, 'w', encoding='utf-8') as out_f:
                    for t_id in sorted(abnormal_ids):
                        out_f.write(json.dumps({"id": t_id}, ensure_ascii=False) + '\n')
                print(f"\nSaved {len(abnormal_ids)} abnormal task IDs to: {retest_path}")
            except Exception as e:
                print(f"Error occurred while saving abnormal task IDs: {e}")

    if overlap_tasks:
        print("\n" + "=" * 65)
        print(f"Premature execution violation trace: Identified {len(overlap_tasks)} tasks with 'one-shot (multi-path overlapping output)' behavior!")
        print("=" * 65)

        os.makedirs(overlap_output_dir, exist_ok=True)
        copied_overlap = 0
        for t_id in overlap_tasks:
            src_log = os.path.join(str(root_path), t_id)
            dst_log = os.path.join(overlap_output_dir, t_id)
            if os.path.exists(src_log):
                if os.path.exists(dst_log):
                    shutil.rmtree(dst_log)
                shutil.copytree(src_log, dst_log)
                copied_overlap += 1
        print(f"Copied and isolated logs of these {copied_overlap} violating tasks to a new folder: {overlap_output_dir}")

    print("\n" + "=" * 65)
    print(" [Deep Analysis] Correlation Analysis between CRASHED State and Behavior")
    print("=" * 65)

    if not crashed_df.empty:
        print("\n1. Root Cause Analysis of CRASHED Tasks:")
        print(
            "   (Explanation: Some CRASHED cases are due to infinite loops leading to forced termination, while others are caused by fatal code errors)")
        crashed_term_counts = crashed_df['termination_type'].value_counts()
        for term, count in crashed_term_counts.items():
            pct = (count / len(crashed_df)) * 100
            print(f"   - [{term}]: Caused {count} CRASHED cases ({pct:.1f}%)")

    else:
        print("\nNo CRASHED tasks in the current batch. The model is running stably!")

    print("\n" + "=" * 65)
    print(" [Cross Analysis] Difficulty × Role Clean TCR Score Matrix (Pivot Table)")
    print("=" * 65)
    if "Unknown" not in difficulties or len(difficulties) > 1:
        pivot_tcr = pd.pivot_table(
            df,
            values='tcr_score_no_hallu',
            index='difficulty',
            columns='role',
            aggfunc='mean',
            fill_value=0
        )
        valid_indices = [d for d in ["Easy", "Middle", "Hard"] if d in pivot_tcr.index]
        if valid_indices:
            pivot_tcr = pivot_tcr.reindex(valid_indices)
        print(pivot_tcr.round(4).to_string())
    else:
        print("Note: All tasks in the current batch are labeled as 'Unknown' difficulty.")

    print("\n" + "=" * 65)
    print(" Overall Summary Statistics (All Roles Aggregated - including crashed tasks with zero scores)")
    print("=" * 65)
    if not df.empty:
        print(f" [1] Global Average TCR: {df['tcr_score'].mean():.4f}")
        print(f" [2] Global Average Clean TCR (no hallucinations): {df['tcr_score_no_hallu'].mean():.4f}")

        valid_global_hallu_df = df.dropna(subset=['hallucination_triggered'])
        global_hallu_mean = valid_global_hallu_df[
            'hallucination_triggered'].mean() if not valid_global_hallu_df.empty else np.nan
        global_hallu_rate_str = f"{global_hallu_mean * 100:.1f}%" if pd.notna(
            global_hallu_mean) else "N/A (no scoring data)"
        print(f" [3] Global Hallucination Rate (valid scored samples only): {global_hallu_rate_str}")

        # Global ask behavior
        global_overall_ask_rate = (df['path_a_count'] > 0).mean() * 100
        global_avg_ask_count = df['path_a_count'].mean()
        print(f" [4] Global Overall Ask Rate: {global_overall_ask_rate:.1f}%")
        print(f" [5] Global Average Ask Count: {global_avg_ask_count:.2f}")

        print(
            f" [6] Global Average PATH Execution Counts: [A Clarify]: {df['path_a_count'].mean():.2f} | [B Implement (dominant)]: {df['path_b_count'].mean():.2f} | [C Verify]: {df['path_c_count'].mean():.2f} | [D Submit]: {df['path_d_count'].mean():.2f}"
        )

        print(
            f" [7] Global Code Conflicts (B only): [2-Path Step Stealing]: {df['steal_step_count'].mean():.2f} | [2-Path Blind Execution]: {df['blind_execution_count'].mean():.2f} | [Multi-Path Loss]: {df['ignored_b_count'].mean():.2f}"
        )

        print(
            f" [8] Global Conflict Diversity: [2-Path]: {df['conflict_2_paths'].sum()} | [3-Path]: {df['conflict_3_paths'].sum()} | [4-Path]: {df['conflict_4_paths'].sum()}"
        )

        print(
            f" [9] Global Other Missed Conflicts: [A Clarify (not asked)]: {df['ignored_a_count'].mean():.2f} | [C Verify (no screenshot)]: {df['ignored_c_count'].mean():.2f} | [D Submit (not submitted)]: {df['ignored_d_count'].mean():.2f}"
        )

        global_violating_mean = df['violating_chitchat_count'].mean()
        global_overlap_mean = df['overlap_actions_count'].mean()
        print(
            f" [10] Global Avg Format Violations: {global_violating_mean:.2f} | Global Avg Premature Execution Violations: {global_overlap_mean:.2f}"
        )

    print("\n\n" + "=" * 90)
    print("=" * 90)

    print("\n► Table 2: Overall performance of evaluated web generation agents.")

    t2_easy = df[df['difficulty'] == 'Easy']['tcr_score_no_hallu'].mean() if not df[
        df['difficulty'] == 'Easy'].empty else 0.0
    t2_mid = df[df['difficulty'] == 'Middle']['tcr_score_no_hallu'].mean() if not df[
        df['difficulty'] == 'Middle'].empty else 0.0
    t2_hard = df[df['difficulty'] == 'Hard']['tcr_score_no_hallu'].mean() if not df[
        df['difficulty'] == 'Hard'].empty else 0.0

    overall_clean_tcr = df['tcr_score_no_hallu'].mean() if not df.empty else 0.0

    valid_overall_hallu_df = df.dropna(subset=['hallucination_triggered'])
    overall_hallu = valid_overall_hallu_df[
        'hallucination_triggered'].mean() if not valid_overall_hallu_df.empty else 0.0

    target_roles = ["P-MIN", "P-RAM", "P-INT", "P-CON"]
    actual_roles = df['role'].unique()
    display_roles = [r for r in target_roles if r in actual_roles]
    for r in actual_roles:
        if r not in display_roles:
            display_roles.append(r)

    header_t2 = f"{'Easy':>8} | {'Middle':>8} | {'Hard':>8} | "
    for r in display_roles:
        header_t2 += f"{r:>8} | "
    header_t2 += f"{'TCR ↑':>8} | {'Hallu. Rate ↓':>14}"

    print("-" * len(header_t2))
    print(header_t2)
    print("-" * len(header_t2))

    row_t2 = f"{t2_easy:>8.4f} | {t2_mid:>8.4f} | {t2_hard:>8.4f} | "
    for r in display_roles:
        r_tcr = df[df['role'] == r]['tcr_score_no_hallu'].mean() if not df[df['role'] == r].empty else 0.0
        row_t2 += f"{r_tcr:>8.4f} | "
    row_t2 += f"{overall_clean_tcr:>8.4f} | {overall_hallu * 100:>13.1f}%"
    print(row_t2)
    print("-" * len(header_t2))

    print("\n► Table 3: Ask Frequency")


    overall_ask_rate = (normal_df['path_a_count'] > 0).mean() * 100 if not normal_df.empty else 0.0

    avg_ask_count = normal_df['path_a_count'].mean() if not normal_df.empty else 0.0

    header_t3 = f"{'Overall Ask Rate ↑':>20} | {'Avg. Ask Count':>16}"
    print("-" * len(header_t3))
    print(header_t3)
    print("-" * len(header_t3))
    print(f"{overall_ask_rate:>19.1f}% | {avg_ask_count:>16.4f}")
    print("-" * len(header_t3))

    print("\n► Table 4: Behavioral profiling based on action trajectories")

    sum_c = normal_df['path_c_count'].sum()
    sum_b_pure = normal_df['path_b_count'].sum()
    sum_b_missing = normal_df['steal_step_count'].sum() + normal_df['blind_execution_count'].sum() + normal_df[
        'ignored_b_count'].sum()
    sum_b_total = sum_b_pure + sum_b_missing
    vci = sum_c / sum_b_total if sum_b_total > 0 else 0.0

    pcr = (normal_df['ask_before_code_count'] > 0).mean() if not normal_df.empty else 0.0
    submit_rate = (normal_df['termination_type'] == '(Submit)').mean() if not normal_df.empty else 0.0

    avg_steps = normal_df['total_steps'].mean() if not normal_df.empty else 0.0

    header_t4 = f"{'VCI ↑':>8} | {'PCR ↑':>8} | {'Submit Rate ↑':>15} | {'Avg. Steps ↓':>14} | {'Clean TCR ↑':>12}"
    print("-" * len(header_t4))
    print(header_t4)
    print("-" * len(header_t4))

    print(
        f"{vci:>8.4f} | {pcr * 100:>7.1f}% | {submit_rate * 100:>14.1f}% | {avg_steps:>14.2f} | {overall_clean_tcr:>12.4f}")
    print("-" * len(header_t4))
    print("=" * 90 + "\n")


    output_csv = os.path.join(str(root_path.parent), f"{root_path.name}_summary_with_roles.csv")
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"Detailed underlying data has been exported to: {output_csv}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze InteractWeb-Bench experiment results.")
    parser.add_argument("--dir", type=str,
                        default="/your_path/experiment_results/qwen3.6-plus/logs",
                        help="Path to the logs directory")
    parser.add_argument("--data", type=str, default="/your_path/data",
                        help="Path to the dataset directory (optional)")
    args = parser.parse_args()

    data_dir = args.data

    potential_files = [
        "low_scores_simulation_labeled.jsonl",
        "mid_scores_simulation_labeled.jsonl",
        "high_scores_simulation_labeled.jsonl",
    ]

    dataset_files = []
    for pf in potential_files:
        path = os.path.join(data_dir, pf)
        if os.path.exists(path):
            dataset_files.append(path)

    if not os.path.exists(args.dir):
        print(f"Log directory not found: {args.dir}")
    else:
        analyze_batch_trajectories(args.dir, dataset_paths=dataset_files or None)