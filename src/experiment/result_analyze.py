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
            print(f"找不到数据集文件: {path}，将跳过。")
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
            print(f"解析数据集 {path} 失败: {e}")

    return diff_map


def analyze_batch_trajectories(root_dir, dataset_paths=None):
    results_dict = {}
    root_path = Path(root_dir)

    overlap_output_dir = root_path.parent / "overlap_logs"
    overlap_tasks = []
    fixed_count = 0

    difficulty_map = load_difficulty_map(dataset_paths)

    print(f"正在递归扫描日志目录: {root_path.absolute()}")

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
                "termination_type": "未知 (Unknown)",
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

                                    # 1. 精确获取各类标签及它们的出现位置
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

                                    # 核心修改：按“实际执行的 PATH”计算步长
                                    # 1. 只要有代码，物理落盘必然发生。实质执行了 PATH_B，步长无条件 +1
                                    if has_code:
                                        new_stats["PATH_B_IMPLEMENT"] += 1
                                        action_sequence.append('B')

                                    # 2. 状态机漏斗，决定这一轮的“后续主导动作”（谁来决定系统的反馈）
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

                                    # 3. 精准捕捉被系统规则吞噬的动作及意图分析
                                    if has_ask and dominant_action != 'A':
                                        new_stats["IGNORED_PATH_A"] += 1
                                    if has_verify and dominant_action != 'C':
                                        new_stats["IGNORED_PATH_C"] += 1
                                    if has_finish and dominant_action != 'D':
                                        new_stats["IGNORED_PATH_D"] += 1

                                    if has_code and dominant_action is not None and dominant_action != 'B':
                                        if action_sum == 2:
                                            # 只有两种冲突时，比较位置来精确判断意图
                                            dom_pos = {'D': pos_finish, 'A': pos_ask, 'C': pos_verify}[dominant_action]
                                            if pos_code < dom_pos:
                                                # 先改代码，再触发主导动作 -> 偷步长
                                                new_stats["STEAL_STEP_COUNT"] += 1
                                            else:
                                                # 触发主导动作后，仍改代码 -> 盲写
                                                new_stats["BLIND_EXECUTION_COUNT"] += 1
                                        else:
                                            # 存在两种以上冲突（即 3 或 4），情况复杂，单独统计一般丢失
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

                            # 更新字典
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
                                results_dict[task_id]["termination_type"] = "主动提交 (Submit)"
                            elif len(action_sequence) >= 15:
                                is_loop = False
                                if len(action_sequence) >= 4:
                                    last_4_actions = action_sequence[-4:]
                                    if all(act in ['B', 'C', 'ERR'] for act in last_4_actions):
                                        is_loop = True
                                results_dict[task_id][
                                    "termination_type"] = "死循环 (Infinite Loop)" if is_loop else "用尽次数 (Max Turns)"
                            elif final_state in ["CRASHED", "ERROR"]:
                                results_dict[task_id]["termination_type"] = "异常崩溃 (Crashed/Error)"
                            else:
                                is_loop = False
                                if len(action_sequence) >= 4:
                                    last_4_actions = action_sequence[-4:]
                                    if all(act in ['B', 'C', 'ERR'] for act in last_4_actions):
                                        is_loop = True
                                results_dict[task_id][
                                    "termination_type"] = "死循环 (Infinite Loop)" if is_loop else "意外中断 (Interrupted)"

        except json.JSONDecodeError:
            pass
        except Exception as e:
            pass

    if fixed_count > 0:
        print(f"\n成功执行自动化修复！共精准覆写纠正了 {fixed_count} 个任务文件的头部记录，轨迹原数据均安全保留。")

    df = pd.DataFrame(list(results_dict.values()))
    if df.empty:
        print("未找到任何有效数据！")
        return

    normal_df = df[df['is_abnormal'] == False]
    abnormal_df = df[df['is_abnormal'] == True]

    print("\n" + "=" * 65)
    print(f" PEBench [{root_path.name}] 全局批次统计报告")
    print("=" * 65)
    print(f"总计聚合任务数: {len(df)} | 正常: {len(normal_df)} | 异常: {len(abnormal_df)}")

    # ---------------------------------------------------------
    # 核心发现 1：按角色身份 (Role) 分组对比
    # ---------------------------------------------------------
    print("\n" + "=" * 65)
    print("[核心发现 1] 按角色身份 (Role) 分组对比")
    print("=" * 65)

    roles = sorted(df['role'].unique())
    for r in roles:
        r_df = df[df['role'] == r]
        print(f"\n角色身份: 【{r}】 (共测试 {len(r_df)} 个任务)")

        avg_tcr = r_df['tcr_score'].mean()
        avg_tcr_no_hallu = r_df['tcr_score_no_hallu'].mean()

        hallu_mean = r_df['hallucination_triggered'].mean()
        hallu_rate_str = f"{hallu_mean * 100:.1f}%" if pd.notna(hallu_mean) else "N/A (无打分数据)"

        print(f"   平均 TCR 得分: {avg_tcr:.4f} (剔除幻觉纯度: {avg_tcr_no_hallu:.4f})")
        print(f"   幻觉触发率: {hallu_rate_str}")

        status_str = " | ".join([f"{k}: {v}个" for k, v in r_df['final_state'].value_counts().items()])
        print(f"   状态分布: {status_str}")

        term_counts = r_df['termination_type'].value_counts()
        term_str = " | ".join([f"{k}: {v}个" for k, v in term_counts.items()])
        print(f"   终止原因: {term_str}")

        print(f"   平均动作步数: {r_df['total_steps'].mean():.1f} 步")

        avg_a = r_df['path_a_count'].mean()
        avg_b = r_df['path_b_count'].mean()
        avg_c = r_df['path_c_count'].mean()
        avg_d = r_df['path_d_count'].mean()
        print(
            f"   实际执行 PATH: [A 澄清]: {avg_a:.1f} | [B 编码(已落盘)]: {avg_b:.1f} | [C 验证]: {avg_c:.1f} | [D 提交]: {avg_d:.1f}")

        avg_steal = r_df['steal_step_count'].mean()
        avg_blind = r_df['blind_execution_count'].mean()
        avg_ig_b_multi = r_df['ignored_b_count'].mean()
        print(
            f"   代码冲突(仅B): [2Path偷步]: {avg_steal:.2f} | [2Path盲写]: {avg_blind:.2f} | [多Path复杂丢失]: {avg_ig_b_multi:.2f}")

        c2, c3, c4 = r_df['conflict_2_paths'].sum(), r_df['conflict_3_paths'].sum(), r_df['conflict_4_paths'].sum()
        print(f"   冲突多样性: [2-Path]: {c2} 次 | [3-Path]: {c3} 次 | [4-Path]: {c4} 次")

        avg_ig_a = r_df['ignored_a_count'].mean()
        avg_ig_c = r_df['ignored_c_count'].mean()
        avg_ig_d = r_df['ignored_d_count'].mean()
        print(
            f"   其他冲突未遂: [A 澄清(未发)]: {avg_ig_a:.2f} | [C 验证(无图)]: {avg_ig_c:.2f} | [D 提交(未交)]: {avg_ig_d:.2f}")

        avg_violating = r_df['violating_chitchat_count'].mean()
        avg_overlap = r_df['overlap_actions_count'].mean()
        print(f"   平均格式违规 (纯文本): {avg_violating:.2f} 次 | 平均抢跑违规 (多Path): {avg_overlap:.2f} 次")

        avg_before_ask = r_df['ask_before_code_count'].mean()
        avg_after_ask = r_df['ask_after_code_count'].mean()
        print(f"   澄清时机拆解: (编码前: {avg_before_ask:.2f} 次 | 编码后: {avg_after_ask:.2f} 次)")

    # ---------------------------------------------------------
    # 核心发现 2：按任务难度 (Difficulty) 分组对比
    # ---------------------------------------------------------
    print("\n" + "=" * 65)
    print("[核心发现 2] 按任务难度 (Difficulty) 分组对比")
    print("=" * 65)

    difficulties = sorted(df['difficulty'].unique(), key=lambda x: DIFFICULTY_ORDER.get(x, 99))

    for d in difficulties:
        d_df = df[df['difficulty'] == d]
        print(f"\n任务难度: 【{d}】 (共 {len(d_df)} 个任务)")

        if d == "Unknown" and len(d_df) > 0:
            print("   注意：以下任务未在提供的数据集中找到难度标签，被归为 Unknown:")
            for _, row in d_df.iterrows():
                print(f"      - {row['task_id']}")

        avg_tcr = d_df['tcr_score'].mean()
        avg_tcr_no_hallu = d_df['tcr_score_no_hallu'].mean()

        hallu_mean = d_df['hallucination_triggered'].mean()
        hallu_rate_str = f"{hallu_mean * 100:.1f}%" if pd.notna(hallu_mean) else "N/A (无打分数据)"

        print(f"   平均 TCR 得分: {avg_tcr:.4f} (剔除幻觉纯度: {avg_tcr_no_hallu:.4f})")
        print(f"   幻觉触发率: {hallu_rate_str}")

        status_str = " | ".join([f"{k}: {v}个" for k, v in d_df['final_state'].value_counts().items()])
        print(f"   状态分布: {status_str}")

        term_counts = d_df['termination_type'].value_counts()
        term_str = " | ".join([f"{k}: {v}个" for k, v in term_counts.items()])
        print(f"    终止原因: {term_str}")

        print(f"    平均动作步数: {d_df['total_steps'].mean():.1f} 步")

        avg_a = d_df['path_a_count'].mean()
        avg_b = d_df['path_b_count'].mean()
        avg_c = d_df['path_c_count'].mean()
        avg_d = d_df['path_d_count'].mean()
        print(
            f"    实际执行 PATH: [A 澄清]: {avg_a:.1f} | [B 编码(已落盘)]: {avg_b:.1f} | [C 验证]: {avg_c:.1f} | [D 提交]: {avg_d:.1f}")

        avg_steal = d_df['steal_step_count'].mean()
        avg_blind = d_df['blind_execution_count'].mean()
        avg_ig_b_multi = d_df['ignored_b_count'].mean()
        print(
            f"    代码冲突(仅B): [2Path偷步]: {avg_steal:.2f} | [2Path盲写]: {avg_blind:.2f} | [多Path复杂丢失]: {avg_ig_b_multi:.2f}")

        c2, c3, c4 = d_df['conflict_2_paths'].sum(), d_df['conflict_3_paths'].sum(), d_df['conflict_4_paths'].sum()
        print(f"    冲突多样性: [2-Path]: {c2} 次 | [3-Path]: {c3} 次 | [4-Path]: {c4} 次")

        avg_ig_a = d_df['ignored_a_count'].mean()
        avg_ig_c = d_df['ignored_c_count'].mean()
        avg_ig_d = d_df['ignored_d_count'].mean()
        print(
            f"    其他冲突未遂: [A 澄清(未发)]: {avg_ig_a:.2f} | [C 验证(无图)]: {avg_ig_c:.2f} | [D 提交(未交)]: {avg_ig_d:.2f}")

        avg_violating = d_df['violating_chitchat_count'].mean()
        avg_overlap = d_df['overlap_actions_count'].mean()
        print(f"    平均格式违规 (纯文本): {avg_violating:.2f} 次 | 平均抢跑违规 (多Path): {avg_overlap:.2f} 次")

        avg_before_ask = d_df['ask_before_code_count'].mean()
        avg_after_ask = d_df['ask_after_code_count'].mean()
        print(f"    澄清时机拆解: (编码前: {avg_before_ask:.2f} 次 | 编码后: {avg_after_ask:.2f} 次)")

    # ---------------------------------------------------------
    # 新增专项分析：主动提交 vs 非主动提交
    # ---------------------------------------------------------
    print("\n" + "=" * 65)
    print(" [专项分析] 主动提交 (Submit) vs 非主动提交 质量对比")
    print("=" * 65)

    submit_df = df[df['termination_type'] == '主动提交 (Submit)']
    non_submit_df = df[df['termination_type'] != '主动提交 (Submit)']

    print(f"【主动提交 (Submit)】 (共 {len(submit_df)} 个任务)")
    if not submit_df.empty:
        print(
            f"   平均 TCR 得分: {submit_df['tcr_score'].mean():.4f} (剔除幻觉纯度: {submit_df['tcr_score_no_hallu'].mean():.4f})")
        submit_hallu = submit_df['hallucination_triggered'].mean()
        print(f"   幻觉触发率: {submit_hallu * 100:.1f}%" if pd.notna(submit_hallu) else "   幻觉触发率: N/A")
    else:
        print("   (无数据)")

    print(f"\n【非主动提交 (其它被动终止)】 (共 {len(non_submit_df)} 个任务)")
    if not non_submit_df.empty:
        print(
            f"   平均 TCR 得分: {non_submit_df['tcr_score'].mean():.4f} (剔除幻觉纯度: {non_submit_df['tcr_score_no_hallu'].mean():.4f})")
        non_submit_hallu = non_submit_df['hallucination_triggered'].mean()
        print(f"   幻觉触发率: {non_submit_hallu * 100:.1f}%" if pd.notna(non_submit_hallu) else "   幻觉触发率: N/A")
    else:
        print("   (无数据)")

    error_df = df[df['final_state'] == 'ERROR']
    crashed_df = df[df['final_state'] == 'CRASHED']

    if not error_df.empty or not crashed_df.empty:
        print("\n" + "=" * 65)
        print("异常溯源：发现以下执行崩溃或打分失败的任务，请重点排查！")
        print("=" * 65)

        if not error_df.empty:
            print("\n【ERROR 状态】(打分系统或浏览器崩溃):")
            for _, row in error_df.iterrows():
                print(f"   - Task ID: {row['task_id']:<18} | Role: {row['role']:<6} | Diff: {row['difficulty']:<6}")

        if not crashed_df.empty:
            print("\n【CRASHED 状态】(模型写出毒代码死循环或自行崩溃，未进入打分):")
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
                        print(f"无法完整复制 workspace {task_id}: {e}")

            print(f"\n已复制 {copied_count} 个异常任务 logs 至: {error_logs_dir}")
            print(f"已复制对应 workspaces 至: {error_workspaces_dir}")

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
                print(f"\n已提取 {extracted_count} 条异常任务 (ERROR+CRASHED) 至: {retest_path}")
            except Exception as e:
                print(f"提取异常任务时出错: {e}")
        elif abnormal_ids and not dataset_paths:
            try:
                with open(retest_path, 'w', encoding='utf-8') as out_f:
                    for t_id in sorted(abnormal_ids):
                        out_f.write(json.dumps({"id": t_id}, ensure_ascii=False) + '\n')
                print(f"\n已保存 {len(abnormal_ids)} 条异常任务 ID 至: {retest_path}")
            except Exception as e:
                print(f"保存异常任务 ID 时出错: {e}")

    if overlap_tasks:
        print("\n" + "=" * 65)
        print(f"抢跑违规溯源：定位到 {len(overlap_tasks)} 个存在“一气呵成(多Path重叠输出)”的任务！")
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
        print(f"已将这 {copied_overlap} 个违规任务的日志隔离复制至新文件夹: {overlap_output_dir}")

    print("\n" + "=" * 65)
    print(" [深度剖析] CRASHED 状态与行为的关联分析")
    print("=" * 65)

    if not crashed_df.empty:
        print("\n1. CRASHED 任务的实际底层死法剖析：")
        print("   (解释：部分 CRASHED 可能是因为陷入死循环被迫中止，有些则是代码致命报错)")
        crashed_term_counts = crashed_df['termination_type'].value_counts()
        for term, count in crashed_term_counts.items():
            pct = (count / len(crashed_df)) * 100
            print(f"   - 【{term}】: 导致了 {count} 次 CRASHED (占比 {pct:.1f}%)")

        print("\n2. 提问行为 (PATH_A) 对 CRASHED (崩溃) 比例的深度影响分析：")

        def print_crashed_stats(sub_df, label):
            if len(sub_df) > 0:
                c_count = len(sub_df[sub_df['final_state'] == 'CRASHED'])
                rate = (c_count / len(sub_df)) * 100
                print(f"      {label}: 共 {len(sub_df):>3} 个任务，发生 CRASHED {c_count:>2} 次 (崩溃率: {rate:>4.1f}%)")
            else:
                print(f"      {label}: 暂无相关任务")

        print("\n   【全局维度：是否与用户产生过交互 (Overall Ask)】")
        asked_any_df = df[df['path_a_count'] > 0]
        no_ask_any_df = df[df['path_a_count'] == 0]

        print_crashed_stats(asked_any_df, "【有提问行为 (Overall PATH_A > 0)】")
        print_crashed_stats(no_ask_any_df, "【无提问行为 (Overall PATH_A = 0)】")

        print("\n   【阶段维度 1：编码前是否主动澄清 (Pre-Code Ask)】")
        asked_before_df = df[df['ask_before_code_count'] > 0]
        no_ask_before_df = df[df['ask_before_code_count'] == 0]

        print_crashed_stats(asked_before_df, "【编码前提问 (Pre-Code PATH_A > 0)】")
        print_crashed_stats(no_ask_before_df, "【编码前未提问 (Pre-Code PATH_A = 0)】")

        print("\n   【阶段维度 2：编码后是否中途打断 (Post-Code Ask)】")
        asked_after_df = df[df['ask_after_code_count'] > 0]
        no_ask_after_df = df[df['ask_after_code_count'] == 0]

        print_crashed_stats(asked_after_df, "【编码后提问 (Post-Code PATH_A > 0)】")
        print_crashed_stats(no_ask_after_df, "【编码后未提问 (Post-Code PATH_A = 0)】")

    else:
        print("\n当前批次没有出现任何 CRASHED 的任务，模型运行稳定！")

    print("\n" + "=" * 65)
    print(" [交叉分析] 难度 × 角色 纯净TCR得分矩阵 (Pivot Table)")
    print("=" * 65)
    if "Unknown" not in difficulties or len(difficulties) > 1:
        pivot_tcr = pd.pivot_table(df, values='tcr_score_no_hallu', index='difficulty', columns='role', aggfunc='mean',
                                   fill_value=0)
        valid_indices = [d for d in ["Easy", "Middle", "Hard"] if d in pivot_tcr.index]
        if valid_indices:
            pivot_tcr = pivot_tcr.reindex(valid_indices)
        print(pivot_tcr.round(4).to_string())
    else:
        print("提示：当前所有任务难度均为 'Unknown'。")

    print("\n" + "=" * 65)
    print(" 宏观大盘数据 (所有角色汇总)")
    print("=" * 65)
    if not normal_df.empty:
        print(f" [1] 全局平均 TCR (综合): {normal_df['tcr_score'].mean():.4f}")
        print(f" [2] 全局平均 TCR (剔除幻觉): {normal_df['tcr_score_no_hallu'].mean():.4f}")

        global_hallu_mean = normal_df['hallucination_triggered'].mean()
        global_hallu_rate_str = f"{global_hallu_mean * 100:.1f}%" if pd.notna(global_hallu_mean) else "N/A (无打分数据)"
        print(f" [3] 全局独立幻觉率: {global_hallu_rate_str}")

        print(
            f" [4] 全局实际执行 PATH: [A 澄清]: {normal_df['path_a_count'].mean():.2f} | [B 编码]: {normal_df['path_b_count'].mean():.2f} | [C 验证]: {normal_df['path_c_count'].mean():.2f} | [D 提交]: {normal_df['path_d_count'].mean():.2f}")

        print(
            f" [5] 全局代码冲突(仅B): [2Path偷步]: {normal_df['steal_step_count'].mean():.2f} | [2Path盲写]: {normal_df['blind_execution_count'].mean():.2f} | [多Path复杂丢失]: {normal_df['ignored_b_count'].mean():.2f}")

        print(
            f" [6] 全局冲突多样性: [2-Path]: {normal_df['conflict_2_paths'].sum()} 次 | [3-Path]: {normal_df['conflict_3_paths'].sum()} 次 | [4-Path]: {normal_df['conflict_4_paths'].sum()} 次")

        print(
            f" [7] 全局其他冲突未遂: [A 澄清(未发)]: {normal_df['ignored_a_count'].mean():.2f} | [C 验证(无图)]: {normal_df['ignored_c_count'].mean():.2f} | [D 提交(未交)]: {normal_df['ignored_d_count'].mean():.2f}")

        global_violating_mean = normal_df['violating_chitchat_count'].mean()
        global_overlap_mean = normal_df['overlap_actions_count'].mean()
        print(f" [8] 全局平均格式违规: {global_violating_mean:.2f} 次 | 全局平均抢跑违规: {global_overlap_mean:.2f} 次")

        output_csv = os.path.join(str(root_path.parent), f"{root_path.name}_summary_with_roles.csv")
        df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"\n明细数据已导出: {output_csv}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze InteractWeb-Bench experiment results.")
    parser.add_argument("--dir", type=str,
                        default="/home/hhr/home/experiment_results/gpt-4.1/logs",
                        help="Path to the logs directory")
    parser.add_argument("--data", type=str, default="/home/hhr/home/data",
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
        print(f"找不到日志目录: {args.dir}")
    else:
        analyze_batch_trajectories(args.dir, dataset_paths=dataset_files or None)