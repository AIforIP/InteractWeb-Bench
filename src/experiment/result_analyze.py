import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 设置中文字体
# plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
# plt.rcParams['axes.unicode_minus'] = False

# 强制难度的逻辑排序权重，保证图表和打印顺序为 Easy -> Middle -> Hard
DIFFICULTY_ORDER = {"Easy": 1, "Middle": 2, "Hard": 3, "Unknown": 99}


def load_difficulty_map(dataset_paths):
    """
    🌟 提取并合并所有 Task ID 到难度的映射 (增加防覆盖严格匹配逻辑)
    """
    diff_map = {}
    if not dataset_paths:
        return diff_map

    if isinstance(dataset_paths, str):
        dataset_paths = [dataset_paths]

    for path in dataset_paths:
        if not os.path.exists(path):
            print(f"⚠️ 找不到数据集文件: {path}，将跳过。")
            continue

        try:
            with open(path, 'r', encoding='utf-8') as f:
                if str(path).endswith('.jsonl'):
                    for line in f:
                        if not line.strip(): continue
                        data = json.loads(line)
                        task_id = str(data.get("original_id") or data.get("id") or data.get("task_id"))
                        # 取出难度
                        diff = data.get("difficulty") or data.get("level") or data.get("task_level")

                        if task_id and task_id != "None":
                            base_id = task_id.split('_')[0] if '_' in task_id else task_id

                            # 💡 核心修复：如果找到了有效的难度标签，才写入；坚决防止被后续的空数据覆盖成 Unknown
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
            print(f"❌ 解析数据集 {path} 失败: {e}")

    return diff_map


def analyze_batch_trajectories(root_dir, dataset_paths=None):
    results_dict = {}
    root_path = Path(root_dir)

    difficulty_map = load_difficulty_map(dataset_paths)

    print(f"🔍 正在递归扫描日志目录: {root_path.absolute()}")

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
                "ask_user_count": 0,
                "ask_before_code_count": 0,
                "ask_after_code_count": 0,
                "screenshot_count": 0,
                "violating_chitchat_count": 0,
                "is_abnormal": True
            }

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                if filepath.suffix == '.jsonl':
                    pass
                else:
                    data = json.load(f)

                    if isinstance(data, list):
                        pass

                    elif isinstance(data, dict):
                        if results_dict[task_id]["difficulty"] == "Unknown":
                            diff = data.get("difficulty") or data.get("task_level") or data.get("level")
                            if diff:
                                results_dict[task_id]["difficulty"] = str(diff).capitalize()

                        stats = data.get("path_distribution_stats", {})
                        if stats:
                            results_dict[task_id]["ask_user_count"] = max(results_dict[task_id]["ask_user_count"],
                                                                          stats.get("PATH_A_CLARIFY", 0))
                            results_dict[task_id]["screenshot_count"] = max(results_dict[task_id]["screenshot_count"],
                                                                            stats.get("PATH_C_VERIFY", 0))
                            results_dict[task_id]["violating_chitchat_count"] = max(
                                results_dict[task_id]["violating_chitchat_count"],
                                stats.get("VIOLATING_CHITCHAT_COUNT", 0))

                        trajectory = data.get("trajectory", [])
                        if isinstance(trajectory, list) and len(trajectory) > 0:
                            first_code_turn = -1
                            ask_before = 0
                            ask_after = 0
                            has_submit = False
                            action_sequence = []

                            for turn in trajectory:
                                if turn.get("role") == "assistant":
                                    content = turn.get("content", "")
                                    turn_num = turn.get("turn", -1)

                                    has_ask = '<boltAction type="ask_user"' in content
                                    has_code = '<boltAction type="file"' in content or '<boltAction type="shell"' in content or '<boltAction type="start"' in content
                                    has_verify = '<boltAction type="screenshot_validated"' in content
                                    has_finish = '<boltAction type="finish"' in content

                                    if has_finish:
                                        has_submit = True
                                        action_sequence.append('FINISH')
                                    if has_ask:
                                        action_sequence.append('A')
                                    if has_code:
                                        action_sequence.append('B')
                                    if has_verify:
                                        action_sequence.append('C')

                                    if has_code and first_code_turn == -1:
                                        first_code_turn = turn_num

                                    if has_ask:
                                        if first_code_turn == -1 or turn_num < first_code_turn:
                                            ask_before += 1
                                        else:
                                            ask_after += 1

                            results_dict[task_id]["total_steps"] = len(action_sequence)
                            results_dict[task_id]["ask_before_code_count"] = max(
                                results_dict[task_id]["ask_before_code_count"], ask_before)
                            results_dict[task_id]["ask_after_code_count"] = max(
                                results_dict[task_id]["ask_after_code_count"], ask_after)

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
                                    if all(act in ['B', 'C'] for act in last_4_actions):
                                        is_loop = True
                                results_dict[task_id][
                                    "termination_type"] = "死循环 (Infinite Loop)" if is_loop else "用尽次数 (Max Turns)"
                            elif final_state in ["CRASHED", "ERROR"]:
                                results_dict[task_id]["termination_type"] = "异常崩溃 (Crashed/Error)"
                            else:
                                is_loop = False
                                if len(action_sequence) >= 4:
                                    last_4_actions = action_sequence[-4:]
                                    if all(act in ['B', 'C'] for act in last_4_actions):
                                        is_loop = True
                                results_dict[task_id][
                                    "termination_type"] = "死循环 (Infinite Loop)" if is_loop else "意外中断 (Interrupted)"

        except json.JSONDecodeError:
            pass
        except Exception as e:
            pass

    df = pd.DataFrame(list(results_dict.values()))
    if df.empty:
        print("未找到任何有效数据！")
        return

    normal_df = df[df['is_abnormal'] == False]
    abnormal_df = df[df['is_abnormal'] == True]

    print("\n" + "=" * 65)
    print(f" PEBench [{root_path.name}] 全局批次统计报告")
    print("=" * 65)
    print(f"总计聚合任务数: {len(df)} | 正常: {len(normal_df)} | ❌ 异常: {len(abnormal_df)}")

    # ---------------------------------------------------------
    # 核心发现 1：按角色身份 (Role) 分组对比
    # ---------------------------------------------------------
    print("\n" + "=" * 65)
    print("[核心发现 1] 按角色身份 (Role) 分组对比")
    print("=" * 65)

    roles = sorted(df['role'].unique())
    for r in roles:
        r_df = df[df['role'] == r]
        print(f"\n🔹 角色身份: 【{r}】 (共测试 {len(r_df)} 个任务)")

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

        avg_violating = r_df['violating_chitchat_count'].mean()
        print(f"   平均格式违规 (输出纯文本): {avg_violating:.2f} 次")

        avg_total_ask = r_df['ask_user_count'].mean()
        avg_before_ask = r_df['ask_before_code_count'].mean()
        avg_after_ask = r_df['ask_after_code_count'].mean()
        print(
            f"   平均反问 (PATH_A): {avg_total_ask:.2f} 次 (编码前: {avg_before_ask:.2f} 次 | 编码后: {avg_after_ask:.2f} 次)")

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
            print("   ⚠️ 注意：以下任务未在提供的数据集中找到难度标签，被归为 Unknown:")
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

        avg_violating = d_df['violating_chitchat_count'].mean()
        print(f"    平均格式违规 (输出纯文本): {avg_violating:.2f} 次")

        avg_total_ask = d_df['ask_user_count'].mean()
        avg_before_ask = d_df['ask_before_code_count'].mean()
        avg_after_ask = d_df['ask_after_code_count'].mean()
        print(
            f"   平均反问 (PATH_A): {avg_total_ask:.2f} 次 (编码前: {avg_before_ask:.2f} 次 | 编码后: {avg_after_ask:.2f} 次)")

    # ---------------------------------------------------------
    # 🌟 新增专项分析：主动提交 vs 非主动提交
    # ---------------------------------------------------------
    print("\n" + "=" * 65)
    print(" 📊 [专项分析] 主动提交 (Submit) vs 非主动提交 质量对比")
    print("=" * 65)

    submit_df = df[df['termination_type'] == '主动提交 (Submit)']
    non_submit_df = df[df['termination_type'] != '主动提交 (Submit)']

    print(f"🔹 【主动提交 (Submit)】 (共 {len(submit_df)} 个任务)")
    if not submit_df.empty:
        print(
            f"   平均 TCR 得分: {submit_df['tcr_score'].mean():.4f} (剔除幻觉纯度: {submit_df['tcr_score_no_hallu'].mean():.4f})")
        submit_hallu = submit_df['hallucination_triggered'].mean()
        print(f"   幻觉触发率: {submit_hallu * 100:.1f}%" if pd.notna(submit_hallu) else "   幻觉触发率: N/A")
    else:
        print("   (无数据)")

    print(f"\n🔹 【非主动提交 (其它被动终止)】 (共 {len(non_submit_df)} 个任务)")
    if not non_submit_df.empty:
        print(
            f"   平均 TCR 得分: {non_submit_df['tcr_score'].mean():.4f} (剔除幻觉纯度: {non_submit_df['tcr_score_no_hallu'].mean():.4f})")
        non_submit_hallu = non_submit_df['hallucination_triggered'].mean()
        print(f"   幻觉触发率: {non_submit_hallu * 100:.1f}%" if pd.notna(non_submit_hallu) else "   幻觉触发率: N/A")
    else:
        print("   (无数据)")

    # ---------------------------------------------------------
    # 🚨 异常溯源 (查找 ERROR / CRASHED 样例并提取) 🚨
    # ---------------------------------------------------------
    error_df = df[df['final_state'] == 'ERROR']
    crashed_df = df[df['final_state'] == 'CRASHED']

    if not error_df.empty or not crashed_df.empty:
        print("\n" + "!" * 65)
        print("🚨 异常溯源：发现以下执行崩溃或打分失败的任务，请重点排查！")
        print("!" * 65)

        if not error_df.empty:
            print("\n❌ 【ERROR 状态】(打分系统或浏览器崩溃):")
            for _, row in error_df.iterrows():
                print(f"   - Task ID: {row['task_id']:<18} | Role: {row['role']:<6} | Diff: {row['difficulty']:<6}")

            if dataset_paths:
                retest_path = r"E:\Agent_work\src\data_generation\re_test1.jsonl"
                print(f"\n🔄 正在精确提取真正的 ERROR 样例至: {retest_path}")
                error_ids = set(error_df['task_id'].tolist())
                extracted_count = 0

                try:
                    os.makedirs(os.path.dirname(retest_path), exist_ok=True)
                    with open(retest_path, 'w', encoding='utf-8') as out_f:
                        extracted_ids = set()
                        for d_path in dataset_paths:
                            if not os.path.exists(d_path): continue
                            with open(d_path, 'r', encoding='utf-8') as in_f:
                                if str(d_path).endswith('.jsonl'):
                                    for line in in_f:
                                        if not line.strip(): continue
                                        item_data = json.loads(line)
                                        t_id = str(item_data.get("id") or item_data.get("task_id"))
                                        if t_id in error_ids and t_id not in extracted_ids:
                                            out_f.write(line.strip() + '\n')
                                            extracted_ids.add(t_id)
                                            extracted_count += 1
                                elif str(d_path).endswith('.json'):
                                    json_data = json.load(in_f)
                                    if isinstance(json_data, list):
                                        for item in json_data:
                                            t_id = str(item.get("id") or item.get("task_id"))
                                            if t_id in error_ids and t_id not in extracted_ids:
                                                out_f.write(json.dumps(item, ensure_ascii=False) + '\n')
                                                extracted_ids.add(t_id)
                                                extracted_count += 1
                                    elif isinstance(json_data, dict):
                                        for k, item in json_data.items():
                                            t_id = str(k)
                                            if t_id in error_ids and t_id not in extracted_ids:
                                                out_f.write(json.dumps(item, ensure_ascii=False) + '\n')
                                                extracted_ids.add(t_id)
                                                extracted_count += 1
                    print(f"✅ 提取成功！共精准提取了 {extracted_count} 条 ERROR 样例。")
                except Exception as e:
                    print(f"⚠️ 提取 re_test1.jsonl 时发生错误: {e}")

        if not crashed_df.empty:
            print("\n⚠️ 【CRASHED 状态】(模型写出毒代码死循环或自行崩溃，未进入打分):")
            for _, row in crashed_df.iterrows():
                print(f"   - Task ID: {row['task_id']:<18} | Role: {row['role']:<6} | Diff: {row['difficulty']:<6}")

    # ---------------------------------------------------------
    # 🔬 新增：深度剖析 CRASHED 状态与行为的关联分析 🔬
    # ---------------------------------------------------------
    print("\n" + "=" * 65)
    print(" 🔬 [深度剖析] CRASHED 状态与行为的关联分析")
    print("=" * 65)

    if not crashed_df.empty:
        print("\n🔹 1. CRASHED 任务的实际底层死法剖析：")
        print("   (解释：部分 CRASHED 可能是因为陷入死循环被迫中止，有些则是代码致命报错)")
        crashed_term_counts = crashed_df['termination_type'].value_counts()
        for term, count in crashed_term_counts.items():
            pct = (count / len(crashed_df)) * 100
            print(f"   - 【{term}】: 导致了 {count} 次 CRASHED (占比 {pct:.1f}%)")

        print("\n🔹 2. 提问行为 (PATH_A) 对 CRASHED (崩溃) 比例的深度影响分析：")

        # 为了代码复用，定义一个小的内部打印函数
        def print_crashed_stats(sub_df, label):
            if len(sub_df) > 0:
                c_count = len(sub_df[sub_df['final_state'] == 'CRASHED'])
                rate = (c_count / len(sub_df)) * 100
                print(f"      {label}: 共 {len(sub_df):>3} 个任务，发生 CRASHED {c_count:>2} 次 (崩溃率: {rate:>4.1f}%)")
            else:
                print(f"      {label}: 暂无相关任务")

        # ------------------- 维度 A: 总体提问 vs 不提问 -------------------
        print("\n   👉 【全局维度：是否与用户产生过交互 (Overall Ask)】")
        asked_any_df = df[df['ask_user_count'] > 0]
        no_ask_any_df = df[df['ask_user_count'] == 0]

        print_crashed_stats(asked_any_df, "🗣️ 【有提问行为 (Overall PATH_A > 0)】")
        print_crashed_stats(no_ask_any_df, "🤐 【无提问行为 (Overall PATH_A = 0)】")

        # ------------------- 维度 B: 编码前提问 vs 未在编码前提问 -------------------
        print("\n   👉 【阶段维度 1：编码前是否主动澄清 (Pre-Code Ask)】")
        asked_before_df = df[df['ask_before_code_count'] > 0]
        no_ask_before_df = df[df['ask_before_code_count'] == 0]

        print_crashed_stats(asked_before_df, "🗣️ 【编码前提问 (Pre-Code PATH_A > 0)】")
        print_crashed_stats(no_ask_before_df, "🤐 【编码前未提问 (Pre-Code PATH_A = 0)】")

        # ------------------- 维度 C: 编码后提问 vs 未在编码后提问 -------------------
        print("\n   👉 【阶段维度 2：编码后是否中途打断 (Post-Code Ask)】")
        asked_after_df = df[df['ask_after_code_count'] > 0]
        no_ask_after_df = df[df['ask_after_code_count'] == 0]

        print_crashed_stats(asked_after_df, "🗣️ 【编码后提问 (Post-Code PATH_A > 0)】")
        print_crashed_stats(no_ask_after_df, "🤐 【编码后未提问 (Post-Code PATH_A = 0)】")

    else:
        print("\n🎉 太棒了，当前批次没有出现任何 CRASHED 的任务，模型运行极其稳定！")

    # ---------------------------------------------------------
    # 交叉分析 (难度 × 角色 TCR 矩阵)
    # ---------------------------------------------------------
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

        global_violating_mean = normal_df['violating_chitchat_count'].mean()
        print(f" [4] 全局平均格式违规: {global_violating_mean:.2f} 次")

    output_csv = f"{root_path.name}_summary_with_roles.csv"
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\n💾 明细数据已导出: {output_csv}")

    # # --- 可视化图表 ---
    # try:
    #     plt.figure(figsize=(20, 16))

    #     plt.subplot(3, 3, 1)
    #     colors_state = {'PASS': '#4CAF50', 'SUCCESS': '#4CAF50', 'FAIL': '#FFC107', 'CRASHED': '#F44336',
    #                     'ERROR': '#E91E63'}
    #     state_counts = df['final_state'].value_counts()
    #     plot_colors_state = [colors_state.get(str(x).upper(), '#9E9E9E') for x in state_counts.index]
    #     state_counts.plot(kind='pie', autopct='%1.1f%%', colors=plot_colors_state, startangle=90)
    #     plt.title('Final State Distribution', fontweight='bold')
    #     plt.ylabel('')

    #     plt.subplot(3, 3, 2)
    #     tcr_data = df.groupby('role')[['tcr_score', 'tcr_score_no_hallu']].mean()
    #     tcr_data.plot(kind='bar', ax=plt.gca(), color=['#00BCD4', '#8BC34A'], edgecolor='black')
    #     plt.title('TCR Score by Persona Role', fontweight='bold')
    #     plt.xlabel('Persona Role')
    #     plt.ylabel('Score')
    #     plt.legend(['With Hallu Penalty', 'Pure Task TCR'])
    #     plt.xticks(rotation=0)

    #     plt.subplot(3, 3, 3)
    #     hallu_data = df.groupby('role')['hallucination_triggered'].mean().dropna() * 100
    #     if not hallu_data.empty:
    #         hallu_data.plot(kind='bar', color='#E91E63', edgecolor='black')
    #     plt.title('Hallucination Rate (%) by Persona Role', fontweight='bold')
    #     plt.xlabel('Persona Role')
    #     plt.ylabel('Rate (%)')
    #     plt.xticks(rotation=0)

    #     plt.subplot(3, 3, 4)
    #     diff_tcr_data = df.groupby('difficulty')['tcr_score_no_hallu'].mean()
    #     valid_diff_indices = [d for d in ["Easy", "Middle", "Hard"] if d in diff_tcr_data.index]
    #     if valid_diff_indices:
    #         diff_tcr_data = diff_tcr_data.reindex(valid_diff_indices)
    #     diff_tcr_data.plot(kind='bar', color='#FF9800', edgecolor='black')
    #     plt.title('Pure TCR by Difficulty', fontweight='bold')
    #     plt.xlabel('Difficulty')
    #     plt.ylabel('Pure TCR Score')
    #     plt.xticks(rotation=0)

    #     plt.subplot(3, 3, 5)
    #     if "Unknown" not in difficulties or len(difficulties) > 1:
    #         pivot_tcr.plot(kind='bar', ax=plt.gca(), edgecolor='black')
    #         plt.title('Pure TCR: Difficulty x Persona', fontweight='bold')
    #         plt.xlabel('Difficulty')
    #         plt.ylabel('Pure TCR Score')
    #         plt.xticks(rotation=0)
    #         plt.legend(title='Persona Role')
    #     else:
    #         plt.text(0.5, 0.5, 'No Difficulty Data', horizontalalignment='center', verticalalignment='center',
    #                  fontsize=12)
    #         plt.title('Pure TCR: Difficulty x Persona', fontweight='bold')
    #         plt.axis('off')

    #     plt.subplot(3, 3, 6)
    #     colors_term = {'主动提交 (Submit)': '#4CAF50', '死循环 (Infinite Loop)': '#FF9800',
    #                    '用尽次数 (Max Turns)': '#2196F3', '异常崩溃 (Crashed/Error)': '#F44336',
    #                    '意外中断 (Interrupted)': '#9C27B0'}
    #     term_counts = df['termination_type'].value_counts()
    #     plot_colors_term = [colors_term.get(str(x), '#9E9E9E') for x in term_counts.index]
    #     term_counts.plot(kind='pie', autopct='%1.1f%%', colors=plot_colors_term, startangle=90)
    #     plt.title('Task Termination Reasons', fontweight='bold')
    #     plt.ylabel('')

    #     # ---------------------------------------------------------
    #     # 🌟 新增的第 7 个图表：主动提交 vs 非主动提交 分数对比
    #     # ---------------------------------------------------------
    #     plt.subplot(3, 3, 7)
    #     submit_compare_df = pd.DataFrame({
    #         'With Hallu Penalty': [submit_df['tcr_score'].mean() if not submit_df.empty else 0,
    #                                non_submit_df['tcr_score'].mean() if not non_submit_df.empty else 0],
    #         'Pure Task TCR': [submit_df['tcr_score_no_hallu'].mean() if not submit_df.empty else 0,
    #                           non_submit_df['tcr_score_no_hallu'].mean() if not non_submit_df.empty else 0]
    #     }, index=['Proactive Submit', 'Non-Submit'])

    #     submit_compare_df.plot(kind='bar', ax=plt.gca(), color=['#00BCD4', '#8BC34A'], edgecolor='black')
    #     plt.title('TCR Score: Submit vs Non-Submit', fontweight='bold')
    #     plt.ylabel('Average Score')
    #     plt.xticks(rotation=0)

    #     plt.tight_layout()
    #     img_name = f"{root_path.name}_roles_analysis_report.png"
    #     plt.savefig(img_name, dpi=300)
    #     print(f"\n 升级版 九宫格全景可视化图表已生成: {img_name}")
    # except Exception as e:
    #     print(f" 绘图时出现问题: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze InteractWeb-Bench experiment results.")
    parser.add_argument("--dir", type=str, required=True, help="Path to the logs directory")
    parser.add_argument("--data", type=str, default=None, help="Path to the dataset directory (optional)")
    args = parser.parse_args()

    # 自动从 --data 目录或 logs 目录同级的 data/ 目录寻找数据集
    data_dir = args.data or os.path.join(os.path.dirname(os.path.abspath(args.dir)), "..", "data")
    data_dir = os.path.abspath(data_dir)

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
        print(f"❌ 找不到日志目录: {args.dir}")
    else:
        analyze_batch_trajectories(args.dir, dataset_paths=dataset_files or None)