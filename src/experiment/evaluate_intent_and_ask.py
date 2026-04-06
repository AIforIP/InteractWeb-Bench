import os
import json
import re
import argparse
import requests
from collections import defaultdict
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

DEFAULT_VLM_URL = os.environ.get("OPENAILIKE_VLM_BASE_URL", "https://api.openai.com/v1")
DEFAULT_API_KEY = os.environ.get("OPENAILIKE_VLM_API_KEY", "")

# 定义 Persona 对应的系统预期行为，用于指导 Judge LLM
PERSONA_EXPECTATIONS = {
    "P-MIN": "Incomplete intent. The barrier is missing core information required to fulfill the ground truth.",
    "P-RAM": "Obscured intent. The barrier is excessive irrelevant noise that masks the ground truth.",
    "P-INT": "Abstract intent. The barrier is non-literal, sensory language that represents the ground truth.",
    "P-CON": "Conflicting intent. The barrier is internal logical paradoxes that contradict the ground truth."
}


def call_llm_judge(prompt: str, system_message: str, model: str = "gpt-4o") -> str:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEFAULT_API_KEY}"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"}
    }

    response = requests.post(f"{DEFAULT_VLM_URL}/chat/completions", headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def evaluate_chr(questions: list, oracle_slots: list, persona: str, model: str):
    """
    计算 CHR (Ask Quality) 及其判定细节。
    如果 Agent 没有提问，返回 None, None。
    """
    if not questions:
        return None, None

    if not oracle_slots:
        return 100.0, {}

    persona_guidance = PERSONA_EXPECTATIONS.get(persona, "")
    questions_text = "\n".join([f"- {q}" for q in questions])
    slots_text = "\n".join(
        [f"{i + 1}. {slot.get('expected_result', slot.get('Expected_Result', str(slot)))}" for i, slot in
         enumerate(oracle_slots)])

    system_message = "You are an expert requirement engineering judge. You must output ONLY a valid JSON object."
    prompt = f"""
    Evaluate if the Agent's questions successfully addressed the hidden constraints.

    [User Persona Context]
    Persona: {persona}
    Challenge: {persona_guidance}

    [Hidden Constraints (Oracle Slots)]
    {slots_text}

    [Agent's Questions]
    {questions_text}

    For EACH hidden constraint, determine if ANY of the Agent's questions successfully hit it (True/False). 
    Hitting a constraint means asking about the missing data, clarifying the contradiction, or confirming the noise/metaphor related to it.
    Output a JSON object where keys are constraint indices (e.g., "1", "2") and values are boolean (true/false).
    Example: {{"1": true, "2": false}}
    """

    try:
        response_str = call_llm_judge(prompt, system_message, model)
        evaluation = json.loads(response_str)

        # 严格校验：按要求遍历 1 到 total_slots，防止大模型伪造额外的 key
        hit_count = 0
        total_slots = len(oracle_slots)
        for i in range(total_slots):
            key = str(i + 1)
            if evaluation.get(key) is True:
                hit_count += 1

        score = (hit_count / total_slots) * 100.0 if total_slots > 0 else 0.0
        return score, evaluation  # 返回分数以及详细字典
    except Exception as e:
        tqdm.write(f"CHR Evaluation Error for {persona}: {e}")
        return 0.0, {"error": str(e)}


def evaluate_ias(reasoning_traces: str, ground_truth: str, persona: str, model: str):
    """
    计算 IAS (Intent Alignment Score) 及其详细推理理由。
    """
    if not reasoning_traces.strip():
        return 1, "No internal reasoning traces provided."

    persona_guidance = PERSONA_EXPECTATIONS.get(persona, "")
    system_message = "You are an expert software engineering evaluator for AI agents. Output ONLY a valid JSON object."
    prompt = f"""
        Compare the Agent's manifest understanding of the requirements against the User's Ground Truth intent. 
        The goal is to measure the final CLARITY and ACCURACY of the identified intent.

        [Task Context]
        Persona Category: {persona} (Barrier: {persona_guidance})

        [Ground Truth - The Target Intent]
        {ground_truth}

        [Agent's Output - Reasoning & Interactive Traces]
        {reasoning_traces}

        [IAS Scoring Rubric - 5-point Likert Scale]
        5 (Perfect Alignment): The Agent's final understood intent is identical to the Ground Truth. All obstacles (gaps, noise, metaphors, or conflicts) have been resolved.
        4 (Substantial Alignment): The Agent has captured the core intent and all major requirements. Only minor, non-functional details are missing or slightly off.
        3 (Moderate Alignment): The Agent understands the main task but fails to fully lock onto several key constraints. The intent is partially obscured by the Persona Barrier.
        2 (Weak Alignment): Significant misalignment. The Agent's understood intent deviates from the Ground Truth in fundamental ways (e.g., following noise or missing primary constraints).
        1 (No Alignment): Complete failure. The Agent's manifest understanding bears no meaningful resemblance to the Ground Truth.

        Output a JSON object: {{"score": int, "reasoning": "string"}}
        """

    try:
        response_str = call_llm_judge(prompt, system_message, model)
        evaluation = json.loads(response_str)
        score = evaluation.get("score", 1)
        reasoning = evaluation.get("reasoning", "No reasoning provided by judge.")
        return score, reasoning  # 返回分数以及具体打分理由
    except Exception as e:
        tqdm.write(f"IAS Evaluation Error for {persona}: {e}")
        return 1, str(e)


def extract_reasoning_and_questions(history: list):
    """
    从真实的 Trajectory 格式中提取 <think> 标签和 <boltAction type="ask_user"> 标签。
    """
    questions = []
    reasoning_traces = []

    for msg in history:
        if msg.get("role") == "assistant":
            content = msg.get("content", "")

            # 1. 提取实际提问
            ask_actions = re.findall(r'<boltAction\s+type="ask_user">(.*?)</boltAction>', content, re.DOTALL)
            questions.extend([q.strip() for q in ask_actions if q.strip()])

            # 2. 提取思维链
            thinks = re.findall(r'<think>(.*?)</think>', content, re.DOTALL)
            if thinks:
                reasoning_traces.extend([t.strip() for t in thinks if t.strip()])
            else:
                # 兼容未输出 <think> 但输出了规划文字的情况 (剔除代码工件)
                clean_content = re.sub(r'<boltArtifact.*?</boltArtifact>', '', content, flags=re.DOTALL)
                if clean_content.strip():
                    reasoning_traces.append(clean_content.strip())

    return questions, "\n---\n".join(reasoning_traces)


def calculate_loc(history_json_path: str) -> int:
    """
    直接读取 history.json，通过最终的虚拟文件系统 (workspace_files 或 workspace_snapshots)
    准确统计所有代码文件的最终总行数 (LOC)。
    """
    if not os.path.exists(history_json_path):
        return 0

    try:
        with open(history_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 优先读取顶级字段 'workspace_files' (这代表最终退出时的文件状态)
        final_files = data.get("workspace_files", {})

        # 容错：如果 workspace_files 为空，去 workspace_snapshots 里找最后一步
        if not final_files:
            snapshots = data.get("workspace_snapshots", {})
            if snapshots:
                # 提取类似 "step_5" 中的数字并找到最大的 step
                highest_step = max(snapshots.keys(), key=lambda x: int(x.split('_')[1]) if '_' in x else 0)
                final_files = snapshots.get(highest_step, {})

        total_loc = 0
        for file_path, file_content in final_files.items():
            # 确保内容不是空且为字符串格式
            if file_content and isinstance(file_content, str):
                lines = len(file_content.splitlines())
                total_loc += lines

        return total_loc
    except Exception as e:
        tqdm.write(f"LOC Calculation Error reading {history_json_path}: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/home/hhr/home/InteractWeb-Bench/data/all.jsonl")
    parser.add_argument("--log_dir", type=str, default="/home/hhr/home/experiment_results")
    parser.add_argument("--target_model", type=str, default="gpt-5-mini")
    parser.add_argument("--judge_model", type=str, default="gpt-5-mini")
    args = parser.parse_args()

    # 1. 加载 Metadata
    tasks_meta = {}
    if os.path.exists(args.data_path):
        with open(args.data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    task = json.loads(line)
                    tasks_meta[task["id"]] = task

    safe_model_name = args.target_model.replace("/", "-").replace(":", "-")
    base_log_path = os.path.join(args.log_dir, safe_model_name, "logs")

    if not os.path.exists(base_log_path):
        print(f"Log directory not found: {base_log_path}")
        return

    results = []
    task_dirs = [d for d in os.listdir(base_log_path) if os.path.isdir(os.path.join(base_log_path, d))]

    print(f"Starting Intent, Ask Quality & Generation (LOC) evaluation for {args.target_model}...")

    for task_id in tqdm(task_dirs):
        interaction_history_path = os.path.join(base_log_path, task_id, "interaction_history.json")
        full_history_path = os.path.join(base_log_path, task_id, "history.json")  # 用于精准读取代码快照

        if not os.path.exists(interaction_history_path):
            continue

        task_info = tasks_meta.get(task_id, {})
        ground_truth = task_info.get("ground_truth_instruction", "")
        persona = task_info.get("persona", "P-MIN")

        with open(interaction_history_path, "r", encoding="utf-8") as f:
            history_data = json.load(f)

        trajectory = history_data.get("trajectory", [])

        # 优先尝试从最后一个 turn 的 debug_info 中获取 oracle_slots
        oracle_slots = task_info.get("oracle_slots", [])
        if trajectory and "debug_info" in trajectory[-1]:
            logged_slots = trajectory[-1]["debug_info"].get("oracle_slots_used_for_grading", [])
            if logged_slots:
                oracle_slots = logged_slots

        # 计算意图和提问质量
        questions, reasoning = extract_reasoning_and_questions(trajectory)
        chr_score, chr_details = evaluate_chr(questions, oracle_slots, persona, args.judge_model)
        ias_score, ias_reasoning = evaluate_ias(reasoning, ground_truth, persona, args.judge_model)

        # 计算生成的代码长度 (Avg. LOC)
        loc = calculate_loc(full_history_path)

        results.append({
            "task_id": task_id,
            "persona": persona,
            "chr": chr_score,
            "chr_details": chr_details,
            "ias": ias_score,
            "ias_reasoning": ias_reasoning,
            "questions_asked": len(questions),
            "loc": loc  # 新增: 代码长度记录
        })

    if results:
        # 计算全局均值 (跳过 CHR 为 None 的情况)
        valid_chr_list = [r["chr"] for r in results if r["chr"] is not None]
        avg_chr = sum(valid_chr_list) / len(valid_chr_list) if valid_chr_list else 0.0
        avg_ias = sum(r["ias"] for r in results) / len(results)
        avg_loc = sum(r["loc"] for r in results) / len(results)

        # 按 Persona 分组统计
        persona_stats = defaultdict(lambda: {"chr_sum": 0, "chr_count": 0, "ias_sum": 0, "ias_count": 0, "loc_sum": 0})
        for r in results:
            p = r["persona"]
            if r["chr"] is not None:
                persona_stats[p]["chr_sum"] += r["chr"]
                persona_stats[p]["chr_count"] += 1
            persona_stats[p]["ias_sum"] += r["ias"]
            persona_stats[p]["ias_count"] += 1
            persona_stats[p]["loc_sum"] += r["loc"]

        aggregated_persona = {}
        for p, stats in persona_stats.items():
            aggregated_persona[p] = {
                "avg_chr": (stats["chr_sum"] / stats["chr_count"]) if stats["chr_count"] > 0 else "N/A",
                "avg_ias": stats["ias_sum"] / stats["ias_count"],
                "avg_loc": stats["loc_sum"] / stats["ias_count"],  # 计算单一人设下的平均代码行数
                "valid_ask_tasks": stats["chr_count"],
                "total_tasks": stats["ias_count"]
            }

        print("\n==== Evaluation Complete ====")
        print(f"Model: {args.target_model}")
        print(f"Total Tasks: {len(results)} | Tasks with valid Questions: {len(valid_chr_list)}")
        print(
            f"Overall Ask Quality (CHR): {avg_chr:.2f}% | Overall Intent (IAS): {avg_ias:.2f} / 5.0 | Overall Avg. LOC: {avg_loc:.1f}")
        for p, stats in aggregated_persona.items():
            chr_str = f"{stats['avg_chr']:.2f}%" if isinstance(stats['avg_chr'], float) else stats['avg_chr']
            print(
                f"  [{p}] CHR: {chr_str} ({stats['valid_ask_tasks']} asked) | IAS: {stats['avg_ias']:.2f} | LOC: {stats['avg_loc']:.1f}")

        summary_path = os.path.join(args.log_dir, safe_model_name, "intent_ask_evaluation.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({
                "aggregate_overall": {
                    "avg_chr": avg_chr,
                    "avg_ias": avg_ias,
                    "avg_loc": avg_loc,
                    "ask_rate": f"{(len(valid_chr_list) / len(results)) * 100:.1f}%"
                },
                "aggregate_by_persona": aggregated_persona,
                "details": results
            }, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()