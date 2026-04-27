import os
import json
import re
import argparse
import requests
from collections import defaultdict
from tqdm import tqdm
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
try:
    from scipy.stats import spearmanr

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("[WARNING] scipy is not installed. Spearman correlation will be skipped. Run `pip install scipy`.")

load_dotenv()

DEFAULT_VLM_URL = os.environ.get("OPENAILIKE_VLM_BASE_URL", "https://api.openai.com/v1")
DEFAULT_API_KEY = os.environ.get("OPENAILIKE_VLM_API_KEY", "")

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
        "top_p": 0.0001,
        "seed": 42,
        "response_format": {"type": "json_object"}
    }

    response = requests.post(f"{DEFAULT_VLM_URL}/chat/completions", headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def evaluate_chr(questions: list, oracle_slots: list, persona: str, instruction: str, ground_truth: str, model: str):
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
    Evaluate if the Agent's questions successfully addressed the target constraints (Oracle Slots).

    [Initial User Instruction (What the agent actually received)]
    {instruction}

    [Task Context - The Complete Ground Truth Application]
    {ground_truth}

    [User Persona Context & Barrier]
    Persona: {persona}
    Barrier Description: {persona_guidance}

    [Target Constraints (Oracle Slots)]
    {slots_text}

    [Agent's Questions]
    {questions_text}

    BACKGROUND: The agent received an imperfect 'Initial User Instruction' (modified by the Persona Barrier). The 'Target Constraints' are the absolute truths derived from the 'Ground Truth'.
    TASK: For EACH Target Constraint, determine if ANY of the Agent's questions successfully hit it. 
    "Hitting" a constraint means the agent asked a question that helps resolve the barrier (e.g., asking for missing info, confirming noisy info, or resolving a conflict) in order to discover or clarify this specific constraint.

    CRITICAL: You MUST output a JSON object exactly matching this schema to enforce step-by-step reasoning:
    {{
      "evaluations": [
        {{
          "constraint_index": 1,
          "constraint_focus": "<briefly state what the constraint requires>",
          "matched_question": "<quote the EXACT question from the agent that addresses this, or write 'None' if missing>",
          "reasoning": "<explain how the agent's question navigates the Persona Barrier in the initial instruction to uncover or confirm this specific target constraint>",
          "hit": true
        }},
        ... (repeat for ALL constraints)
      ]
    }}
    """
    try:
        response_str = call_llm_judge(prompt, system_message, model)
        evaluation = json.loads(response_str)

        hit_count = 0
        total_slots = len(oracle_slots)

        eval_list = evaluation.get("evaluations", [])
        if isinstance(eval_list, list):
            eval_map = {str(item.get("constraint_index")): item.get("hit", False) for item in eval_list if
                        isinstance(item, dict)}
            for i in range(total_slots):
                is_hit = eval_map.get(str(i + 1), False)
                if is_hit is True or str(is_hit).lower() == "true":
                    hit_count += 1

        return (hit_count / total_slots) * 100.0 if total_slots > 0 else 0.0, evaluation
    except Exception as e:
        tqdm.write(f"CHR Evaluation Error for {persona}: {e}")
        return 0.0, {"error": str(e)}


def evaluate_ias(reasoning_traces: str, instruction: str, ground_truth: str, persona: str, model: str):
    if not reasoning_traces.strip():
        return 1, "No internal reasoning traces provided."

    persona_guidance = PERSONA_EXPECTATIONS.get(persona, "")
    system_message = "You are an expert software engineering evaluator for AI agents. Output ONLY a valid JSON object."

    prompt = f"""
        Compare the Agent's manifest understanding of the requirements against the User's Ground Truth intent. 

        [Task Context]
        Persona Category: {persona} (Barrier: {persona_guidance})

        [Initial User Instruction (What the agent actually received)]
        {instruction}

        [Ground Truth - The Target Intent]
        {ground_truth}

        [Agent's Output - Reasoning & Interactive Traces]
        {reasoning_traces}

        [IAS Scoring Rubric - 5-point Likert Scale]
        5: Perfect Alignment (Agent fully bypassed the barrier and aligned perfectly with the ground truth).
        4: Substantial Alignment.
        3: Moderate Alignment.
        2: Weak Alignment.
        1: No Alignment (Agent was completely misled by the initial instruction and failed to find the ground truth).

        CRITICAL: You MUST output a JSON object exactly matching this schema to enforce step-by-step reasoning:
        {{
          "identified_ground_truth": "<summarize user's true intent>",
          "identified_agent_intent": "<summarize what the agent thought the intent was, based on how it interpreted the Initial Instruction in its traces>",
          "reasoning": "<compare them step-by-step. Consider the Persona Barrier: did the agent overcome the flawed Initial Instruction to reach the Ground Truth?>",
          "score": 4
        }}
        """
    try:
        response_str = call_llm_judge(prompt, system_message, model)
        evaluation = json.loads(response_str)
        return int(evaluation.get("score", 1)), evaluation.get("reasoning", "No reasoning.")
    except Exception as e:
        tqdm.write(f"IAS Evaluation Error for {persona}: {e}")
        return 1, str(e)


def extract_reasoning_and_questions(history: list):
    questions = []
    reasoning_traces = []
    for msg in history:
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            ask_actions = re.findall(r'<boltAction\s+type="ask_user">(.*?)</boltAction>', content, re.DOTALL)
            questions.extend([q.strip() for q in ask_actions if q.strip()])
            thinks = re.findall(r'<think>(.*?)</think>', content, re.DOTALL)
            if thinks:
                reasoning_traces.extend([t.strip() for t in thinks if t.strip()])
            else:
                clean_content = re.sub(r'<boltArtifact.*?</boltArtifact>', '', content, flags=re.DOTALL)
                if clean_content.strip():
                    reasoning_traces.append(clean_content.strip())
    return questions, "\n---\n".join(reasoning_traces)


def calculate_loc(history_json_path: str) -> int:
    if not os.path.exists(history_json_path): return 0
    try:
        with open(history_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        final_files = data.get("workspace_files", {})
        if not final_files:
            snapshots = data.get("workspace_snapshots", {})
            if snapshots:
                highest_step = max(snapshots.keys(), key=lambda x: int(x.split('_')[1]) if '_' in x else 0)
                final_files = snapshots.get(highest_step, {})
        total_loc = 0
        for file_path, file_content in final_files.items():
            if file_content and isinstance(file_content, str):
                total_loc += len(file_content.splitlines())
        return total_loc
    except Exception:
        return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/your_path/InteractWeb-Bench/data/all.jsonl")
    parser.add_argument("--log_dir", type=str, default="/your_path/experiment_results")
    parser.add_argument("--target_model", type=str, default="qwen3.6-plus")
    parser.add_argument("--judge_model", type=str, default="gpt-5-mini")
    parser.add_argument("--workers", type=int, default=10, help="Number of concurrent workers for API calls")
    args = parser.parse_args()

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

    task_dirs = [d for d in os.listdir(base_log_path) if os.path.isdir(os.path.join(base_log_path, d))]

    print(f"[INFO] Starting Intent, Ask Quality, LOC & Hallucination evaluation for {args.target_model}...")
    print(f"[INFO] Running with {args.workers} concurrent workers...")

    def process_single_task(task_id):
        interaction_history_path = os.path.join(base_log_path, task_id, "interaction_history.json")
        full_history_path = os.path.join(base_log_path, task_id, "history.json")

        if not os.path.exists(interaction_history_path):
            return None

        task_info = tasks_meta.get(task_id, {})
        instruction = task_info.get("instruction", "")
        ground_truth = task_info.get("ground_truth_instruction", "")
        persona = task_info.get("persona", "P-MIN")

        with open(interaction_history_path, "r", encoding="utf-8") as f:
            history_data = json.load(f)

        trajectory = history_data.get("trajectory", [])
        oracle_slots = task_info.get("oracle_slots", [])

        hallu_triggered = None
        for item in reversed(trajectory):
            if isinstance(item, dict) and "debug_info" in item:
                debug_info = item["debug_info"]
                if not isinstance(debug_info, dict): continue

                logged_slots = debug_info.get("oracle_slots_used_for_grading", [])
                if logged_slots: oracle_slots = logged_slots

                eval_detail = debug_info.get("evaluation_detail", {})
                raw_metrics = eval_detail.get("raw_metrics", {})

                if raw_metrics and oracle_slots:
                    flag = False
                    details_list = raw_metrics.get("Details", []) if isinstance(raw_metrics, dict) else (
                        raw_metrics if isinstance(raw_metrics, list) else [])

                    for idx, slot in enumerate(oracle_slots):
                        passed = False
                        if details_list and idx < len(details_list):
                            detail_item = details_list[idx]
                            passed = bool(detail_item.get("passed", False)) if "passed" in detail_item else str(
                                detail_item.get("status", "")).lower() == "pass"
                        elif isinstance(raw_metrics, dict):
                            v = raw_metrics.get(str(idx)) or raw_metrics.get(f"Checklist ID [{idx}]")
                            if v: passed = str(v.get("status", "fail") if isinstance(v, dict) else v).lower() == "pass"

                        if slot.get("assertion_type", "").upper() == "NEGATIVE" and not passed:
                            flag = True
                    hallu_triggered = 1 if flag else 0
                    break

        questions, reasoning = extract_reasoning_and_questions(trajectory)

        chr_score, chr_details = evaluate_chr(
            questions=questions,
            oracle_slots=oracle_slots,
            persona=persona,
            instruction=instruction,
            ground_truth=ground_truth,
            model=args.judge_model
        )

        ias_score, ias_reasoning = evaluate_ias(
            reasoning_traces=reasoning,
            instruction=instruction,
            ground_truth=ground_truth,
            persona=persona,
            model=args.judge_model
        )

        loc = calculate_loc(full_history_path)

        return {
            "task_id": task_id,
            "persona": persona,
            "chr": chr_score,
            "chr_details": chr_details,
            "ias": ias_score,
            "ias_reasoning": ias_reasoning,
            "questions_asked": len(questions),
            "loc": loc,
            "hallu_triggered": hallu_triggered
        }

    results = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_task = {executor.submit(process_single_task, task_id): task_id for task_id in task_dirs}

        for future in tqdm(as_completed(future_to_task), total=len(task_dirs), desc="Evaluating"):
            try:
                res = future.result()
                if res is not None:
                    results.append(res)
            except Exception as exc:
                task_id = future_to_task[future]
                tqdm.write(f"[ERROR] Task {task_id} generated an exception: {exc}")

    if results:
        valid_chr_list = [r["chr"] for r in results if r["chr"] is not None]
        avg_chr = sum(valid_chr_list) / len(valid_chr_list) if valid_chr_list else 0.0
        avg_ias = sum(r["ias"] for r in results) / len(results)
        avg_loc = sum(r["loc"] for r in results) / len(results)

        valid_hallu = [r["hallu_triggered"] for r in results if r["hallu_triggered"] is not None]
        avg_hallu = (sum(valid_hallu) / len(valid_hallu) * 100) if valid_hallu else 0.0

        persona_stats = defaultdict(
            lambda: {"chr_sum": 0, "chr_count": 0, "ias_sum": 0, "ias_count": 0, "loc_sum": 0, "hal_sum": 0,
                     "hal_count": 0})
        for r in results:
            p = r["persona"]
            if r["chr"] is not None:
                persona_stats[p]["chr_sum"] += r["chr"]
                persona_stats[p]["chr_count"] += 1
            persona_stats[p]["ias_sum"] += r["ias"]
            persona_stats[p]["ias_count"] += 1
            persona_stats[p]["loc_sum"] += r["loc"]
            if r["hallu_triggered"] is not None:
                persona_stats[p]["hal_sum"] += r["hallu_triggered"]
                persona_stats[p]["hal_count"] += 1

        aggregated_persona = {}
        for p, stats in persona_stats.items():
            aggregated_persona[p] = {
                "avg_chr": (stats["chr_sum"] / stats["chr_count"]) if stats["chr_count"] > 0 else "N/A",
                "avg_ias": stats["ias_sum"] / stats["ias_count"],
                "avg_loc": stats["loc_sum"] / stats["ias_count"],
                "hallu_rate": (stats["hal_sum"] / stats["hal_count"] * 100) if stats["hal_count"] > 0 else "N/A",
                "valid_ask_tasks": stats["chr_count"],
                "total_tasks": stats["ias_count"]
            }

        spearman_result = {"correlation": None, "p_value": None, "status": "Not calculated"}
        if SCIPY_AVAILABLE and valid_hallu:
            paired_data = [(r["loc"], r["hallu_triggered"]) for r in results if r["hallu_triggered"] is not None]
            loc_array = [d[0] for d in paired_data]
            hal_array = [d[1] for d in paired_data]

            if len(loc_array) > 1:
                corr, p_val = spearmanr(loc_array, hal_array)
                spearman_result = {
                    "correlation": float(corr) if not pd.isna(corr) else None,
                    "p_value": float(p_val) if not pd.isna(p_val) else None,
                    "status": "Success"
                }

        print("\n==== Evaluation Complete ====")
        print(f"Model: {args.target_model}")
        print(
            f"Total Tasks: {len(results)} | Valid Ask: {len(valid_chr_list)} | Valid Eval (Hallu): {len(valid_hallu)}")
        print(
            f"Overall CHR: {avg_chr:.2f}% | IAS: {avg_ias:.2f} | LOC: {avg_loc:.1f} | Hallucination: {avg_hallu:.1f}%")

        for p, stats in aggregated_persona.items():
            chr_str = f"{stats['avg_chr']:.2f}%" if isinstance(stats['avg_chr'], float) else stats['avg_chr']
            hal_str = f"{stats['hallu_rate']:.1f}%" if isinstance(stats['hallu_rate'], float) else stats['hallu_rate']
            print(
                f"  [{p}] CHR: {chr_str} | IAS: {stats['avg_ias']:.2f} | LOC: {stats['avg_loc']:.1f} | Hallu: {hal_str}")

        if spearman_result["status"] == "Success":
            sig = "Significant" if spearman_result["p_value"] < 0.05 else "Not Significant"
            print("\n==== Spearman Correlation (LOC vs Hallucination) ====")
            print(f"Correlation (ρ): {spearman_result['correlation']:.4f}")
            print(f"P-value: {spearman_result['p_value']:.4e} ({sig})")

        summary_path = os.path.join(args.log_dir, safe_model_name, "intent_ask_evaluation.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({
                "aggregate_overall": {
                    "avg_chr": avg_chr,
                    "avg_ias": avg_ias,
                    "avg_loc": avg_loc,
                    "avg_hallu_rate": avg_hallu,
                    "ask_rate": f"{(len(valid_chr_list) / len(results)) * 100:.1f}%"
                },
                "spearman_correlation": spearman_result,
                "aggregate_by_persona": aggregated_persona,
                "details": results
            }, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()