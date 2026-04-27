import os
import json
import base64
import argparse
import requests
import re
from tqdm import tqdm
from dotenv import load_dotenv


load_dotenv()
DEFAULT_VLM_URL = os.environ.get("OPENAILIKE_VLM_BASE_URL", "")
DEFAULT_API_KEY = os.environ.get("OPENAILIKE_VLM_API_KEY", "")


def encode_image_to_base64(image_path: str) -> str:

    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def extract_json_from_text(text: str) -> dict:


    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        json_str = match.group(1)
    else:

        match = re.search(r"```\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:

            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                json_str = text[start:end + 1]
            else:
                json_str = "{}"

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:

        return {
            "has_visual_bug": False,
            "visual_layout_score": 3,
            "creative_alignment_score": 3,
            "overall_aesthetics_score": 3,
            "reasoning": "JSON Parsing Error. Raw output: " + text
        }


def call_vlm_api(image_base64: str, model_name: str, prompt: str) -> dict:

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEFAULT_API_KEY}"
    }


    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": "You are an expert web designer and visual aesthetics evaluator. You must output ONLY a valid JSON object."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                ]
            }
        ],
        "temperature": 0.0,
        "top_p": 0.0001,
        "seed": 42,
        "response_format": {"type": "json_object"}
    }

    response = requests.post(
        f"{DEFAULT_VLM_URL}/chat/completions",
        headers=headers,
        json=payload,
        timeout=120
    )
    response.raise_for_status()
    response_data = response.json()
    content = response_data["choices"][0]["message"]["content"]

    return extract_json_from_text(content)


def process_task(task_id, base_log_path, judge_model, eval_prompt):

    image_path = os.path.join(base_log_path, task_id, "visual_step_0.png")

    if not os.path.exists(image_path):
        return None

    try:
        base64_image = encode_image_to_base64(image_path)
        eval_data = call_vlm_api(base64_image, judge_model, eval_prompt)

        return {
            "task_id": task_id,
            "has_visual_bug": bool(eval_data.get("has_visual_bug", False)),
            "visual_layout": float(eval_data.get("visual_layout_score", 3)),
            "creative_alignment": float(eval_data.get("creative_alignment_score", 3)),
            "overall_aesthetics": float(eval_data.get("overall_aesthetics_score", 3)),
            "reasoning": eval_data.get("reasoning", "")
        }
    except Exception as e:
        return {"error": str(e), "task_id": task_id}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="/your_path/experiment_results",
                        help="Root directory for experiment results")

    parser.add_argument("--target_model", type=str, default="qwen3.5-397b-a17b",
                        help="Name of the agent model to be evaluated")

    parser.add_argument("--target_task", type=str, default="000070_P-RAM",
                        help="Task ID for re-evaluating a specific sample")
    parser.add_argument("--judge_model", type=str, default="gpt-5-mini", help="VLM API model name used for scoring")
    args = parser.parse_args()

    safe_model_name = args.target_model.replace("/", "-").replace(":", "-")
    base_log_path = os.path.join(args.log_dir, safe_model_name, "logs")
    summary_path = os.path.join(args.log_dir, safe_model_name, "api_aesthetic_evaluation.json")

    if not os.path.exists(base_log_path):
        print(f"[ERROR] Log directory not found: {base_log_path}")
        return

    eval_prompt = """
You are a highly critical expert web designer and visual aesthetics evaluator based on the "ArtiMuse" framework.
Evaluate the clean screenshot of the generated webpage (ignore any external testing markers or debug text if present).

1. Objective Defect (has_visual_bug) [Boolean]:
   Are there obvious UI rendering failures (text overlapping, container overflow, broken images, unstyled HTML, severely misaligned elements)?

2. Visual Layout (visual_layout_score) [Scale: 1-5]:
   [5] Flawless, pixel-perfect alignment, masterful contrast and spacing.
   [4] Good and functional, but uses standard/generic layouts. Minor spacing inconsistencies.
   [3] Mediocre. Visibly unbalanced, awkward whitespace, or clashing colors.
   [2] Poor structure. Elements feel randomly placed but usable.
   [1] Completely broken layout.

3. Creative Alignment (creative_alignment_score) [Scale: 1-5]:
   [5] Highly unique, artistic, and emotionally engaging. Looks like a top-tier premium theme.
   [4] Cohesive and on-theme, but slightly predictable or template-like.
   [3] Generic, boring, or lacks a distinct visual identity.
   [2] Confused theme. Colors and typography do not match the intended topic.
   [1] Zero creativity. Barebones default styling.

4. Overall Aesthetics (overall_aesthetics_score) [Scale: 1-5]:
   [5] Breathtaking overall visual impact.
   [4] Visually pleasing and professional.
   [3] Average, looks like a beginner's draft.
   [2] Unattractive, hard to look at.
   [1] Visually offensive or completely broken.

CRITICAL: You MUST output ONLY a valid JSON object. 
To enforce step-by-step thinking, you MUST output the "reasoning" key FIRST, before assigning any scores.
Match this exact structure:
{
  "reasoning": "<Be extremely critical. Explicitly mention why it didn't get a 5, justify any low scores, and analyze each of the 3 dimensions before outputting the scores below.>",
  "has_visual_bug": true or false,
  "visual_layout_score": <int 1-5>,
  "creative_alignment_score": <int 1-5>,
  "overall_aesthetics_score": <int 1-5>
}
""".strip()

    print(f"\n[INFO] Re-requesting evaluation for model {args.target_model} on specific task [{args.target_task}]...")


    existing_data = {"aggregate_metrics": {}, "trajectory_details": []}
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            existing_data = json.load(f)

    trajectory_details = existing_data.get("trajectory_details", [])


    res = process_task(args.target_task, base_log_path, args.judge_model, eval_prompt)

    if res is None:
        print(f"[ERROR] Screenshot visual_step_0.png for task {args.target_task} not found. Please check the path.")
        return
    if "error" in res:
        print(f"[ERROR] VLM evaluation failed for task {args.target_task}: {res['error']}")
        return

    print(f"\n Re-evaluation successful! New result:\n{json.dumps(res, indent=2, ensure_ascii=False)}")

    updated = False
    for i, item in enumerate(trajectory_details):
        if item.get("task_id") == args.target_task:
            trajectory_details[i] = res
            updated = True
            break

    if not updated:
        print(f"[INFO] Task {args.target_task} not found in history. Appending it as new data.")
        trajectory_details.append(res)
    else:
        print(f"[INFO] Successfully replaced the old score for {args.target_task} in the history.")


    if trajectory_details:
        valid_tasks_count = len(trajectory_details)
        bugs_count = sum(1 for r in trajectory_details if r["has_visual_bug"])

        vbr_percentage = (bugs_count / valid_tasks_count) * 100.0
        avg_vl = sum(r["visual_layout"] for r in trajectory_details) / valid_tasks_count
        avg_ca = sum(r["creative_alignment"] for r in trajectory_details) / valid_tasks_count
        avg_oa = sum(r["overall_aesthetics"] for r in trajectory_details) / valid_tasks_count

        total_las_score = avg_vl + avg_ca + avg_oa

        print("\n" + "=" * 55)
        print(f" Refreshed overall metrics for {args.target_model} (based on {valid_tasks_count} samples)")
        print("-" * 55)
        print(f" Objective Defect (VBR) : {vbr_percentage:.1f}%")
        print(f" Visual Layout (LAS-VL) : {avg_vl:.2f} / 5.0")
        print(f" Creative Alignment (CA) : {avg_ca:.2f} / 5.0")
        print(f" Overall Aesthetics (OA) : {avg_oa:.2f} / 5.0")
        print(f" Total Macro Score       : {total_las_score:.2f} / 15.0")
        print("=" * 55)


        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({
                "aggregate_metrics": {
                    "evaluated_samples": valid_tasks_count,
                    "vbr_percentage": vbr_percentage,
                    "avg_visual_layout": avg_vl,
                    "avg_creative_alignment": avg_ca,
                    "avg_overall_aesthetics": avg_oa,
                    "total_las_score": total_las_score
                },
                "trajectory_details": trajectory_details
            }, f, indent=2, ensure_ascii=False)

        print(f" Data has been successfully overwritten and saved to:\n  -> {summary_path}")


if __name__ == "__main__":
    main()