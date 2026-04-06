import os
import json
import base64
import argparse
import requests
import re
from tqdm import tqdm
from dotenv import load_dotenv

# 加载环境变量中的 API 配置
load_dotenv()
DEFAULT_VLM_URL = os.environ.get("OPENAILIKE_VLM_BASE_URL", "")
DEFAULT_API_KEY = os.environ.get("OPENAILIKE_VLM_API_KEY", "")


def encode_image_to_base64(image_path: str) -> str:
    """将本地图片读取并转换为 Base64 字符串"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def extract_json_from_text(text: str) -> dict:
    """从 API 返回的文本中安全提取 JSON（兼容 markdown 代码块）"""
    # 优先匹配 ```json ... ``` 格式
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        json_str = match.group(1)
    else:
        # 再尝试匹配普通 ``` ... ``` 代码块
        match = re.search(r"```\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            # 尝试直接查找最外层大括号
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                json_str = text[start:end + 1]
            else:
                json_str = "{}"

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # 解析失败时的保底默认值
        return {
            "has_visual_bug": False,
            "visual_layout_score": 3,
            "creative_alignment_score": 3,
            "overall_aesthetics_score": 3,
            "reasoning": "JSON Parsing Error. Raw output: " + text
        }


def call_vlm_api(image_base64: str, model_name: str, prompt: str) -> dict:
    """通过 OpenAI 兼容接口调用视觉大模型进行打分"""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="/home/hhr/home/experiment_results", help="实验结果根目录")
    parser.add_argument("--target_model", type=str, default="gpt-5-mini", help="被评测的 Agent 模型名称")
    parser.add_argument("--judge_model", type=str, default="gpt-5-mini", help="用于打分的 VLM API 模型名称")
    args = parser.parse_args()

    safe_model_name = args.target_model.replace("/", "-").replace(":", "-")
    base_log_path = os.path.join(args.log_dir, safe_model_name, "logs")

    if not os.path.exists(base_log_path):
        print(f"[ERROR] Log directory not found: {base_log_path}")
        return

    task_dirs = [
        d for d in os.listdir(base_log_path)
        if os.path.isdir(os.path.join(base_log_path, d))
    ]

    # 修改点 1：替换为极度严苛的高区分度 Prompt
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

Output ONLY a valid JSON object matching this exact structure:
{
  "has_visual_bug": true or false,
  "visual_layout_score": <int 1-5>,
  "creative_alignment_score": <int 1-5>,
  "overall_aesthetics_score": <int 1-5>,
  "reasoning": "<Be extremely critical. Explicitly mention why it didn't get a higher score, and justify if you gave a low score.>"
}
""".strip()

    results = []
    print(f"\n[INFO] Starting API-based Aesthetic Evaluation for model: {args.target_model}")
    print(f"[INFO] Using Judge Model via API: {args.judge_model}")

    for task_id in tqdm(task_dirs, desc="Evaluating Images"):
        # 修改点 2：将 target image 替换为 `visual_step_0.png` 纯净版截图
        image_path = os.path.join(base_log_path, task_id, "visual_step_0.png")

        # 如果截图不存在，跳过
        if not os.path.exists(image_path):
            continue

        try:
            base64_image = encode_image_to_base64(image_path)
            eval_data = call_vlm_api(base64_image, args.judge_model, eval_prompt)

            results.append({
                "task_id": task_id,
                "has_visual_bug": bool(eval_data.get("has_visual_bug", False)),
                "visual_layout": float(eval_data.get("visual_layout_score", 3)),
                "creative_alignment": float(eval_data.get("creative_alignment_score", 3)),
                "overall_aesthetics": float(eval_data.get("overall_aesthetics_score", 3)),
                "reasoning": eval_data.get("reasoning", "")
            })
        except Exception as e:
            tqdm.write(f"[ERROR] Failed to evaluate task {task_id}: {e}")

    if results:
        valid_tasks_count = len(results)
        bugs_count = sum(1 for r in results if r["has_visual_bug"])

        vbr_percentage = (bugs_count / valid_tasks_count) * 100.0
        avg_vl = sum(r["visual_layout"] for r in results) / valid_tasks_count
        avg_ca = sum(r["creative_alignment"] for r in results) / valid_tasks_count
        avg_oa = sum(r["overall_aesthetics"] for r in results) / valid_tasks_count

        total_las_score = avg_vl + avg_ca + avg_oa

        print("\n" + "=" * 55)
        print(f"✅ Aesthetic Evaluation Complete for: {args.target_model}")
        print(f"📊 Valid Evaluated Images: {valid_tasks_count} (Crashed samples safely skipped)")
        print("-" * 55)
        print(f"📉 Objective Defect (VBR) $\\downarrow$: {vbr_percentage:.1f}%")
        print(f"📈 Visual Layout (LAS-VL) $\\uparrow$: {avg_vl:.2f} / 5.0")
        print(f"📈 Creative Alignment (CA) $\\uparrow$: {avg_ca:.2f} / 5.0")
        print(f"📈 Overall Aesthetics (OA) $\\uparrow$: {avg_oa:.2f} / 5.0")
        print(f"⭐ Total Macro Score $\\uparrow$     : {total_las_score:.2f} / 15.0")
        print("=" * 55)

        summary_path = os.path.join(args.log_dir, safe_model_name, "api_aesthetic_evaluation.json")
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
                "trajectory_details": results
            }, f, indent=2, ensure_ascii=False)

        print(f"💾 Detailed trajectory and scores saved to:\n  -> {summary_path}")
    else:
        print("[WARNING] No valid images found or all evaluated tasks failed.")


if __name__ == "__main__":
    main()