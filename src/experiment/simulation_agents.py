import os
import json
import sys

# 引入项目根目录
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from utils.llm_generation import llm_generation
from utils.vlm_generation import vlm_generation, encode_image

# --- 1. 角色的特定表达风格 (Persona Rules) ---
PERSONA_RULES = {
    "P-MIN": """
**Persona: Minimalist (P-MIN)**
- You are extremely impatient but prioritize task completion.
- Rule: Provide the bare minimum data requested. Avoid all adjectives or fluff.
- Rule: If asked for clarification, give the exact accurate value from Ground Truth but in the shortest possible form (e.g., "Deep blue" instead of "I want a very deep and professional blue").
""",
    "P-RAM": """
**Persona: Rambler (P-RAM)**
- You are talkative and disorganized, but ultimately helpful if asked.
- Rule: Wrap the accurate answer from Ground Truth in at least 70% irrelevant noise (daily life, weather, etc.).
- Rule: When encountering technical jargon, always remember that you have no technical background and no concept of technical terms.
""",
    "P-INT": """
**Persona: Intuitive (P-INT)**
- You think in metaphors but will provide clarity when pressed.
- Rule: Translate facts into artistic metaphors. 
- Rule: If developers ask for technical details (e.g., hexadecimal code or pixels), provide metaphorical and artistic descriptions that align with the "Ground Truth."
""",
    "P-CON": """
**Persona: Conflicting (P-CON)**
- You initially hold cognitive biases, but defer to the expert's professional judgment upon questioning.
- Rule: If the developer identifies a contradiction, first defend your original stance briefly ("Are you sure? I thought it looked good..."), but then immediately provide the ACCURATE requirement from the Ground Truth to allow the project to move forward.
- Rule: For non-conflicting questions, provide rigid and accurate data.
"""
}

# --- 2. 核心双层架构 System Prompt ---
USER_SIMULATION_SYSTEM_PROMPT = """
You are a non-technical client communicating with a web developer.

**Your Requirements (Ground Truth)**:
{ground_truth}

---
**STEP 1: INFORMATION CONTAINMENT (STRICTLY ENFORCED)**
Before drafting your answer, you must mentally perform an "Information Filtering" process:
1. **Analyze**: What exactly is the developer asking for based on the conversation history?
2. **Scan**: Find the specific parts of the Ground Truth that match the question.
3. **Filter**: Identify any information in the Ground Truth that was NOT asked for. 
4. **CRITICAL RULE**: You must treat the unasked information as "HIDDEN". NEVER volunteer "HIDDEN" information. Only answer what was explicitly asked. If asked about "Data", DO NOT mention "Colors".

**STEP 2: APPLY YOUR PERSONA**
Construct your response using ONLY the matched information from Step 1, while strictly applying the following persona rules:
{persona_rules}

---
**Output Requirement**:
Reply directly with your answer in English. Do not output your internal thinking process.
"""


class UserSimulator:
    def __init__(self, ground_truth_instruction, initial_instruction, evaluation_checklist, persona="P-MIN",
                 model="gpt-4o", vlm_model="gpt-4o", base_url=None, api_key=None):  # 🌟 新增专属配置参数
        self.ground_truth_instruction = ground_truth_instruction
        self.evaluation_checklist = evaluation_checklist
        self.persona = persona
        self.model = model
        self.vlm_model = vlm_model

        # 🌟 挂载专属路由和密钥
        self.base_url = base_url
        self.api_key = api_key

        # 记忆系统初始化：将批量生成的初始指令 (L1) 作为第一条发出的消息记录下来
        self.conversation_history = [
            {"role": "assistant", "content": f"I initially told the developer: \"{initial_instruction}\""}
        ]

    def answer_question(self, question):
        """
        基于“核心法则+角色滤镜+历史记忆”回答问题
        """
        # 1. 获取对应角色的规则并组装 System Prompt
        persona_rules = PERSONA_RULES.get(self.persona, PERSONA_RULES["P-MIN"])
        system_prompt = USER_SIMULATION_SYSTEM_PROMPT.format(
            ground_truth=self.ground_truth_instruction,
            persona_rules=persona_rules
        )

        # 2. 组装发送给大模型的消息列表
        messages = [{"role": "system", "content": system_prompt}]

        # 载入之前的交互记忆（包含初始指令）
        messages.extend(self.conversation_history)

        # 载入开发者当前的新问题
        current_question_msg = {"role": "user", "content": f"The Developer asks: \"{question}\""}
        messages.append(current_question_msg)

        # 3. 🌟 调用大模型生成回答 (透传专属路由配置)
        response = llm_generation(
            messages=messages,
            model=self.model,
            temperature=0.4,
            base_url=self.base_url,  # 透传专属 URL
            api_key=self.api_key  # 透传专属 Key
        )
        answer = response.strip()

        # 4. 将本轮问答写入记忆，供下一轮使用
        self.conversation_history.append(current_question_msg)
        self.conversation_history.append({"role": "assistant", "content": answer})

        return answer

    def evaluate_with_hybrid_oracle(self, target_url, screenshot_path, oracle_slots):
        """
        调用全新的双轨客观评估系统 (Playwright + VQA)
        注意：外部在调用此方法前，需要确保服务器已启动并已截取 screenshot_path
        """
        from experiment.hybrid_evaluator import execute_hybrid_feedback

        # 执行双轨评测
        metrics = execute_hybrid_feedback(target_url, screenshot_path, oracle_slots, self.vlm_model)

        sr = metrics.get("Success_Rate_SR", 0)
        tcr = metrics.get("Task_Completion_Rate_TCR", 0.0)
        details = metrics.get("Details", [])

        # 组装给 Agent 看的最终反馈报告 (或者仅用于论文数据统计)
        status = "SATISFIED" if sr == 1 else "REJECTED"

        final_report_text = f"**User Acceptance Test Report (Hybrid Evaluation)**\n"
        final_report_text += f"Final Status: {status}\n"
        final_report_text += f"Task Completion Rate (TCR): {tcr * 100}%\n\n"
        final_report_text += "Detailed Feedback:\n"

        for item in details:
            mark = "[PASS]" if item.get("passed") else "[FAIL]"
            task_desc = item.get("task", "Unknown Task")
            track_type = item.get("track", "Unknown Track")

            final_report_text += f"{mark} [{track_type}] {task_desc}\n"
            if not item.get("passed"):
                final_report_text += f"   Reason: {item.get('error', 'No error detail provided')}\n"

        return {
            "status": status,
            "sr": sr,
            "tcr": tcr,
            "text": final_report_text,
            "raw_metrics": metrics
        }