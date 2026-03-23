import os
import json
import shutil
import time
import sys
import re
from typing import Dict, Tuple, Any

# 引入项目根目录
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from prompts import system_prompt, reminders_prompt
from utils import (
    llm_generation,
    extract_and_write_files,
    directory_to_dict,
    dict_to_directory,
    restore_from_last_step,
    generate_gui_agent_instruction,
    current_timestamp
)
from utils.vlm_generation import vlm_generation, encode_image, compress_and_encode_image
from utils.execute_for_feedback import BrowserEnv, run_commands, execute_for_feedback

# ==============================================================================
# [核心逻辑] INTERNAL_TEST_PROMPT (无标签纯视觉 + 键盘支持版)
# ==============================================================================
INTERNAL_TEST_PROMPT = """
You are the "Visual Copilot" for a web developer.
Your Goal: Verify the implementation based on the **User Instruction** AND the **Developer's Verification Criteria**.

**User Instruction**: "{instruction}"

**Developer's Self-Defined Criteria** (Specific features implemented in this step):
{criteria}

**Detailed Interaction History**:
{context_summary}

**Current Progress**: Step {step_idx}/{max_steps}

**Verification Strategies (CRITICAL)**:
1. **Focus on Criteria**: Check specifically for the features mentioned in the Criteria.
2. **EARLY STOP (FAIL FAST)**: Your purpose is to provide feedback to the developer. If a required element is missing, styling is wrong, or a feature fails, DO NOT keep trying or Waiting. IMMEDIATELY output `Action: Fail; [reason]` so the developer can fix it.
3. **Avoid Repetition**: Remember your Interaction History. If an action didn't work previously, fail immediately. Do not test the same broken thing twice.
4. Critical - Browser Logs: Review any provided "BROWSER CONSOLE ERRORS". If they are critical errors (e.g., Uncaught TypeError, API 404 Not Found) that directly block the UI functionality, output 'Fail'. However, you MUST ignore harmless warnings (e.g., React deprecation notices, unique key warnings) if the visual Criteria are successfully met.

**Available Actions** :
- `Click ["exact text OR icon meaning"]`: For standard web elements.
- `Click [x%, y%]`: For elements inside images/canvas.
- `Type ["exact text"]; [text]`: Input test data.
- `Type [x%, y%]; [text]`: Click coordinate to focus, then type.
- `PressKey ["key_name"]`: Press a keyboard key.
- `Scroll [down/up]`: Check hidden areas.
- `Wait`: Wait for loading ONLY if you expect a network delay.
- `Finish`: ALL criteria are met perfectly and there are no bugs.
- `Fail; [reason]`: Found a bug, console error, or criteria mismatch. STOP and feedback immediately!

**Response Format**:
Thought: [Analyze: 1. Is the criterion met? 2. Are there errors? 3. If failed, I must FAIL IMMEDIATELY.]
Action: [One action command]
"""
# --- 死循环故障分析 Prompt ---
FAILURE_ANALYSIS_PROMPT = """
You are a "Post-Mortem Analyst" AI. 
The Web Agent has got stuck in a verification loop (failed {limit} times in a row).

Your Goal: Analyze the provided "Action Trace" and the "Last Screenshot" to determine the ROOT CAUSE of the failure.

**Context**:
- The agent was trying to verify: "{instruction}"
- It kept failing visual verification.

**Analyze the following**:
1. **Visual State Stagnation**: Did the agent keep clicking but the screen didn't change?
2. **Logic Paradox**: Is the instruction contradictory?
3. **Technical Error**: Are there critical console errors?

**Output Format**:
Summary: [1-sentence summary]
Root Cause: [Detailed explanation]
Category: [Visual_Stagnation / Logic_Paradox / Code_Error / Hallucination]
"""

def remove_dir(directory):
    for _ in range(5):
        try:
            shutil.rmtree(directory)
            return True
        except:
            time.sleep(5)
    return False

class WebGenAgent:
    def __init__(self, model, vlm_model, fb_model, workspace_dir, log_dir, instruction, max_iter, overwrite,
                 error_limit, max_tokens=-1, max_completion_tokens=-1, temperature=0.5, custom_system_prompt=None,
                 difficulty="middle", max_simulation_steps=15):
        self.model = model
        self.vlm_model = vlm_model
        self.fb_model = fb_model
        self.workspace_dir = workspace_dir
        self.log_dir = log_dir
        self.instruction = instruction
        self.max_iter = max_iter
        self.max_tokens = max_tokens
        self.max_completion_tokens = max_completion_tokens
        self.temperature = temperature

        base_prompt = custom_system_prompt if custom_system_prompt else system_prompt



        # [核心修改]：补充了“多功能合并写入”的提示
        self.system_prompt = base_prompt + (
            "\n\n<IMPORTANT_CONSTRAINT>\n"
            "1. TESTABILITY: You MUST add descriptive `data-testid` attributes to all key interactive elements.\n"
            "2. **AUTONOMOUS VERIFICATION**:\n"
            "   - The goal is to ensure the webpage code runs correctly, balancing efficiency and reasonableness of the test task.\n"
            "   - When writing the `<TestCriteria>` block, there's no need to check every function; only verify the functions you deem necessary.\n"
            "   - If a specific test item causes `<boltAction type=\"screenshot_validated\">` to fail repeatedly, its validation steps can be deferred when writing the `<TestCriteria>` block. It is more important to complete various functions as comprehensively as possible than to perfectly fix a stubborn error.\n"
            "   - Immediately BEFORE using `<boltAction type=\"screenshot_validated\">`, you MUST provide a `<TestCriteria>` block.\n"
            "   - Design the criteria to match what you just implemented. You can include static visual checks, interactive functional checks (e.g., clicking, typing), or a combination of both.\n"
            "   - If you implemented multiple features in this step, you can combine their verification steps into a single `<TestCriteria>` block.\n"
            "   Example:\n"
            "   <TestCriteria>\n"
            "   1. Verify the header is dark orange.\n"
            "   2. Click the 'Book' button.\n"
            "   3. Verify the booking modal appears and click 'Cancel'.\n"
            "   4. (Can be deferred if this step often fails) Type 'Honda' into the search input and press Enter.\n"
            "   5. (Can be deferred if this step often fails) Verify the car grid updates to show only Honda cars.\n"
            "   </TestCriteria>\n"
            "   <boltAction type=\"screenshot_validated\">/dashboard</boltAction>\n"
            "</IMPORTANT_CONSTRAINT>"
        )

        self.difficulty = difficulty.lower()
        self.max_visual_steps = max_simulation_steps
        self.error_limit = error_limit

        self.id = os.path.basename(log_dir)
        print(f"[{self.id}] Config: Steps={self.max_visual_steps}, ErrorLimit={self.error_limit}")

        if os.path.exists(workspace_dir): remove_dir(workspace_dir)
        os.makedirs(workspace_dir)
        if overwrite and os.path.exists(log_dir): remove_dir(log_dir)
        if not os.path.exists(log_dir): os.makedirs(log_dir)

        self.is_finished = False
        self.messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": instruction}]
        self.gui_instruction = generate_gui_agent_instruction(instruction, fb_model, -1, -1)
        self.step_idx = -1
        self.consecutive_failures = 0
        self.nodes = {}

        restored = restore_from_last_step(log_dir, workspace_dir, max_iter)
        if restored[0]:
            self.messages, self.gui_instruction, self.step_idx, _, _, self.nodes = restored
            if self.messages[-1].get("info", {}).get("is_finish", False):
                self.is_finished = True

    def get_concise_messages(self):
        return [{"role": m["role"], "content": m["content"]} for m in self.messages]

    def _get_context_summary(self):
        summary_lines = ["=== CONVERSATION HISTORY (Agent & User) ==="]
        latest_artifact = ""
        past_audits = []  # 【新增 1】：专门用于收集过往的视觉检查报告

        # 使用 enumerate 获取索引，方便我们回头看“上一条消息”
        for i, msg in enumerate(self.messages):
            role = msg['role']

            # 1. 过滤系统提示词
            if role == 'system':
                continue

            # 2. 健壮的文本提取
            raw_content = msg['content']
            if isinstance(raw_content, list):
                text_parts = [item['text'] for item in raw_content if item.get('type') == 'text']
                content = "\n".join(text_parts)
            else:
                content = str(raw_content)

            # 3. 提取 User Agent 对话（采用状态机/行为追溯法）
            if role == 'user':
                # 【新增 2】：拦截视觉审计报告，将其存入记忆列表，并跳过普通对话处理
                if "**Visual Process Audit**" in content:
                    clean_audit = content.replace("**Visual Process Audit**", "").strip()
                    past_audits.append(clean_audit)
                    continue

                # 显式过滤执行报错，避免污染对话流
                if "Execution Feedback:" in content:
                    continue

                is_real_user = False

                # 情况 A：这是整个交互的开端（初始需求 Instruction）
                # 通常 messages[0] 是 system prompt，messages[1] 是初始指令
                if i == 1:
                    is_real_user = True
                # 情况 B：判断它的“上一句话”是不是大模型发出的 ask_user 提问
                elif i > 0:
                    prev_msg = self.messages[i - 1]
                    if prev_msg['role'] == 'assistant' and '<boltAction type="ask_user"' in str(prev_msg['content']):
                        is_real_user = True

                # 如果是真实用户，才加入摘要
                if is_real_user:
                    summary_lines.append(f"[USER AGENT]:\n{content.strip()}\n")

            # 4. 提取 Coding Agent 的对话与思考过程
            elif role == 'assistant':
                if '<boltArtifact' in content:
                    parts = content.split('<boltArtifact')
                    chat_part = parts[0].strip()

                    if chat_part:
                        summary_lines.append(f"[CODING AGENT]:\n{chat_part}\n")

                    summary_lines.append(f"[CODING AGENT]: [Generated/Updated Code...]\n")
                    latest_artifact = '<boltArtifact' + '<boltArtifact'.join(parts[1:])
                else:
                    summary_lines.append(f"[CODING AGENT]:\n{content.strip()}\n")

        # ==========================================================
        # 【新增 3】：在代码状态之前，附加上所有历史测试结果
        # ==========================================================
        if past_audits:
            summary_lines.append("\n=== PREVIOUS VISUAL AUDIT RESULTS (Do not repeat failed checks blindly) ===")
            for idx, audit in enumerate(past_audits):
                summary_lines.append(f"[Audit Round {idx + 1}]:\n{audit}\n")

        # 5. 附上最新代码
        if latest_artifact:
            summary_lines.append("\n=== LATEST CODE STATE (For Visual Verification) ===")
            summary_lines.append(latest_artifact)
            summary_lines.append("===================================================")

        return "\n".join(summary_lines)

    def _run_autonomous_test(self, target_path, criteria=None):
        if not criteria:
            criteria = f"Verify that the page meets the User Instruction: {self.instruction}"

        print(f"\033[95m[Agent]: Audit Goal -> {criteria[:100]}...\033[0m")

        env = BrowserEnv(self.workspace_dir, self.log_dir)
        trace = []
        status = "unknown"
        error_msg = ""
        last_screenshot = None
        context_summary = self._get_context_summary()
        step_idx = 0

        import re  # 确保引入了正则库

        try:
            env.start(target_path)
            history_text = ""

            while step_idx < self.max_visual_steps:
                print(f"  > Audit Step {step_idx + 1}/{self.max_visual_steps}")

                # 传入 draw_som=False，明确要求内部视觉验证阶段不要画框
                img_path = env.capture_observation(step_idx, draw_som=False)
                last_screenshot = img_path
                b64_img = encode_image(img_path)

                # 【护城河 1】防御性拦截：避免截图为空导致引发 API 400 Bad Request
                if not b64_img or len(b64_img) < 100:
                    print("\033[91m[致命错误] 截图 Base64 过短或失败，拦截 VLM 调用！\033[0m")
                    status = "error"
                    error_msg = "Screenshot capture failed or image is corrupted."
                    break

                # 获取当前步骤的控制台日志
                console_logs = []
                if hasattr(env, 'get_console_logs'):
                    console_logs = env.get_console_logs()

                log_feedback = ""
                if console_logs:
                    log_feedback = "\n [BROWSER CONSOLE ERRORS DETECTED]:\n"
                    # 【护城河 2】日志净化：只提取真正的报错 (SEVERE/error)，屏蔽 React 黄字警告
                    has_real_error = False
                    for log in console_logs:
                        if log.get('level') in ['SEVERE', 'ERROR'] or log.get('type') == 'error':
                            msg = log.get('text', log.get('message', str(log)))
                            log_feedback += f"- {msg}\n"
                            has_real_error = True

                    if not has_real_error:
                        log_feedback = ""  # 如果全是无用警告，就当没看见
                    else:
                        print(f"\033[91m{log_feedback}\033[0m")

                sys_msg = INTERNAL_TEST_PROMPT.format(
                    instruction=self.instruction,
                    criteria=criteria,
                    context_summary=context_summary,
                    step_idx=step_idx + 1,
                    max_steps=self.max_visual_steps
                )

                step_context = f"Action History:\n{history_text}\n"
                if log_feedback:
                    step_context += f"{log_feedback}\n"

                response = vlm_generation(
                    model=self.vlm_model,
                    messages=[
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": [
                            {"type": "text", "text": step_context},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}}
                        ]}
                    ]
                )

                # ==========================================
                # 【核心修改点】使用正则更健壮地同时提取 Thought 和 Action
                # 这样即使大模型在 Thought 里换了行，也能完整抓取
                # ==========================================
                action = "Wait"
                thought_process = "No specific thought process provided."

                thought_match = re.search(r'Thought:\s*(.*?)(?=\n*Action:|$)', response, re.IGNORECASE | re.DOTALL)
                if thought_match:
                    thought_process = thought_match.group(1).strip()

                action_match = re.search(r'Action:\s*(.*)', response, re.IGNORECASE)
                if action_match:
                    action = action_match.group(1).strip()

                trace.append({"step": step_idx, "thought": response, "action": action, "logs": log_feedback,
                              "screenshot_path": img_path})
                history_text += f"Step {step_idx}: {action}\n"

                if "Finish" in action:
                    status = "success"
                    break

                if "Fail" in action:
                    status = "failed"
                    # 【核心修改点】将大模型的具体推理（Thought）组装进错误信息，反馈给 Coding Agent
                    error_msg = f"Visual Audit Failed: {action}\n[Visual Agent Observation]: {thought_process}"
                    break

                env.execute_action(action)
                time.sleep(2)
                step_idx += 1

            if status == "unknown":
                status = "failed"
                error_msg = f"Audit timed out after {self.max_visual_steps} steps."

        except Exception as e:
            status = "error"
            error_msg = f"Internal Audit Error: {str(e)}"
        finally:
            env.close()

        # =========================================================
        # 核心逻辑：仅在失败或报错时，才附加当前步骤捕获到的控制台日志
        # =========================================================
        raw_console_errors = ""
        if 'log_feedback' in locals() and log_feedback and status in ["failed", "error"]:
            raw_console_errors = f"\n\n--- BROWSER CONSOLE LOGS (For Debugging) ---\n{log_feedback}"

        report = f"**Visual Process Audit**\nGoal: {criteria}\nStatus: {status}\nDetails: {error_msg}{raw_console_errors}"

        return report, (status == "failed" or status == "error"), last_screenshot, trace

    def _analyze_failure(self, trace, last_screenshot_path):
        if not last_screenshot_path: return "No snapshot."
        b64_img = encode_image(last_screenshot_path)
        sys_msg = FAILURE_ANALYSIS_PROMPT.format(instruction=self.instruction, limit=self.error_limit)
        try:
            return vlm_generation(model=self.vlm_model, messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": [{"type": "text", "text": f"Trace: {json.dumps(trace)}"},
                                             {"type": "image_url",
                                              "image_url": {"url": f"data:image/png;base64,{b64_img}"}}]}
            ])
        except:
            return "Analysis failed."

    def step(self, i, user_feedback=None, simulation_mode=False):
        if user_feedback:
            self.messages.append({"role": "user", "content": user_feedback})

        concise_messages = self.get_concise_messages()
        output = llm_generation(concise_messages, self.model, max_tokens=self.max_tokens,
                                max_completion_tokens=self.max_completion_tokens, temperature=self.temperature)

        # 1. Ask User (明确的提问标签)
        if re.search(r'<boltAction\s+type\s*=\s*["\']ask_user["\'].*?>', output, re.DOTALL | re.IGNORECASE):
            match = re.search(r'<boltAction\s+type\s*=\s*["\']ask_user["\'].*?>(.*?)(?:</boltAction>|$)', output,
                              re.DOTALL | re.IGNORECASE)
            question = match.group(1).strip() if match else output

            # 为 Assistant 创建独立的 info
            info_assistant = {"is_question": True}
            self.messages.append({"role": "assistant", "content": output, "info": info_assistant})
            return {"type": "question", "content": question, "is_finish": False}, False

        # 2. Finish (明确的结束标签)
        elif re.search(r'<boltAction\s+type\s*=\s*["\']finish["\'].*?>', output, re.DOTALL | re.IGNORECASE):
            extract_and_write_files(output, self.workspace_dir)

            info_assistant = {"is_finish": True}
            self.messages.append({"role": "assistant", "content": output, "info": info_assistant})
            return {"type": "submitted" if simulation_mode else "finish", "content": output,
                    "is_finish": not simulation_mode}, False

        # 3. Visual Verification (明确的视觉测试标签)
        elif re.search(r'<boltAction\s+type\s*=\s*["\']screenshot_validated["\'].*?>', output,
                       re.DOTALL | re.IGNORECASE):

            info_assistant = {}
            self.messages.append({"role": "assistant", "content": output, "info": info_assistant})

            extract_and_write_files(output, self.workspace_dir)
            run_commands(["npm install"], self.workspace_dir)

            criteria_match = re.search(r'<TestCriteria>(.*?)</TestCriteria>', output, re.DOTALL | re.IGNORECASE)
            criteria_text = criteria_match.group(1).strip() if criteria_match else None

            action_match = re.search(
                r'<boltAction\s+type\s*=\s*["\']screenshot_validated["\'][^>]*>(.*?)(?:</boltAction>|$)', output,
                re.DOTALL | re.IGNORECASE)
            target_path = action_match.group(1).strip() if action_match and action_match.group(1).strip() else "/"

            if not criteria_text:
                criteria_text = f"Verify Instruction: {self.instruction}"

            feedback_str, failed, last_screenshot_path, detailed_trace = self._run_autonomous_test(target_path,
                                                                                                   criteria=criteria_text)

            # 为 User 视觉反馈创建完全独立的 info
            info_user = {"internal_test_trace": detailed_trace}

            if failed:
                self.consecutive_failures += 1
                print(f"\033[93m[System] Audit Failed. Streak: {self.consecutive_failures}/{self.error_limit}\033[0m")
            else:
                self.consecutive_failures = 0

            if self.consecutive_failures >= self.error_limit:
                force_stop_analysis = self._analyze_failure(detailed_trace, last_screenshot_path)
                self.is_finished = True
                feedback_str += f"\n\n[SYSTEM ABORT]: Failed verification {self.error_limit} times.\nAnalysis: {force_stop_analysis}"
                info_user["is_deadlock"] = True
                info_user["failure_analysis"] = force_stop_analysis

            user_content = [{"type": "text", "text": feedback_str}]
            if last_screenshot_path and os.path.exists(last_screenshot_path):
                try:
                    compressed_b64 = compress_and_encode_image(last_screenshot_path, max_size=(800, 800), quality=60)
                    user_content.append({"type": "image_url",
                                         "image_url": {"url": f"data:image/jpeg;base64,{compressed_b64}",
                                                       "detail": "low"}})
                except:
                    pass

            self.messages.append({"role": "user", "content": user_content, "info": info_user})
            return {"type": "internal_test", "content": target_path, "is_finish": self.is_finished}, failed

        # 4. Coding
        elif "<boltArtifact" in output:
            info_assistant = {}
            self.messages.append({"role": "assistant", "content": output, "info": info_assistant})

            extract_and_write_files(output, self.workspace_dir)
            f_dict = execute_for_feedback(self.workspace_dir, self.log_dir)

            # 为 User 执行反馈创建完全独立的 info
            info_user = {"environment_feedback": f_dict}

            fb = "Execution Feedback:\n"
            if f_dict.get("install_error"): fb += f"Install Error: {f_dict['install_error']}\n"
            if f_dict.get("start_error"):
                fb += f"Runtime Error: {f_dict['start_results']}\n"
            else:
                fb += "Environment Ready. Verify UI or Submit."

            self.messages.append({"role": "user", "content": fb, "info": info_user})
            return {"type": "coding", "content": output, "is_finish": False}, False

        # 5. Fallback
        else:
            info_assistant = {"is_question": True}
            self.messages.append({"role": "assistant", "content": output, "info": info_assistant})
            return {"type": "question", "content": output, "is_finish": False}, False

    def save_history(self, i, pre=None, has_error=False):
        output_file = os.path.join(self.log_dir, "history.json")
        self.nodes[f"step{i}.json"] = {"has_error": has_error, "pre": pre}

        existing_snapshots = {}
        if os.path.exists(output_file):
            try:
                with open(output_file, "r", encoding="utf-8") as f:
                    old_data = json.load(f)
                    existing_snapshots = old_data.get("workspace_snapshots", {})
            except:
                pass

        current_workspace = directory_to_dict(self.workspace_dir)
        existing_snapshots[f"step_{i}"] = current_workspace

        # 保存时，self.messages 已经包含了所有 turn 的 info 信息
        data = {
            "messages": self.messages,
            "nodes": self.nodes,
            "step_idx": i,
            "workspace_files": current_workspace,
            "workspace_snapshots": existing_snapshots
        }
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _extract_step_index(self, filename: str) -> int:
        m = re.search(r"step(\d+)\.json$", filename)
        return int(m.group(1)) if m else -1

    def get_error_count(self, file_name):
        return 0

    def choose_best_node(self):
        if not self.nodes: return None, None, False
        last_key = list(self.nodes.keys())[-1]
        return last_key, self.nodes[last_key], True

    def run(self):
        pass