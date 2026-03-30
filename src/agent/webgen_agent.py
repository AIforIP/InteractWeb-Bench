import os
import json
import shutil
import time
import sys
import re
import socket
from contextlib import closing
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
    current_timestamp
)
from utils.vlm_generation import vlm_generation, encode_image, compress_and_encode_image
from utils.execute_for_feedback import BrowserEnv, run_commands, execute_for_feedback


# --- 动态端口辅助函数 ---
def find_free_port():
    """向操作系统借一个绝对空闲的随机端口，用作 Vite 的首选起点"""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


# ==============================================================================
# [核心逻辑] INTERNAL_TEST_PROMPT (无标签纯视觉 + 键盘支持版)
# ==============================================================================
INTERNAL_TEST_PROMPT = """
You are the "Visual Copilot" for a web developer.
Your Goal: Verify the implementation based on the **User Instruction** AND the **Developer's Verification Criteria**.

**User Instruction**: "{instruction}"

**Developer's Self-Defined Criteria**: 
{criteria}

**Detailed Interaction History**:
{context_summary}

**Current Progress**: Step {step_idx}/{max_steps}

**Verification Strategies (CRITICAL)**:
1. **Focus on Criteria**: Check specifically for the features mentioned in the Criteria.
2. **AUTO-FILLED DIALOGS (NEW)**: Our testing framework instantly auto-fills native browser prompt dialogs in the background to prevent hanging. If you click an 'Add' or 'Create' button and a new item (e.g., 'Test Input Value') immediately appears on the page in the next step, consider the function **PASSED**. Do NOT fail the test just because you didn't see the input form.
3. **EARLY STOP (FAIL FAST)**: If a required element is clearly missing, styling is wrong, or a feature fails, IMMEDIATELY output `Action: Fail; [reason]`.
4. **Avoid Repetition**: If an action didn't work previously (check Interaction History), fail immediately.
5. **Browser Logs**: Review "BROWSER CONSOLE ERRORS". Fail on critical errors (404, TypeError). Ignore harmless warnings (React keys, deprecations).

**Available Actions**:
- `Click ["exact text OR icon meaning"]`: For standard web elements.
- `Click [x%, y%]`: For elements inside images/canvas.
- `Type ["exact text"]; [text]`: Input test data.
- `Type [x%, y%]; [text]`: Focus and type.
- `PressKey ["key_name"]`: Press a keyboard key.
- `Scroll [down/up]`: Check hidden areas.
- `Wait`: Only if expecting a network delay.
- `Finish`: ALL criteria are met perfectly.
- `Fail; [reason]`: Found a bug or criteria mismatch. STOP and feedback!

**Response Format (STRICT)**:
Thought: [Your analysis in a single paragraph. Mention if you detected an auto-filled result.]
Action: [One action command from the list above]
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
        self.step_idx = -1
        self.consecutive_failures = 0
        self.format_error_count = 0  # 增加全局格式违规计数器
        self.nodes = {}

        restored = restore_from_last_step(log_dir, workspace_dir, max_iter)
        if restored[0]:
            self.messages, _, self.step_idx, _, _, self.nodes = restored
            if self.messages[-1].get("info", {}).get("is_finish", False):
                self.is_finished = True

    def get_concise_messages(self):
        """获取压缩后的消息上下文，并强制限制全局图片数量，防止触发 API 400 错误"""
        MAX_IMAGES = 3  # 安全阈值，只保留上下文中最新的 3 张图片
        image_count = 0
        concise_msgs = []

        # 逆序遍历消息，优先保留最新的图片
        for m in reversed(self.messages):
            new_msg = {"role": m["role"]}

            # 如果 content 是列表（可能包含文本和图片）
            if isinstance(m["content"], list):
                new_content = []
                # 再次逆序遍历 content 列表中的元素
                for item in reversed(m["content"]):
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        if image_count < MAX_IMAGES:
                            new_content.insert(0, item)
                            image_count += 1
                        # 如果图片数量已达上限，则直接丢弃该图片对象
                    else:
                        new_content.insert(0, item)
                new_msg["content"] = new_content
            else:
                new_msg["content"] = m["content"]

            concise_msgs.insert(0, new_msg)

        return concise_msgs

    def _get_context_summary(self):
        summary_lines = ["=== CONVERSATION HISTORY (Agent & User) ==="]
        latest_artifact = ""
        past_audits = []

        for i, msg in enumerate(self.messages):
            role = msg['role']

            if role == 'system':
                continue

            raw_content = msg['content']
            if isinstance(raw_content, list):
                text_parts = [item['text'] for item in raw_content if item.get('type') == 'text']
                content = "\n".join(text_parts)
            else:
                content = str(raw_content)

            if role == 'user':
                if "**Visual Process Audit**" in content:
                    clean_audit = content.replace("**Visual Process Audit**", "").strip()
                    past_audits.append(clean_audit)
                    continue

                if "Execution Feedback:" in content:
                    continue

                is_real_user = False
                if i == 1:
                    is_real_user = True
                elif i > 0:
                    prev_msg = self.messages[i - 1]
                    if prev_msg['role'] == 'assistant' and '<boltAction type="ask_user"' in str(prev_msg['content']):
                        is_real_user = True

                if is_real_user:
                    summary_lines.append(f"[USER AGENT]:\n{content.strip()}\n")

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

        if past_audits:
            summary_lines.append("\n=== PREVIOUS VISUAL AUDIT RESULTS (Do not repeat failed checks blindly) ===")
            for idx, audit in enumerate(past_audits):
                summary_lines.append(f"[Audit Round {idx + 1}]:\n{audit}\n")

        if latest_artifact:
            summary_lines.append("\n=== LATEST CODE STATE (For Visual Verification) ===")
            summary_lines.append(latest_artifact)
            summary_lines.append("===================================================")

        return "\n".join(summary_lines)

    def _run_autonomous_test(self, target_path, criteria=None):
        if not criteria:
            criteria = f"Verify that the page meets the User Instruction: {self.instruction}"

        print(f"\033[95m[Agent]: Audit Goal -> {criteria[:100]}...\033[0m")

        # 🌟 随机分配首选端口，让 Vite 优先使用，避免并行任务都在抢 5173
        preferred_port = find_free_port()
        dynamic_start_cmd = f"npm run dev -- --port {preferred_port}"

        # 🌟 修改 1：在实例化 BrowserEnv 时，下放指令、模型和调用函数
        env = BrowserEnv(
            project_dir=self.workspace_dir,
            log_dir=self.log_dir,
            start_cmd=dynamic_start_cmd,
            instruction=self.instruction,  # 传任务
            builder_model=self.model,  # 传当前的 Builder 模型 (无状态)
            llm_caller=llm_generation  # 传生成函数
        )

        trace = []
        status = "unknown"
        error_msg = ""
        last_screenshot = None
        context_summary = self._get_context_summary()
        step_idx = 0

        try:
            env.start(target_path)
            history_text = ""

            while step_idx < self.max_visual_steps:
                print(f"  > Audit Step {step_idx + 1}/{self.max_visual_steps}")

                img_path = env.capture_observation(step_idx, draw_som=False)
                last_screenshot = img_path
                b64_img = encode_image(img_path)

                if not b64_img or len(b64_img) < 100:
                    print("\033[91m[致命错误] 截图 Base64 过短或失败，拦截 VLM 调用！\033[0m")
                    status = "error"
                    error_msg = "Screenshot capture failed or image is corrupted."
                    break

                console_logs = []
                if hasattr(env, 'get_console_logs'):
                    console_logs = env.get_console_logs()

                log_feedback = ""
                if console_logs:
                    log_feedback = "\n [BROWSER CONSOLE ERRORS DETECTED]:\n"
                    has_real_error = False
                    for log in console_logs:
                        if log.get('level') in ['SEVERE', 'ERROR'] or log.get('type') == 'error':
                            msg = log.get('text', log.get('message', str(log)))
                            log_feedback += f"- {msg}\n"
                            has_real_error = True

                    if not has_real_error:
                        log_feedback = ""
                    else:
                        print(f"\033[91m{log_feedback}\033[0m")

                # 🌟 修改 2：从“专线信箱”提取系统通知 (后台弹窗处理记录)
                system_notes = []
                if hasattr(env, 'get_and_clear_system_notes'):
                    system_notes = env.get_and_clear_system_notes()

                sys_feedback = ""
                if system_notes:
                    sys_feedback = "\n [SYSTEM NOTIFICATIONS (DO NOT FAIL TEST BASED ON THIS)]:\n"
                    for note in system_notes:
                        sys_feedback += f"- {note}\n"

                sys_msg = INTERNAL_TEST_PROMPT.format(
                    instruction=self.instruction,
                    criteria=criteria,
                    context_summary=context_summary,
                    step_idx=step_idx + 1,
                    max_steps=self.max_visual_steps
                )

                # 🌟 修改 3：组装包含系统通知的上下文
                step_context = f"Action History:\n{history_text}\n"
                if sys_feedback:
                    step_context += f"{sys_feedback}\n"
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

        # 1. Ask User
        if re.search(r'<boltAction\s+type\s*=\s*["\']ask_user["\'].*?>', output, re.DOTALL | re.IGNORECASE):
            match = re.search(r'<boltAction\s+type\s*=\s*["\']ask_user["\'].*?>(.*?)(?:</boltAction>|$)', output,
                              re.DOTALL | re.IGNORECASE)
            question = match.group(1).strip() if match else output

            info_assistant = {"is_question": True}
            self.messages.append({"role": "assistant", "content": output, "info": info_assistant})
            return {"type": "question", "content": question, "is_finish": False}, False

        # 2. Finish
        elif re.search(r'<boltAction\s+type\s*=\s*["\']finish["\'].*?>', output, re.DOTALL | re.IGNORECASE):
            extract_and_write_files(output, self.workspace_dir)

            info_assistant = {"is_finish": True}
            self.messages.append({"role": "assistant", "content": output, "info": info_assistant})
            return {"type": "submitted" if simulation_mode else "finish", "content": output,
                    "is_finish": not simulation_mode}, False

        # 3. Visual Verification
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

            # 🌟 同理，分配首选端口，不再依赖底层去猜
            preferred_port = find_free_port()
            dynamic_start_cmd = f"npm run dev -- --port {preferred_port}"

            f_dict = execute_for_feedback(
                self.workspace_dir,
                self.log_dir,
                start_cmd=dynamic_start_cmd
            )

            info_user = {"environment_feedback": f_dict}

            fb = "Execution Feedback:\n"
            if f_dict.get("install_error"): fb += f"Install Error: {f_dict['install_error']}\n"
            if f_dict.get("start_error"):
                fb += f"Runtime Error: {f_dict['start_results']}\n"
            else:
                fb += "Environment Ready. Verify UI or Submit."

            self.messages.append({"role": "user", "content": fb, "info": info_user})
            return {"type": "coding", "content": output, "is_finish": False}, False

        # 5. Fallback (格式违规兜底纠错)
        else:
            self.format_error_count += 1
            info_assistant = {"is_format_error": True}
            self.messages.append({"role": "assistant", "content": output, "info": info_assistant})

            print(
                f"\033[93m[Warn] LLM Format Error (Total: {self.format_error_count}). Requesting regeneration...\033[0m")

            # 向模型发送强约束纠错提示
            correction_msg = (
                "SYSTEM WARNING: Your previous output did not match any expected action path. "
                "You MUST choose exactly ONE path (A, B, C, or D). "
                "If you are attempting to write code or run shell commands (like npm install), "
                "you MUST wrap them STRICTLY inside opening `<boltArtifact>` and closing `</boltArtifact>` tags. "
                "Do NOT output bare `<boltAction>` tags outside of an artifact. "
                "Please evaluate the situation and generate your response again using the correct format."
            )

            info_user = {"is_correction": True}
            self.messages.append({"role": "user", "content": correction_msg, "info": info_user})

            return {"type": "format_error", "content": output, "is_finish": False}, True

    def save_history(self, i, pre=None, has_error=False):
        output_file = os.path.join(self.log_dir, "history.json")

        # 将累计格式错误次数保存进历史节点数据中
        self.nodes[f"step{i}.json"] = {
            "has_error": has_error,
            "pre": pre,
            "format_errors_up_to_now": getattr(self, 'format_error_count', 0)
        }

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