import subprocess
import os
import sys
import time
import re
import signal
import platform
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

# 引入项目根目录
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from .timestamp import current_timestamp

# --- Set-of-Marks (SoM) 脚本 ---
JS_SOM_SCRIPT = """
(function() {
    var oldLabels = document.querySelectorAll('.som-label');
    oldLabels.forEach(l => l.remove());
    var items = document.querySelectorAll('button, a, input, textarea, select, [role="button"], [onclick]');
    var index = 0;
    items.forEach(function(item) {
        var rect = item.getBoundingClientRect();
        if (rect.width < 5 || rect.height < 5 || item.offsetParent === null) return;
        var label = document.createElement('div');
        label.className = 'som-label';
        label.innerText = index;
        label.style.position = 'absolute';
        label.style.background = '#FFD700';
        label.style.color = 'black';
        label.style.fontSize = '12px';
        label.style.fontWeight = 'bold';
        label.style.padding = '1px 3px';
        label.style.border = '1px solid black';
        label.style.borderRadius = '3px';
        label.style.zIndex = '2147483647';
        label.style.pointerEvents = 'none';
        label.style.left = (rect.left + window.scrollX) + 'px';
        label.style.top = (rect.top + window.scrollY) + 'px';
        document.body.appendChild(label);
        item.setAttribute('data-som-id', index);
        index++;
    });
})();
"""


# ==============================================================================
#  核心进程与网络管理工具 (嗅探 + 彻底清理)
# ==============================================================================
def stop_process_tree(proc: subprocess.Popen, timeout: float = 5.0):
    """跨平台彻底杀死进程组，防止僵尸进程占用端口"""
    if proc is None or proc.poll() is not None:
        return
    try:
        if platform.system() == "Windows":
            proc.send_signal(signal.CTRL_BREAK_EVENT)
            try:
                proc.wait(timeout)
            except subprocess.TimeoutExpired:
                proc.terminate()
        else:
            # POSIX: 杀死整个进程组
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGTERM)
            try:
                proc.wait(timeout)
            except subprocess.TimeoutExpired:
                os.killpg(pgid, signal.SIGKILL)
    except Exception as e:
        print(f"[WARN] 无法彻底清理进程树: {e}")


def wait_for_url_in_log(log_path, timeout=30):
    """动态读取日志文件，嗅探前端框架实际绑定的 URL"""
    print(f"Waiting for URL to appear in log ({log_path})...")
    # 匹配 Vite/CRA 常见的 URL 输出格式
    url_pattern = re.compile(r"http://(?:localhost|127\.0\.0\.1|0\.0\.0\.0|(?:\d{1,3}\.){3}\d{1,3}):\d+/?")
    deadline = time.time() + timeout

    while time.time() < deadline:
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                match = url_pattern.search(content)
                if match:
                    url = match.group(0)
                    # 统一替换为 localhost 以兼容 Playwright 访问
                    url = url.replace("0.0.0.0", "localhost").replace("127.0.0.1", "localhost")
                    print(f"Found service URL: {url}")
                    return url
        time.sleep(1)
    raise TimeoutError(f"在 {timeout} 秒内未能在日志中嗅探到服务 URL。")


def start_background_service(start_cmd, cwd, log_file="service.log"):
    """使用独立进程组启动后台服务，以便后续能够一锅端"""
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "w", encoding="utf-8")

    # 关键：创建新的进程组
    kwargs = {'start_new_session': True} if platform.system() != "Windows" else {
        'creationflags': getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 512)}

    process = subprocess.Popen(
        start_cmd,
        shell=True,
        cwd=cwd,
        stdout=log,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        **kwargs
    )
    return process, str(log_path)


def run_commands(cmds, cwd):
    results = []
    for cmd in cmds:
        print(f"Running: {cmd}")
        process = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
        output = process.stdout + (process.stderr or "")
        if "EBADENGINE" in output or "Unsupported engine" in output:
            print("\033[93m[Warning] Node.js version mismatch detected in output.\033[0m")
        print(output)
        results.append((cmd, output))
    return results


# ==============================================================================
#  BrowserEnv: 视觉代理交互环境
# ==============================================================================
class BrowserEnv:
    # 移除了死板的 app_port，直接接收动态嗅探到的 target_url
    def __init__(self, project_dir, log_dir, start_cmd="npm run dev"):
        self.project_dir = project_dir
        self.log_dir = log_dir
        self.start_cmd = start_cmd
        self.base_url = None
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        self.process = None
        self.console_logs = []

    def start(self, target_path=None):
        # 1. 启动服务并动态嗅探 URL
        log_file = os.path.join(self.log_dir, "service.log")
        self.process, log_path_str = start_background_service(self.start_cmd, self.project_dir, log_file)
        self.base_url = wait_for_url_in_log(log_path_str, timeout=30)

        # 2. 启动 Playwright 浏览器
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"]
        )
        self.context = self.browser.new_context(viewport={'width': 1200, 'height': 800})
        self.page = self.context.new_page()

        def handle_dialog(dialog):
            print(f"  > [Visual Copilot] Auto-accepted Dialog: {dialog.message}")
            dialog.accept()

        self.page.on("dialog", handle_dialog)
        self.page.on("console",
                     lambda msg: self.console_logs.append({"type": msg.type, "text": msg.text}) if msg.type in ["error",
                                                                                                                "warning"] else None)
        self.page.on("pageerror", lambda exc: self.console_logs.append({"type": "exception", "text": str(exc)}))

        # 3. 访问动态获取到的地址
        url = f"{self.base_url.rstrip('/')}/{target_path.lstrip('/')}" if target_path else self.base_url
        print(f"  > [BrowserEnv] Probing: {url}")

        try:
            self.page.goto(url, wait_until="domcontentloaded", timeout=20000)
            time.sleep(2)
        except Exception as e:
            print(f"  > [BrowserEnv] Load Timeout/Error: {e}")

    def get_console_logs(self):
        return [l for l in self.console_logs if l['type'] in ['error', 'exception']]

    def is_page_empty(self):
        try:
            content = self.page.inner_text("body").strip()
            return len(content) == 0
        except:
            return True

    def capture_observation(self, step_idx, draw_som=True):
        if not self.page: return None
        img_path = os.path.join(self.log_dir, f"visual_step_{step_idx}.png")
        try:
            if draw_som: self.page.evaluate(JS_SOM_SCRIPT)
            self.page.screenshot(path=img_path)
            if draw_som: self.page.evaluate(
                "var oldLabels = document.querySelectorAll('.som-label'); oldLabels.forEach(l => l.remove());")
        except Exception as e:
            print(f"  > [BrowserEnv] Capture Failed: {e}")
        return img_path

    def execute_action(self, action_str):
        # 动作执行逻辑保持不变
        if not self.page: return "Page not loaded"
        viewport_size = self.page.viewport_size
        width, height = viewport_size['width'], viewport_size['height']

        def parse_coordinates(coord_str):
            match = re.search(r'\[\s*([\d\.]+)%?\s*,\s*([\d\.]+)%?\s*\]', coord_str)
            if match:
                x_val, y_val = float(match.group(1)), float(match.group(2))
                if "%" in coord_str or (x_val <= 100 and y_val <= 100):
                    return (x_val / 100.0) * width, (y_val / 100.0) * height
                return x_val, y_val
            return None, None

        try:
            if "Click" in action_str:
                x, y = parse_coordinates(action_str)
                if x is not None and y is not None:
                    self.page.mouse.click(x, y)
                    return f"Clicked coordinate: ({int(x)}, {int(y)})"

                text_match = re.search(r'Click\s*\["(.*?)"\]', action_str)
                if text_match:
                    target_text = text_match.group(1)
                    locator = self.page.locator(f"text='{target_text}'").first
                    if locator.count() > 0:
                        locator.click(timeout=3000, force=True)
                        return f"Clicked text: {target_text}"
                    return f"Action failed: Could not find '{target_text}'"

            elif "Type" in action_str:
                match_id = re.search(r"\[(\d+)\];\s*(.*)", action_str)
                if match_id:
                    idx, text = match_id.group(1), match_id.group(2).strip()
                    locator = self.page.locator(f"[data-som-id='{idx}']")
                    if locator.count() > 0:
                        locator.fill(text, timeout=3000, force=True)
                        self.page.keyboard.press("Enter")
                        return f"Typed in [{idx}]"
                    return f"Action failed: Could not find element [{idx}]"

            elif "Scroll" in action_str:
                direction = 500 if "down" in action_str.lower() else -500
                self.page.mouse.wheel(0, direction)
                return "Scrolled"

            elif "Wait" in action_str:
                time.sleep(2)
                return "Waited"

        except Exception as e:
            return f"Action failed: {str(e)}"
        return "Unknown Action"

    def close(self):
        if self.browser: self.browser.close()
        if self.playwright: self.playwright.stop()
        if self.process: stop_process_tree(self.process)


# ==============================================================================
#  核心改进：无硬编码端口反馈收集
# ==============================================================================
def execute_for_feedback(project_dir, log_dir, cmds=["npm install"], start_cmd="npm run dev", step_idx=None):
    feedback = {"install_error": [], "start_results": "", "start_error": False, "screenshot_path": ""}

    install_res = run_commands(cmds, cwd=project_dir)
    for cmd, out in install_res:
        if "npm ERR!" in out or "EBADENGINE" in out or "Unsupported engine" in out:
            feedback["install_error"].append(f"Issue in '{cmd}':\n{out[-600:]}")

    # 环境初始化：不再传递 app_port
    env = BrowserEnv(project_dir, log_dir, start_cmd)

    try:
        env.start()
        logs = env.get_console_logs()
        if env.is_page_empty() or logs:
            feedback["start_error"] = True
            feedback[
                "start_results"] = f"Runtime Issue! Empty Page: {env.is_page_empty()}, Console Logs: {json.dumps(logs)}"
        else:
            feedback["start_results"] = "Success"
            if step_idx is not None:
                feedback["screenshot_path"] = env.capture_observation(step_idx)

    except Exception as e:
        feedback["start_error"] = True
        log_tail = ""
        log_path = os.path.join(log_dir, "service.log")
        if os.path.exists(log_path):
            try:
                with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                    log_tail = "".join(f.readlines()[-30:])
            except:
                log_tail = "Failed to read service.log"

        feedback["start_results"] = (
            f"CRITICAL: Failed to dynamically start or sniff service.\n"
            f"Technical Error: {str(e)}\n\n"
            f"--- BACKEND SERVICE LOG (Potential compilation/port issues inside) ---\n"
            f"{log_tail}\n"
        )
    finally:
        env.close()

    return feedback


def execute_for_webvoyager_feedback(instruction, project_dir, log_dir, vlm_model, model, cmds=["npm install"],
                                    start_cmd="npm run dev", step_idx=None, max_tokens=-1, max_completion_tokens=-1,
                                    target_path=None):
    from .vlm_generation import vlm_generation, encode_image
    run_commands(cmds, cwd=project_dir)

    env = BrowserEnv(project_dir, log_dir, start_cmd)
    trace, status, grade, suggestions = [], "unknown", 1.0, "Simulation failed."

    try:
        env.start(target_path)
        history_text = ""
        for i in range(6):
            img_path = env.capture_observation(f"user_eval_{i}")
            b64_img = encode_image(img_path)
            response = vlm_generation(
                messages=[
                    {"role": "system", "content": f"You are a QA. Task: {instruction}"},
                    {"role": "user", "content": [{"type": "text", "text": f"History: {history_text}"},
                                                 {"type": "image_url",
                                                  "image_url": {"url": f"data:image/png;base64,{b64_img}"}}]}
                ],
                model=vlm_model
            )
            action = "Wait"
            for line in response.split('\n'):
                if "Action:" in line: action = line.split("Action:", 1)[1].strip(); break
            trace.append({"step": i, "thought": response, "action": action, "screenshot": img_path})
            history_text += f"Step {i}: {action}\n"
            if "Finish" in action:
                status, grade, suggestions = "success", 5.0, "Met."
                break
            env.execute_action(action)
            time.sleep(2)
    except Exception as e:
        status, suggestions = "error", str(e)
    finally:
        env.close()

    return {
        "install_results": [], "install_error": [], "start_results": "", "start_error": False,
        "webvoyager_feedback": {"grade": grade, "improvement_suggestions": suggestions, "trace": trace},
        "webvoyager_text": json.dumps(trace, indent=2),
        "webvoyager_error": "" if status != "error" else suggestions
    }