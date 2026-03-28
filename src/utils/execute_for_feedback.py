import subprocess
import os
import sys
import time
import re
import socket
import json
import signal
import platform
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


# --- 基础工具函数 ---
def stop_process_tree(proc: subprocess.Popen, timeout: float = 10.0):
    if proc is None or proc.poll() is not None: return
    try:
        if platform.system() == "Windows":
            subprocess.run(f"taskkill /F /T /PID {proc.pid}", shell=True, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            try:
                proc.wait(timeout)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception as e:
        print(f"[WARN] could not stop process: {e}")


def is_port_open(port, host='127.0.0.1'):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex((host, port)) == 0


def force_kill_port(port: int):
    try:
        if platform.system() == "Windows":
            cmd = f'FOR /F "tokens=5" %a in (\'netstat -aon ^| findstr :{port}\') do taskkill /F /PID %a'
            subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.run(f"lsof -ti:{port} | xargs kill -9", shell=True, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
    except Exception:
        pass


def run_commands(cmds, cwd):
    results = []
    for cmd in cmds:
        print(f"Running: {cmd}")
        process = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
        output = process.stdout + (process.stderr or "")

        # 实时探测环境不匹配警告
        if "EBADENGINE" in output or "Unsupported engine" in output:
            print("\033[93m[Warning] Node.js version mismatch detected in output.\033[0m")

        print(output)
        results.append((cmd, output))
    return results


def start_background_service(start_cmd, cwd, port, log_file="service.log"):  # 增加 port 参数
    force_kill_port(port)  # 替换原来的 force_kill_port_3000
    time.sleep(0.5)
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "w", encoding="utf-8")
    kwargs = {'start_new_session': True} if platform.system() != "Windows" else {}
    process = subprocess.Popen(start_cmd, shell=True, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                               stdin=subprocess.DEVNULL, **kwargs)
    return process, log_path


# --- 核心 BrowserEnv ---
class BrowserEnv:
    # 【修复 1】__init__ 增加 app_port，并保存到 self
    def __init__(self, project_dir, log_dir, start_cmd="npm run dev", app_port=3000):
        self.project_dir = project_dir
        self.log_dir = log_dir
        self.start_cmd = start_cmd
        self.app_port = app_port  # <--- 关键！保存传入的端口
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        self.process = None
        self.console_logs = []

    def start(self, target_path=None):
        if not is_port_open(self.app_port):  # 使用动态端口
            log_file = os.path.join(self.log_dir, "service.log")
            self.process, _ = start_background_service(self.start_cmd, self.project_dir, self.app_port, log_file)
            print(f"Waiting for service to start on port {self.app_port} (Log: {log_file})...")
            for _ in range(10):
                if is_port_open(self.app_port): break  # 使用动态端口
                time.sleep(1)

        self.playwright = sync_playwright().start()
        # 1. 对齐沙盒参数，防止在 Docker 或 Linux 环境下崩溃
        self.browser = self.playwright.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"]
        )
        # 2. 对齐 WebVoyager 的分辨率 (1200x800)，确保坐标系绝对一致
        self.context = self.browser.new_context(viewport={'width': 1200, 'height': 800})
        self.page = self.context.new_page()

        # 3. 注入与 WebVoyager 完全一致的全局弹窗拦截器，防止死锁
        def handle_dialog(dialog):
            print(f"  > [Visual Copilot] Auto-accepted Dialog: {dialog.message}")
            dialog.accept()

        self.page.on("dialog", handle_dialog)

        self.page.on("console",
                     lambda msg: self.console_logs.append({"type": msg.type, "text": msg.text}) if msg.type in ["error",
                                                                                                                "warning"] else None)
        self.page.on("pageerror", lambda exc: self.console_logs.append({"type": "exception", "text": str(exc)}))

        # 替换 base_url 中的 3000
        base_url = f"http://localhost:{self.app_port}"
        url = f"{base_url}/{target_path.lstrip('/')}" if target_path else base_url
        print(f"  > [BrowserEnv] Probing: {url}")

        try:
            # 优化等待机制：使用 domcontentloaded 替代 networkidle，防止因一直有后台轮询导致超时
            self.page.goto(url, wait_until="domcontentloaded", timeout=20000)
            time.sleep(2)  # 给予前端框架（React/Vue）水合渲染的时间
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
            if draw_som:
                self.page.evaluate(JS_SOM_SCRIPT)
            self.page.screenshot(path=img_path)
            if draw_som:
                self.page.evaluate(
                    "var oldLabels = document.querySelectorAll('.som-label'); oldLabels.forEach(l => l.remove());")
        except Exception as e:
            print(f"  > [BrowserEnv] Capture Failed: {e}")
        return img_path

    def execute_action(self, action_str):
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
                        # 4. 强制点击：无视 z-index 遮挡层
                        locator.click(timeout=3000, force=True)
                        return f"Clicked text: {target_text}"
                    return f"Action failed: Could not find '{target_text}'"

            elif "Type" in action_str:
                match_id = re.search(r"\[(\d+)\];\s*(.*)", action_str)
                if match_id:
                    idx, text = match_id.group(1), match_id.group(2).strip()
                    locator = self.page.locator(f"[data-som-id='{idx}']")
                    if locator.count() > 0:
                        # 强制清空并输入
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


# --- 核心改进：非阻断式环境反馈 ---
def execute_for_feedback(project_dir, log_dir, cmds=["npm install"], start_cmd="npm run dev", step_idx=None,
                         app_port=3000):
    feedback = {"install_error": [], "start_results": "", "start_error": False, "screenshot_path": ""}
    # 1. 安装环境并嗅探
    install_res = run_commands(cmds, cwd=project_dir)
    for cmd, out in install_res:
        # 捕获 ERR 和关键的环境警告
        if "npm ERR!" in out or "EBADENGINE" in out or "Unsupported engine" in out:
            # 截取最后一部分输出，保证模型能看到版本冲突的详情
            feedback["install_error"].append(f"Issue in '{cmd}':\n{out[-600:]}")

    # 【修复 2】把 app_port 透传给 BrowserEnv!
    env = BrowserEnv(project_dir, log_dir, start_cmd, app_port=app_port)

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
                    lines = f.readlines()
                    log_tail = "".join(lines[-30:])
            except:
                log_tail = "Failed to read service.log"

        feedback["start_results"] = (
            f"CRITICAL: Browser could not connect to localhost:{app_port}.\n"
            f"Technical Error: {str(e)}\n\n"
            f"--- BACKEND SERVICE LOG (Potential version issues inside) ---\n"
            f"{log_tail}\n"
        )
    finally:
        env.close()

    return feedback


# 【修复 3】为兼容函数的签名也加上 app_port=3000，并透传给 BrowserEnv
def execute_for_webvoyager_feedback(instruction, project_dir, log_dir, vlm_model, model, cmds=["npm install"],
                                    start_cmd="npm run dev", step_idx=None, max_tokens=-1, max_completion_tokens=-1,
                                    target_path=None, app_port=3000):
    from .vlm_generation import vlm_generation, encode_image
    run_commands(cmds, cwd=project_dir)

    # 将 app_port 传给 BrowserEnv
    env = BrowserEnv(project_dir, log_dir, start_cmd, app_port=app_port)
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