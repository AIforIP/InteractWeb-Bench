import os
import sys
import json
import time
import argparse
import subprocess
import atexit
import shutil  # [新增] 用于压缩文件
from flask import Flask, request, Response, stream_with_context, send_file  # [新增] send_file
from flask_cors import CORS

# --- 路径配置 ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 确保能导入 src 下的模块
sys.path.append(os.path.abspath(os.path.join(CURRENT_DIR, "src")))

from agent.webgen_agent import WebGenAgent

app = Flask(__name__)
CORS(app)

# 全局变量
global_agent = None
args = None
preview_process = None


def parse_args():
    parser = argparse.ArgumentParser(description="WebGenAgent Interactive Server")
    parser.add_argument("--workspace_dir", type=str, default=os.path.join(CURRENT_DIR, "workspace"))
    parser.add_argument("--log_dir", type=str, default=os.path.join(CURRENT_DIR, "server_logs"))
    parser.add_argument("--builder_model", type=str, default="gpt-4o")
    parser.add_argument("--webvoyager_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--user_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--max_turns", type=int, default=50)
    return parser.parse_args()


# --- 强力清场函数 ---
def kill_node_processes():
    print("[System] Cleaning up: Killing active Node.js processes...")
    try:
        subprocess.run("taskkill /F /IM node.exe", shell=True, stderr=subprocess.DEVNULL)
    except Exception as e:
        pass


atexit.register(kill_node_processes)


def init_agent(instruction):
    global global_agent, args

    kill_node_processes()
    os.makedirs(args.log_dir, exist_ok=True)

    print(f"[System] Initializing Agent for task: {instruction}")

    # =========================================================================
    # [核心] 强制中文 + 提问策略
    # =========================================================================
    enhanced_instruction = (
        f"User Requirement: {instruction}\n\n"
        "🔴 **CRITICAL INSTRUCTION (READ CAREFULLY)**:\n"
        "You are acting as a Senior Product Manager & Developer.\n"
        "1. **DO NOT** start coding immediately if the user's requirement is vague.\n"
        "2. You **MUST** use the `<boltAction type='ask_user'>` tool to clarify specific details first.\n"
        "3. **LANGUAGE RULE**: You MUST ask questions in **CHINESE** (中文).\n"
        "4. Ask about: **Visual Style**, **Key Features**, or **Layout Preference**.\n"
        "5. **ONLY** start coding after you have received user's confirmation.\n"
        "6. If you absolutely have no questions, start coding."
    )

    global_agent = WebGenAgent(
        model=args.builder_model,
        vlm_model=args.webvoyager_model,
        fb_model=args.user_model,
        workspace_dir=args.workspace_dir,
        log_dir=args.log_dir,
        instruction=enhanced_instruction,
        max_iter=args.max_turns,
        overwrite=True,
        error_limit=5,
        difficulty="hard"
    )
    global_agent.step_idx = 0


# --- [新增接口] 下载代码包 ---
@app.route('/api/download', methods=['GET'])
def download_project():
    global args
    try:
        # 1. 定义压缩包路径 (放在 server 根目录下，叫 project_code.zip)
        zip_filename = "project_code"
        zip_path = os.path.join(CURRENT_DIR, zip_filename)

        # 2. 检查 workspace 是否存在
        if not os.path.exists(args.workspace_dir):
            return Response(json.dumps({"error": "No project found"}), status=404, mimetype='application/json')

        print(f"[System] Zipping workspace: {args.workspace_dir} -> {zip_path}.zip")

        # 3. 创建压缩包 (shutil 会自动加 .zip 后缀)
        shutil.make_archive(zip_path, 'zip', args.workspace_dir)

        # 4. 发送文件
        return send_file(
            f"{zip_path}.zip",
            as_attachment=True,
            download_name="project_code.zip",
            mimetype='application/zip'
        )
    except Exception as e:
        print(f"[Error] Download failed: {e}")
        return Response(json.dumps({"error": str(e)}), status=500, mimetype='application/json')


# --- 手动启动预览服务接口 ---
@app.route('/api/preview/start', methods=['POST'])
def start_preview():
    global args, preview_process
    kill_node_processes()
    try:
        print(f"[System] Starting preview server in: {args.workspace_dir}")
        preview_process = subprocess.Popen(
            ["npm", "run", "dev", "--", "--port", "3000", "--host"],
            cwd=args.workspace_dir,
            shell=True
        )
        return Response(json.dumps({"status": "started", "msg": "Preview started"}), mimetype='application/json')
    except Exception as e:
        return Response(json.dumps({"status": "error", "msg": str(e)}), mimetype='application/json', status=500)


# --- 聊天主接口 ---
@app.route('/api/chat', methods=['POST'])
def chat():
    global global_agent, args
    data = request.json
    user_prompt = data.get('prompt', '')

    if user_prompt.lower() in ["reset", "restart", "clear"]:
        global_agent = None
        kill_node_processes()
        return Response(json.dumps({"content": "会话已重置。"}), mimetype='application/json')

    def generate():
        feedback_payload = None

        if global_agent is None:
            init_agent(user_prompt)
            yield json.dumps({"content": f"🚀 任务开始： {user_prompt}\n"}) + "\n"
            feedback_payload = None
        else:
            yield json.dumps({"content": f"🗣️ 收到回复： {user_prompt}\n"}) + "\n"
            feedback_payload = user_prompt

        while global_agent.step_idx < args.max_turns:
            current_step = global_agent.step_idx
            step_feedback = feedback_payload if (feedback_payload and current_step == global_agent.step_idx) else None
            if step_feedback: feedback_payload = None

            try:
                result, is_failed = global_agent.step(
                    current_step,
                    user_feedback=step_feedback,
                    simulation_mode=False
                )

                res_type = result.get('type', '')
                content = result.get('content', '') or result.get('raw_output', '')

                if res_type == 'question':
                    yield json.dumps({
                        "content": f"\n❓ Agent 提问：\n{content}\n",
                        "is_question": True
                    }) + "\n"
                    print(f"[Agent Question]: {content}")
                    break

                elif result.get('is_finish') or res_type == 'submitted':
                    final_msg = f"\n🎉 任务已完成！\n"
                    if content:
                        final_msg += f"{content}\n"

                    yield json.dumps({
                        "content": final_msg,
                        "action": {"type": "finish", "status": "completed"}
                    }) + "\n"
                    print("[Agent]: Task Finished.")
                    global_agent.is_finished = True
                    break

                else:
                    global_agent.step_idx += 1
                    log_msg = ""
                    if res_type == 'internal_test':
                        status = "不通过" if is_failed else "通过"
                        log_msg = f"👁️ 视觉测试 ({status}): 检查页面 {result.get('content')}\n"
                    elif "npm install" in content:
                        log_msg = "📦 正在安装依赖...\n"
                    elif "Created:" in content or "Modified:" in content:
                        log_msg = f"📝 代码已更新 (请查看右侧预览)\n"
                    else:
                        log_msg = f"🔨 步骤 {current_step}: Coding...\n"

                    yield json.dumps({"content": log_msg}) + "\n"
                    global_agent.save_history(current_step)
                    time.sleep(0.5)
                    continue

            except Exception as e:
                import traceback
                traceback.print_exc()
                yield json.dumps({"content": f"❌ 出错: {str(e)}"}) + "\n"
                break

    return Response(stream_with_context(generate()), mimetype='application/json')


if __name__ == '__main__':
    args = parse_args()
    print(f"Server running on http://127.0.0.1:{args.port}")
    app.run(port=args.port, debug=False)