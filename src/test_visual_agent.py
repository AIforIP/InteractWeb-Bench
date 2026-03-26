import os
import time
import json
from playwright.sync_api import sync_playwright

# 引入修改过的模块以及底层现成的环境管理工具
from utils.vlm_generation import vlm_generation, encode_image
from utils.execute_for_feedback import (
    BrowserEnv,
    is_port_open,
    start_background_service,
    stop_process_tree
)
from agent.webgen_agent import INTERNAL_TEST_PROMPT


def extract_instruction_from_jsonl(jsonl_path, task_index=0):
    """从 JSONL 文件中提取指定行的 ground_truth_instruction"""
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"找不到 JSONL 文件: {jsonl_path}")

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    if task_index >= len(lines):
        raise IndexError(f"指定的索引 {task_index} 超出了数据行数 ({len(lines)})")

    task_data = json.loads(lines[task_index])
    instruction = task_data.get("ground_truth_instruction", task_data.get("instruction", ""))
    task_id = task_data.get("id", "Unknown_ID")

    return task_id, instruction


def run_standalone_visual_test(target_url, ground_truth_instruction, project_dir=None, start_cmd=None,
                               vlm_model="gpt-4o-mini", max_steps=5):
    print(f"🚀 启动纯视觉 Agent 单测...")
    print(f"📜 测试指令: {ground_truth_instruction}")

    log_dir = "./standalone_visual_logs"
    os.makedirs(log_dir, exist_ok=True)

    env = BrowserEnv(project_dir=".", log_dir=log_dir, start_cmd="")
    local_server_process = None

    # 动态覆盖 start 方法，融入【自动启动 npm 服务】和【关闭无头模式】的逻辑
    def custom_start(url):
        nonlocal local_server_process

        # 1. 自动管理本地 Node 服务
        if project_dir and start_cmd and not is_port_open(3000):
            print(f"📦 正在目录 {project_dir} 自动执行 '{start_cmd}'...")
            log_file = os.path.join(log_dir, "standalone_service.log")
            local_server_process, _ = start_background_service(start_cmd, project_dir, log_file)

            # 轮询等待端口 3000 开放
            for _ in range(20):
                if is_port_open(3000):
                    print("✅ 本地测试服务已在端口 3000 成功就绪！")
                    break
                time.sleep(1)
            else:
                print("⚠️ 警告: 等待服务启动超时，页面可能无法正常加载。")

        # 2. 启动可见的浏览器界面
        print(f"🔗 正在访问目标网址: {url}")
        env.playwright = sync_playwright().start()
        env.browser = env.playwright.chromium.launch(headless=True)
        env.context = env.browser.new_context(viewport={'width': 1280, 'height': 800})
        env.page = env.context.new_page()
        env.page.goto(url, wait_until="networkidle", timeout=15000)
        env.console_logs = []

    env.start = custom_start

    try:
        env.start(target_url)
        history_text = ""

        for step_idx in range(max_steps):
            print(f"\n--- 🔄 测试步骤 {step_idx + 1}/{max_steps} ---")

            # 传入 draw_som=False，确保截图纯净
            img_path = env.capture_observation(step_idx, draw_som=False)
            b64_img = encode_image(img_path)
            print(f"📸 截图已保存至: {img_path}")

            strict_criteria = (
            )

            sys_msg = INTERNAL_TEST_PROMPT.format(
                instruction=ground_truth_instruction,
                criteria=strict_criteria,
                context_summary="No previous context (Standalone Test).",
                step_idx=step_idx + 1,
                max_steps=max_steps
            )

            step_context = f"Action History:\n{history_text}\n"

            print("🧠 正在等待 VLM 思考决策...")
            response = vlm_generation(
                model=vlm_model,
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": [
                        {"type": "text", "text": step_context},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}}
                    ]}
                ]
            )

            action = "Wait"
            for line in response.split('\n'):
                if "Action:" in line:
                    action = line.split("Action:", 1)[1].strip()
                    break

            # 打印完整的思考过程
            print(f"🧠 Agent 的内心独白:\n\033[93m{response}\033[0m")
            print(f"🤖 Agent 决定执行: \033[92m{action}\033[0m")

            if "Finish" in action or "Fail" in action:
                print(f"🏁 测试结束，状态: {action}")
                break

            # 调用底层的双引擎解析器执行动作
            result = env.execute_action(action)
            print(f"⚙️ 底层执行反馈: {result}")

            history_text += f"Step {step_idx}: {action} (Result: {result})\n"
            time.sleep(2)

    except Exception as e:
        print(f"❌ 测试发生错误: {e}")
    finally:
        print("\n🧹 正在执行环境清理...")
        env.close()
        # 3. 自动化收尾：强制关闭 npm run dev 后台进程
        if local_server_process:
            print("🛑 正在关闭本地 Node 服务进程...")
            stop_process_tree(local_server_process)
            print("✅ 清理完毕。")


if __name__ == "__main__":
    # ================= 配置区域 =================

    # 1. 网页项目源代码目录 (指向你刚刚生成的 000014_P-RAM 工作区)
    PROJECT_DIR = "/home/hhr/home/hhr/experiment_results/gpt-5-mini/workspaces/000014_P-RAM"

    # 2. 启动服务的命令
    START_CMD = "npm run dev"

    # 3. 浏览器要访问的目标地址
    TARGET_WEB_URL = "http://localhost:3000"

    # 4. 数据集路径与索引 (指向你实际的 jsonl 文件)
    JSONL_FILE_PATH = "/home/hhr/home/hhr/src/data/test_mini.jsonl"

    # 注意：你需要将这里的 Index 改为 000014_P-RAM 在 jsonl 文件中实际的行号（从0开始计）
    # 如果只是为了测试视觉能力，你也可以直接将下面 instruction 写死成字符串
    TEST_DATA_INDEX = 0

    # 5. 指定本地视觉模型名称 (必须包含 qwen 以触发你们的动态路由)
    LOCAL_VLM_MODEL = "Qwen/Qwen3-VL-2B-Thinking"

    # ===========================================

    try:
        # 如果 jsonl 提取报错，可以直接将 instruction = "用户具体的网页需求文本"
        task_id, instruction = extract_instruction_from_jsonl(JSONL_FILE_PATH, TEST_DATA_INDEX)
        print(f"=====================================")
        print(f"成功加载任务 ID: {task_id}")
        print(f"=====================================")

        run_standalone_visual_test(
            target_url=TARGET_WEB_URL,
            ground_truth_instruction=instruction,
            project_dir=PROJECT_DIR,
            start_cmd=START_CMD,
            vlm_model=LOCAL_VLM_MODEL  # 强制传入本地模型名称
        )
    except Exception as err:
        print(f"初始化失败: {err}")