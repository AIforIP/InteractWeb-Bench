# InteractWeb-Bench

InteractWeb-Bench 是一个专为网页生成智能体（Web Generation Agents）设计的交互式评估基准测试。

## 🛠️ 安装与环境配置

请按照以下步骤设置运行环境并安装所需的依赖项。

### 1. 创建并激活 Conda 环境

```bash
conda create -n InteractWeb-Bench python=3.10 -y
conda activate InteractWeb-Bench
pip install -r requirements.txt
playwright install chromium
```
安装Node.js
```bash
chmod +x install_node.sh
./install_node.sh
```
或者直接用 bash 运行（不需要赋权）：
```bash
bash install_node.sh
```
### 2. 🔑 环境变量配置

本项目使用 `.env` 文件来管理 API 密钥。请参考根目录下的 `.env.example` 文件进行配置。

1. **创建配置文件**：
```bash
cp .env.example .env
```

2. **编辑 `.env` 文件**，填入你的 API 信息：

```text
# 1. 定义底层计算节点端口 (代表服务器上启动的独立 vLLM 实例)
LOCAL_NODE_1_PORT=8024
LOCAL_NODE_2_PORT=8025

# 2. 动态路由映射表 (将具体的模型名指向对应的计算节点)
LOCAL_MODELS_MAP="Qwen3-VL-2B-Thinking=http://localhost:${LOCAL_NODE_1_PORT}/v1,Qwen3.5-9B=http://localhost:${LOCAL_NODE_2_PORT}/v1"

# ====== 兜底大锅饭配置 (Fallback) ======
OPENAILIKE_API_KEY="your_key"
OPENAILIKE_BASE_URL="api_url"

OPENAILIKE_VLM_API_KEY="your_key"
OPENAILIKE_VLM_BASE_URL="api_url"

ANTHROPIC_VLM_API_KEY="your_key"
ANTHROPIC_VLM_BASE_URL="api_url"
#独立配置不用请注释掉
# ====== 1. Builder (写代码) 独立配置 ======
BUILDER_API_KEY="your_builder_key"
BUILDER_BASE_URL="builder_api_url"

# ====== 2. Visual Copilot (看图改Bug) 独立配置 ======
COPILOT_API_KEY="your_copilot_key"
COPILOT_BASE_URL="copilot_api_url"

# ====== 3. WebVoyager (最终打分) 独立配置 ======
WEBVOYAGER_API_KEY="your_key"
WEBVOYAGER_BASE_URL="api_url"

# ====== 4. UserSimulator (模拟甲方) 独立配置 ======
USER_MODEL_API_KEY="your_key"
USER_MODEL_BASE_URL="api_url"
```

### 3. 修改配置文件 (config.yaml)：
在项目根目录下编辑 config.yaml 文件，指定数据路径与模型参数：
```YALM
# 数据路径配置
data_path: "数据jsonl路径"

# 模型选择配置
models:
  builder_model: "测试模型"
  visual_copilot_model: "测试模型"
  webvoyager_model: "打分模型"
  user_model: "用户模型"
```
### 4. 本地模型部署 (可选)：
如果使用本地模型进行评测，需提前安装 vllm 包：
```bash
pip install vllm
```
目前底层代码已自动适配 qwen（端口 8024）和 llama（端口 8025）。请在项目中编写对应的 .sh 启动脚本。以 Qwen 为例（例如保存为 src/scripts/deploy_qwen_3b_thinking.sh）：
```bash
#!/bin/bash
# 自动清理 8024 端口，防止启动冲突
fuser -k 8024/tcp >/dev/null 2>&1

MODEL_PATH="/your_model_address/"

# 使用 Python 模块的绝对路径启动
python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --served-model-name "Qwen/Qwen3-VL-2B-Thinking" \
    --port 8024 \
    --host 0.0.0.0 \
    --trust-remote-code \
    --dtype bfloat16 \
    --max-model-len 128000 \
    --limit-mm-per-prompt '{"image": 5}' \
    --gpu-memory-utilization 0.7 \
    --enforce-eager
```
### 5. 启动模拟测试
确保当前处于项目根目录下（即 src 的上一级文件夹）。

方式一：纯云端模型调用，直接运行主程序：
```bash
# 确保你当前处于项目根目录下
python src/experiment/run_simulation.py --config /your_config_address/
```
方式二：使用本地模型 (双终端模式)，需要打开两个终端窗口，均激活环境并进入项目根目录：
终端 1（启动本地大模型服务）：
```bash
bash src/scripts/deploy_qwen_3b_thinking.sh
```
终端 2（启动主测试程序）：
```bash
python src/experiment/run_simulation.py
```
### 6. Docker 部署与评测指南
请获取系统配套的 Docker 镜像压缩包 `interactweb-bench_v1.0.tar`，并在目标服务器的终端执行以下命令加载镜像：

```bash
docker load -i interactweb-bench_v1.0.tar
```
配置运行容器
```bash
docker_run.sh
```
进入docker后激活环境后按照
```bash
# 确保你当前处于项目根目录(InteractWeb-Bench)下
python src/experiment/run_simulation.py --config /your_config_address/
```