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

### 2. 🔑 环境变量配置

本项目使用 `.env` 文件来管理 API 密钥。请参考根目录下的 `.env.example` 文件进行配置。

1. **创建配置文件**：
   ```bash
   cp .env.example .env
   ```

2. **编辑 `.env` 文件**，填入你的 API 信息：

```text
# LLM 基础配置
OPENAILIKE_API_KEY="your_key"
OPENAILIKE_BASE_URL="api_url"

# VLM 配置
OPENAILIKE_VLM_API_KEY="your_key"
OPENAILIKE_VLM_BASE_URL="api_url"

# 反馈模型配置
OPENAILIKE_FB_API_KEY="your_key"
OPENAILIKE_FB_BASE_URL="api_url"

# Anthropic 视觉模型配置 (可选)
ANTHROPIC_VLM_API_KEY="your_key"
ANTHROPIC_VLM_BASE_URL="api_url"
```

### 3. 在 src/experiment/run_simulation.py 文件中，修改参数：
```python
DEFAULT_DATA_PATH = r"生成网页指令的jsonl地址"
parser.add_argument("--builder_model", type=str, default="你的测试模型")
parser.add_argument("--visual_copilot_model", type=str, default="你的测试模型")
parser.add_argument("--webvoyager_model", type=str, default="打分模型")
parser.add_argument("--user_model", type=str, default="模拟用户的模型")
```

### 4. 启动模拟测试
```bash
# 确保你当前处于项目根目录下
python src/experiment/run_simulation.py
# 后台运行方式
nohup python src/experiment/run_simulation.py > run_simulation.log 2>&1 &
```
