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
### 2. 在 src/experiment/run_simulation.py 文件中，修改以下参数：
```
DEFAULT_DATA_PATH = r"生成网页指令的jsonl地址"
parser.add_argument("--builder_model", type=str, default="你的测试模型")
parser.add_argument("--visual_copilot_model", type=str, default="你的测试模型")
parser.add_argument("--webvoyager_model", type=str, default="打分模型")
parser.add_argument("--user_model", type=str, default="模拟用户的模型")
```
### 3.  启动模拟测试
# 确保你当前处于项目根目录下
```
python src/experiment/run_simulation.py
# 后台运行方式
nohup python src/experiment/run_simulation.py > run_simulation.log 2>&1 &
```