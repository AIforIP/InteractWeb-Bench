# 1. 基础镜像
FROM python:3.10-slim

# 2. 设置环境变量，防止交互式安装卡住
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 3. 设置工作目录
WORKDIR /app

# 4. 安装系统基础依赖和 Node.js v22
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    bash \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# 5. 复制依赖清单并安装 Python 库
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. 安装 Playwright 及其 Chromium 内核与系统依赖
RUN playwright install chromium --with-deps

# 7. 复制项目剩余源代码 (包含 src/scripts)
COPY . .

# 8. 默认启动命令
CMD ["python", "src/experiment/run_simulation.py", "--config", "/app/config.yaml"]