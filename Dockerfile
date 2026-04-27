# 1. Base image
FROM python:3.10-slim

# 2. Set environment variables to prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 3. Set working directory
WORKDIR /app

# 4. Install system dependencies and Node.js v22
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    bash \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# 5. Copy dependency file and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Install Playwright along with Chromium and required system dependencies
RUN playwright install chromium --with-deps

# 7. Copy the rest of the project source code (including src/scripts)
COPY . .

# 8. Default startup command
CMD ["python", "src/experiment/run_simulation.py", "--config", "/app/config.yaml"]