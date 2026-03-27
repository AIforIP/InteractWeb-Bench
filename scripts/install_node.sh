#!/bin/bash

# 1. 安装 NVM (如果尚未安装)
if [ ! -d "$HOME/.nvm" ]; then
    echo "正在安装 NVM..."
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
fi

# 2. 加载 NVM 环境
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

# 3. 安装项目需要的指定版本 (v22.14.0)
echo "正在安装 Node v22.14.0..."
nvm install 22.14.0

# 4. 验证版本切换
nvm use 22.14.0
echo "当前项目使用的 Node 版本: $(node -v)"
echo "系统原始版本依然存在，你可以通过 'nvm use system' 切回 v12"