#!/bin/bash

# 1. Install NVM (if not already installed)
if [ ! -d "$HOME/.nvm" ]; then
    echo "Installing NVM..."
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
fi

# 2. Load NVM environment
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

# 3. Install the required Node.js version for the project (v22.14.0)
echo "Installing Node v22.14.0..."
nvm install 22.14.0

# 4. Verify version switch
nvm use 22.14.0
echo "Current Node version for this project: $(node -v)"
echo "The system default Node version is still available. You can switch back using 'nvm use system'"