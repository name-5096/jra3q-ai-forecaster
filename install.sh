#!/bin/bash
set -e

echo -e "\033[36m==========================================================\033[0m"
echo -e "\033[36m 🌐 JRA-3Q 4D Mesoscale & 3D Spherical AI Forecaster Setup\033[0m"
echo -e "\033[36m==========================================================\033[0m"

# 1. Check Python 3
if ! command -v python3 &> /dev/null; then
    echo -e "\033[31m❌ Error: python3 is not installed.\033[0m"
    exit 1
fi

PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo -e "\033[32m✅ Detected Python Version: ${PY_VER}\033[0m"

# 2. Directory prompt (default: $HOME/jra3q-ai-forecaster)
DEFAULT_DIR="$HOME/jra3q-ai-forecaster"
echo ""
read -p "📁 Enter install directory [Press Enter for default: ${DEFAULT_DIR}]: " INPUT_DIR
TARGET_DIR="${INPUT_DIR:-$DEFAULT_DIR}"

echo -e "\033[36m📂 Target directory: ${TARGET_DIR}\033[0m"
mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

# 3. Download core files
echo -e "\033[33m📥 Downloading forecaster core files...\033[0m"
BASE_URL="https://raw.githubusercontent.com/name-5096/jra3q-ai-forecaster/main"

curl -fsSL "${BASE_URL}/converter.py" -o converter.py
curl -fsSL "${BASE_URL}/local_ai.py" -o local_ai.py
curl -fsSL "${BASE_URL}/requirements.txt" -o requirements.txt

# 4. Check already installed packages (Skip if present)
echo -e "\033[33m🔍 Checking installed dependencies...\033[0m"
MISSING_PKGS=$(python3 -c "
import importlib.util
pkgs = {
    'streamlit': 'streamlit',
    'numpy': 'numpy',
    'pandas': 'pandas',
    'plotly': 'plotly',
    'requests': 'requests',
    'tabulate': 'tabulate',
    'google.generativeai': 'google-generativeai',
    'netCDF4': 'netCDF4'
}
missing = [pip_name for mod, pip_name in pkgs.items() if importlib.util.find_spec(mod) is None]
print(' '.join(missing))
")

if [ -z "$MISSING_PKGS" ]; then
    echo -e "\033[32m⚡ All required libraries are already installed! (Skipping pip install)\033[0m"
else
    echo -e "\033[33m📦 Installing missing dependencies: ${MISSING_PKGS}...\033[0m"
    python3 -m pip install $MISSING_PKGS --quiet
fi

echo -e "\033[32m==========================================================\033[0m"
echo -e "\033[32m ✅ Setup Complete! Launching Forecaster UI...\033[0m"
echo -e "\033[32m==========================================================\033[0m"

# 5. Launch Streamlit
python3 -m streamlit run converter.py
