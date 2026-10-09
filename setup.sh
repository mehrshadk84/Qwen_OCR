#!/bin/bash
echo "==================================================="
echo "     Qwen2.5-VL Persian OCR - Auto Setup (Linux)   "
echo "==================================================="

# 1. Check for Python
if ! command -v python3 &> /dev/null
then
    echo "[!] Python3 is not installed. Installing automatically..."
    sudo apt-get update
    sudo apt-get install -y python3 python3-pip python3-venv
fi

# 2. Create Virtual Environment
echo "[1/4] Creating Virtual Environment..."
python3 -m venv venv
source venv/bin/activate

# 3. Upgrade pip
echo "[2/4] Upgrading pip..."
python3 -m pip install --upgrade pip

# 4. Install requirements (Linux standard pip installs PyTorch correctly by default)
echo "[3/4] Installing project dependencies..."
pip install -r requirements.txt

echo "==================================================="
echo "[OK] Installation Complete!"
echo "To run the code, execute:"
echo "source venv/bin/activate && python3 main.py --mode all"
echo "==================================================="