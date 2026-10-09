@echo off
echo ===================================================
echo     Qwen2.5-VL Persian OCR - Auto Setup (Windows)
echo ===================================================

:: 1. Check if Python is installed
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [!] Python is not installed on this system.
    echo Attempting to install Python via Winget...
    winget install -e --id Python.Python.3.11
    echo.
    echo [!] Python has been installed! 
    echo Please close this window, open a new one, and run setup.bat again.
    pause
    exit /b
)

:: 2. Create and activate Virtual Environment
echo [1/4] Creating Virtual Environment...
python -m venv venv
call venv\Scripts\activate.bat

:: 3. Upgrade pip
echo [2/4] Upgrading pip...
python -m pip install --upgrade pip

:: 4. Install PyTorch with CUDA (Essential for Windows GPU)
echo [3/4] Installing PyTorch (CUDA 12.1 for Nvidia GPUs)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

:: 5. Install all project libraries
echo [4/4] Installing project dependencies...
pip install -r requirements.txt

echo ===================================================
echo [OK] Installation Complete!
echo To run the code, use the following command in CMD:
echo venv\Scripts\python main.py --mode all
echo ===================================================
pause