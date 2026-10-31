
@echo off
title NeuroSeg AI - Brain Tumor Segmentation
color 0B

cd /d "%~dp0"

echo.
echo ==========================================
echo          NeuroSeg AI Platform
echo       Brain Tumor MRI Segmentation
echo ==========================================
echo.

REM Check virtual environment
if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Virtual environment not found!
    echo Please create .venv first.
    pause
    exit /b 1
)

REM Activate environment
call ".venv\Scripts\activate.bat"

echo [1/3] Checking Python environment...
python --version

echo.
echo [2/3] Checking model checkpoint...

if not exist "models\brain_tumor_pretrained_unet_resnet34.pth" (
    echo ERROR: Model checkpoint not found!
    pause
    exit /b 1
)

echo Model checkpoint found.

echo.
echo [3/3] Starting NeuroSeg AI...
echo.
echo Your browser will open at:
echo http://localhost:8501
echo.
echo Keep this window open while using the platform.
echo Press CTRL+C to stop the server.
echo.

python -m streamlit run app.py --server.port 8501

echo.
echo NeuroSeg AI has stopped.
pause

