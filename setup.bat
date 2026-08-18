@echo off
chcp 65001 > nul
echo ========================================
echo   開始建置 Python 虛擬環境與安裝套件
echo ========================================

if not exist venv (
    echo [1/3] 建立 venv 虛擬環境...
    python -m venv venv
) else (
    echo [1/3] venv 環境已存在，跳過建立。
)

echo [2/3] 升級 pip...
.\venv\Scripts\python.exe -m pip install --upgrade pip

if exist requirements.txt (
    echo [3/3] 安裝第三方套件...
    .\venv\Scripts\python.exe -m pip install -r requirements.txt
) else (
    echo [ERR] 找不到 requirements.txt！
    pause
    exit /b 1
)

echo.
echo ========================================
echo   環境安裝成功！執行 run_gui.bat 即可啟動
echo ========================================
pause