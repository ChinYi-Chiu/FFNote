@echo off
title Auto Setup Environment

echo ========================================
echo   [1/4] Checking Python Environment...
echo ========================================

set "PYTHON_EXE="

:: 1. Check default Python 3.11 path
if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
    set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    echo [OK] Found Python 3.11 in AppData
    goto CREATE_VENV
)

:: 2. Check py launcher for 3.11
py -3.11 -c "import sys" >nul 2>&1
if %errorlevel% equ 0 (
    set "PYTHON_EXE=py -3.11"
    echo [OK] Found Python 3.11 via py launcher
    goto CREATE_VENV
)

:: 3. Check system default python
python -c "import sys; sys.exit(0)" >nul 2>&1
if %errorlevel% equ 0 (
    set "PYTHON_EXE=python"
    echo [OK] Found System Python
    goto CREATE_VENV
)

:: ----------------------------------------------------
:: Download and Install Python 3.11
:: ----------------------------------------------------
echo.
echo [INFO] Python 3.11 not found. Downloading Python 3.11 installer...

set "PY_INSTALLER=python_installer.exe"
set "PY_URL=https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe"

curl -L -o "%PY_INSTALLER%" "%PY_URL%"

if not exist "%PY_INSTALLER%" (
    echo [ERROR] Download failed. Please check your internet connection.
    pause
    exit /b 1
)

echo [INFO] Installing Python 3.11 quietly (takes ~1 min)...
"%PY_INSTALLER%" /quiet SimpleInstall=1 PrependPath=1 Include_test=0
del "%PY_INSTALLER%" >nul 2>&1

set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python auto-installation failed. Please install Python manually.
    pause
    exit /b 1
)

echo [OK] Python 3.11 installed successfully!

:: ----------------------------------------------------
:: Create Virtual Environment (venv)
:: ----------------------------------------------------
:CREATE_VENV
echo.
echo ========================================
echo   [2/4] Checking / Creating venv...
echo ========================================

if exist venv (
    echo [OK] venv directory already exists.
    goto INSTALL_DEP
)

echo [INFO] Creating venv...
%PYTHON_EXE% -m venv venv
if %errorlevel% neq 0 (
    echo [ERROR] Failed to create venv!
    pause
    exit /b 1
)

:INSTALL_DEP
echo.
echo ========================================
echo   [3/4] Installing dependencies...
echo ========================================

echo [INFO] Upgrading pip, setuptools, and wheel...
.\venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
if %errorlevel% neq 0 (
    echo [ERROR] Failed to upgrade pip build tools.
    pause
    exit /b 1
)

if not exist requirements.txt (
    echo [ERROR] requirements.txt not found!
    pause
    exit /b 1
)

echo [INFO] Installing packages from requirements.txt...
.\venv\Scripts\python.exe -m pip install --prefer-binary -r requirements.txt
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Package installation failed!
    echo Please check the error messages above.
    pause
    exit /b 1
)

echo.
echo ========================================
echo   [4/4] Setup Complete!
echo ========================================
echo Run 'run_gui.bat' to start the web interface.
echo.
pause