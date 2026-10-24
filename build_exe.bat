@echo off
setlocal

set PROJECT=C:\Users\BlueDarkUP\PycharmProjects\Zero2YoloYard
set PYTHON=%PROJECT%\.venv\Scripts\python.exe
set PYINSTALLER=%PROJECT%\.venv\Scripts\pyinstaller.exe

set KMP_DUPLICATE_LIB_OK=TRUE
set TF_ENABLE_ONEDNN_OPTS=0

echo ============================================================
echo   Zero2YoloYard - Single File EXE Build
echo   Python: %PYTHON%
echo ============================================================

REM Clean build directories
echo [1/3] Cleaning build directories...
if exist "%PROJECT%\build\Zero2YoloYard" rd /s /q "%PROJECT%\build\Zero2YoloYard"
if exist "%PROJECT%\dist\Zero2YoloYard" rd /s /q "%PROJECT%\dist\Zero2YoloYard"
if exist "%PROJECT%\dist\Zero2YoloYard.exe" del /f /q "%PROJECT%\dist\Zero2YoloYard.exe"

REM Run PyInstaller
echo [2/3] Running PyInstaller...
"%PYINSTALLER%" --clean --noconfirm "%PROJECT%\Zero2YoloYard.spec"

if %errorlevel% neq 0 (
    echo [ERROR] Build failed!
    exit /b 1
)

echo [3/3] Build finished successfully!
echo Output: %PROJECT%\dist\Zero2YoloYard\Zero2YoloYard.exe
