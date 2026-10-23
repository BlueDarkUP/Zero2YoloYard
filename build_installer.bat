@echo off
setlocal
echo ============================================================
echo   Zero2YoloYard - Professional Inno Setup Installer Builder
echo ============================================================

set ISCC="C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
if not exist %ISCC% (
    echo [ERROR] Inno Setup 6 not found at %ISCC%
    exit /b 1
)

if not exist "%~dp0dist\Zero2YoloYard.exe" (
    echo [ERROR] dist\Zero2YoloYard.exe does not exist! Please run build_exe.bat first.
    exit /b 1
)

echo [1/2] Compiling Inno Setup script...
%ISCC% "%~dp0installer\Zero2YoloYard.iss"

if %errorlevel% neq 0 (
    echo [ERROR] Compilation failed!
    exit /b 1
)

echo [2/2] Installer successfully generated!
echo Output: %~dp0installer_output\Zero2YoloYard_Installer_v2.0.exe
echo ============================================================
pause
