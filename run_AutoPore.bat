@echo off
setlocal
cd /d "%~dp0"
set "PYTHON_EXE=%USERPROFILE%\.conda\envs\pore-network-viz\python.exe"
if exist "%PYTHON_EXE%" (
    "%PYTHON_EXE%" AutoPore.py
) else (
    python AutoPore.py
)
endlocal
