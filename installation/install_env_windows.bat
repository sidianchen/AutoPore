@echo off
setlocal

echo ========================================
echo Creating Conda environment: pore-network-viz
echo ========================================

where conda >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Conda was not found in PATH.
    echo Open Anaconda Prompt or Miniconda Prompt and run this script again.
    pause
    exit /b 1
)

echo.
echo Checking Ollama installation...
where ollama >nul 2>nul
if errorlevel 1 (
    where winget >nul 2>nul
    if errorlevel 1 (
        echo [WARN] winget was not found, so Ollama was not installed automatically.
        echo Install Ollama manually from https://ollama.com/download/windows if you want to use the local provider in AutoPore.py.
    ) else (
        echo Installing Ollama with winget...
        winget install --id Ollama.Ollama -e --source winget --accept-source-agreements --accept-package-agreements
        if errorlevel 1 (
            echo [WARN] Ollama installation failed.
            echo Install Ollama manually from https://ollama.com/download/windows if you want to use the local provider in AutoPore.py.
        ) else (
            echo Ollama installation finished. Open a new shell if the ollama command is not found immediately.
        )
    )
) else (
    echo Ollama is already available.
)

call conda env remove -n pore-network-viz -y >nul 2>nul
call conda env create -f environment.yml
if errorlevel 1 (
    echo [ERROR] Failed to create the environment.
    pause
    exit /b 1
)

call conda activate pore-network-viz
if errorlevel 1 (
    echo [ERROR] Failed to activate the environment.
    echo Try running manually:
    echo   conda activate pore-network-viz
    pause
    exit /b 1
)

echo.
echo Running verification script...
python verify_install.py

echo.
echo Done.
echo To use this environment later:
 echo   conda activate pore-network-viz
echo For local LLM mode, install a tool-capable Ollama model, for example:
echo   ollama pull devstral:24b
pause
