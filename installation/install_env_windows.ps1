$ErrorActionPreference = 'Stop'

Write-Host "========================================"
Write-Host "Creating Conda environment: pore-network-viz"
Write-Host "========================================"

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] Conda was not found in PATH. Open Anaconda Prompt or initialize Conda in PowerShell first."
    exit 1
}

Write-Host ""
Write-Host "Checking Ollama installation..."
if (Get-Command ollama -ErrorAction SilentlyContinue) {
    Write-Host "Ollama is already available."
} elseif (Get-Command winget -ErrorAction SilentlyContinue) {
    try {
        Write-Host "Installing Ollama with winget..."
        winget install --id Ollama.Ollama -e --source winget --accept-source-agreements --accept-package-agreements
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Ollama installation finished. Open a new shell if the ollama command is not found immediately."
        } else {
            Write-Host "[WARN] Ollama installation failed with exit code $LASTEXITCODE."
            Write-Host "Install Ollama manually from https://ollama.com/download/windows if you want to use the local provider in AutoPore.py."
        }
    } catch {
        Write-Host "[WARN] Ollama installation failed: $($_.Exception.Message)"
        Write-Host "Install Ollama manually from https://ollama.com/download/windows if you want to use the local provider in AutoPore.py."
    }
} else {
    Write-Host "[WARN] winget was not found, so Ollama was not installed automatically."
    Write-Host "Install Ollama manually from https://ollama.com/download/windows if you want to use the local provider in AutoPore.py."
}

try {
    conda env remove -n pore-network-viz -y | Out-Null
} catch {
}

conda env create -f environment.yml

$condaHook = conda shell.powershell hook | Out-String
Invoke-Expression $condaHook
conda activate pore-network-viz

Write-Host ""
Write-Host "Running verification script..."
python verify_install.py

Write-Host ""
Write-Host "Done. Activate later with: conda activate pore-network-viz"
Write-Host "For local LLM mode, install a tool-capable Ollama model, for example: ollama pull devstral:24b"
