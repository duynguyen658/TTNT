# PowerShell script to start Python Backend + ngrok
# Usage: .\start-backend.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Starting Plant Disease AI Backend" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python
Write-Host "Checking Python..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "OK: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python not found!" -ForegroundColor Red
    exit 1
}

# Check ngrok
Write-Host ""
Write-Host "Checking ngrok..." -ForegroundColor Yellow
$ngrokAvailable = $false

if (Get-Command ngrok -ErrorAction SilentlyContinue) {
    $ngrokAvailable = $true
    Write-Host "OK: ngrok installed" -ForegroundColor Green
} else {
    Write-Host "WARNING: ngrok not found!" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "ngrok is required to expose your local backend to Vercel." -ForegroundColor Cyan
    Write-Host ""

    $installNgrok = Read-Host "Do you want to install ngrok now? (y/n)"

    if ($installNgrok -eq "y" -or $installNgrok -eq "Y") {
        Write-Host ""
        Write-Host "Attempting to install ngrok..." -ForegroundColor Yellow

        # Check if Chocolatey is available
        if (Get-Command choco -ErrorAction SilentlyContinue) {
            # Check if running as administrator
            $isAdmin = ([System.Security.Principal.WindowsPrincipal] [System.Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([System.Security.Principal.WindowsBuiltInRole]::Administrator)

            if (-not $isAdmin) {
                Write-Host "WARNING: PowerShell is not running as Administrator!" -ForegroundColor Yellow
                Write-Host "Chocolatey installation requires admin privileges." -ForegroundColor Yellow
                Write-Host ""
                Write-Host "Please do one of the following:" -ForegroundColor Cyan
                Write-Host "1. Right-click PowerShell and select 'Run as Administrator'" -ForegroundColor White
                Write-Host "   Then run this script again" -ForegroundColor White
                Write-Host "2. Or install ngrok manually from: https://ngrok.com/download" -ForegroundColor White
                Write-Host ""
                $manualInstall = Read-Host "Open download page in browser? (y/n)"
                if ($manualInstall -eq "y" -or $manualInstall -eq "Y") {
                    Start-Process "https://ngrok.com/download"
                }
                exit 1
            }

            Write-Host "Installing ngrok via Chocolatey..." -ForegroundColor Cyan
            Write-Host "This may take a few minutes..." -ForegroundColor Yellow
            Write-Host ""

            # Run choco install and capture output
            $chocoOutput = choco install ngrok -y 2>&1
            $chocoExitCode = $LASTEXITCODE

            # Check if ngrok is now available
            Start-Sleep -Seconds 2
            $ngrokInstalled = Get-Command ngrok -ErrorAction SilentlyContinue

            if ($ngrokInstalled -and $chocoExitCode -eq 0) {
                Write-Host ""
                Write-Host "OK: ngrok installed successfully!" -ForegroundColor Green
                Write-Host "Refreshing PATH..." -ForegroundColor Yellow
                $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
                Write-Host "You can now continue with the script." -ForegroundColor Green
                $ngrokAvailable = $true
            } else {
                Write-Host ""
                Write-Host "ERROR: Failed to install ngrok via Chocolatey" -ForegroundColor Red
                Write-Host ""
                Write-Host "Common issues:" -ForegroundColor Yellow
                Write-Host "- Lock file error: Another process may be using Chocolatey" -ForegroundColor White
                Write-Host "- Access denied: Need admin privileges" -ForegroundColor White
                Write-Host "- Network issues: Check internet connection" -ForegroundColor White
                Write-Host ""
                Write-Host "Please try one of these:" -ForegroundColor Cyan
                Write-Host "1. Close other PowerShell/Chocolatey windows and try again" -ForegroundColor White
                Write-Host "2. Install manually from: https://ngrok.com/download" -ForegroundColor White
                Write-Host "3. Run: choco install ngrok -y (in admin PowerShell)" -ForegroundColor White
                Write-Host ""
                $manualInstall = Read-Host "Open download page in browser? (y/n)"
                if ($manualInstall -eq "y" -or $manualInstall -eq "Y") {
                    Start-Process "https://ngrok.com/download"
                }
                exit 1
            }
        } else {
            Write-Host "Chocolatey not found. Manual installation required." -ForegroundColor Yellow
            Write-Host ""
            Write-Host "Please install ngrok manually:" -ForegroundColor Cyan
            Write-Host "1. Download from: https://ngrok.com/download" -ForegroundColor White
            Write-Host "2. Extract and add to PATH" -ForegroundColor White
            Write-Host "3. Or install Chocolatey first: https://chocolatey.org/install" -ForegroundColor White
            Write-Host "   Then run: choco install ngrok" -ForegroundColor White
            Write-Host ""
            $openBrowser = Read-Host "Open download page in browser? (y/n)"
            if ($openBrowser -eq "y" -or $openBrowser -eq "Y") {
                Start-Process "https://ngrok.com/download"
            }
            Write-Host ""
            Write-Host "After installing, restart PowerShell and run this script again." -ForegroundColor Yellow
            exit 1
        }
    } else {
        Write-Host ""
        Write-Host "ngrok installation skipped." -ForegroundColor Yellow
        $continueWithout = Read-Host "Continue without ngrok? (y/n)"

        if ($continueWithout -ne "y" -and $continueWithout -ne "Y") {
            Write-Host "Exiting. Please install ngrok and run again." -ForegroundColor Yellow
            exit 1
        }

        Write-Host ""
        Write-Host "WARNING: Backend will run locally only." -ForegroundColor Yellow
        Write-Host "You won't be able to connect it to Vercel without ngrok." -ForegroundColor Yellow
    }
}

# Check api_server.py
if (-not (Test-Path "api_server.py")) {
    Write-Host "ERROR: api_server.py not found!" -ForegroundColor Red
    exit 1
}

# Check and install requirements
if (-not (Test-Path "requirements.txt")) {
    Write-Host "WARNING: requirements.txt not found!" -ForegroundColor Yellow
} else {
    Write-Host ""
    Write-Host "Checking dependencies..." -ForegroundColor Yellow

    # Check if critical packages are installed
    $missingPackages = @()
    $criticalPackages = @("uvicorn", "fastapi", "multipart")

    foreach ($package in $criticalPackages) {
        $checkResult = python -c "import $package" 2>&1
        if ($LASTEXITCODE -ne 0 -or $checkResult -match "ModuleNotFoundError|ImportError") {
            $missingPackages += $package
        }
    }

    if ($missingPackages.Count -gt 0) {
        Write-Host "WARNING: Missing required packages: $($missingPackages -join ', ')" -ForegroundColor Yellow
        Write-Host ""
        $installDeps = Read-Host "Do you want to install dependencies now? (y/n)"

        if ($installDeps -eq "y" -or $installDeps -eq "Y") {
            Write-Host ""
            Write-Host "Installing dependencies from requirements.txt..." -ForegroundColor Cyan
            Write-Host "This may take a few minutes..." -ForegroundColor Yellow
            Write-Host ""

            try {
                python -m pip install --upgrade pip

                # Check for NumPy 2.x compatibility issue and fix it
                Write-Host "Checking NumPy version compatibility..." -ForegroundColor Cyan
                $numpyVersion = python -c "import numpy; print(numpy.__version__)" 2>&1
                if ($numpyVersion -match "^2\.") {
                    Write-Host "WARNING: NumPy 2.x detected. Downgrading to NumPy 1.x for compatibility..." -ForegroundColor Yellow
                    python -m pip install "numpy>=1.24.0,<2.0.0" --force-reinstall
                }

                python -m pip install -r requirements.txt

                if ($LASTEXITCODE -eq 0) {
                    Write-Host ""
                    Write-Host "OK: Dependencies installed successfully!" -ForegroundColor Green
                } else {
                    Write-Host ""
                    Write-Host "ERROR: Failed to install some dependencies" -ForegroundColor Red
                    Write-Host "Please check the error messages above" -ForegroundColor Yellow
                    Write-Host ""
                    $continueAnyway = Read-Host "Continue anyway? (y/n)"
                    if ($continueAnyway -ne "y" -and $continueAnyway -ne "Y") {
                        exit 1
                    }
                }
            } catch {
                Write-Host ""
                Write-Host "ERROR: Failed to install dependencies" -ForegroundColor Red
                Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
                Write-Host ""
                Write-Host "Please install manually: pip install -r requirements.txt" -ForegroundColor Yellow
                exit 1
            }
        } else {
            Write-Host ""
            Write-Host "Dependencies installation skipped." -ForegroundColor Yellow
            Write-Host "The server may fail to start without required packages." -ForegroundColor Yellow
            Write-Host ""
            $continueAnyway = Read-Host "Continue anyway? (y/n)"
            if ($continueAnyway -ne "y" -and $continueAnyway -ne "Y") {
                exit 1
            }
        }
    } else {
        Write-Host "OK: Required packages are installed" -ForegroundColor Green
    }
}

# Check for NumPy compatibility issue before starting
Write-Host ""
Write-Host "Checking NumPy compatibility..." -ForegroundColor Yellow
try {
    $numpyCheck = python -c "import numpy; import pandas; print('OK')" 2>&1
    if ($LASTEXITCODE -ne 0 -or $numpyCheck -match "ImportError|numpy.*2\.") {
        Write-Host "WARNING: NumPy compatibility issue detected!" -ForegroundColor Yellow
        Write-Host "Fixing NumPy version..." -ForegroundColor Cyan
        python -m pip install "numpy>=1.24.0,<2.0.0" --force-reinstall --quiet
        Write-Host "OK: NumPy version fixed" -ForegroundColor Green
    }
} catch {
    # Ignore if check fails, will be caught when starting server
}

# Start Python API
Write-Host ""
Write-Host "Starting Python API Server..." -ForegroundColor Green
Write-Host "API will run at: http://localhost:8000" -ForegroundColor Cyan
Write-Host ""

# Start Python API in background
$pythonProcess = Start-Process python -ArgumentList "api_server.py" -PassThru -NoNewWindow

# Wait for API to start
Write-Host "Waiting for API to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# Check API health
try {
    $healthCheck = Invoke-WebRequest -Uri "http://localhost:8000/health" -TimeoutSec 5 -UseBasicParsing
    if ($healthCheck.StatusCode -eq 200) {
        Write-Host "OK: API is running!" -ForegroundColor Green
    }
} catch {
    Write-Host "WARNING: API might still be starting..." -ForegroundColor Yellow
}

# Start ngrok if available
if ($ngrokAvailable) {
    Write-Host ""
    Write-Host "Starting ngrok tunnel..." -ForegroundColor Green
    Write-Host "ngrok dashboard: http://localhost:4040" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "1. Copy the ngrok URL from above" -ForegroundColor White
    Write-Host "2. Add to Vercel: PYTHON_API_URL=<ngrok-url>" -ForegroundColor White
    Write-Host "3. Press Ctrl+C to stop both services" -ForegroundColor White
    Write-Host ""

    # Start ngrok
    ngrok http 8000
} else {
    Write-Host ""
    Write-Host "Backend is running at: http://localhost:8000" -ForegroundColor Green
    Write-Host "Start ngrok manually: ngrok http 8000" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Press Ctrl+C to stop the API server" -ForegroundColor White

    # Wait for user to stop
    Wait-Process -Id $pythonProcess.Id
}
