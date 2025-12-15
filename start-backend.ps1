# Script để khởi động Backend Python và Cloudflare Tunnel
# Usage: .\start-backend.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "🚀 Starting Plant Disease AI Backend" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Kiểm tra Python có được cài đặt không
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found! Please install Python first." -ForegroundColor Red
    exit 1
}

# Kiểm tra Cloudflare Tunnel có được cài đặt không
try {
    $cloudflaredVersion = cloudflared --version 2>&1
    Write-Host "✅ Cloudflare Tunnel found: $cloudflaredVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Cloudflare Tunnel not found! Please install cloudflared first." -ForegroundColor Red
    Write-Host "   Download from: https://github.com/cloudflare/cloudflared/releases" -ForegroundColor Yellow
    Write-Host "   Or install via: choco install cloudflared" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "Starting Python backend on port 8000..." -ForegroundColor Yellow
Start-Process python -ArgumentList "api_server.py" -WindowStyle Normal

Write-Host "Waiting 5 seconds for backend to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

Write-Host ""
Write-Host "Starting Cloudflare Tunnel..." -ForegroundColor Yellow
Write-Host "Choose tunnel mode:" -ForegroundColor Cyan
Write-Host "  1. Quick Tunnel (URL changes each time)" -ForegroundColor White
Write-Host "  2. Named Tunnel (URL fixed, requires setup)" -ForegroundColor White
Write-Host ""
$choice = Read-Host "Enter choice (1 or 2, default: 1)"

if ($choice -eq "2") {
    $tunnelName = Read-Host "Enter tunnel name (default: plant-disease-backend)"
    if ([string]::IsNullOrWhiteSpace($tunnelName)) {
        $tunnelName = "plant-disease-backend"
    }
    Write-Host "Starting named tunnel: $tunnelName" -ForegroundColor Yellow
    Start-Process cloudflared -ArgumentList "tunnel", "run", $tunnelName -WindowStyle Normal
} else {
    Write-Host "Starting quick tunnel..." -ForegroundColor Yellow
    Start-Process cloudflared -ArgumentList "tunnel", "--url", "http://localhost:8000" -WindowStyle Normal
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "✅ Backend and Cloudflare Tunnel are running!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host '📋 Next steps:' -ForegroundColor Yellow
Write-Host '   1. Copy the Cloudflare Tunnel URL from the new window' -ForegroundColor Cyan
Write-Host '   2. Get tunnel URL by running: .\get-tunnel-url.ps1' -ForegroundColor Cyan
Write-Host '   3. Update PYTHON_API_URL in Vercel Environment Variables' -ForegroundColor Cyan
Write-Host ""
