# Script để lấy Ngrok public URL
# Usage: .\get-ngrok-url.ps1

Write-Host "Fetching Ngrok tunnel information..." -ForegroundColor Yellow
Write-Host ""

try {
    $response = Invoke-RestMethod -Uri "http://localhost:4040/api/tunnels" -Method Get -ErrorAction Stop

    if ($response.tunnels.Count -eq 0) {
        Write-Host "❌ No active Ngrok tunnels found!" -ForegroundColor Red
        Write-Host "   Make sure Ngrok is running: ngrok http 8000" -ForegroundColor Yellow
        exit 1
    }

    $tunnel = $response.tunnels[0]
    $publicUrl = $tunnel.public_url
    $config = $tunnel.config

    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "🌐 Ngrok Tunnel Information" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Public URL: " -NoNewline -ForegroundColor Yellow
    Write-Host "$publicUrl" -ForegroundColor Green
    Write-Host ""
    Write-Host "Forwarding: " -NoNewline -ForegroundColor Yellow
    Write-Host "$($tunnel.config.addr) -> $publicUrl" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "📋 Update this URL in Vercel:" -ForegroundColor Yellow
    Write-Host "   1. Go to Vercel Dashboard" -ForegroundColor Cyan
    Write-Host "   2. Select your project" -ForegroundColor Cyan
    Write-Host "   3. Settings → Environment Variables" -ForegroundColor Cyan
    Write-Host "   4. Update PYTHON_API_URL = $publicUrl" -ForegroundColor Cyan
    Write-Host "   5. Redeploy your project" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Or set it directly:" -ForegroundColor Yellow
    Write-Host "   `$env:PYTHON_API_URL='$publicUrl'" -ForegroundColor Green
    Write-Host ""

    # Copy to clipboard (optional)
    try {
        $publicUrl | Set-Clipboard
        Write-Host "✅ URL copied to clipboard!" -ForegroundColor Green
    } catch {
        # Clipboard not available, skip
    }

} catch {
    Write-Host "❌ Error connecting to Ngrok API!" -ForegroundColor Red
    Write-Host "   Error: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host ""
    Write-Host "Make sure:" -ForegroundColor Yellow
    Write-Host "   1. Ngrok is running: ngrok http 8000" -ForegroundColor Cyan
    Write-Host "   2. Ngrok web interface is accessible at http://localhost:4040" -ForegroundColor Cyan
    exit 1
}
