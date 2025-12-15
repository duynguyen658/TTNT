# Script để lấy Cloudflare Tunnel URL
# Usage: .\get-tunnel-url.ps1

Write-Host "Fetching Cloudflare Tunnel information..." -ForegroundColor Yellow
Write-Host ""

# Kiểm tra xem có quick tunnel đang chạy không
$quickTunnelProcess = Get-Process -Name "cloudflared" -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like "*--url*" }

if ($quickTunnelProcess) {
    Write-Host "⚠️  Quick tunnel detected. URL is displayed in the cloudflared window." -ForegroundColor Yellow
    Write-Host "   Please check the cloudflared window for the URL (format: https://xxxx-xxxx-xxxx.trycloudflare.com)" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "   Or check the cloudflared logs for the URL." -ForegroundColor Cyan
    Write-Host ""
} else {
    # Kiểm tra named tunnel
    $namedTunnelProcess = Get-Process -Name "cloudflared" -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like "*tunnel run*" }

    if ($namedTunnelProcess) {
        Write-Host "✅ Named tunnel detected!" -ForegroundColor Green
        Write-Host ""
        Write-Host "📋 To get your tunnel URL:" -ForegroundColor Yellow
        Write-Host "   1. Check your Cloudflare Dashboard" -ForegroundColor Cyan
        Write-Host "   2. Or check your DNS records for the hostname" -ForegroundColor Cyan
        Write-Host "   3. Or check the cloudflared window for connection status" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "   Named tunnel URL format: https://your-subdomain.yourdomain.com" -ForegroundColor White
        Write-Host ""
    } else {
        Write-Host "❌ No active Cloudflare Tunnel found!" -ForegroundColor Red
        Write-Host ""
        Write-Host "Make sure:" -ForegroundColor Yellow
        Write-Host "   1. Cloudflare Tunnel is running" -ForegroundColor Cyan
        Write-Host "   2. For quick tunnel: cloudflared tunnel --url http://localhost:8000" -ForegroundColor Cyan
        Write-Host "   3. For named tunnel: cloudflared tunnel run <tunnel-name>" -ForegroundColor Cyan
        Write-Host ""
        exit 1
    }
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "📋 Update this URL in Vercel:" -ForegroundColor Yellow
Write-Host "   1. Go to Vercel Dashboard" -ForegroundColor Cyan
Write-Host "   2. Select your project" -ForegroundColor Cyan
Write-Host "   3. Settings → Environment Variables" -ForegroundColor Cyan
Write-Host "   4. Update PYTHON_API_URL with the tunnel URL" -ForegroundColor Cyan
Write-Host "   5. Redeploy your project" -ForegroundColor Cyan
Write-Host ""
