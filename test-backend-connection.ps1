# Script để test kết nối backend
# Usage: .\test-backend-connection.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Testing Backend Connection" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Lấy PYTHON_API_URL từ env hoặc prompt
$pythonApiUrl = $env:PYTHON_API_URL

if ([string]::IsNullOrWhiteSpace($pythonApiUrl)) {
    Write-Host "PYTHON_API_URL not set in environment." -ForegroundColor Yellow
    $pythonApiUrl = Read-Host "Enter Python API URL"
}

if ([string]::IsNullOrWhiteSpace($pythonApiUrl)) {
    Write-Host "[ERROR] No URL provided!" -ForegroundColor Red
    exit 1
}

Write-Host "Testing URL: $pythonApiUrl" -ForegroundColor Cyan
Write-Host ""

# Test 1: Health Check
Write-Host "[1] Testing Health Endpoint..." -ForegroundColor Yellow
try {
    $healthUrl = "$pythonApiUrl/health"
    Write-Host "   GET $healthUrl" -ForegroundColor Gray

    $response = Invoke-RestMethod -Uri $healthUrl -Method Get -ErrorAction Stop -TimeoutSec 10

    Write-Host "   [OK] Health check passed!" -ForegroundColor Green
    Write-Host "   Response: $($response | ConvertTo-Json -Compress)" -ForegroundColor Gray
} catch {
    Write-Host "   [FAIL] Health check failed!" -ForegroundColor Red
    Write-Host "   Error: $($_.Exception.Message)" -ForegroundColor Red

    if ($_.Exception.Message -like "*Unable to connect*" -or
        $_.Exception.Message -like "*Connection refused*" -or
        $_.Exception.Message -like "*Name or service not known*") {
        Write-Host ""
        Write-Host "   Possible issues:" -ForegroundColor Yellow
        Write-Host "      - Backend is not running" -ForegroundColor Cyan
        Write-Host "      - Cloudflare Tunnel is not running" -ForegroundColor Cyan
        Write-Host "      - URL is incorrect" -ForegroundColor Cyan
    }

    Write-Host ""
    exit 1
}

Write-Host ""

# Test 2: CORS Check
Write-Host "[2] Testing CORS..." -ForegroundColor Yellow
try {
    $corsUrl = "$pythonApiUrl/health"
    $headers = @{
        "Origin" = "https://ttnt-henna.vercel.app"
    }

    Write-Host "   GET $corsUrl (with Origin header)" -ForegroundColor Gray

    $response = Invoke-WebRequest -Uri $corsUrl -Method Get -Headers $headers -ErrorAction Stop -TimeoutSec 10

    $corsHeader = $response.Headers["Access-Control-Allow-Origin"]
    if ($corsHeader) {
        Write-Host "   [OK] CORS is configured!" -ForegroundColor Green
        Write-Host "   Access-Control-Allow-Origin: $corsHeader" -ForegroundColor Gray
    } else {
        Write-Host "   [WARN] CORS header not found (might be OK if using wildcard)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "   [WARN] CORS test failed (might be OK)" -ForegroundColor Yellow
    Write-Host "   Error: $($_.Exception.Message)" -ForegroundColor Gray
}

Write-Host ""

# Test 3: API Chat Endpoint (Simple Test)
Write-Host "[3] Testing API Chat Endpoint..." -ForegroundColor Yellow
try {
    $chatUrl = "$pythonApiUrl/api/chat"
    $body = @{
        user_query = "test"
        user_context = @{}
    } | ConvertTo-Json

    Write-Host "   POST $chatUrl" -ForegroundColor Gray

    $response = Invoke-RestMethod -Uri $chatUrl -Method Post -Body $body -ContentType "application/json" -ErrorAction Stop -TimeoutSec 30

    Write-Host "   [OK] API chat endpoint is working!" -ForegroundColor Green
    Write-Host "   Response status: OK" -ForegroundColor Gray
} catch {
    Write-Host "   [FAIL] API chat endpoint failed!" -ForegroundColor Red
    Write-Host "   Error: $($_.Exception.Message)" -ForegroundColor Red

    if ($_.Exception.Response) {
        $statusCode = $_.Exception.Response.StatusCode.value__
        Write-Host "   Status Code: $statusCode" -ForegroundColor Yellow

        try {
            $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
            $responseBody = $reader.ReadToEnd()
            Write-Host "   Response: $responseBody" -ForegroundColor Gray
        } catch {
            # Ignore
        }
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "[OK] Connection Test Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Summary
Write-Host "Summary:" -ForegroundColor Yellow
Write-Host "   Backend URL: $pythonApiUrl" -ForegroundColor Cyan
Write-Host ""
Write-Host "   If all tests passed:" -ForegroundColor Green
Write-Host "   [OK] Backend is running and accessible" -ForegroundColor Green
Write-Host "   [OK] Update PYTHON_API_URL on Vercel if needed" -ForegroundColor Cyan
Write-Host ""
Write-Host "   If tests failed:" -ForegroundColor Red
Write-Host "   [FAIL] Check backend is running: python api_server.py" -ForegroundColor Cyan
Write-Host "   [FAIL] Check Cloudflare Tunnel is running" -ForegroundColor Cyan
Write-Host "   [FAIL] Verify URL is correct" -ForegroundColor Cyan
Write-Host ""
