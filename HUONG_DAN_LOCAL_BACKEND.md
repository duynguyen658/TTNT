# 🖥️ Hướng Dẫn Chạy Backend Local + Vercel Frontend

## 📋 Tổng Quan

- **Backend Python**: Chạy trên máy local của bạn
- **Frontend Next.js**: Deploy trên Vercel
- **Kết nối**: Dùng tunneling service (ngrok/Cloudflare Tunnel) để expose local backend ra internet

## ⚠️ Lưu Ý Quan Trọng

Vercel (cloud) **KHÔNG THỂ** gọi trực tiếp về `localhost:8000` vì:

- Localhost chỉ accessible từ máy local
- Vercel chạy trên cloud, không thể reach local network

**Giải pháp**: Dùng tunneling service để tạo public URL trỏ về localhost.

## 🚀 Cách 1: Dùng ngrok (Khuyến nghị - Dễ nhất)

### Bước 1: Cài đặt ngrok

**Windows:**

```powershell
# Download từ https://ngrok.com/download
# Hoặc dùng Chocolatey
choco install ngrok

# Hoặc dùng Scoop
scoop install ngrok
```

**Hoặc download trực tiếp:**

1. Vào https://ngrok.com/download
2. Download Windows version
3. Giải nén và thêm vào PATH

### Bước 2: Đăng ký ngrok (Free)

1. Tạo account tại https://dashboard.ngrok.com/signup
2. Lấy **authtoken** từ dashboard
3. Chạy lệnh:

```bash
ngrok config add-authtoken YOUR_AUTH_TOKEN
```

### Bước 3: Start Backend Python

```bash
# Terminal 1: Start Python API
python api_server.py
```

Backend sẽ chạy tại: `http://localhost:8000`

### Bước 4: Start ngrok Tunnel

```bash
# Terminal 2: Start ngrok
ngrok http 8000
```

ngrok sẽ cung cấp URL công khai, ví dụ:

```
Forwarding  https://abc123.ngrok-free.app -> http://localhost:8000
```

**Copy URL này** (ví dụ: `https://abc123.ngrok-free.app`)

### Bước 5: Cấu hình Vercel

1. Vào Vercel Dashboard → Project → Settings → Environment Variables
2. Thêm biến:
   ```
   PYTHON_API_URL=https://abc123.ngrok-free.app
   ```
3. Redeploy project

### Bước 6: Test

1. Mở Vercel app
2. Test chat với AI
3. Kiểm tra logs trong Terminal 1 (Python) và Terminal 2 (ngrok)

## 🔄 Cách 2: Dùng Cloudflare Tunnel (Free, Ổn định hơn)

### Bước 1: Cài đặt cloudflared

**Windows:**

```powershell
# Download từ https://github.com/cloudflare/cloudflared/releases
# Hoặc dùng Chocolatey
choco install cloudflared
```

### Bước 2: Login Cloudflare

```bash
cloudflared tunnel login
```

### Bước 3: Tạo Tunnel

```bash
# Tạo tunnel mới
cloudflared tunnel create my-backend

# List tunnels
cloudflared tunnel list
```

### Bước 4: Chạy Tunnel

```bash
# Terminal 1: Start Python API
python api_server.py

# Terminal 2: Start Cloudflare Tunnel
cloudflared tunnel --url http://localhost:8000
```

Cloudflare sẽ cung cấp URL, ví dụ:

```
https://random-name.trycloudflare.com
```

### Bước 5: Cấu hình Vercel

Thêm `PYTHON_API_URL` với Cloudflare URL vào Vercel environment variables.

## 📝 Script Tự Động (Windows PowerShell)

Tạo file `start-backend.ps1`:

```powershell
# Start Backend + ngrok tự động
Write-Host "🚀 Starting Python Backend..." -ForegroundColor Green

# Start Python API trong background
Start-Process python -ArgumentList "api_server.py" -WindowStyle Minimized

# Đợi 3 giây để API khởi động
Start-Sleep -Seconds 3

Write-Host "🌐 Starting ngrok tunnel..." -ForegroundColor Green

# Start ngrok
ngrok http 8000
```

Chạy:

```powershell
.\start-backend.ps1
```

## 🔧 Cấu Hình Backend

### CORS Configuration

Backend đã được cấu hình để cho phép tất cả origins. Nếu muốn giới hạn:

```python
# Trong api_server.py, thay đổi:
cors_origins = [
    "https://your-app.vercel.app",
    "https://*.vercel.app",
    "https://*.ngrok-free.app",
    "https://*.trycloudflare.com",
]
```

Hoặc set environment variable:

```bash
$env:CORS_ORIGINS="https://your-app.vercel.app,https://*.vercel.app"
```

### Environment Variables (Local)

Tạo file `.env` trong thư mục gốc:

```env
# LLM Provider
LLM_PROVIDER=groq

# API Keys
GROQ_API_KEY=your-groq-api-key
# HOẶC
OPENAI_API_KEY=your-openai-api-key

# YOLO Model
YOLO_MODEL_PATH=models/yolo_detection_s.pt

# CORS (optional)
CORS_ORIGINS=https://your-app.vercel.app,https://*.vercel.app
```

## ⚡ Tips & Best Practices

### 1. Giữ ngrok/Cloudflare Tunnel chạy

- Đảm bảo tunnel luôn chạy khi Vercel cần gọi API
- Nếu tunnel dừng, Vercel sẽ không thể kết nối

### 2. ngrok Free Plan Limitations

- URL thay đổi mỗi lần restart (trừ khi dùng paid plan)
- Cần update Vercel env var mỗi lần URL thay đổi
- Có rate limits

**Giải pháp**: Dùng ngrok paid plan hoặc Cloudflare Tunnel (free, ổn định hơn)

### 3. Security

- ngrok free plan có warning page (có thể bỏ qua)
- Cloudflare Tunnel an toàn hơn
- Có thể thêm authentication nếu cần

### 4. Monitoring

Xem logs:

- **Python API**: Terminal 1
- **ngrok**: Terminal 2 hoặc http://localhost:4040 (ngrok dashboard)
- **Vercel**: Vercel dashboard → Logs

## 🐛 Troubleshooting

### Lỗi: "Connection refused"

- **Nguyên nhân**: Backend chưa chạy hoặc tunnel chưa start
- **Giải pháp**:
  1. Kiểm tra `python api_server.py` đang chạy
  2. Kiểm tra ngrok/Cloudflare tunnel đang chạy
  3. Test: `curl http://localhost:8000/health`

### Lỗi: "CORS error"

- **Nguyên nhân**: Backend không cho phép Vercel origin
- **Giải pháp**:
  1. Kiểm tra `CORS_ORIGINS` trong `.env`
  2. Hoặc set `CORS_ORIGINS=*` để cho phép tất cả

### Lỗi: "ngrok URL changed"

- **Nguyên nhân**: ngrok free plan tạo URL mới mỗi lần restart
- **Giải pháp**:
  1. Update `PYTHON_API_URL` trong Vercel
  2. Redeploy Vercel app
  3. Hoặc dùng ngrok paid plan để có static domain

### Lỗi: "Tunnel timeout"

- **Nguyên nhân**: Tunnel bị disconnect
- **Giải pháp**:
  1. Restart ngrok/Cloudflare tunnel
  2. Kiểm tra internet connection
  3. Dùng Cloudflare Tunnel (ổn định hơn)

## 📊 So Sánh Tunneling Services

| Service           | Free Plan | Static URL | Stability | Setup      |
| ----------------- | --------- | ---------- | --------- | ---------- |
| ngrok             | ✅        | ❌         | ⭐⭐⭐    | Dễ         |
| Cloudflare Tunnel | ✅        | ✅         | ⭐⭐⭐⭐  | Trung bình |
| localtunnel       | ✅        | ❌         | ⭐⭐      | Dễ         |

## 🎯 Next Steps

1. ✅ Chọn tunneling service (ngrok hoặc Cloudflare)
2. ✅ Start backend local
3. ✅ Start tunnel
4. ✅ Cấu hình Vercel environment variable
5. ✅ Test integration

## 💡 Alternative: Chạy Cả Frontend + Backend Local

Nếu không muốn dùng tunneling, có thể chạy cả frontend local:

```bash
# Terminal 1: Backend
python api_server.py

# Terminal 2: Frontend
cd FE/ai-chatbot-main
npm run dev
```

Frontend sẽ chạy tại `http://localhost:3000` và tự động kết nối với `http://localhost:8000`.
