# 🧪 Test Backend Endpoint

## ✅ Đã Tạo Endpoint

**Route:** `/api/test-backend`

**Method:** `GET`

**Mục đích:** Test kết nối đến Python backend từ Vercel

## 🧪 Cách Test

### Test Local (Development)

1. **Start dev server:**

   ```powershell
   cd FE/ai-chatbot-main
   npm run dev
   ```

2. **Mở browser:**

   ```
   http://localhost:3000/api/test-backend
   ```

   Hoặc từ terminal:

   ```powershell
   curl http://localhost:3000/api/test-backend
   ```

### Test trên Vercel (Production)

1. **Sau khi deploy, mở:**

   ```
   https://ttnt-henna.vercel.app/api/test-backend
   ```

   Hoặc từ terminal:

   ```powershell
   curl https://ttnt-henna.vercel.app/api/test-backend
   ```

## 📋 Kết Quả Mong Đợi

### ✅ Success Response:

```json
{
  "PYTHON_API_URL": "https://examinations-movements-directors-assists.trycloudflare.com",
  "timestamp": "2026-01-14T...",
  "tests": [
    {
      "name": "Health Check",
      "status": "ok",
      "message": "Backend is healthy. Agents: 5"
    },
    {
      "name": "Chat Endpoint",
      "status": "ok",
      "message": "Chat endpoint is working"
    }
  ],
  "overall": "ok",
  "message": "All backend tests passed!"
}
```

### ❌ Error Response:

```json
{
  "PYTHON_API_URL": "http://localhost:8000",
  "timestamp": "2026-01-14T...",
  "tests": [
    {
      "name": "Health Check",
      "status": "error",
      "message": "Failed to connect to backend"
    }
  ],
  "overall": "error",
  "message": "Some backend tests failed. Check details above."
}
```

## 🔍 Troubleshooting

### Lỗi 404: "Không tìm thấy trang"

**Nguyên nhân:**

- Route chưa được deploy
- Đang test trên production nhưng chưa push code

**Giải pháp:**

1. **Nếu test local:**
   - Đảm bảo dev server đang chạy (`npm run dev`)
   - Check route có trong build output không

2. **Nếu test trên Vercel:**
   - Push code lên GitHub
   - Đợi Vercel deploy xong
   - Test lại

### Lỗi: "PYTHON_API_URL" = "http://localhost:8000"

**Nguyên nhân:**

- `PYTHON_API_URL` chưa được set trên Vercel

**Giải pháp:**

1. Vercel Dashboard → Settings → Environment Variables
2. Add `PYTHON_API_URL` = `https://examinations-movements-directors-assists.trycloudflare.com`
3. Redeploy

### Lỗi: "Failed to connect to backend"

**Nguyên nhân:**

- Backend không chạy
- Cloudflare Tunnel không hoạt động
- URL sai

**Giải pháp:**

1. Check backend: `python api_server.py`
2. Check Cloudflare Tunnel: `cloudflared tunnel --url http://localhost:8000`
3. Test backend connection: `.\test-backend-connection.ps1`

## 📝 Next Steps

1. **Test local trước:**

   ```powershell
   # Terminal 1: Start backend
   python api_server.py

   # Terminal 2: Start Cloudflare Tunnel
   cloudflared tunnel --url http://localhost:8000

   # Terminal 3: Start frontend
   cd FE/ai-chatbot-main
   npm run dev

   # Browser: Test endpoint
   http://localhost:3000/api/test-backend
   ```

2. **Nếu local OK, deploy lên Vercel:**
   - Push code
   - Update `PYTHON_API_URL` trên Vercel
   - Redeploy
   - Test: `https://ttnt-henna.vercel.app/api/test-backend`

3. **Nếu vẫn 404 trên Vercel:**
   - Check Vercel build logs
   - Verify route có trong build output
   - Check middleware không chặn route
