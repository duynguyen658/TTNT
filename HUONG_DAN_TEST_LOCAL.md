# 🧪 Hướng Dẫn Test Local (FE + BE)

## 📋 Tổng Quan

Test toàn bộ hệ thống trên máy local trước khi deploy lên Vercel:

- **Frontend (Next.js)**: Chạy tại `http://localhost:3000`
- **Backend (Python)**: Chạy tại `http://localhost:8000`
- **Kết nối**: FE gọi trực tiếp BE qua `http://localhost:8000`

## 🚀 Bước 1: Start Backend Python

### Cách 1: Dùng Script Tự Động (Khuyến nghị)

```powershell
# Từ thư mục gốc D:\TTNT2
.\start-backend.ps1
```

Script sẽ:

- ✅ Kiểm tra Python
- ✅ Kiểm tra dependencies
- ✅ Tự động cài đặt nếu thiếu
- ✅ Start Python API tại `http://localhost:8000`

### Cách 2: Chạy Thủ Công

```powershell
# Terminal 1: Start Python API
python api_server.py
```

Backend sẽ chạy tại: `http://localhost:8000`

**Kiểm tra backend:**

```powershell
# Test health check
curl http://localhost:8000/health
# Hoặc mở browser: http://localhost:8000/health
```

## 🎨 Bước 2: Cấu Hình Frontend

### 2.1. Tạo file `.env.local`

Tạo file `FE/ai-chatbot-main/.env.local`:

```env
# Python Backend URL (local)
PYTHON_API_URL=http://localhost:8000

# NextAuth
AUTH_SECRET=your-secret-key-here
NEXTAUTH_URL=http://localhost:3000

# Database (nếu cần)
POSTGRES_URL=your-postgres-url

# AI Gateway (nếu dùng)
AI_GATEWAY_API_KEY=your-key
```

**Lưu ý**:

- `PYTHON_API_URL=http://localhost:8000` - Kết nối trực tiếp với BE local
- Không cần ngrok khi test local

### 2.2. Cài đặt Dependencies

```powershell
cd FE/ai-chatbot-main
npm install
# hoặc
pnpm install
```

## 🚀 Bước 3: Start Frontend

```powershell
# Từ thư mục FE/ai-chatbot-main
npm run dev
# hoặc
pnpm dev
```

Frontend sẽ chạy tại: `http://localhost:3000`

## ✅ Bước 4: Test Kết Nối

### 4.1. Test Backend Trực Tiếp

```powershell
# Health check
curl http://localhost:8000/health

# Test API chat
curl -X POST http://localhost:8000/api/chat `
  -H "Content-Type: application/json" `
  -d '{"user_query": "Cây cà chua bị vàng lá", "user_context": {}}'
```

### 4.2. Test Frontend API Route

Mở browser:

- `http://localhost:3000/api/plant-ai` (GET) - Health check
- `http://localhost:3000/api/test` - Test route

### 4.3. Test Toàn Bộ Flow

1. Mở browser: `http://localhost:3000`
2. Đăng nhập hoặc tạo account
3. Tạo chat mới
4. Gửi message: "Cây cà chua của tôi bị vàng lá, xin tư vấn"
5. Kiểm tra:
   - Frontend có gọi `/api/plant-ai` không?
   - Backend có nhận request không?
   - Response có trả về đúng không?

## 🔍 Debug Nếu Có Lỗi

### Lỗi: "Connection refused" hoặc "ECONNREFUSED"

**Nguyên nhân**: Backend chưa chạy hoặc port sai

**Giải pháp**:

1. Kiểm tra backend có đang chạy không:
   ```powershell
   curl http://localhost:8000/health
   ```
2. Kiểm tra port 8000 có bị chiếm không:
   ```powershell
   netstat -ano | findstr :8000
   ```
3. Restart backend nếu cần

### Lỗi: "CORS error"

**Nguyên nhân**: Backend không cho phép origin từ `http://localhost:3000`

**Giải pháp**:
Kiểm tra `api_server.py` có cấu hình CORS đúng không:

```python
# Trong api_server.py
cors_origins = ["*"]  # Cho phép tất cả trong development
# hoặc
cors_origins = ["http://localhost:3000"]
```

### Lỗi: "Module not found" trong Frontend

**Nguyên nhân**: Dependencies chưa được cài đặt

**Giải pháp**:

```powershell
cd FE/ai-chatbot-main
npm install
# hoặc
pnpm install
```

### Lỗi: "Environment variable not found"

**Nguyên nhân**: File `.env.local` chưa được tạo hoặc sai tên

**Giải pháp**:

1. Kiểm tra file `FE/ai-chatbot-main/.env.local` có tồn tại không
2. Kiểm tra tên biến có đúng không: `PYTHON_API_URL`
3. Restart dev server sau khi thêm env var

## 📊 Kiểm Tra Logs

### Backend Logs

Xem trong Terminal chạy `api_server.py`:

- Request logs
- Error messages
- Response status

### Frontend Logs

Xem trong Terminal chạy `npm run dev`:

- Next.js build logs
- API route logs
- Error messages

### Browser Console

Mở DevTools (F12) → Console:

- JavaScript errors
- Network requests
- API call logs

## ✅ Checklist Trước Khi Deploy

- [ ] Backend chạy ổn định tại `http://localhost:8000`
- [ ] Frontend chạy ổn định tại `http://localhost:3000`
- [ ] Test `/api/plant-ai` thành công
- [ ] Test chat với AI thành công
- [ ] Upload hình ảnh hoạt động (nếu có)
- [ ] Không có lỗi trong console
- [ ] Không có lỗi CORS
- [ ] Response time hợp lý (< 5s)

## 🚀 Sau Khi Test Thành Công

Khi đã test local thành công, bạn có thể:

1. Deploy Frontend lên Vercel
2. Cấu hình `PYTHON_API_URL` trong Vercel env vars
3. Dùng ngrok để expose backend local
4. Test lại trên Vercel

## 💡 Tips

1. **Giữ cả 2 terminal mở**: Một cho backend, một cho frontend
2. **Dùng ngrok dashboard**: `http://localhost:4040` để xem requests
3. **Check logs thường xuyên**: Để phát hiện lỗi sớm
4. **Test từng bước**: Đừng test tất cả cùng lúc
