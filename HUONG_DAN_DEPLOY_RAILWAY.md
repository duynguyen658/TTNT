# 🚂 Hướng Dẫn Deploy lên Railway

## 📋 Tổng Quan

Railway là một platform deploy đơn giản, hỗ trợ Docker và tự động detect Dockerfile.

## ✅ Đã Tạo Các File

1. **`Dockerfile`** - Multi-stage build để tối ưu kích thước
2. **`.dockerignore`** - Loại bỏ file không cần thiết
3. **`railway.json`** - Cấu hình Railway (optional)
4. **`requirements.txt`** - Đã thêm `fastapi` và `uvicorn`

## 🚀 Các Bước Deploy

### Bước 1: Chuẩn Bị Code

1. **Đảm bảo có YOLO model**:

   ```bash
   # Model phải có trong thư mục models/
   ls models/yolo_detection_s.pt
   ```

2. **Commit code lên GitHub**:
   ```bash
   git add .
   git commit -m "Add Dockerfile for Railway deployment"
   git push origin main
   ```

### Bước 2: Tạo Project trên Railway

1. Truy cập [railway.app](https://railway.app)
2. Đăng nhập với GitHub
3. Click **"New Project"**
4. Chọn **"Deploy from GitHub repo"**
5. Chọn repository của bạn

### Bước 3: Cấu Hình Environment Variables

Trong Railway dashboard, thêm các biến môi trường:

#### Bắt Buộc:

```env
# LLM Provider (groq hoặc openai)
LLM_PROVIDER=groq

# API Keys (chọn một trong hai)
GROQ_API_KEY=your-groq-api-key
# HOẶC
OPENAI_API_KEY=your-openai-api-key

# YOLO Model Path (optional, mặc định: models/yolo_detection_s.pt)
YOLO_MODEL_PATH=models/yolo_detection_s.pt
```

#### Tùy Chọn:

```env
# Port (Railway tự động set, không cần set thủ công)
PORT=8000

# Python environment
PYTHONUNBUFFERED=1
```

### Bước 4: Deploy

1. Railway sẽ tự động detect `Dockerfile`
2. Click **"Deploy"** hoặc push code mới sẽ tự động deploy
3. Đợi build xong (có thể mất 5-10 phút lần đầu)

### Bước 5: Lấy URL

1. Sau khi deploy xong, Railway sẽ cung cấp URL
2. Ví dụ: `https://your-app-name.up.railway.app`
3. Copy URL này để dùng trong frontend

## 🔧 Cấu Hình Frontend

Cập nhật `FE/ai-chatbot-main/.env.local`:

```env
PYTHON_API_URL=https://your-app-name.up.railway.app
```

Hoặc trong Vercel, thêm environment variable:

```
PYTHON_API_URL=https://your-app-name.up.railway.app
```

## 📊 Monitoring

### Xem Logs:

1. Vào Railway dashboard
2. Click vào service
3. Tab **"Deployments"** → Click vào deployment mới nhất
4. Tab **"Logs"** để xem real-time logs

### Health Check:

```bash
curl https://your-app-name.up.railway.app/health
```

Expected response:

```json
{
  "status": "healthy",
  "agents": {...}
}
```

## 🐛 Troubleshooting

### Lỗi: "Module not found"

- **Nguyên nhân**: Thiếu dependencies
- **Giải pháp**: Kiểm tra `requirements.txt` có đầy đủ không

### Lỗi: "YOLO model not found"

- **Nguyên nhân**: Model file không được copy vào Docker
- **Giải pháp**:
  1. Kiểm tra model có trong `models/` folder
  2. Đảm bảo `.dockerignore` không ignore `models/`

### Lỗi: "Out of memory"

- **Nguyên nhân**: TensorFlow/YOLO cần nhiều RAM
- **Giải pháp**:
  1. Upgrade Railway plan (có thêm RAM)
  2. Hoặc optimize model (dùng model nhỏ hơn)

### Build quá lâu

- **Nguyên nhân**: TensorFlow và ultralytics rất nặng
- **Giải pháp**:
  1. Sử dụng multi-stage build (đã có trong Dockerfile)
  2. Cache dependencies nếu có thể

## 💰 Cost Optimization

### Railway Pricing:

- **Free tier**: $5 credit/tháng
- **Hobby**: $20/tháng
- **Pro**: $100/tháng

### Tips:

1. **Tắt service khi không dùng** (Railway có auto-sleep)
2. **Optimize Dockerfile** (đã dùng multi-stage build)
3. **Dùng model nhỏ hơn** nếu có thể

## 🔄 Update Code

Mỗi khi push code mới lên GitHub:

1. Railway tự động detect changes
2. Tự động build và deploy
3. Có thể xem progress trong dashboard

## 📝 Notes

- Railway tự động set `PORT` environment variable
- Dockerfile đã được optimize với multi-stage build
- Health check endpoint: `/health`
- API docs: `https://your-app-name.up.railway.app/docs` (FastAPI auto-generated)

## 🎯 Next Steps

1. ✅ Deploy backend lên Railway
2. ✅ Cập nhật frontend với Railway URL
3. ✅ Test integration
4. ✅ Monitor logs và performance
