# Hướng Dẫn Tích Hợp AI Model với Frontend

## ✅ Đã Tạo Các File

1. **`api_server.py`** - Python FastAPI server
2. **`FE/ai-chatbot-main/app/(chat)/api/plant-ai/route.ts`** - Next.js API route
3. **`FE/ai-chatbot-main/lib/ai/tools/plant-diagnosis.ts`** - AI SDK Tool
4. **`requirements-api.txt`** - Python dependencies

## 🚀 Quick Start

### Bước 1: Cài đặt Python Dependencies

```bash
pip install -r requirements-api.txt
```

### Bước 2: Start Python API Server

```bash
# Terminal 1
python api_server.py
```

Server sẽ chạy tại: `http://localhost:8000`

### Bước 3: Cấu hình Environment Variable

Thêm vào `FE/ai-chatbot-main/.env.local`:

```env
PYTHON_API_URL=http://localhost:8000
```

### Bước 4: Start Next.js Frontend

```bash
# Terminal 2
cd FE/ai-chatbot-main
npm run dev
```

### Bước 5: Test

1. Mở browser: http://localhost:3000
2. Chat với AI và hỏi về bệnh cây trồng
3. AI sẽ tự động sử dụng `plantDiagnosis` tool để gọi Python backend

## 📝 Cách Sử Dụng

### Trong Chat:

AI sẽ tự động nhận diện khi bạn hỏi về bệnh cây trồng và sử dụng tool `plantDiagnosis`.

**Ví dụ:**

- "Cây cà chua của tôi bị vàng lá, xin tư vấn"
- "Phân tích hình ảnh này" (kèm hình ảnh)
- "Cây lúa bị đốm nâu ở miền Bắc mùa mưa"

### Upload Hình Ảnh:

1. Click vào icon upload trong chat
2. Chọn hình ảnh cây bị bệnh
3. Gửi message kèm hình ảnh
4. AI sẽ tự động gọi YOLO model để phân tích

## 🌐 Deploy lên Production

### Deploy Python Backend (Railway):

1. Tạo account tại [railway.app](https://railway.app)
2. New Project → Deploy from GitHub
3. Select repo → Add service
4. Set:
   - **Build Command**: `pip install -r requirements-api.txt`
   - **Start Command**: `python api_server.py`
5. Add environment variables
6. Deploy!

### Deploy Frontend (Vercel):

1. Push code lên GitHub
2. Import project vào Vercel
3. Add environment variable:
   - `PYTHON_API_URL=https://your-api.railway.app`
4. Deploy!

## 🔍 Kiểm Tra

### Health Check:

```bash
# Python API
curl http://localhost:8000/health

# Next.js API
curl http://localhost:3000/api/plant-ai
```

### Test API:

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "user_query": "Cây cà chua bị vàng lá",
    "user_context": {"plant_type": "cà chua"}
  }'
```

## 📚 Chi Tiết

Xem file `HUONG_DAN_TICH_HOP_AI_MODEL.md` ở thư mục gốc để biết thêm chi tiết.
