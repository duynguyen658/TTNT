# Hệ Thống AI Multi-Agent Nhận dạng và Tư vấn Bệnh Cây Trồng

Hệ thống gồm 5 agents phối hợp để nhận dạng và tư vấn điều trị bệnh cây trồng dựa trên hình ảnh. Tích hợp YOLOv8 Detection Model để phát hiện bệnh với độ chính xác cao.

## ⚡ Quick Start

### Local Development

```bash
# 1. Cài đặt Backend
pip install -r requirements.txt

# 2. Tạo file .env và thêm GROQ_API_KEY
echo "GROQ_API_KEY=your_key_here" > .env

# 3. Chạy Backend (Terminal 1)
python api_server.py

# 4. Cài đặt Frontend (Terminal 2)
cd FE/ai-chatbot-main
pnpm install

# 5. Tạo file .env.local và thêm POSTGRES_URL
echo "POSTGRES_URL=your_postgres_url" > .env.local
echo "PYTHON_API_URL=http://localhost:8000" >> .env.local

# 6. Chạy migration
pnpm db:migrate

# 7. Chạy Frontend
pnpm dev
```

Mở browser: `http://localhost:3000` để sử dụng hệ thống.

### Với Cloudflare Tunnel (Cho Production/Deploy)

```powershell
# 1. Cài đặt Cloudflare Tunnel
choco install cloudflared

# 2. Chạy Backend + Tunnel (tự động)
.\start-backend.ps1

# 3. Copy Cloudflare URL (ví dụ: https://abc123.trycloudflare.com)

# 4. Cập nhật .env.local
echo "PYTHON_API_URL=https://abc123.trycloudflare.com" >> .env.local

# 5. Chạy Frontend
cd FE/ai-chatbot-main
pnpm dev
```

## 🧠 5 Agents

### 🟢 Agent 1 – Thu thập & Phân tích Yêu cầu

- Hiểu câu hỏi của người dùng
- Trích xuất thông tin (loại cây, triệu chứng, mục tiêu)
- Quyết định agent nào cần gọi tiếp

### 🟢 Agent 2 – Chẩn đoán Bệnh từ Hình ảnh (Vision / YOLO)

- Phân tích hình ảnh cây trồng
- Nhận dạng loại bệnh
- Trả về kết quả chẩn đoán ban đầu và độ tin cậy

### 🟡 Agent 3 – Thẩm định Chẩn đoán & Xác định Tác nhân gây bệnh

- Đánh giá độ tin cậy của kết quả chẩn đoán
- Cảnh báo nguy cơ nhầm lẫn
- Xác định tác nhân gây bệnh (nấm, vi khuẩn, virus, dinh dưỡng, môi trường)

### 🔵 Agent 4 – Kiến thức & Kinh nghiệm Thực tế

- Bổ sung kiến thức nông học và kinh nghiệm từ thực tế
- Đưa ra lưu ý, kinh nghiệm phòng và xử lý bệnh
- Hỗ trợ Agent 5 tư vấn đúng hướng

### 🟢 Agent 5 – Tổng hợp & Tư vấn Điều trị

- Tổng hợp kết quả từ tất cả các agent
- Đưa ra chẩn đoán cuối cùng
- Tư vấn xử lý và phòng ngừa cho người nông dân bằng ngôn ngữ dễ hiểu

## 🚀 Cài Đặt và Chạy Hệ Thống

### Yêu Cầu Hệ Thống

- **Python**: 3.8 trở lên
- **Node.js**: 18.x trở lên
- **pnpm**: 9.x trở lên (hoặc npm/yarn)
- **PostgreSQL**: Database cho chat history (có thể dùng Vercel Postgres hoặc local)
- **CUDA-capable GPU** (khuyến nghị cho YOLO) hoặc CPU
- **Groq API key** hoặc **OpenAI API key**

### Bước 1: Cài Đặt Backend (Python)

1. **Cài đặt dependencies:**

```bash
pip install -r requirements.txt
```

2. **Tạo file `.env` trong thư mục gốc và thêm cấu hình:**

```env
# LLM Provider
GROQ_API_KEY=your_groq_api_key
# hoặc
OPENAI_API_KEY=your_openai_api_key
LLM_PROVIDER=groq  # hoặc "openai"

# YOLO Model
YOLO_MODEL_PATH=models/yolo_detection_s.pt

# CORS (cho production)
CORS_ORIGINS=https://your-frontend-domain.com,https://*.vercel.app
```

3. **(Optional) Cài đặt pre-commit hooks:**

```bash
pre-commit install
```

### Bước 2: Cài Đặt Frontend (Next.js)

1. **Di chuyển vào thư mục frontend:**

```bash
cd FE/ai-chatbot-main
```

2. **Cài đặt dependencies:**

```bash
pnpm install
# hoặc
npm install
```

3. **Tạo file `.env.local` trong thư mục `FE/ai-chatbot-main`:**

```env
# Database (PostgreSQL)
POSTGRES_URL=your_postgres_connection_string
# hoặc nếu dùng Vercel Postgres
POSTGRES_URL=postgresql://user:password@host:port/database

# Python Backend URL
PYTHON_API_URL=http://localhost:8000

# NextAuth (nếu cần)
NEXTAUTH_SECRET=your_secret_key
NEXTAUTH_URL=http://localhost:3000
```

4. **Chạy database migration:**

```bash
pnpm db:migrate
# hoặc
npm run db:migrate
```

## 💻 Chạy Hệ Thống

### Cách 1: Chạy Thủ Công (2 Terminal)

**Terminal 1 - Backend (Python):**

```bash
# Từ thư mục gốc D:\TTNT2
python api_server.py
```

Backend sẽ chạy tại: `http://localhost:8000`

**Terminal 2 - Frontend (Next.js):**

```bash
# Từ thư mục FE/ai-chatbot-main
cd FE/ai-chatbot-main
pnpm dev
# hoặc
npm run dev
```

Frontend sẽ chạy tại: `http://localhost:3000`

### Cách 2: Chạy với Cloudflare Tunnel (Cho Production/Deploy)

**Khi nào cần Cloudflare Tunnel?**

- Khi deploy frontend lên Vercel/Netlify và cần kết nối với backend local
- Khi muốn test frontend production với backend local
- Khi cần expose backend ra internet

**Cài đặt Cloudflare Tunnel:**

```powershell
# Windows (với Chocolatey)
choco install cloudflared

# Hoặc download từ: https://github.com/cloudflare/cloudflared/releases
```

**Chạy Backend + Cloudflare Tunnel:**

```powershell
# Từ thư mục gốc D:\TTNT2
.\start-backend.ps1
```

Script này sẽ:

1. Kiểm tra Python và Cloudflare Tunnel
2. Khởi động Python backend tại `http://localhost:8000`
3. Khởi động Cloudflare Tunnel (Quick Tunnel hoặc Named Tunnel)
4. Hiển thị URL public (ví dụ: `https://abc123.trycloudflare.com`)

**Sau khi có Cloudflare URL:**

1. Copy URL từ cửa sổ Cloudflare Tunnel (ví dụ: `https://abc123.trycloudflare.com`)
2. Cập nhật `PYTHON_API_URL` trong Vercel Environment Variables:
   ```
   PYTHON_API_URL=https://abc123.trycloudflare.com
   ```
3. Hoặc cập nhật `.env.local` của frontend local:
   ```env
   PYTHON_API_URL=https://abc123.trycloudflare.com
   ```

**Chạy Frontend:**

```powershell
# Terminal khác
cd FE/ai-chatbot-main
pnpm dev
```

**Lưu ý:**

- Quick Tunnel: URL thay đổi mỗi lần chạy (phù hợp test)
- Named Tunnel: URL cố định (cần setup trước, phù hợp production)
- Cloudflare Tunnel chỉ chạy khi terminal còn mở

### Kiểm Tra Hệ Thống

1. **Kiểm tra Backend:**
   - Mở browser: `http://localhost:8000/docs` (Swagger UI)
   - Hoặc: `http://localhost:8000/health` (health check)

2. **Kiểm tra Frontend:**
   - Mở browser: `http://localhost:3000`
   - Gửi message test hoặc upload ảnh

3. **Test Flow:**
   - Gửi text: "Xin chào" → Nhận simple response
   - Gửi câu hỏi nông nghiệp: "Thuốc trừ sâu nào tốt?" → Chạy 5 agents
   - Upload ảnh cây bị bệnh → Chạy 5 agents với YOLO detection

## 📋 Lưu Ý Quan Trọng

### Environment Variables

**Backend (`.env` trong thư mục gốc):**

- `GROQ_API_KEY` hoặc `OPENAI_API_KEY`: Bắt buộc
- `LLM_PROVIDER`: `groq` hoặc `openai`
- `YOLO_MODEL_PATH`: Đường dẫn đến model YOLO (mặc định: `models/yolo_detection_s.pt`)
- `CORS_ORIGINS`: Danh sách origins được phép (production)

**Frontend (`.env.local` trong `FE/ai-chatbot-main`):**

- `POSTGRES_URL`: Connection string đến PostgreSQL database (bắt buộc)
- `PYTHON_API_URL`: URL của Python backend (mặc định: `http://localhost:8000`)
- `NEXTAUTH_SECRET`: Secret key cho NextAuth (nếu dùng authentication)
- `NEXTAUTH_URL`: URL của frontend (mặc định: `http://localhost:3000`)

### Troubleshooting

**Lỗi: Backend không kết nối được**

- Kiểm tra Python backend đang chạy tại `http://localhost:8000`
- Kiểm tra `PYTHON_API_URL` trong `.env.local` của frontend
- Kiểm tra CORS settings trong `api_server.py`

**Lỗi: Database connection failed**

- Kiểm tra `POSTGRES_URL` trong `.env.local`
- Đảm bảo đã chạy `pnpm db:migrate` để tạo tables
- Kiểm tra PostgreSQL server đang chạy

**Lỗi: YOLO model not found**

- Kiểm tra file model tại `models/yolo_detection_s.pt`
- Hoặc download model và đặt đúng đường dẫn trong `.env`

**Lỗi: API key không hợp lệ**

- Kiểm tra `GROQ_API_KEY` hoặc `OPENAI_API_KEY` trong `.env`
- Đảm bảo API key còn hiệu lực và có đủ quota

## 🔍 Sử Dụng Trong Code (Python)

```python
import asyncio
from orchestrator import AgentOrchestrator

async def example():
    orchestrator = AgentOrchestrator()

    user_input = {
        "user_query": "Cây cà chua của tôi có lá bị vàng và đốm nâu, xin tư vấn",
        "user_context": {
            "conversation_history": [
                {"role": "user", "content": "Câu hỏi trước"},
                {"role": "assistant", "content": "Câu trả lời trước"}
            ]
        },
        "image_data": "base64_encoded_image",  # Optional
    }

    result = await orchestrator.execute(user_input)
    print(result["final_advice"])

asyncio.run(example())
```

## 📁 Cấu Trúc Thư Mục

```
TTNT2/
├── agents/                    # 5 agents
│   ├── agent1_user_collector.py
│   ├── agent2_image_diagnosis.py
│   ├── agent3_diagnosis_validator.py
│   ├── agent4_knowledge_experience.py
│   └── agent5_final_synthesis.py
├── yolo/                     # YOLO model scripts
├── models/                   # Trained models
├── config.py                 # Configuration
├── orchestrator.py           # Agent orchestrator
├── api_server.py            # FastAPI server
└── requirements.txt         # Dependencies
```

## ✨ Tính Năng

- ✅ 5 agents chuyên biệt cho bệnh cây trồng
- ✅ Nhận dạng bệnh từ hình ảnh (YOLO + Vision API)
- ✅ Thẩm định và xác định tác nhân gây bệnh
- ✅ Bổ sung kiến thức nông học và kinh nghiệm thực tế
- ✅ Tư vấn điều trị cụ thể, dễ hiểu cho nông dân
- ✅ Flow tuần tự: Agent 1 → 2 → 3 → 4 → 5
- ✅ FastAPI server để tích hợp với frontend
- ✅ Hỗ trợ Groq và OpenAI

## 🔧 YOLO Model

### Training

```bash
python yolo/train_yolo_detection.py --data data/data.yaml --epochs 100
```

### Inference

```bash
python yolo/inference_yolo.py --model models/yolo_detection_s.pt --image test.jpg
```

## 📚 Tài Liệu

- **YOLO Training**: `yolo/HUONG_DAN_TRAIN_YOLO_DETECTION.md`
- **YOLO Inference**: `yolo/HUONG_DAN_INFERENCE.md`
- **Tích hợp Frontend**: `FE/ai-chatbot-main/README_TICH_HOP_AI.md`
- **Test Local**: `HUONG_DAN_TEST_LOCAL.md`

## 📝 Yêu Cầu Hệ Thống

### Backend

- **Python**: 3.8 trở lên
- **CUDA-capable GPU** (khuyến nghị cho YOLO) hoặc CPU
- **Groq API key** hoặc **OpenAI API key**
- **Tối thiểu 4GB VRAM** cho YOLO model size 's'
- **RAM**: Tối thiểu 8GB (khuyến nghị 16GB)

### Frontend

- **Node.js**: 18.x trở lên
- **pnpm**: 9.x trở lên (hoặc npm/yarn)
- **PostgreSQL**: Database cho chat history
  - Có thể dùng Vercel Postgres (free tier)
  - Hoặc PostgreSQL local/remote

### Tùy Chọn

- **Cloudflare Tunnel** (khuyến nghị): Để expose backend cho frontend production
  - Download: https://github.com/cloudflare/cloudflared/releases
  - Hoặc: `choco install cloudflared` (Windows)
- **ngrok**: Alternative cho Cloudflare Tunnel
- **Redis**: Cho resumable streams (optional)

## 📧 Liên Hệ

Nếu có thắc mắc, vui lòng gửi email: nguyenphanhongduy658@gmail.com
