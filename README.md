# Hệ Thống AI Multi-Agent Nhận dạng và Tư vấn Bệnh Cây Trồng

Hệ thống gồm 5 agents phối hợp để nhận dạng và tư vấn điều trị bệnh cây trồng dựa trên hình ảnh. Tích hợp YOLOv8 Detection Model để phát hiện bệnh với độ chính xác cao.

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

## 🚀 Cài Đặt

1. Cài đặt dependencies:

```bash
pip install -r requirements.txt
```

2. Tạo file `.env` và thêm API key:

```env
GROQ_API_KEY=your_groq_api_key
# hoặc
OPENAI_API_KEY=your_openai_api_key
LLM_PROVIDER=groq  # hoặc "openai"
YOLO_MODEL_PATH=models/yolo_detection_s.pt
```

3. (Optional) Cài đặt pre-commit hooks:

```bash
pre-commit install
```

## 💻 Sử Dụng

### Chạy Backend API Server

```bash
python api_server.py
```

Server sẽ chạy tại `http://localhost:8000`

### Sử dụng trong Code

```python
import asyncio
from orchestrator import AgentOrchestrator

async def example():
    orchestrator = AgentOrchestrator()

    user_input = {
        "user_query": "Cây cà chua của tôi có lá bị vàng và đốm nâu, xin tư vấn",
        "user_context": {
            "plant_type": "cà chua",
            "location": "miền Bắc",
            "season": "mùa mưa"
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

## 📝 Yêu Cầu

- Python 3.8+
- CUDA-capable GPU (khuyến nghị cho YOLO) hoặc CPU
- Groq API key hoặc OpenAI API key
- Tối thiểu 4GB VRAM cho YOLO model size 's'

## 📧 Liên Hệ

Nếu có thắc mắc, vui lòng gửi email: nguyenphanhongduy658@gmail.com
