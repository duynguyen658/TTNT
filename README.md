# Hệ Thống AI Multi-Agent Nhận dạng và Tư vấn Bệnh Cây Trồng

Hệ thống gồm 5 agents độc lập, phối hợp để nhận dạng và tư vấn điều trị bệnh cây trồng dựa trên hình ảnh, dataset và thông tin từ cộng đồng. Hệ thống tích hợp YOLOv8 Detection Model để phát hiện và định vị bệnh cây trồng với độ chính xác cao.

## Cấu Trúc Hệ Thống

### Agent 1 - Thu thập & Tiền xử lý Thông tin Bệnh Cây Trồng

- Thu thập và phân tích câu hỏi về bệnh cây trồng từ người dùng
- Trích xuất từ khóa liên quan đến bệnh cây (triệu chứng, loại cây, v.v.)
- Phân loại loại câu hỏi (nhận dạng, điều trị, phòng ngừa)
- Xác định các agent tiếp theo cần gọi dựa trên ngữ cảnh
- Làm giàu thông tin sử dụng LLM chuyên về nông nghiệp

### Agent 2 - Chẩn đoán Bệnh từ Hình ảnh Cây Trồng

- Phân tích hình ảnh cây trồng bị bệnh (lá, thân, rễ, quả)
- Sử dụng Vision Model để nhận dạng triệu chứng bệnh
- Xác định loại bệnh (nấm, vi khuẩn, virus, thiếu dinh dưỡng)
- Đánh giá mức độ nghiêm trọng và đưa ra chẩn đoán ban đầu
- Khuyến nghị điều trị dựa trên hình ảnh

### Agent 3 - Phân tích Dataset Bệnh Cây Trồng

- Phân tích dataset bệnh cây trồng (CSV, Excel, JSON)
- Tính toán thống kê về phân bố bệnh, loại cây, triệu chứng
- Phát hiện patterns và xu hướng bệnh
- Phân tích mối tương quan giữa các yếu tố
- Đưa ra insights và chẩn đoán dựa trên dữ liệu lịch sử

### Agent 4 - Tìm kiếm Thông tin Bệnh Cây trên Mạng Xã Hội

- Tìm kiếm thông tin từ cộng đồng nông dân, diễn đàn nông nghiệp
- Tóm tắt kinh nghiệm thực tế từ người dùng
- Phân tích sentiment và đánh giá hiệu quả của các giải pháp
- Trích xuất thông tin quan trọng về cách điều trị và phòng ngừa

### Agent 5 - Tổng hợp Tư vấn Điều trị Bệnh Cây

- Tổng hợp kết quả từ tất cả các agent (hình ảnh, dataset, mạng xã hội)
- Phát hiện xung đột và tìm đồng thuận giữa các nguồn
- Đưa ra chẩn đoán cuối cùng và tư vấn điều trị cụ thể
- Cung cấp biện pháp phòng ngừa và các bước tiếp theo

## Cài Đặt

1. Cài đặt các dependencies:

```bash
pip install -r requirements.txt
```

2. Cài đặt development dependencies (cho code quality tools):

```bash
pip install -r requirements-dev.txt
```

3. Tạo file `.env` và thêm API key:

```
OPENAI_API_KEY=your_api_key_here
```

4. Tạo các thư mục cần thiết (tự động tạo khi chạy):

- `data/images/` - Lưu hình ảnh
- `data/datasets/` - Lưu datasets
- `output/` - Lưu kết quả

5. (Optional) Setup pre-commit hooks:

```bash
pre-commit install
```

## Sử Dụng

### Chế độ Demo

```bash
python main.py
```

### Chế độ Tương tác

```bash
python main.py --interactive
```

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
        # Tùy chọn:
        # "image_path": "data/images/cay_ca_chua_benh.jpg",  # Hình ảnh cây bị bệnh
        # "dataset_path": "data/datasets/benh_cay_troong.csv"  # Dataset bệnh cây trồng
    }

    result = await orchestrator.execute(user_input)
    print(result["final_advice"])

asyncio.run(example())
```

## Cấu Trúc Thư Mục

```
TTNT2/
├── agents/
│   ├── __init__.py
│   ├── base_agent.py
│   ├── agent1_user_collector.py
│   ├── agent2_image_diagnosis.py
│   ├── agent3_dataset_diagnosis.py
│   ├── agent4_social_media.py
│   └── agent5_final_synthesis.py
├── yolo/
│   ├── train_yolo_detection.py      # Script training YOLO model
│   ├── inference_yolo.py             # Script inference/test model
│   ├── check_dataset.py              # Script kiểm tra dataset
│   ├── HUONG_DAN_TRAIN_YOLO_DETECTION.md
│   └── HUONG_DAN_INFERENCE.md
├── models/
│   ├── yolo_detection_s.pt           # Model đã train
│   └── yolo_detection_s_config.json  # Config model
├── data/
│   ├── train/                        # Training data
│   │   ├── images/
│   │   └── labels/
│   ├── valid/                        # Validation data
│   │   ├── images/
│   │   └── labels/
│   ├── test/                         # Test data
│   │   ├── images/
│   │   └── labels/
│   ├── data.yaml                     # Dataset config
│   ├── images/                       # Ảnh input
│   └── datasets/                     # CSV/Excel datasets
├── runs/
│   └── detect/
│       └── plant_disease_detection/  # Training results
│           ├── weights/
│           │   ├── best.pt
│           │   └── last.pt
│           ├── results.png
│           └── confusion_matrix.png
├── output/                           # Kết quả output
├── scripts/                          # Utility scripts
│   ├── format_code.py                # Format code script
│   └── lint_code.py                  # Lint code script
├── config.py
├── orchestrator.py
├── main.py
├── requirements.txt                  # Production dependencies
├── requirements-dev.txt              # Development dependencies
├── pyproject.toml                    # Python tool configs
├── .flake8                           # Flake8 config
├── .pre-commit-config.yaml          # Pre-commit hooks
├── .eslintrc.json                    # ESLint config
├── .prettierrc.json                  # Prettier config
├── package.json                      # Node.js dependencies
└── README.md
```

## Tính Năng

### Multi-Agent System

- ✅ 5 agents độc lập chuyên về bệnh cây trồng
- ✅ Nhận dạng bệnh từ hình ảnh cây trồng (lá, thân, rễ, quả)
- ✅ Phân tích dataset bệnh cây trồng để tìm patterns và xu hướng
- ✅ Tìm kiếm kinh nghiệm từ cộng đồng nông dân
- ✅ Tổng hợp tư vấn điều trị cụ thể và thực tế
- ✅ Điều phối thông minh: chỉ gọi các agent cần thiết
- ✅ Xử lý song song các agent khi có thể
- ✅ Phát hiện xung đột và tìm đồng thuận giữa các nguồn
- ✅ Hỗ trợ nhiều loại input: text, image, dataset
- ✅ Logging và tracking quá trình thực thi

### YOLOv8 Detection Model

- ✅ Training YOLOv8 Detection model với dataset tùy chỉnh
- ✅ Phát hiện và định vị bệnh cây trồng với bounding boxes
- ✅ Hỗ trợ 30+ loại bệnh cây trồng (Apple, Tomato, Potato, Pepper, etc.)
- ✅ Inference và test model trên ảnh thực tế
- ✅ Batch processing cho nhiều ảnh
- ✅ Đánh giá model trên test set
- ✅ Visualization kết quả với bounding boxes và confidence scores

## YOLO Detection Model

### Training Model

Huấn luyện YOLOv8 Detection model để phát hiện bệnh cây trồng:

```bash
# Training với dataset
python yolo/train_yolo_detection.py --data data/data.yaml --epochs 100 --batch 16

# Hoặc với các tùy chọn
python yolo/train_yolo_detection.py \
    --data data/data.yaml \
    --model-size s \
    --epochs 100 \
    --batch 16 \
    --device cuda
```

**Xem chi tiết**: `yolo/HUONG_DAN_TRAIN_YOLO_DETECTION.md`

### Inference/Test Model

Kiểm tra model đã train trên ảnh thực tế:

```bash
# Test một ảnh
python yolo/inference_yolo.py \
    --model runs/detect/plant_disease_detection/weights/best.pt \
    --image path/to/image.jpg

# Test thư mục ảnh
python yolo/inference_yolo.py \
    --model runs/detect/plant_disease_detection/weights/best.pt \
    --folder data/test/images \
    --save results/

# Đánh giá trên test set
python yolo/inference_yolo.py \
    --model runs/detect/plant_disease_detection/weights/best.pt \
    --test data/test/images
```

**Xem chi tiết**: `yolo/HUONG_DAN_INFERENCE.md`

### Dataset Format

Dataset YOLO Detection cần có cấu trúc:

```
data/
    train/
        images/    # Ảnh training
        labels/    # Annotations (.txt files)
    valid/
        images/
        labels/
    test/
        images/
        labels/
    data.yaml      # Config file
```

Format label file (`.txt`): `class_id x_center y_center width height` (normalized 0-1)

## Dataset Bệnh Cây Trồng

Hệ thống hỗ trợ phân tích dataset bệnh cây trồng với các định dạng:

- CSV: `data/datasets/benh_cay_troong.csv`
- Excel: `data/datasets/benh_cay_troong.xlsx`
- JSON: `data/datasets/benh_cay_troong.json`

Dataset nên chứa các cột như:

- Loại cây trồng
- Tên bệnh
- Triệu chứng
- Nguyên nhân
- Cách điều trị
- Thời gian, địa điểm (nếu có)

## Yêu Cầu Hệ Thống

### Dependencies

```bash
pip install -r requirements.txt
```

### Yêu Cầu Cho YOLO Training

- Python 3.8+
- CUDA-capable GPU (khuyến nghị) hoặc CPU
- Ultralytics: `pip install ultralytics`
- OpenCV: `pip install opencv-python`
- Tối thiểu 4GB VRAM cho model size 's'
- Tối thiểu 8GB RAM

### Yêu Cầu Cho Multi-Agent System

- OpenAI API key (cho LLM features)
- Python 3.8+
- Các thư viện trong `requirements.txt`

## Lưu Ý

### Multi-Agent System

- Cần API key của OpenAI để sử dụng đầy đủ tính năng
- Một số tính năng có thể hoạt động ở chế độ giới hạn nếu không có API key
- Agent 4 (Social Media) hiện đang mô phỏng - cần tích hợp API thật để sử dụng thực tế
- Đặt dataset vào thư mục `data/datasets/` để phân tích
- Hình ảnh cây bị bệnh nên rõ ràng, chụp đủ ánh sáng để có kết quả tốt nhất

### YOLO Model

- Dataset cần có annotations (bounding boxes) để training
- Format dataset phải đúng YOLO format
- Kiểm tra dataset trước khi train: `python yolo/check_dataset.py`
- Training trên GPU nhanh hơn nhiều so với CPU (10-50x)
- Model size 's' là lựa chọn tốt cho hầu hết trường hợp (cân bằng tốc độ và độ chính xác)

## Quick Start

### 1. Sử dụng Multi-Agent System

```bash
# Chế độ demo
python main.py

# Chế độ tương tác
python main.py --interactive
```

### 2. Training YOLO Model

```bash
# Kiểm tra dataset trước
python yolo/check_dataset.py

# Training
python yolo/train_yolo_detection.py --data data/data.yaml --epochs 100
```

### 3. Test/Inference Model

```bash
# Test một ảnh
python yolo/inference_yolo.py \
    --model runs/detect/plant_disease_detection/weights/best.pt \
    --image test.jpg
```

## Phát Triển Thêm

### Multi-Agent System

- Tích hợp API mạng xã hội thật (Facebook, diễn đàn nông nghiệp)
- Thêm database bệnh cây trồng Việt Nam
- Cải thiện xử lý xung đột và đồng thuận
- Thêm caching và optimization
- Web interface với upload ảnh trực tiếp
- Mobile app cho nông dân

### YOLO Model

- Tích hợp model vào Agent 2 (Image Diagnosis)
- Fine-tuning với dataset Việt Nam
- Hỗ trợ real-time detection qua webcam
- Export model sang ONNX/TensorRT để tối ưu
- Thêm segmentation model cho phân tích chi tiết hơn
- Multi-class detection với confidence threshold động

## Code Quality Tools

Dự án đã được cấu hình sẵn các công cụ chất lượng code:

### Python Tools

- **Black**: Code formatter (PEP 8)
- **Flake8**: Linter
- **isort**: Import sorter
- **Pre-commit**: Git hooks tự động

### JavaScript Tools

- **ESLint**: JavaScript linter
- **Prettier**: Code formatter

### Sử Dụng Nhanh

#### Với Makefile (Khuyến nghị)

```bash
# Xem tất cả commands
make help

# Format code
make format

# Lint code
make lint

# Format + Lint
make check

# Setup pre-commit
make setup-precommit
```

#### Với Scripts Python

```bash
# Format code Python
python scripts/format_code.py

# Lint code Python
python scripts/lint_code.py
```

#### Với npm

```bash
# Format code JavaScript
npm run format:js

# Lint code JavaScript
npm run lint:js
```

#### Thủ công

```bash
# Format Python
black . && isort .

# Lint Python
flake8 .

# Format JavaScript
prettier --write "**/*.{js,jsx,ts,tsx,json}"
```

**Xem chi tiết**: `CODE_QUALITY.md`

## Tài Liệu Tham Khảo

- **YOLO Training**: `yolo/HUONG_DAN_TRAIN_YOLO_DETECTION.md`
- **YOLO Inference**: `yolo/HUONG_DAN_INFERENCE.md`
- **Code Quality**: `CODE_QUALITY.md`
- **Ultralytics Docs**: https://docs.ultralytics.com/
- **YOLOv8 Paper**: https://arxiv.org/abs/2304.00501

## License

Nếu bạn có thắc mắc vui lòng gửi gmail cho mình: nguyenphanhongduy658@gmail.com
