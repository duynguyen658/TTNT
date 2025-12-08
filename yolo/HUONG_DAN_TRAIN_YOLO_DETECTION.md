# Hướng Dẫn Huấn Luyện YOLOv8 Detection Model

## Tổng Quan

Script `train_yolo_detection.py` giúp bạn huấn luyện YOLOv8 Detection model để phát hiện và định vị bệnh cây trồng với bounding boxes từ dataset của bạn.

## Yêu Cầu

- Python 3.8+
- Ultralytics YOLO: `pip install ultralytics`
- Dataset YOLO format với annotations (bounding boxes)

## Cấu Trúc Dataset YOLO Detection

Dataset cần có cấu trúc như sau:

```
data/
    train/
        images/
            img1.jpg
            img2.jpg
            ...
        labels/
            img1.txt
            img2.txt
            ...
    valid/ (hoặc val/)
        images/
            img1.jpg
            ...
        labels/
            img1.txt
            ...
    test/ (optional)
        images/
            img1.jpg
            ...
        labels/
            img1.txt
            ...
    data.yaml
```

### Format File Label (.txt)

Mỗi file `.txt` tương ứng với một ảnh, mỗi dòng là một object:

```
class_id x_center y_center width height
```

Tất cả giá trị được normalize về [0, 1]:
- `x_center`, `y_center`: Tọa độ tâm của bounding box (tính theo chiều rộng/cao của ảnh)
- `width`, `height`: Chiều rộng và cao của bounding box (tính theo chiều rộng/cao của ảnh)

Ví dụ:
```
0 0.5 0.5 0.3 0.4
1 0.2 0.3 0.15 0.2
```

### Format File data.yaml

```yaml
path: /path/to/dataset
train: train/images
val: valid/images
test: test/images  # optional

nc: 30  # số lượng classes
names: ['Apple Scab Leaf', 'Apple leaf', 'Apple rust leaf', ...]  # tên các classes
```

## Cách Sử Dụng

### 1. Train với data.yaml được chỉ định

```bash
python train_yolo_detection.py --data data/data.yaml --epochs 100
```

### 2. Train từ thư mục dataset (tự động tìm data.yaml)

```bash
python train_yolo_detection.py --dataset data --epochs 100
```

### 3. Train với model lớn hơn (độ chính xác cao hơn)

```bash
python train_yolo_detection.py --data data/data.yaml --model-size m --epochs 150
```

Các kích thước model:
- `n` (nano): Nhỏ nhất, nhanh nhất (~6MB)
- `s` (small): Cân bằng, khuyến nghị (~22MB)
- `m` (medium): Tốt hơn (~52MB)
- `l` (large): Rất tốt (~87MB)
- `x` (xlarge): Tốt nhất, chậm nhất (~136MB)

### 4. Train trên GPU

```bash
python train_yolo_detection.py --data data/data.yaml --device cuda
```

### 5. Train với batch size nhỏ hơn (nếu GPU memory không đủ)

```bash
python train_yolo_detection.py --data data/data.yaml --batch 8
```

### 6. Tùy chỉnh đầy đủ

```bash
python train_yolo_detection.py \
    --data data/data.yaml \
    --model-size s \
    --epochs 100 \
    --imgsz 640 \
    --batch 16 \
    --device cuda \
    --output-dir runs/detect \
    --project-name plant_disease_detection
```

## Tham Số

- `--data` hoặc `--data-yaml`: Đường dẫn đến file data.yaml (bắt buộc nếu không dùng --dataset)
- `--dataset`: Đường dẫn đến thư mục dataset (tự động tìm data.yaml)
- `--model-size`: Kích thước model - n, s, m, l, x (mặc định: `s`)
- `--epochs`: Số epochs để train (mặc định: `100`)
- `--imgsz`: Kích thước ảnh (mặc định: `640` cho detection)
- `--batch`: Batch size (mặc định: `16`)
- `--device`: Device - cpu, cuda, 0, 1, etc. (mặc định: auto)
- `--output-dir`: Thư mục lưu kết quả (mặc định: `runs/detect`)
- `--project-name`: Tên project (mặc định: `plant_disease_detection`)

## Kết Quả

Sau khi training hoàn tất:

1. **Model tốt nhất**: `runs/detect/plant_disease_detection/weights/best.pt`
2. **Model cuối cùng**: `runs/detect/plant_disease_detection/weights/last.pt`
3. **Model đã copy**: `models/yolo_detection_s.pt`
4. **Config file**: `models/yolo_detection_s_config.json`
5. **Training logs**: `runs/detect/plant_disease_detection/`
6. **Validation results**: `runs/detect/plant_disease_detection/val_batch*.jpg`

## Sử Dụng Model Đã Train

### Sử dụng trong code:

```python
from models.yolo_disease_model import YOLODiseaseModel

# Load model detection đã train
model = YOLODiseaseModel(model_path="models/yolo_detection_s.pt")

# Predict với detection
result = model.predict("path/to/image.jpg", conf=0.25, iou=0.45)

print(f"Disease: {result['disease']}")
print(f"Confidence: {result['confidence']}")
print(f"Bounding boxes: {result['all']}")
```

### Kết quả predict:

```python
{
    "disease": "Tomato leaf bacterial spot",
    "confidence": 0.95,
    "top3": [
        {
            "disease": "Tomato leaf bacterial spot",
            "confidence": 0.95,
            "bbox": [100, 150, 300, 400]  # [x1, y1, x2, y2]
        },
        ...
    ],
    "all": [...],
    "num": 3,
    "inference_time(s)": 0.023,
    "FPS": 43.478
}
```

## Sử Dụng Trong Code (Train từ Class)

```python
from models.yolo_disease_model import YOLODiseaseModel

# Khởi tạo model
model = YOLODiseaseModel()

# Train detection model
results = model.train(
    dataset_path="data/data.yaml",  # Đường dẫn đến data.yaml
    model_size="s",
    epochs=100,
    imgsz=640,
    batch=16,
    mode="detection"  # Quan trọng: phải là "detection"
)
```

## Chuyển Đổi Dataset Từ Format Khác

### Từ COCO Format:

```python
from ultralytics.data.converter import convert_coco

convert_coco(
    labels_dir="path/to/coco/labels",
    save_dir="path/to/yolo/labels"
)
```

### Từ Pascal VOC Format:

```python
from ultralytics.data.converter import convert_voc

convert_voc(
    images_dir="path/to/voc/images",
    labels_dir="path/to/voc/annotations",
    output_dir="path/to/yolo"
)
```

## Lưu Ý

1. **Dataset size**: Cần ít nhất vài trăm ảnh với annotations để có kết quả tốt
2. **GPU**: Training trên GPU nhanh hơn nhiều so với CPU (10-50x)
3. **Memory**: 
   - Model `n`: ~2GB VRAM
   - Model `s`: ~4GB VRAM
   - Model `m`: ~6GB VRAM
   - Model `l`: ~8GB VRAM
   - Model `x`: ~10GB VRAM
4. **Time**: Training có thể mất vài giờ đến vài ngày tùy dataset size và model size
5. **Image size**: 640 là mặc định tốt, có thể tăng lên 1280 cho độ chính xác cao hơn (chậm hơn)

## Troubleshooting

### Lỗi: "YOLO not installed"
```bash
pip install ultralytics
```

### Lỗi: "CUDA out of memory"
- Giảm `--batch` size (ví dụ: `--batch 8` hoặc `--batch 4`)
- Giảm `--imgsz` (ví dụ: `--imgsz 416`)
- Dùng model nhỏ hơn (`--model-size n`)
- Hoặc train trên CPU (`--device cpu`)

### Lỗi: "Dataset not found" hoặc "data.yaml not found"
- Kiểm tra đường dẫn dataset
- Đảm bảo cấu trúc thư mục đúng format
- Kiểm tra file data.yaml có đúng format không

### Lỗi: "No labels found"
- Đảm bảo có thư mục `labels/` với file `.txt`
- Kiểm tra format file label (class_id x y w h, normalized)

### Lỗi: "Mismatch number of images and labels"
- Đảm bảo mỗi ảnh có file label tương ứng
- Tên file phải khớp (ví dụ: `img1.jpg` và `img1.txt`)

## Ví Dụ Đầy Đủ

```bash
# 1. Kiểm tra dataset
python train_yolo_detection.py --dataset data

# 2. Train model
python train_yolo_detection.py \
    --data data/data.yaml \
    --model-size s \
    --epochs 100 \
    --batch 16 \
    --device cuda

# 3. Sử dụng model
python -c "
from models.yolo_disease_model import YOLODiseaseModel
model = YOLODiseaseModel('models/yolo_detection_s.pt')
result = model.predict('test_image.jpg')
print(f'Disease: {result[\"disease\"]}')
print(f'Confidence: {result[\"confidence\"]}')
print(f'Bounding box: {result[\"all\"][0][\"bbox\"]}')
"
```

## So Sánh Classification vs Detection

| Feature | Classification | Detection |
|---------|---------------|-----------|
| Output | Class name | Class name + Bounding box |
| Dataset | Chỉ cần ảnh theo thư mục | Cần ảnh + annotations (bbox) |
| Use case | Nhận dạng toàn bộ ảnh | Phát hiện và định vị object |
| Accuracy | Cao cho toàn ảnh | Cao cho từng object |
| Speed | Nhanh hơn | Chậm hơn một chút |
| Model size | Nhỏ hơn | Lớn hơn |

## Tài Liệu Tham Khảo

- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
- [YOLOv8 Training Guide](https://docs.ultralytics.com/modes/train/)
- [YOLO Dataset Format](https://docs.ultralytics.com/datasets/)
