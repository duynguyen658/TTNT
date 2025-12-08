# Hướng Dẫn Inference (Dự Đoán) YOLO Detection Model

## Tổng Quan

Sau khi train model xong, bạn cần test model bằng inference để xem model hoạt động như thế nào trên ảnh thực tế.

## Cài Đặt

```bash
pip install ultralytics opencv-python pillow
```

## Cách Sử Dụng

### 1. Test Nhanh (Dùng Script Có Sẵn)

```bash
python test_model.py
```

Script này sẽ:

- Tự động tìm model trong `runs/detect/plant_disease_detection/weights/best.pt`
- Test trên 3 ảnh đầu tiên trong test set
- Hiển thị kết quả với bounding boxes

### 2. Dự Đoán Một Ảnh

```bash
python inference_yolo.py --model runs/detect/plant_disease_detection/weights/best.pt --image path/to/image.jpg
```

### 3. Dự Đoán Thư Mục Ảnh

```bash
python inference_yolo.py --model runs/detect/plant_disease_detection/weights/best.pt --folder data/test/images
```

### 4. Dự Đoán và Lưu Kết Quả

```bash
python inference_yolo.py \
    --model runs/detect/plant_disease_detection/weights/best.pt \
    --folder data/test/images \
    --save results/
```

### 5. Đánh Giá Trên Test Set

```bash
python inference_yolo.py --model runs/detect/plant_disease_detection/weights/best.pt --test data/test/images
```

### 6. Tùy Chỉnh Threshold

```bash
# Tăng confidence threshold (chỉ hiển thị detections chắc chắn)
python inference_yolo.py \
    --model runs/detect/plant_disease_detection/weights/best.pt \
    --image test.jpg \
    --conf 0.5 \
    --iou 0.5
```

## Tham Số

- `--model`: Đường dẫn đến model .pt (mặc định: `runs/detect/plant_disease_detection/weights/best.pt`)
- `--image`: Đường dẫn đến ảnh cần dự đoán
- `--folder`: Thư mục chứa ảnh cần dự đoán
- `--test`: Thư mục test images để đánh giá
- `--save`: Thư mục lưu kết quả (ảnh với bounding boxes)
- `--conf`: Confidence threshold (0-1, mặc định: 0.25)
- `--iou`: IoU threshold cho NMS (0-1, mặc định: 0.45)
- `--no-show`: Không hiển thị ảnh kết quả (chỉ lưu file)

## Kết Quả

### Output Console

```
🔍 Đang phân tích: test.jpg

📊 Kết quả phát hiện:
   Tìm thấy 2 object(s):
   1. Tomato leaf bacterial spot: 95.23% bbox: [100, 150, 300, 400]
   2. Tomato leaf: 78.45% bbox: [50, 200, 250, 350]
```

### Ảnh Kết Quả

Ảnh sẽ được vẽ với:

- Bounding boxes (hộp giới hạn)
- Class names
- Confidence scores
- Màu sắc khác nhau cho mỗi class

## Sử Dụng Trong Code

```python
from inference_yolo import YOLOInference

# Khởi tạo
inference = YOLOInference(
    model_path="runs/detect/plant_disease_detection/weights/best.pt",
    conf_threshold=0.25,
    iou_threshold=0.45
)

# Dự đoán một ảnh
result = inference.predict_single("test_image.jpg", save_path="result.jpg")

# Kết quả
print(f"Số detections: {result['num_detections']}")
for det in result['detections']:
    print(f"{det['class_name']}: {det['confidence']:.2%}")
    print(f"  Bbox: {det['bbox']}")

# Dự đoán nhiều ảnh
results = inference.predict_folder("data/test/images", output_dir="results/")

# Đánh giá test set
inference.evaluate_on_test_set("data/test/images")
```

## Format Kết Quả

```python
{
    "image_path": "test.jpg",
    "num_detections": 2,
    "detections": [
        {
            "class_id": 27,
            "class_name": "Tomato leaf bacterial spot",
            "confidence": 0.9523,
            "bbox": [100, 150, 300, 400]  # [x1, y1, x2, y2]
        },
        ...
    ],
    "top_detection": {
        "class_id": 27,
        "class_name": "Tomato leaf bacterial spot",
        "confidence": 0.9523,
        "bbox": [100, 150, 300, 400]
    }
}
```

## Tips

1. **Confidence Threshold**:
   - Thấp (0.25): Phát hiện nhiều hơn, có thể có false positives
   - Cao (0.5-0.7): Chỉ phát hiện những gì chắc chắn, có thể bỏ sót

2. **IoU Threshold**:
   - Thấp (0.3): Loại bỏ ít overlapping boxes
   - Cao (0.5-0.7): Loại bỏ nhiều overlapping boxes (chỉ giữ box tốt nhất)

3. **Test trên nhiều ảnh**:
   - Test trên test set để đánh giá tổng thể
   - Test trên ảnh thực tế để xem performance trong production

4. **So sánh với Ground Truth**:
   - Nếu có labels, so sánh predictions với ground truth
   - Tính precision, recall, mAP

## Troubleshooting

### Lỗi: "Model not found"

```bash
# Kiểm tra đường dẫn model
ls runs/detect/plant_disease_detection/weights/
```

### Lỗi: "No module named 'cv2'"

```bash
pip install opencv-python
```

### Lỗi: "Image not found"

- Kiểm tra đường dẫn ảnh
- Đảm bảo file tồn tại

### Kết quả không hiển thị

- Thử giảm `--conf` threshold
- Kiểm tra ảnh có objects không
- Xem log để biết có detections không

## Ví Dụ Đầy Đủ

```bash
# 1. Test nhanh
python test_model.py

# 2. Test một ảnh cụ thể
python inference_yolo.py \
    --model runs/detect/plant_disease_detection/weights/best.pt \
    --image data/test/images/sample.jpg \
    --save results/ \
    --conf 0.3

# 3. Test toàn bộ test set
python inference_yolo.py \
    --model runs/detect/plant_disease_detection/weights/best.pt \
    --test data/test/images \
    --save results/test_results/

# 4. Đánh giá và xem thống kê
python inference_yolo.py \
    --model runs/detect/plant_disease_detection/weights/best.pt \
    --test data/test/images
```

## So Sánh Với Metrics Từ Training

Sau khi train, bạn có thể xem metrics trong:

- `runs/detect/plant_disease_detection/results.csv`
- `runs/detect/plant_disease_detection/results.png`
- `runs/detect/plant_disease_detection/confusion_matrix.png`

Inference giúp bạn xem model hoạt động trên ảnh thực tế như thế nào!
