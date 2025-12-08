import os
import sys
import json
from pathlib import Path
from typing import List, Union, Optional
import argparse

# Kiểm tra YOLO
try:
    from ultralytics import YOLO
    import cv2
    import numpy as np
    from PIL import Image
    YOLO_AVAILABLE = True
except ImportError as e:
    YOLO_AVAILABLE = False
    print(f"❌ Thiếu thư viện: {e}")
    print("👉 Cài đặt: pip install ultralytics opencv-python pillow")
    sys.exit(1)


class YOLOInference:
    def __init__(self, model_path: str, conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        """
        Khởi tạo YOLO Inference
        
        Args:
            model_path: Đường dẫn đến model .pt
            conf_threshold: Ngưỡng confidence (0-1)
            iou_threshold: Ngưỡng IoU cho NMS
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Không tìm thấy model: {model_path}")
        
        print(f"📥 Đang tải model: {model_path}")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Load class names
        self.class_names = list(self.model.names.values()) if hasattr(self.model, 'names') else []
        print(f"✅ Đã tải model với {len(self.class_names)} classes")
    
    def predict_single(self, image_path: str, save_path: Optional[str] = None, show: bool = True):
        """
        Dự đoán trên một ảnh
        
        Args:
            image_path: Đường dẫn đến ảnh
            save_path: Đường dẫn lưu ảnh kết quả (optional)
            show: Có hiển thị ảnh không
        
        Returns:
            dict với kết quả detection
        """
        if not os.path.exists(image_path):
            print(f"❌ Không tìm thấy ảnh: {image_path}")
            return None
        
        print(f"\n🔍 Đang phân tích: {image_path}")
        
        # Predict
        results = self.model.predict(
            image_path,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        result = results[0]
        
        # Parse kết quả
        detections = []
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                conf = float(boxes.conf[i])
                bbox = boxes.xyxy[i].cpu().numpy().tolist()  # [x1, y1, x2, y2]
                
                detections.append({
                    "class_id": cls_id,
                    "class_name": self.class_names[cls_id] if cls_id < len(self.class_names) else f"Class_{cls_id}",
                    "confidence": conf,
                    "bbox": bbox
                })
        
        # Sắp xếp theo confidence
        detections.sort(key=lambda x: x["confidence"], reverse=True)
        
        # In kết quả
        print(f"\n📊 Kết quả phát hiện:")
        if len(detections) == 0:
            print("   Không phát hiện được object nào")
        else:
            print(f"   Tìm thấy {len(detections)} object(s):")
            for i, det in enumerate(detections[:10], 1):  # Hiển thị top 10
                print(f"   {i}. {det['class_name']}: {det['confidence']:.2%} "
                      f"bbox: [{int(det['bbox'][0])}, {int(det['bbox'][1])}, "
                      f"{int(det['bbox'][2])}, {int(det['bbox'][3])}]")
        
        # Lưu ảnh kết quả
        if save_path:
            result.save(save_path)
            print(f"\n💾 Đã lưu ảnh kết quả: {save_path}")
        
        # Hiển thị ảnh
        if show and len(detections) > 0:
            try:
                plotted = result.plot()
                cv2.imshow("YOLO Detection Result", plotted)
                print("\n⏸️  Nhấn phím bất kỳ để đóng cửa sổ...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except Exception as e:
                print(f"⚠️  Không thể hiển thị ảnh: {e}")
        
        return {
            "image_path": image_path,
            "num_detections": len(detections),
            "detections": detections,
            "top_detection": detections[0] if detections else None
        }
    
    def predict_batch(self, image_paths: List[str], output_dir: Optional[str] = None):
        """
        Dự đoán trên nhiều ảnh
        
        Args:
            image_paths: Danh sách đường dẫn ảnh
            output_dir: Thư mục lưu kết quả (optional)
        
        Returns:
            List các kết quả
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        results = []
        for i, img_path in enumerate(image_paths):
            save_path = None
            if output_dir:
                save_path = os.path.join(output_dir, f"result_{i+1}_{Path(img_path).name}")
            
            result = self.predict_single(img_path, save_path=save_path, show=False)
            if result:
                results.append(result)
        
        return results
    
    def predict_folder(self, folder_path: str, output_dir: Optional[str] = None, 
                      extensions: List[str] = None):
        """
        Dự đoán trên tất cả ảnh trong thư mục
        
        Args:
            folder_path: Đường dẫn thư mục
            output_dir: Thư mục lưu kết quả
            extensions: Danh sách đuôi file (mặc định: ['.jpg', '.jpeg', '.png'])
        
        Returns:
            List các kết quả
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        
        folder = Path(folder_path)
        if not folder.exists():
            print(f"❌ Không tìm thấy thư mục: {folder_path}")
            return []
        
        # Tìm tất cả ảnh
        image_paths = []
        for ext in extensions:
            image_paths.extend(folder.glob(f"*{ext}"))
        
        if len(image_paths) == 0:
            print(f"⚠️  Không tìm thấy ảnh nào trong {folder_path}")
            return []
        
        print(f"📁 Tìm thấy {len(image_paths)} ảnh trong {folder_path}")
        
        return self.predict_batch([str(p) for p in image_paths], output_dir)
    
    def evaluate_on_test_set(self, test_images_dir: str, test_labels_dir: Optional[str] = None):
        """
        Đánh giá model trên test set
        
        Args:
            test_images_dir: Thư mục chứa ảnh test
            test_labels_dir: Thư mục chứa labels test (optional, để so sánh)
        """
        print("\n" + "=" * 60)
        print("ĐÁNH GIÁ MODEL TRÊN TEST SET")
        print("=" * 60)
        
        results = self.predict_folder(test_images_dir)
        
        if len(results) == 0:
            print("❌ Không có kết quả để đánh giá")
            return
        
        # Thống kê
        total_images = len(results)
        images_with_detections = sum(1 for r in results if r['num_detections'] > 0)
        total_detections = sum(r['num_detections'] for r in results)
        
        print(f"\n📊 Thống kê:")
        print(f"   Tổng số ảnh: {total_images}")
        print(f"   Ảnh có detection: {images_with_detections} ({images_with_detections/total_images:.1%})")
        print(f"   Tổng số detections: {total_detections}")
        print(f"   Trung bình detections/ảnh: {total_detections/total_images:.2f}")
        
        # Thống kê theo class
        class_counts = {}
        for result in results:
            for det in result['detections']:
                class_name = det['class_name']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        if class_counts:
            print(f"\n📈 Phân bố theo class (top 10):")
            sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
            for class_name, count in sorted_classes[:10]:
                print(f"   {class_name}: {count}")
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description='Inference YOLO Detection Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ sử dụng:

1. Dự đoán một ảnh:
   python inference_yolo.py --model runs/detect/plant_disease_detection/weights/best.pt --image test.jpg

2. Dự đoán thư mục ảnh:
   python inference_yolo.py --model runs/detect/plant_disease_detection/weights/best.pt --folder data/test/images

3. Dự đoán và lưu kết quả:
   python inference_yolo.py --model runs/detect/plant_disease_detection/weights/best.pt --image test.jpg --save results/

4. Đánh giá trên test set:
   python inference_yolo.py --model runs/detect/plant_disease_detection/weights/best.pt --test data/test/images

5. Tùy chỉnh threshold:
   python inference_yolo.py --model runs/detect/plant_disease_detection/weights/best.pt --image test.jpg --conf 0.5 --iou 0.5
        """
    )
    
    parser.add_argument('--model', type=str, 
                       default='runs/detect/plant_disease_detection/weights/best.pt',
                       help='Đường dẫn đến model .pt')
    parser.add_argument('--image', type=str, default=None,
                       help='Đường dẫn đến ảnh cần dự đoán')
    parser.add_argument('--folder', type=str, default=None,
                       help='Thư mục chứa ảnh cần dự đoán')
    parser.add_argument('--test', type=str, default=None,
                       help='Thư mục test images để đánh giá')
    parser.add_argument('--save', type=str, default=None,
                       help='Thư mục lưu kết quả')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold (0-1)')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IoU threshold cho NMS (0-1)')
    parser.add_argument('--no-show', action='store_true',
                       help='Không hiển thị ảnh kết quả')
    
    args = parser.parse_args()
    
    # Khởi tạo inference
    try:
        inference = YOLOInference(args.model, conf_threshold=args.conf, iou_threshold=args.iou)
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        return
    
    # Chạy inference
    if args.image:
        # Single image
        inference.predict_single(args.image, save_path=args.save, show=not args.no_show)
    
    elif args.folder:
        # Folder
        results = inference.predict_folder(args.folder, output_dir=args.save)
        print(f"\n✅ Đã xử lý {len(results)} ảnh")
    
    elif args.test:
        # Test set evaluation
        inference.evaluate_on_test_set(args.test)
    
    else:
        print("❌ Cần chỉ định --image, --folder, hoặc --test")
        parser.print_help()


if __name__ == "__main__":
    main()
