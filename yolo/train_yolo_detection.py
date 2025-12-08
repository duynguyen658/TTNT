"""
Script training YOLOv8 Detection model nhận dạng bệnh cây trồng
Hỗ trợ dataset YOLO format với bounding boxes annotations
"""
import os
import sys
import yaml
import json
import shutil
from pathlib import Path
from typing import Optional, List, Dict
import random

# Kiểm tra YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print(f"✅ Ultralytics YOLO đã sẵn sàng")
except ImportError as e:
    YOLO_AVAILABLE = False
    print(f"❌ Ultralytics YOLO chưa được cài đặt: {e}")
    print("👉 Giải pháp: pip install ultralytics")
    sys.exit(1)


def validate_yolo_dataset(dataset_path: str, data_yaml_path: Optional[str] = None):
    """
    Kiểm tra và validate dataset YOLO detection format
    
    YOLO Detection format:
    dataset/
        train/
            images/
                img1.jpg
                img2.jpg
            labels/
                img1.txt
                img2.txt
        val/ (hoặc valid/)
            images/
                img1.jpg
            labels/
                img1.txt
        test/ (optional)
            images/
                img1.jpg
            labels/
                img1.txt
        data.yaml
    """
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        print(f"❌ Không tìm thấy dataset tại: {dataset_path}")
        return False, None, None
    
    # Tìm file data.yaml
    if data_yaml_path:
        yaml_path = Path(data_yaml_path)
    else:
        yaml_path = dataset_path / "data.yaml"
        if not yaml_path.exists():
            # Thử tìm trong thư mục cha
            yaml_path = dataset_path.parent / "data.yaml"
    
    if not yaml_path.exists():
        print(f"⚠️  Không tìm thấy data.yaml, sẽ tạo tự động")
        return True, None, None
    
    # Đọc data.yaml
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data_config = yaml.safe_load(f)
        
        class_names = data_config.get('names', [])
        num_classes = data_config.get('nc', len(class_names))
        
        print(f"\n📋 Dataset config:")
        print(f"   Số classes: {num_classes}")
        print(f"   Classes: {class_names[:5]}{'...' if len(class_names) > 5 else ''}")
        
        # Kiểm tra cấu trúc thư mục
        train_path = Path(data_config.get('train', 'train'))
        val_path = Path(data_config.get('val', 'valid'))
        
        # Nếu là relative path, resolve từ dataset_path
        if not train_path.is_absolute():
            train_path = dataset_path.parent / train_path if '..' in str(train_path) else dataset_path / train_path
        if not val_path.is_absolute():
            val_path = dataset_path.parent / val_path if '..' in str(val_path) else dataset_path / val_path
        
        # Kiểm tra images và labels
        train_images = train_path / "images" if (train_path / "images").exists() else train_path
        train_labels = train_path / "labels" if (train_path / "labels").exists() else train_path
        
        val_images = val_path / "images" if (val_path / "images").exists() else val_path
        val_labels = val_path / "labels" if (val_path / "labels").exists() else val_path
        
        # Đếm số ảnh
        train_img_files = list(train_images.glob("*.jpg")) + list(train_images.glob("*.JPG")) + \
                         list(train_images.glob("*.png")) + list(train_images.glob("*.PNG"))
        val_img_files = list(val_images.glob("*.jpg")) + list(val_images.glob("*.JPG")) + \
                      list(val_images.glob("*.png")) + list(val_images.glob("*.PNG"))
        
        print(f"\n📊 Thống kê dataset:")
        print(f"   Train images: {len(train_img_files)}")
        print(f"   Val images: {len(val_img_files)}")
        
        # Kiểm tra labels
        train_label_files = list(train_labels.glob("*.txt"))
        val_label_files = list(val_labels.glob("*.txt"))
        
        print(f"   Train labels: {len(train_label_files)}")
        print(f"   Val labels: {len(val_label_files)}")
        
        if len(train_img_files) == 0:
            print(f"❌ Không tìm thấy ảnh training")
            return False, None, None
        
        if len(train_label_files) == 0:
            print(f"⚠️  Không tìm thấy labels, dataset có thể chỉ có ảnh (classification)")
            return False, None, None
        
        return True, str(yaml_path), data_config
        
    except Exception as e:
        print(f"❌ Lỗi khi đọc data.yaml: {e}")
        return False, None, None


def create_data_yaml(
    dataset_path: str,
    class_names: List[str],
    train_path: str = "train/images",
    val_path: str = "valid/images",
    test_path: Optional[str] = None,
    output_path: Optional[str] = None
):
    """
    Tạo file data.yaml cho YOLO detection
    """
    dataset_path = Path(dataset_path)
    
    if output_path is None:
        output_path = dataset_path / "data.yaml"
    else:
        output_path = Path(output_path)
    
    data_config = {
        'path': str(dataset_path.absolute()),
        'train': train_path,
        'val': val_path,
        'nc': len(class_names),
        'names': class_names
    }
    
    if test_path:
        data_config['test'] = test_path
    
    # Lưu file
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(data_config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"✅ Đã tạo data.yaml tại: {output_path}")
    return str(output_path)


def train_yolo_detection(
    data_yaml_path: str,
    model_size: str = "s",  # n, s, m, l, x
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: Optional[str] = None,
    output_dir: str = "runs/detect",
    project_name: str = "plant_disease_detection",
    **kwargs
):
    """
    Train YOLOv8 Detection model
    
    Args:
        data_yaml_path: Đường dẫn đến file data.yaml
        model_size: Kích thước model (n, s, m, l, x)
        epochs: Số epochs
        imgsz: Kích thước ảnh (mặc định 640 cho detection)
        batch: Batch size
        device: Device ('cpu', 'cuda', '0', '1', etc.)
        output_dir: Thư mục lưu kết quả
        project_name: Tên project
        **kwargs: Các tham số khác cho model.train()
    """
    if not YOLO_AVAILABLE:
        print("❌ YOLO chưa được cài đặt")
        return None
    
    print("=" * 60)
    print("TRAINING YOLOv8 DETECTION MODEL")
    print("=" * 60)
    print()
    
    # Kiểm tra data.yaml
    yaml_path = Path(data_yaml_path)
    if not yaml_path.exists():
        print(f"❌ Không tìm thấy data.yaml tại: {data_yaml_path}")
        return None
    
    # Đọc và fix data.yaml nếu cần (convert relative paths to absolute)
    temp_yaml = None
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data_config = yaml.safe_load(f)
        
        # Fix relative paths trong data.yaml
        yaml_dir = yaml_path.parent
        if 'path' not in data_config or not Path(data_config['path']).is_absolute():
            data_config['path'] = str(yaml_dir.absolute())
        
        # Fix train/val paths và validate
        for split in ['train', 'val', 'test']:
            if split in data_config:
                split_path = Path(data_config[split])
                if not split_path.is_absolute():
                    # Nếu là relative path, resolve từ yaml_dir
                    if '..' in str(split_path):
                        resolved_path = (yaml_dir.parent / split_path).resolve()
                    else:
                        resolved_path = (yaml_dir / split_path).resolve()
                    data_config[split] = str(resolved_path)
                
                # Kiểm tra labels folder
                split_path = Path(data_config[split])
                labels_path = split_path.parent / "labels" if split_path.name == "images" else split_path / "labels"
                
                if labels_path.exists():
                    label_files = list(labels_path.glob("*.txt"))
                    if len(label_files) == 0:
                        print(f"⚠️  Cảnh báo: Không tìm thấy file .txt trong {labels_path}")
                    else:
                        print(f"✅ {split}: Tìm thấy {len(label_files)} label files")
                else:
                    print(f"⚠️  Cảnh báo: Không tìm thấy thư mục labels tại {labels_path}")
        
        # Lưu lại data.yaml đã fix (tạm thời)
        temp_yaml = yaml_path.with_suffix('.temp.yaml')
        with open(temp_yaml, 'w', encoding='utf-8') as f:
            yaml.dump(data_config, f, default_flow_style=False, allow_unicode=True)
        
        # Sử dụng temp yaml
        yaml_path = temp_yaml
        print(f"📝 Đã fix paths trong data.yaml")
        
    except Exception as e:
        print(f"⚠️  Không thể fix data.yaml: {e}, sử dụng file gốc")
    
    # Load pretrained model
    model_name = f"yolov8{model_size}.pt"
    print(f"📥 Đang tải pretrained model: {model_name}")
    
    try:
        model = YOLO(model_name)
        print(f"✅ Đã tải model: {model_name}")
    except Exception as e:
        print(f"❌ Không thể tải model: {e}")
        return None
    
    # Training parameters
    print(f"\n⚙️  Cấu hình training:")
    print(f"   Data config: {data_yaml_path}")
    print(f"   Model: {model_name}")
    print(f"   Epochs: {epochs}")
    print(f"   Image size: {imgsz}")
    print(f"   Batch size: {batch}")
    print(f"   Device: {device or 'auto'}")
    print()
    
    # Training parameters cho detection
    train_params = {
        "data": str(yaml_path.absolute()),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "project": output_dir,
        "name": project_name,
        "exist_ok": True,
        "pretrained": True,
        "verbose": True,
        "seed": 42,
        "deterministic": True,
        "amp": True,  # Automatic Mixed Precision
        # Augmentation parameters
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        # Learning rate
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        # Loss weights
        "box": 7.5,
        "cls": 0.5,
        "dfl": 1.5,
        # Other
        "close_mosaic": 10,
        "resume": False,
        "fraction": 1.0,
        "profile": False,
        "freeze": None,
        # Windows fix: set workers=0 để tránh lỗi multiprocessing/paging file
        "workers": 0 if os.name == 'nt' else 8,  # Windows: 0, Linux/Mac: 8
        **kwargs
    }
    
    # Train
    try:
        print("🚀 Bắt đầu training...")
        results = model.train(**train_params)
        
        print("\n" + "=" * 60)
        print("✅ TRAINING HOÀN THÀNH!")
        print("=" * 60)
        
        # Tìm file model tốt nhất
        best_model_path = Path(output_dir) / project_name / "weights" / "best.pt"
        last_model_path = Path(output_dir) / project_name / "weights" / "last.pt"
        
        if best_model_path.exists():
            print(f"\n📦 Model tốt nhất: {best_model_path}")
            print(f"📦 Model cuối cùng: {last_model_path}")
            
            # Copy model vào thư mục models
            models_dir = Path("models")
            models_dir.mkdir(exist_ok=True)
            
            final_model_path = models_dir / f"yolo_detection_{model_size}.pt"
            shutil.copy2(best_model_path, final_model_path)
            print(f"📦 Đã copy model vào: {final_model_path}")
            
            # Lưu class names và config
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data_config = yaml.safe_load(f)
            
            classes_file = final_model_path.with_suffix('.json').with_name(
                final_model_path.stem + '_config.json'
            )
            with open(classes_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "class_names": data_config.get('names', []),
                    "num_classes": data_config.get('nc', 0),
                    "model_size": model_size,
                    "mode": "detection"
                }, f, indent=2, ensure_ascii=False)
            
            print(f"📝 Đã lưu config: {classes_file}")
            
            # Hiển thị metrics nếu có
            if hasattr(results, 'results_dict'):
                print(f"\n📊 Metrics:")
                metrics = results.results_dict
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        print(f"   {key}: {value:.4f}")
        
        # Cleanup temp yaml file
        if temp_yaml and Path(temp_yaml).exists():
            try:
                Path(temp_yaml).unlink()
            except:
                pass
        
        return results
        
    except Exception as e:
        print(f"❌ Lỗi khi training: {e}")
        import traceback
        traceback.print_exc()
        
        # Cleanup temp yaml file
        if temp_yaml and Path(temp_yaml).exists():
            try:
                Path(temp_yaml).unlink()
            except:
                pass
        
        return None


def train_from_dataset_path(
    dataset_path: str = "data",
    data_yaml_path: Optional[str] = None,
    model_size: str = "s",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: Optional[str] = None,
    output_dir: str = "runs/detect",
    project_name: str = "plant_disease_detection"
):
    """
    Train YOLO detection từ đường dẫn dataset (tự động tìm data.yaml)
    """
    print("=" * 60)
    print("KIỂM TRA DATASET")
    print("=" * 60)
    print()
    
    # Validate dataset
    is_valid, yaml_path, data_config = validate_yolo_dataset(dataset_path, data_yaml_path)
    
    if not is_valid:
        print("\n❌ Dataset không hợp lệ hoặc không tìm thấy")
        print("\n👉 Dataset cần có cấu trúc:")
        print("   dataset/")
        print("       train/")
        print("           images/")
        print("           labels/")
        print("       valid/")
        print("           images/")
        print("           labels/")
        print("       data.yaml")
        return None
    
    if yaml_path is None:
        print("\n⚠️  Không tìm thấy data.yaml, cần tạo thủ công")
        return None
    
    # Train
    results = train_yolo_detection(
        data_yaml_path=yaml_path,
        model_size=model_size,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        output_dir=output_dir,
        project_name=project_name
    )
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Train YOLOv8 Detection model nhận dạng bệnh cây trồng',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ sử dụng:

1. Train với data.yaml mặc định:
   python train_yolo_detection.py --data data/data.yaml --epochs 100

2. Train từ thư mục dataset (tự động tìm data.yaml):
   python train_yolo_detection.py --dataset data --epochs 100

3. Train với model lớn hơn:
   python train_yolo_detection.py --data data/data.yaml --model-size m --epochs 150

4. Train trên GPU:
   python train_yolo_detection.py --data data/data.yaml --device cuda

5. Train với batch size nhỏ hơn (nếu GPU memory không đủ):
   python train_yolo_detection.py --data data/data.yaml --batch 8
        """
    )
    
    parser.add_argument('--data', '--data-yaml', type=str, default=None,
                       dest='data_yaml',
                       help='Đường dẫn đến file data.yaml')
    parser.add_argument('--dataset', type=str, default='data',
                       help='Đường dẫn đến thư mục dataset (tự động tìm data.yaml)')
    parser.add_argument('--model-size', type=str, default='s',
                       choices=['n', 's', 'm', 'l', 'x'],
                       help='Kích thước model: n (nano), s (small), m (medium), l (large), x (xlarge)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Số epochs để train (mặc định: 100)')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Kích thước ảnh (mặc định: 640 cho detection)')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size (mặc định: 16)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cpu, cuda, 0, 1, etc.). Mặc định: auto')
    parser.add_argument('--output-dir', type=str, default='runs/detect',
                       help='Thư mục lưu kết quả (mặc định: runs/detect)')
    parser.add_argument('--project-name', type=str, default='plant_disease_detection',
                       help='Tên project (mặc định: plant_disease_detection)')
    
    args = parser.parse_args()
    
    # Train
    if args.data_yaml:
        # Train với data.yaml được chỉ định
        results = train_yolo_detection(
            data_yaml_path=args.data_yaml,
            model_size=args.model_size,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            output_dir=args.output_dir,
            project_name=args.project_name
        )
    else:
        # Train từ dataset path (tự động tìm data.yaml)
        results = train_from_dataset_path(
            dataset_path=args.dataset,
            model_size=args.model_size,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            output_dir=args.output_dir,
            project_name=args.project_name
        )
