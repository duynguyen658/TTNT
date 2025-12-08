"""
Script kiểm tra dataset YOLO detection trước khi train
"""

from pathlib import Path

import yaml


def check_dataset(data_yaml_path: str = "data/data.yaml"):
    """Kiểm tra dataset YOLO detection"""
    print("=" * 60)
    print("KIỂM TRA DATASET YOLO DETECTION")
    print("=" * 60)
    print()

    yaml_path = Path(data_yaml_path)
    if not yaml_path.exists():
        print(f"❌ Không tìm thấy file: {yaml_path}")
        return False

    # Đọc data.yaml
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            data_config = yaml.safe_load(f)
    except Exception as e:
        print(f"❌ Lỗi đọc data.yaml: {e}")
        return False

    print("📋 Thông tin dataset:")
    print(f"   Số classes: {data_config.get('nc', 'N/A')}")
    print(
        f"   Classes: {data_config.get('names', [])[:5]}{'...' if len(data_config.get('names', [])) > 5 else ''}"
    )
    print()

    # Kiểm tra từng split
    all_ok = True
    for split in ["train", "val", "test"]:
        if split not in data_config:
            if split == "test":
                print(f"⚠️  {split}: Không có (optional)")
                continue
            else:
                print(f"❌ {split}: Thiếu trong data.yaml")
                all_ok = False
                continue

        split_path_str = data_config[split]
        split_path = Path(split_path_str)

        # Resolve relative path
        if not split_path.is_absolute():
            yaml_dir = yaml_path.parent
            if ".." in str(split_path):
                split_path = (yaml_dir.parent / split_path).resolve()
            else:
                split_path = (yaml_dir / split_path).resolve()

        print(f"\n📁 {split.upper()}:")
        print(f"   Path: {split_path}")

        # Kiểm tra images
        if split_path.name == "images":
            images_path = split_path
            labels_path = split_path.parent / "labels"
        else:
            images_path = split_path / "images"
            labels_path = split_path / "labels"

        if not images_path.exists():
            print(f"   ❌ Images folder không tồn tại: {images_path}")
            all_ok = False
        else:
            image_files = (
                list(images_path.glob("*.jpg"))
                + list(images_path.glob("*.JPG"))
                + list(images_path.glob("*.png"))
                + list(images_path.glob("*.PNG"))
            )
            print(f"   ✅ Images: {len(image_files)} files")

        # Kiểm tra labels
        if not labels_path.exists():
            print(f"   ❌ Labels folder không tồn tại: {labels_path}")
            all_ok = False
        else:
            label_files = list(labels_path.glob("*.txt"))
            print(f"   ✅ Labels: {len(label_files)} files")

            # Kiểm tra matching
            if images_path.exists():
                image_names = {f.stem for f in image_files}
                label_names = {f.stem for f in label_files}

                missing_labels = image_names - label_names
                missing_images = label_names - image_names

                if missing_labels:
                    print(f"   ⚠️  {len(missing_labels)} images không có labels")
                    if len(missing_labels) <= 5:
                        for name in list(missing_labels)[:5]:
                            print(f"      - {name}")

                if missing_images:
                    print(f"   ⚠️  {len(missing_images)} labels không có images")

            # Kiểm tra format label
            if label_files:
                sample_label = label_files[0]
                try:
                    with open(sample_label, "r") as f:
                        lines = f.readlines()
                    if lines:
                        first_line = lines[0].strip().split()
                        if len(first_line) == 5:
                            print(f"   ✅ Label format OK (sample: {first_line})")
                        else:
                            print(
                                f"   ⚠️  Label format có vẻ sai (expected 5 values, got {len(first_line)})"
                            )
                except Exception as e:
                    print(f"   ⚠️  Không thể đọc label: {e}")

    print("\n" + "=" * 60)
    if all_ok:
        print("✅ Dataset OK, sẵn sàng để train!")
    else:
        print("❌ Dataset có vấn đề, vui lòng sửa trước khi train")
    print("=" * 60)

    return all_ok


if __name__ == "__main__":
    import sys

    yaml_file = sys.argv[1] if len(sys.argv) > 1 else "data/data.yaml"
    check_dataset(yaml_file)
