"""
Dataset Preparation Script
Chuyển đổi dataset từ Classification format sang YOLO format cho phone detection.

Lưu ý: Script này giả định điện thoại chiếm phần lớn ảnh,
nên sẽ tạo bounding box chiếm ~80% trung tâm ảnh.
"""

import os
import shutil
from pathlib import Path
from PIL import Image
import random

def create_yolo_dataset(source_dir: str, output_dir: str):
    """
    Chuyển đổi dataset classification sang YOLO format.
    
    Args:
        source_dir: Đường dẫn đến dataset gốc (chứa train/val folders)
        output_dir: Đường dẫn output cho YOLO dataset
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Tạo cấu trúc thư mục YOLO
    for split in ['train', 'val']:
        (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Class mapping: 0 = smartphone
    classes = ['smartphone']
    
    # Xử lý train và val
    for split in ['train', 'val']:
        smartphone_dir = source_path / split / 'smartphone'
        
        if not smartphone_dir.exists():
            print(f"Warning: {smartphone_dir} không tồn tại!")
            continue
        
        image_files = list(smartphone_dir.glob('*.jpg')) + list(smartphone_dir.glob('*.png'))
        print(f"Đang xử lý {len(image_files)} ảnh trong {split}/smartphone...")
        
        for img_file in image_files:
            try:
                # Mở ảnh để lấy kích thước
                with Image.open(img_file) as img:
                    width, height = img.size
                
                # Copy ảnh sang thư mục YOLO
                dest_img = output_path / 'images' / split / img_file.name
                shutil.copy2(img_file, dest_img)
                
                # Tạo YOLO label
                # Format: class_id x_center y_center width height (normalized 0-1)
                # Giả định phone chiếm ~70-90% trung tâm ảnh (random để tăng diversity)
                box_width = random.uniform(0.7, 0.9)
                box_height = random.uniform(0.7, 0.9)
                x_center = 0.5
                y_center = 0.5
                
                # Tạo file label .txt
                label_file = output_path / 'labels' / split / (img_file.stem + '.txt')
                with open(label_file, 'w') as f:
                    # class_id x_center y_center width height
                    f.write(f"0 {x_center} {y_center} {box_width} {box_height}\n")
                
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
    
    # Tạo data.yaml cho YOLO
    yaml_content = f"""# Dataset configuration for YOLO training
path: {output_path.absolute()}
train: images/train
val: images/val

# Classes
names:
  0: smartphone

# Number of classes
nc: 1
"""
    
    yaml_file = output_path / 'data.yaml'
    with open(yaml_file, 'w') as f:
        f.write(yaml_content)
    
    print(f"\n✅ Dataset YOLO đã được tạo tại: {output_path}")
    print(f"📄 Config file: {yaml_file}")
    
    # Thống kê
    train_images = len(list((output_path / 'images' / 'train').glob('*')))
    val_images = len(list((output_path / 'images' / 'val').glob('*')))
    print(f"\n📊 Thống kê:")
    print(f"   - Train images: {train_images}")
    print(f"   - Val images: {val_images}")


if __name__ == "__main__":
    # Đường dẫn
    source_dataset = r"c:\Users\Admin\Desktop\detection\dataset"
    yolo_dataset = r"c:\Users\Admin\Desktop\detection\yolo_dataset"
    
    print("=" * 50)
    print("🔄 Đang chuyển đổi dataset sang YOLO format...")
    print("=" * 50)
    
    create_yolo_dataset(source_dataset, yolo_dataset)
    
    print("\n✨ Hoàn tất! Bạn có thể bắt đầu training.")
