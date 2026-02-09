"""
Training Script for Phone Detector
Fine-tune YOLOv8n để detect smartphone trong ảnh.

Tối ưu cho máy 8GB RAM, không có GPU.
"""

from ultralytics import YOLO
from pathlib import Path
import os

def train_phone_detector():
    """
    Train YOLOv8n để detect smartphone.
    """
    # Đường dẫn
    project_dir = Path(r"c:\Users\Admin\Desktop\detection")
    data_yaml = project_dir / "yolo_dataset" / "data.yaml"
    models_dir = project_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Kiểm tra dataset
    if not data_yaml.exists():
        print("❌ Lỗi: Chưa có dataset YOLO!")
        print("   Vui lòng chạy: python prepare_dataset.py")
        return
    
    print("=" * 60)
    print("🚀 Bắt đầu training YOLOv8n Phone Detector")
    print("=" * 60)
    print(f"📁 Dataset: {data_yaml}")
    print(f"💾 Models dir: {models_dir}")
    print()
    
    # Load pretrained YOLOv8n
    model = YOLO('yolov8n.pt')
    
    # Training config - tối ưu cho 8GB RAM, CPU
    results = model.train(
        data=str(data_yaml),
        epochs=50,              # 50 epochs cho kết quả tốt
        imgsz=640,              # Image size
        batch=16,               # Batch size lớn hơn cho GPU
        patience=10,            # Early stopping
        save=True,
        save_period=10,         # Save checkpoint mỗi 10 epochs
        device=0,               # Chạy trên GPU (CUDA:0)
        workers=2,              # Số worker thấp để tiết kiệm RAM
        project=str(models_dir),
        name='phone_detector',
        exist_ok=True,
        pretrained=True,
        optimizer='AdamW',
        lr0=0.001,              # Learning rate
        lrf=0.01,               # Final learning rate
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        box=7.5,                # Box loss gain
        cls=0.5,                # Cls loss gain
        dfl=1.5,                # DFL loss gain
        hsv_h=0.015,            # Augmentation
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        verbose=True,
    )
    
    print("\n" + "=" * 60)
    print("✅ Training hoàn tất!")
    print("=" * 60)
    
    # Copy best model
    best_model = models_dir / "phone_detector" / "weights" / "best.pt"
    final_model = models_dir / "phone_detector_best.pt"
    
    if best_model.exists():
        import shutil
        shutil.copy2(best_model, final_model)
        print(f"📦 Best model saved: {final_model}")
    
    # Validation
    print("\n🔍 Đang validate model...")
    metrics = model.val()
    print(f"   mAP50: {metrics.box.map50:.4f}")
    print(f"   mAP50-95: {metrics.box.map:.4f}")
    
    return model


def export_model():
    """
    Export model sang ONNX để inference nhanh hơn trên CPU.
    """
    models_dir = Path(r"c:\Users\Admin\Desktop\detection\models")
    model_path = models_dir / "phone_detector_best.pt"
    
    if not model_path.exists():
        print("❌ Lỗi: Chưa có trained model!")
        print("   Vui lòng train trước.")
        return
    
    print("🔄 Đang export model sang ONNX...")
    
    model = YOLO(str(model_path))
    model.export(format='onnx', imgsz=640, simplify=True)
    
    print("✅ Export ONNX hoàn tất!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'export':
        export_model()
    else:
        train_phone_detector()
