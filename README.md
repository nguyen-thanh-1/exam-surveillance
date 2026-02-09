# Exam Surveillance System
Hệ thống giám sát thi cử với Face Detection + Phone Detection

## Tính năng
- 👤 Đếm số người (face) trong khung hình
- 📱 Phát hiện điện thoại
- 🚨 Cảnh báo tự động khi:
  - Không có người (0 face)
  - Có nhiều hơn 1 người (2+ faces)
  - Phát hiện điện thoại
- 📸 Tự động chụp màn hình khi có vi phạm

## Cài đặt

```bash
# Clone repository
git clone https://github.com/nguyen-thanh-1/exam-surveillance.git
cd exam-surveillance

# Tạo virtual environment
python -m venv .venv

# Activate (Windows)
.\.venv\Scripts\activate

# Cài đặt dependencies
pip install -r requirements.txt
```

## Sử dụng

```bash
# Chạy với webcam
python exam_surveillance.py --source 0

# Chạy với video file
python exam_surveillance.py --source path/to/video.mp4
```

## Model
- **Phone Detector**: YOLOv8n fine-tuned (6.2MB)
- **Face Detector**: OpenCV Haar Cascade (built-in)

## Training (Optional)
Nếu muốn train lại model:

```bash
# Chuẩn bị dataset
python prepare_dataset.py

# Train model
python train_phone_detector.py
```

## Kết quả Training
- **mAP50**: 99.39%
- **mAP50-95**: 76.19%
