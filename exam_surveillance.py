"""
Exam Surveillance System
Hệ thống giám sát thi cử với Face Detection + Phone Detection

Tính năng:
- Đếm số người (face) trong khung hình
- Phát hiện điện thoại
- Cảnh báo tự động khi:
  + Không có người (0 face)
  + Có nhiều hơn 1 người (2+ faces)
  + Phát hiện điện thoại

Author: AI Assistant
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from datetime import datetime
import time
import os

# MediaPipe new API (0.10.x)
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class ExamSurveillance:
    """
    Hệ thống giám sát thi cử.
    """
    
    def __init__(self, phone_model_path: str = None):
        """
        Khởi tạo hệ thống.
        
        Args:
            phone_model_path: Đường dẫn đến model phone detector
        """
        print("🔄 Đang khởi tạo hệ thống giám sát...")
        
        # Tạo thư mục lưu cảnh báo
        self.alerts_dir = Path(r"c:\Users\Admin\Desktop\detection\alerts")
        self.alerts_dir.mkdir(exist_ok=True)
        
        # Download và load MediaPipe Face Detector model
        print("   📍 Loading Face Detection...")
        self.face_detector = self._init_face_detector()
        
        # Khởi tạo Phone Detector (YOLOv8)
        print("   📱 Loading Phone Detector...")
        if phone_model_path and Path(phone_model_path).exists():
            self.phone_detector = YOLO(phone_model_path)
            print(f"      ✅ Loaded: {phone_model_path}")
        else:
            # Thử tìm model trong thư mục mặc định
            default_path = Path(r"c:\Users\Admin\Desktop\detection\models\phone_detector_best.pt")
            if default_path.exists():
                self.phone_detector = YOLO(str(default_path))
                print(f"      ✅ Loaded: {default_path}")
            else:
                print("      ⚠️ Không tìm thấy phone model! Chỉ detect face.")
                self.phone_detector = None
        
        # Thống kê
        self.alert_count = 0
        self.last_alert_time = 0
        self.alert_cooldown = 3  # Giây giữa các cảnh báo
        
        print("✅ Hệ thống đã sẵn sàng!\n")
    
    def _init_face_detector(self):
        """
        Khởi tạo face detector sử dụng OpenCV Haar Cascade (backup).
        MediaPipe tasks API cần download model, nên dùng OpenCV cho đơn giản.
        """
        # Sử dụng OpenCV Haar Cascade - nhẹ và nhanh
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        return face_cascade
    
    def detect_faces(self, frame):
        """
        Detect faces trong frame sử dụng OpenCV Haar Cascade.
        
        Returns:
            Tuple (số face, list các bounding box)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces_detected = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        faces = []
        for (x, y, w, h) in faces_detected:
            faces.append({
                'bbox': (x, y, w, h),
                'confidence': 0.9  # Haar cascade không có confidence score
            })
        
        return len(faces), faces
    
    def detect_phones(self, frame):
        """
        Detect phones trong frame sử dụng YOLOv8.
        
        Returns:
            Tuple (có phone hay không, list các detections)
        """
        if self.phone_detector is None:
            return False, []
        
        # Inference
        results = self.phone_detector(frame, conf=0.4, verbose=False)
        
        phones = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                confidence = float(box.conf[0])
                phones.append({
                    'bbox': (x1, y1, x2 - x1, y2 - y1),
                    'confidence': confidence
                })
        
        return len(phones) > 0, phones
    
    def check_violation(self, face_count: int, has_phone: bool):
        """
        Kiểm tra vi phạm.
        
        Returns:
            Tuple (có vi phạm không, loại vi phạm)
        """
        violations = []
        
        if face_count == 0:
            violations.append("KHONG CO NGUOI")
        elif face_count > 1:
            violations.append(f"PHAT HIEN {face_count} NGUOI")
        
        if has_phone:
            violations.append("PHAT HIEN DIEN THOAI")
        
        return len(violations) > 0, violations
    
    def save_alert(self, frame, violations: list):
        """
        Lưu screenshot khi có vi phạm.
        """
        current_time = time.time()
        
        # Cooldown để không lưu quá nhiều ảnh
        if current_time - self.last_alert_time < self.alert_cooldown:
            return None
        
        self.last_alert_time = current_time
        self.alert_count += 1
        
        # Tạo tên file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        violation_text = "_".join(v.replace(" ", "-") for v in violations)
        filename = f"alert_{timestamp}_{violation_text[:50]}.jpg"
        filepath = self.alerts_dir / filename
        
        # Lưu ảnh
        cv2.imwrite(str(filepath), frame)
        print(f"[ALERT] Da luu canh bao: {filepath}")
        
        return filepath
    
    def draw_overlay(self, frame, face_count: int, faces: list, 
                     has_phone: bool, phones: list, violations: list):
        """
        Vẽ overlay lên frame.
        """
        h, w, _ = frame.shape
        
        # Màu dựa trên trạng thái
        if violations:
            status_color = (0, 0, 255)  # Đỏ
            status_text = "CANH BAO!"
            border_color = (0, 0, 255)
        else:
            status_color = (0, 255, 0)  # Xanh
            status_text = "BINH THUONG"
            border_color = (0, 255, 0)
        
        # Vẽ viền trạng thái
        cv2.rectangle(frame, (0, 0), (w-1, h-1), border_color, 8)
        
        # Header background
        cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, 0), (w, 80), border_color, 3)
        
        # Status text
        cv2.putText(frame, status_text, (20, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        # Info text
        info_text = f"Faces: {face_count} | Phone: {'YES' if has_phone else 'NO'}"
        cv2.putText(frame, info_text, (20, 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        text_size = cv2.getTextSize(timestamp, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        cv2.putText(frame, timestamp, (w - text_size[0] - 20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Vẽ face bounding boxes
        for face in faces:
            x, y, fw, fh = face['bbox']
            cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)
            cv2.putText(frame, f"Face", 
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Vẽ phone bounding boxes
        for phone in phones:
            x, y, pw, ph = phone['bbox']
            cv2.rectangle(frame, (x, y), (x + pw, y + ph), (0, 0, 255), 3)
            cv2.putText(frame, f"PHONE {phone['confidence']:.2f}", 
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Vẽ danh sách vi phạm
        if violations:
            y_offset = 100
            for i, v in enumerate(violations):
                cv2.putText(frame, f"[!] {v}", (20, y_offset + i*30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Footer
        footer_text = f"Alerts: {self.alert_count} | Press 'Q' to quit"
        cv2.putText(frame, footer_text, (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        return frame
    
    def run(self, source=0):
        """
        Chạy hệ thống giám sát.
        
        Args:
            source: 0 cho webcam, hoặc path đến video file
        """
        print("=" * 60)
        print("HE THONG GIAM SAT THI CU")
        print("=" * 60)
        print(f"Source: {'Webcam' if source == 0 else source}")
        print(f"Alerts folder: {self.alerts_dir}")
        print("\nNhan 'Q' de thoat")
        print("=" * 60)
        
        # Mở camera/video
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print("ERROR: Khong the mo camera/video!")
            return
        
        # Cấu hình camera
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        fps_time = time.time()
        fps = 0
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("WARNING: Khong the doc frame!")
                    break
                
                # Detect faces
                face_count, faces = self.detect_faces(frame)
                
                # Detect phones
                has_phone, phones = self.detect_phones(frame)
                
                # Check violations
                is_violation, violations = self.check_violation(face_count, has_phone)
                
                # Save alert if violation
                if is_violation:
                    self.save_alert(frame.copy(), violations)
                
                # Draw overlay
                display_frame = self.draw_overlay(
                    frame.copy(), face_count, faces, 
                    has_phone, phones, violations
                )
                
                # Calculate FPS
                frame_count += 1
                if frame_count >= 10:
                    fps = frame_count / (time.time() - fps_time)
                    fps_time = time.time()
                    frame_count = 0
                
                # Draw FPS
                cv2.putText(display_frame, f"FPS: {fps:.1f}", 
                           (display_frame.shape[1] - 120, display_frame.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                
                # Show
                cv2.imshow("Exam Surveillance System", display_frame)
                
                # Check for quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    print("\nDang thoat...")
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            print("\n" + "=" * 60)
            print("THONG KE PHIEN LAM VIEC")
            print("=" * 60)
            print(f"   Tong so canh bao: {self.alert_count}")
            print(f"   Thu muc luu: {self.alerts_dir}")
            print("=" * 60)


def main():
    """
    Entry point.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Exam Surveillance System")
    parser.add_argument('--source', type=str, default='0',
                       help='Video source (0 for webcam, or path to video)')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to phone detector model')
    
    args = parser.parse_args()
    
    # Parse source
    source = int(args.source) if args.source.isdigit() else args.source
    
    # Create and run system
    system = ExamSurveillance(phone_model_path=args.model)
    system.run(source=source)


if __name__ == "__main__":
    main()
