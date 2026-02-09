"""
Exam Surveillance System v2
Hệ thống giám sát thi cử với Face Detection + Phone Detection

Cải tiến:
- Sử dụng YOLOv8 pretrained trên COCO dataset
- Class "cell phone" đã được train với hàng ngàn ảnh thực tế
- Độ chính xác cao hơn, ít false positive

Author: AI Assistant
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from datetime import datetime
import time
import os


class ExamSurveillance:
    """
    Hệ thống giám sát thi cử.
    """
    
    # COCO class ID for cell phone
    CELL_PHONE_CLASS_ID = 67
    
    def __init__(self):
        """
        Khởi tạo hệ thống.
        """
        print("=" * 60)
        print("EXAM SURVEILLANCE SYSTEM v2")
        print("=" * 60)
        print("Dang khoi tao he thong...")
        
        # Tạo thư mục lưu cảnh báo
        self.alerts_dir = Path(__file__).resolve().parent / "alerts"
        self.alerts_dir.mkdir(exist_ok=True, parents=True)
        
        # Khởi tạo Face Detector (OpenCV Haar Cascade)
        print("   [+] Loading Face Detection...")
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_detector = cv2.CascadeClassifier(cascade_path)
        print("       OK - OpenCV Haar Cascade")
        
        # Khởi tạo Phone Detector (YOLOv8 pretrained COCO)
        # COCO dataset có class "cell phone" (class 67) với annotation thực tế
        print("   [+] Loading Phone Detection (YOLOv8-COCO)...")
        self.phone_detector = YOLO('yolov8n.pt')  # Pretrained on COCO
        print("       OK - YOLOv8n pretrained (cell phone class)")
        
        # Thống kê
        self.alert_count = 0
        self.last_alert_time = 0
        self.alert_cooldown = 3  # Giây giữa các cảnh báo
        
        print("\n[OK] He thong da san sang!")
        print("=" * 60)
    
    def detect_faces(self, frame):
        """
        Detect faces trong frame.
        
        Returns:
            Tuple (số face, list các bounding box)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces_detected = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        faces = []
        for (x, y, w, h) in faces_detected:
            faces.append({
                'bbox': (x, y, w, h),
                'confidence': 0.9
            })
        
        return len(faces), faces
    
    def detect_phones(self, frame):
        """
        Detect phones trong frame sử dụng YOLOv8 COCO pretrained.
        Chỉ lấy class "cell phone" (class 67).
        
        Returns:
            Tuple (có phone hay không, list các detections)
        """
        # Inference với confidence threshold cao hơn
        results = self.phone_detector(
            frame, 
            conf=0.5,           # Confidence threshold
            classes=[67],       # Chỉ detect class 67 = cell phone
            verbose=False
        )
        
        phones = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                
                # Double check class ID
                if class_id == self.CELL_PHONE_CLASS_ID:
                    phones.append({
                        'bbox': (x1, y1, x2 - x1, y2 - y1),
                        'confidence': confidence
                    })
        
        return len(phones) > 0, phones
    
    def check_violation(self, face_count: int, has_phone: bool):
        """
        Kiểm tra vi phạm.
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
        
        if current_time - self.last_alert_time < self.alert_cooldown:
            return None
        
        self.last_alert_time = current_time
        self.alert_count += 1
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        violation_text = "_".join(v.replace(" ", "-") for v in violations)
        filename = f"alert_{timestamp}_{violation_text[:50]}.jpg"
        filepath = self.alerts_dir / filename
        
        cv2.imwrite(str(filepath), frame)
        print(f"[ALERT] Saved: {filepath}")
        
        return filepath
    
    def draw_overlay(self, frame, face_count: int, faces: list, 
                     has_phone: bool, phones: list, violations: list):
        """
        Vẽ overlay lên frame.
        """
        h, w, _ = frame.shape
        
        # Calculate dynamic scales based on width
        # Base on 640 width
        scale_factor = w / 640.0
        # Ensure minimum readability for very small windows
        font_scale_header = max(0.4, 0.7 * scale_factor)
        font_scale_info = max(0.3, 0.6 * scale_factor)
        font_scale_box = max(0.3, 0.5 * scale_factor)
        thickness = max(1, int(2 * scale_factor))
        margin = max(5, int(20 * scale_factor))
        line_height = max(15, int(30 * scale_factor))
        
        # Màu dựa trên trạng thái
        if violations:
            status_color = (0, 0, 255)  # Đỏ
            status_text = "CANH BAO!"
            border_color = (0, 0, 255)
        else:
            status_color = (0, 255, 0)  # Xanh
            status_text = "BINH THUONG"
            border_color = (0, 255, 0)
        
        # Vẽ viền trạng thái (nếu muốn bỏ viền đen header thì vẫn giữ viền màu status)
        cv2.rectangle(frame, (0, 0), (w-1, h-1), border_color, thickness*2)
        
        # NOTE: Đã bỏ phần vẽ viền đen (header background) theo yêu cầu
        
        # Status text (Vẽ trực tiếp lên hình, thêm shadow/outline đen để dễ đọc)
        # Helper để vẽ text có viền
        def draw_text_with_outline(img, text, pos, scale, color, thick):
            cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thick+1) # Viền đen
            cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)
            
        # Status text position
        draw_text_with_outline(frame, status_text, (margin, margin + int(10*scale_factor) + 10), 
                             font_scale_header, status_color, thickness)
        
        # Info text
        info_text = f"Faces: {face_count} | Phone: {'YES' if has_phone else 'NO'}"
        draw_text_with_outline(frame, info_text, (margin, margin + int(10*scale_factor) + 10 + line_height),
                             font_scale_info, (255, 255, 255), thickness)
        
        # Timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        text_size = cv2.getTextSize(timestamp, cv2.FONT_HERSHEY_SIMPLEX, font_scale_info, thickness)[0]
        draw_text_with_outline(frame, timestamp, (w - text_size[0] - margin, margin + int(10*scale_factor) + 10),
                             font_scale_info, (255, 255, 255), thickness)
        
        # Vẽ face bounding boxes
        for face in faces:
            x, y, fw, fh = face['bbox']
            cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), thickness)
            # Label Face
            # cv2.putText(frame, "Face", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale_box, (0, 255, 0), thickness)
        
        # Vẽ phone bounding boxes
        for phone in phones:
            x, y, pw, ph = phone['bbox']
            cv2.rectangle(frame, (x, y), (x + pw, y + ph), (0, 0, 255), thickness)
            cv2.putText(frame, f"PHONE", 
                       (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale_box, (0, 0, 255), thickness)
        
        # Vẽ danh sách vi phạm
        if violations:
            y_offset = margin + int(10*scale_factor) + 10 + line_height * 2
            for i, v in enumerate(violations):
                draw_text_with_outline(frame, f"[!] {v}", (margin, y_offset + i*line_height),
                                     font_scale_info, (0, 0, 255), thickness)
        
        # Footer
        footer_text = f"Alerts: {self.alert_count} | 'Q' to quit"
        draw_text_with_outline(frame, footer_text, (margin, h - margin),
                             font_scale_box, (200, 200, 200), 1)
        
        return frame
    
    def run(self, source=0, width=640, height=480):
        """
        Chạy hệ thống giám sát.
        """
        print(f"\nSource: {'Webcam' if source == 0 else source}")
        print(f"Resolution: {width}x{height}")
        print(f"Alerts folder: {self.alerts_dir.absolute()}")
        print("\nNhan 'Q' de thoat\n")
        
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print("ERROR: Khong the mo camera/video!")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
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
                
                # Detect phones (COCO cell phone class)
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
                cv2.imshow("Exam Surveillance System v2", display_frame)
                
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
            print(f"   Thu muc luu: {self.alerts_dir.absolute()}")
            print("=" * 60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Exam Surveillance System v2")
    parser.add_argument('--source', type=str, default='0',
                       help='Video source (0 for webcam, or path to video)')
    parser.add_argument('--width', type=int, default=90,
                       help='Camera width (default: 90)')
    parser.add_argument('--height', type=int, default=90,
                       help='Camera height (default: 90)')
    
    args = parser.parse_args()
    
    source = int(args.source) if args.source.isdigit() else args.source
    
    system = ExamSurveillance()
    system.run(source=source, width=args.width, height=args.height)


if __name__ == "__main__":
    main()
