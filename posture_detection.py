#Instal dependencies
'''
pip install mediapipe
pip install opencv-python
pip install ultralytics
pip install numpy
'''

#Import
import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
import math

class PostureDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        self.yolo_model = YOLO('yolov8n.pt')
                
        self.neck_angle_threshold_min = 5.0 
        self.neck_angle_threshold_max = 30.0  
        self.back_angle_threshold_min = 120.0
        self.back_angle_threshold_max = 170.0
        
        self.shoulder_distance_threshold = 0.1

    def calculate_vertical_angle(self, top_point, bottom_point):
        """
        Menghitung sudut terhadap garis vertikal
        Return sudut dalam derajat (0-180)
        """
        if any(math.isnan(coord) for point in [top_point, bottom_point] for coord in point):
            return 0.0
            
        # Vektor dari bottom ke top point
        vector = np.array([
            top_point[0] - bottom_point[0],
            top_point[1] - bottom_point[1]
        ])
        
        # Vektor vertikal (ke atas)
        vertical = np.array([0, -1])
        
        # Hitung sudut menggunakan dot product
        dot_product = np.dot(vector, vertical)
        norms = np.linalg.norm(vector) * np.linalg.norm(vertical)
        
        if norms == 0:
            return 0.0
            
        cos_angle = dot_product / norms
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        
        return angle

    def calculate_angle(self, point1, point2, point3):
        """Menghitung sudut antara tiga titik"""
        if any(math.isnan(coord) for point in [point1, point2, point3] for coord in point):
            return 0.0
            
        a = np.array(point1)
        b = np.array(point2)
        c = np.array(point3)
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)

    def detect_orientation(self, landmarks):
        """Mendeteksi orientasi tubuh (samping atau depan)"""
        left_shoulder = np.array([landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                                landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].z])
        right_shoulder = np.array([landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                                 landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].z])
        
        shoulder_distance = abs(left_shoulder[0] - right_shoulder[0])
        return "samping" if shoulder_distance < self.shoulder_distance_threshold else "depan"

    def detect_posture(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = self.pose.process(frame_rgb)
        yolo_results = self.yolo_model(frame)
        
        if pose_results.pose_landmarks:
            self.mp_draw.draw_landmarks(
                frame,
                pose_results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )
            
            landmarks = pose_results.pose_landmarks.landmark
            h, w, _ = frame.shape
            
            orientation = self.detect_orientation(landmarks)
            
            if orientation == "samping":
                ear = np.array([landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR].x * w,
                              landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR].y * h])
                nose = np.array([landmarks[self.mp_pose.PoseLandmark.NOSE].x * w,
                               landmarks[self.mp_pose.PoseLandmark.NOSE].y * h])
                shoulder = np.array([landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w,
                                   landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h])
                hip = np.array([landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].x * w,
                              landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].y * h])
                knee = np.array([landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE].x * w,
                               landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE].y * h])
                
                # Hitung sudut leher terhadap vertikal
                neck_angle = self.calculate_vertical_angle(ear, shoulder)
                back_angle = self.calculate_angle(shoulder, hip, knee)
                
                # Analisis postur dengan threshold yang sudah disesuaikan
                if neck_angle > self.neck_angle_threshold_max:
                    posture_status = "Peringatan: Leher Terlalu Menunduk!"
                    color = (0, 0, 255)
                elif neck_angle < self.neck_angle_threshold_min:
                    posture_status = "Peringatan: Leher Terlalu Mendongak!"
                    color = (0, 0, 255)
                elif back_angle < self.back_angle_threshold_min:
                    posture_status = "Peringatan: Punggung Terlalu Membungkuk!"
                    color = (0, 0, 255)
                else:
                    posture_status = "Postur Baik"
                    color = (0, 255, 0)
                
                # Visualisasi
                cv2.putText(frame, f"Orientasi: {orientation}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Sudut Leher: {neck_angle:.1f}°", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Sudut Punggung: {back_angle:.1f}°", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, posture_status, (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Gambar garis referensi vertikal untuk leher
                vertical_top = (int(shoulder[0]), int(shoulder[1] - 100))
                cv2.line(frame, tuple(map(int, shoulder)), vertical_top, (0, 255, 255), 2)  # Garis vertikal
                cv2.line(frame, tuple(map(int, shoulder)), tuple(map(int, ear)), (255, 0, 0), 2)  # Garis leher
                cv2.line(frame, tuple(map(int, shoulder)), tuple(map(int, hip)), (0, 255, 0), 2)  # Garis punggung
                cv2.line(frame, tuple(map(int, hip)), tuple(map(int, knee)), (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Silakan menghadap ke samping", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Gambar deteksi YOLO
        for result in yolo_results:
            boxes = result.boxes
            for box in boxes:
                if box.cls[0].item() == 0:  
                    x1, y1, x2, y2 = box.xyxy[0]
                    cv2.rectangle(frame, 
                                (int(x1), int(y1)), 
                                (int(x2), int(y2)),
                                (255, 0, 0), 2)
        
        return frame

def main():
    cap = cv2.VideoCapture(0)
    detector = PostureDetector()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = detector.detect_posture(frame)
        cv2.imshow('Side View Posture Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()