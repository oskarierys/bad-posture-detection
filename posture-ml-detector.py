import cv2
import mediapipe as mp
import numpy as np
import joblib
import json
import glob
import os
from collections import deque

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

class MLPostureDetector:
    def __init__(self, model_path=None, scaler_path=None, metadata_path=None):
        if model_path is None:
            model_path, scaler_path, metadata_path = self.find_latest_model()
        
        print(f"Loading modelu: {model_path}")
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.feature_names = metadata['features']
        self.label_names = metadata['label_names']
    
        print(f"Model loaded!")
        print(f"Features: {self.feature_names}")
        print(f"Classes: {self.label_names}")
        
        self.prediction_history = deque(maxlen=10)
        self.confidence_threshold = 0.6
        
    def find_latest_model(self):
        model_files = glob.glob('models/posture_model_*.joblib')
        if not model_files:
            raise FileNotFoundError(
                "No trained model!\n"
            )
        
        latest_model = max(model_files, key=os.path.getctime)
        
        scaler_path = latest_model.replace('posture_model_', 'scaler_')
        metadata_path = latest_model.replace('posture_model_', 'model_metadata_').replace('.joblib', '.json')
        
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"File not found: {scaler_path}")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"File not found: {metadata_path}")
        
        return latest_model, scaler_path, metadata_path
    
    def extract_features(self, landmarks):
        try:
            nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
            left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
            right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value]
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            
            shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
            
            head_tilt = abs(left_ear.y - right_ear.y)
            
            average_ear_x = (left_ear.x + right_ear.x) / 2
            average_shoulder_x = (left_shoulder.x + right_shoulder.x) / 2
            head_forward = abs(average_ear_x - average_shoulder_x)
            
            shoulder_rotation = abs(left_shoulder.x - right_shoulder.x)
            
            average_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
            nose_y = nose.y
            torso_lean = abs(nose_y - average_shoulder_y)
            
            nose_x = nose.x
            torso_lean_side = abs(nose_x - average_shoulder_x)
            
            shoulder_width = abs(left_shoulder.x - right_shoulder.x)
            
            head_to_shoulder_ratio = abs(average_ear_x - average_shoulder_x)
            
            avg_visibility = np.mean([
                nose.visibility,
                left_ear.visibility,
                right_ear.visibility,
                left_shoulder.visibility,
                right_shoulder.visibility
            ])
            
            features = [
                shoulder_diff,
                head_tilt,
                head_forward,
                shoulder_rotation,
                torso_lean,
                torso_lean_side,
                shoulder_width,
                head_to_shoulder_ratio,
                avg_visibility
            ]
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def predict(self, landmarks):
        features = self.extract_features(landmarks)
        
        if features is None:
            return None, None, None
        
        features_scaled = self.scaler.transform(features)
        
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        confidence = probabilities[prediction]
        
        label = self.label_names[prediction]
        
        self.prediction_history.append(prediction)
        
        if len(self.prediction_history) >= 5:
            smoothed_prediction = max(set(self.prediction_history), 
                                     key=self.prediction_history.count)
            smoothed_label = self.label_names[smoothed_prediction]
        else:
            smoothed_label = label
        
        return smoothed_label, confidence, probabilities
    
    def draw_results(self, image, label, confidence, probabilities):
        h, w = image.shape[:2]
        
        colors = {
            'Correct': (0, 255, 0),
            'Bad Head': (0, 165, 255),
            'Bad Upper Body': (0, 0, 255)
        }
        
        bar_y = 180
        bar_height = 40
        bar_width = 400
        
        cv2.putText(image, "Probabilities:", (10, bar_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        for i, (name, prob) in enumerate(zip(self.label_names, probabilities)):
            y = bar_y + 30 + i * (bar_height + 15)
            
            cv2.rectangle(image, (10, y), (10 + bar_width, y + bar_height),
                         (120, 120, 120), -1)
            cv2.rectangle(image, (10, y), (10 + bar_width, y + bar_height),
                         (255, 255, 255), 1)
            
            fill_width = int(bar_width * prob)
            bar_color = colors.get(name, (255, 255, 255))
            cv2.rectangle(image, (10, y), (10 + fill_width, y + bar_height),
                         bar_color, -1)
            
            text = f"{name}: {prob*100:.1f}%"
            cv2.putText(image, text, (20, y + 27),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
    
    def draw_feature_values(self, image, landmarks):
        features = self.extract_features(landmarks)
        if features is None:
            return
        
        h, w = image.shape[:2]
        panel_x = w - 350
        panel_y = 10
        
        cv2.rectangle(image, (panel_x, panel_y), (w - 10, panel_y + 200),
                     (0, 0, 0), -1)
        cv2.rectangle(image, (panel_x, panel_y), (w - 10, panel_y + 200),
                     (255, 255, 255), 2)
        
        cv2.putText(image, "Feature Values:", (panel_x + 10, panel_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        feature_values = features[0]
        y = panel_y + 60
        for name, value in zip(self.feature_names, feature_values):
            text = f"{name}: {value:.4f}"
            cv2.putText(image, text, (panel_x + 10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            y += 30
    
    def run(self, camera_id=0, show_features=False):
        capture = cv2.VideoCapture(camera_id)
        
        print("ML POSTURE DETECTOR - Real-time Detection")
        print(f"Model: {self.label_names}")
        print(f"Features: {len(self.feature_names)}")
        print("Press 'q' to quit")
        
        with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        ) as pose:
            
            while capture.isOpened():
                success, frame = capture.read()
                if not success:
                    break
                
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                result = pose.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                if result.pose_landmarks is not None:
                    mp_drawing.draw_landmarks(
                        image,
                        result.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(
                            color=(0, 255, 0), thickness=2, circle_radius=3
                        ),
                        connection_drawing_spec=mp_drawing.DrawingSpec(
                            color=(0, 255, 255), thickness=2
                        )
                    )
                    
                    label, confidence, probabilities = self.predict(
                        result.pose_landmarks.landmark
                    )
                    
                    if label is not None:
                        self.draw_results(image, label, confidence, probabilities)
                        
                        if show_features:
                            self.draw_feature_values(image, result.pose_landmarks.landmark)
                else:
                    h, w = image.shape[:2]
                    cv2.putText(image, "No person detected", (w//2 - 150, h//2),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                cv2.imshow('ML Posture Detector', image)
                
                key = cv2.waitKey(10) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('f'):
                    show_features = not show_features
                    print(f"Feature display: {'ON' if show_features else 'OFF'}")
        
        capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        detector = MLPostureDetector()
        detector.run(camera_id=0, show_features=False)
    except FileNotFoundError as e:
        print(f"\n{e}")
        print("\nSteps to follow:")
        print("1. Run: python dataset_collector.py")
        print("2. Collect data (min 50-100 samples each type)")
        print("3. Run: python model_trainer.py")
        print("4. Run: python ml_posture_detector.py")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

# TODO: Add logger and plotter for thesis