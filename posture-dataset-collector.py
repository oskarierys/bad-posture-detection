import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from datetime import datetime
import os
import json

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

class PostureDetectionDatasetCollector:
    def __init__(self, save_dir="data-set"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.dataset = []

        self.counts = {
            'correct': 0,
            'bad_head_position': 0,
            'bad_upper_body_position': 0
        }

        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

    def posture_check(self, landmarks):
        try:
            nose = landmarks[self.mp_pose.PoseLandmark.NOSE.value]
            left_ear = landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value]
            right_ear = landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value]
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

            # Shoulder asymetry
            shoulder_diff = abs(left_shoulder.y - right_shoulder.y)

            # Head tilt
            head_tilt = abs(left_ear.y - right_ear.y)

            # Head forward
            average_ear_x = (left_ear.x + right_ear.x) / 2
            average_shoulder_x = (left_shoulder.x + right_shoulder.x) / 2
            head_forward = abs(average_ear_x - average_shoulder_x)

            # Shoulder rotation
            shoulder_rotation = abs(left_shoulder.x - right_shoulder.x)

            # Torso lean - (forward/backward and sideways)
            average_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
            nose_y = nose.y
            nose_x = nose.x
            torso_lean = abs(nose_y - average_shoulder_y)
            torso_lean_side = abs(nose_x - average_shoulder_x)
 
            # Shoulder width ratio
            shoulder_width = abs(left_shoulder.x - right_shoulder.x)

            # Head to shoulder ratio
            head_to_shoulder_ratio = abs(average_ear_x - average_shoulder_x)

            # Visibility
            avg_visibility = np.mean([
                nose.visibility,
                left_ear.visibility,
                right_ear.visibility,
                left_shoulder.visibility,
                right_shoulder.visibility
            ])

            features = {
                'shoulder_asymetry': shoulder_diff,
                'head_tilt': head_tilt,
                'head_forward': head_forward,
                'shoulder_rotation': shoulder_rotation,
                'torso_lean': torso_lean,
                'torso_lean_side': torso_lean_side,
                'shoulder_width': shoulder_width,
                'head_to_shoulder_ratio': head_to_shoulder_ratio,
                'avg_visibility': avg_visibility
            }

            return features
        
        except Exception as e:
            print(f"Error in festure extraction: {e}")
            return None
    
    def add_data(self, landmarks, label):
        features = self.posture_check(landmarks)

        if features is not None:
            features['label'] = label
            features['timestamp'] = datetime.now().isoformat()
            self.dataset.append(features)

            self.counts[label] += 1
            print(f"Added data for label '{label}'. Current counts: {self.counts}.")
            print(f"TOTAL SAMPLES: {len(self.dataset)}")

            return True
        return False
    
    def save_dataset(self):
        if len(self.dataset) == 0:
            print("Dataset is empty. Nothing to save.")
            return
        
        df = pd.DataFrame(self.dataset)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(self.save_dir, f"posture_dataset_{timestamp}.csv")
        df.to_csv(csv_path, index=False)

        metadata = {
            'timestamp': timestamp,
            'total_samples': len(self.dataset),
            'counts': self.counts,
            'features': list(df.columns.drop(['label', 'timestamp'])) 
        }

        json_path = os.path.join(self.save_dir, f"posture_metadata_{timestamp}.json")
        with open(json_path, 'w') as json_file:
            json.dump(metadata, json_file, indent=3)

        print(f"Dataset saved to {csv_path}")
        print(f"Metadata saved to {json_path}")
        print("Statistics:")
        print(f"  Correct: {self.counts['correct']}")
        print(f"  Bad head position: {self.counts['bad_head_position']}")
        print(f"  Bad upper body position: {self.counts['bad_upper_body_position']}")
        print(f"  TOTAL: {len(self.dataset)}")

    def create_ui(self, image, current_label=None):
        h, w = image.shape[:2]

        stats_y = 250
        cv2.rectangle(image, (10, stats_y), (300, stats_y+120), (50, 50, 50), -1)
        cv2.putText(image, "Dataset Stats:", (20, stats_y + 25), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 2)

        stats_text = [
            f"Correct Posture: {self.counts['correct']}",
            f"Bad Head Position: {self.counts['bad_head_position']}",
            f"Bad Upper Body Position: {self.counts['bad_upper_body_position']}"
        ] 

        y = stats_y + 60
        for stat in stats_text:
            cv2.putText(image, stat, (20, y), cv2.FONT_HERSHEY_DUPLEX, 0.5, (200, 200, 200), 1)
            y += 20

        if current_label:
            label_colours = {
                'correct': (0, 255, 0),
                'bad_head_position': (0, 165, 255),
                'bad_upper_body_position': (0, 0, 255)
            }
            colour = label_colours.get(current_label, (255, 255, 255))
            cv2.putText(image, f"Saved: {current_label}", (w-400, h-30), cv2.FONT_HERSHEY_DUPLEX, 0.7, colour, 2)

    def run(self, camera_id=0):
        capture = cv2.VideoCapture(camera_id)
        current_label = None
        label_timer = 0

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
                        mp_pose.POSE_CONNECTIONS
                    )

                if label_timer > 0:
                    label_timer -= 1
                    if label_timer == 0:
                        current_label = None

                self.create_ui(image, current_label)
                cv2.imshow('Posture Dataset Collector', image)

                key = cv2.waitKey(10) & 0xFF

                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.save_dataset()
                elif key == ord('1') and result.pose_landmarks is not None:
                    if self.add_data(result.pose_landmarks.landmark, 'correct'):
                        current_label = 'correct'
                        label_timer = 30
                elif key == ord('2') and result.pose_landmarks is not None:
                    if self.add_data(result.pose_landmarks.landmark, 'bad_head_position'):
                        current_label = 'bad_head_position'
                        label_timer = 30
                elif key == ord('3') and result.pose_landmarks is not None:
                    if self.add_data(result.pose_landmarks.landmark, 'bad_upper_body_position'):
                        current_label = 'bad_upper_body_position'
                        label_timer = 30

        capture.release()
        cv2.destroyAllWindows()

        if len(self.dataset) > 0:
            print("Don't forget to save your dataset before exiting!")
            self.save_dataset()

if __name__ == "__main__":
    collector = PostureDetectionDatasetCollector()
    collector.run()


            