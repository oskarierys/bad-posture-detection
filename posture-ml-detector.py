import cv2
import mediapipe as mp
import numpy as np
import joblib
import json
import glob
import os
from collections import deque
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import time

try:
    from win10toast import ToastNotifier
    TOAST_AVAILABLE = True
    toaster = ToastNotifier()
except ImportError:
    print("win10toast not installed.")
    print("Run: pip install win10toast")
    TOAST_AVAILABLE = False
    toaster = None

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

class PostureToastNotifier:
    def __init__(self, log_interval=30 ,toast_interval=600):
        self.log_interval = log_interval
        self.toast_interval = toast_interval
        self.last_log_time = time.time()
        self.last_toast_time = time.time()
        self.session_start = datetime.now()

        self.timestamps = []
        self.predictions = []
        self.confidences = []
        self.all_probabilities = []

        self.counts = {
            'Correct': 0,
            'Bad head position': 0,
            'Bad upperbody position': 0
        }

        self.total_frames = 0

    def log_prediction(self, label, confidence, probabilities):
        self.total_frames += 1
        self.counts[label] += 1

        current_time = time.time()

        if current_time - self.last_log_time >= self.log_interval:
            self.timestamps.append(datetime.now())
            
            label_map = {'Correct': 0, 'Bad head position': 1, 'Bad upperbody position': 2}
            self.predictions.append(label_map[label])
            self.confidences.append(confidence)
            self.all_probabilities.append(probabilities.copy())
            
            self.last_log_time = current_time
            self.print_summary()

        if current_time - self.last_toast_time >= self.toast_interval:
            self.show_toast_notification()
            self.last_toast_time = current_time

    def print_summary(self):
        elapsed = (datetime.now() - self.session_start).total_seconds()
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        
        print(f"\n{'='*60}")
        print(f"SUMMARY - {minutes}m {seconds}s")
        print(f"{'='*60}")
        
        if self.total_frames > 0:
            for label, count in self.counts.items():
                percentage = (count / self.total_frames) * 100
                bar = '█' * int(percentage / 2)
                print(f"{label:20s}: {percentage:5.1f}% {bar}")
        
        print(f"{'='*60}\n")
    
    def show_toast_notification(self):
        if not TOAST_AVAILABLE or toaster is None:
            return
        
        elapsed = (datetime.now() - self.session_start).total_seconds()  
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)

        if self.total_frames > 0:
            percentages = {
                label: (count / self.total_frames) * 100 
                for label, count in self.counts.items()
            }
            
            message = f""" Sesja: {minutes}m {seconds}s

            Correct: {percentages['Correct']:.1f}%
            Bad Head: {percentages['Bad head position']:.1f}%
            Bad Body: {percentages['Bad upperbody position']:.1f}%

            Keep up the good posture!"""
            
            try:
                toaster.show_toast(
                    "Posture Analysis Report",
                    message,
                    duration=20,
                    icon_path=None,
                    threaded=False
                )
                print("Toast notification sent!")
            except Exception as e:
                print(f"Toast notification failed: {e}")
    
    def get_statistics(self):
        if self.total_frames == 0:
            return None
        
        stats = {
            'duration_seconds': (datetime.now() - self.session_start).total_seconds(),
            'total_frames': self.total_frames,
            'counts': self.counts.copy(),
            'percentages': {
                label: (count / self.total_frames) * 100 
                for label, count in self.counts.items()
            }
        }
        return stats
    
    def save_session_data(self, save_dir="posture_logs"):
        if len(self.timestamps) == 0:
            print("No data to save.")
            return None
        
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        label_names = ['Correct', 'Bad head position', 'Bad upperbody position']
        data = {
            'timestamp': self.timestamps,
            'prediction': [label_names[p] for p in self.predictions],
            'confidence': self.confidences
        }
        
        for i, label in enumerate(label_names):
            data[f'prob_{label.lower().replace(" ", "_")}'] = [
                probs[i] for probs in self.all_probabilities
            ]
        
        df = pd.DataFrame(data)
        csv_path = os.path.join(save_dir, f"posture_session_{timestamp}.csv")
        df.to_csv(csv_path, index=False)
        
        print(f"Session data saved: {csv_path}")
        return csv_path
    
class PostureRaportGenerator:
    def __init__(self, logger):
        self.logger = logger
        self.colors = {0: '#00FF00', 1: '#FFA500', 2: '#FF0000'}
        self.label_names = ['Correct', 'Bad head position', 'Bad upperbody position']

    def generate_raport(self, save_dir="posture-raports"):
        if len(self.logger.timestamps) < 2:
            print("Not enough data to generate raport.")
            return
        
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        stats = self.logger.get_statistics()
        
        print("Generating plots...")
        
        # 1. Timeline
        self._save_timeline(save_dir, timestamp)
        
        # 2. Pie Chart
        self._save_pie_chart(save_dir, timestamp)
        
        # 3. Bar Chart
        self._save_bar_chart(save_dir, timestamp)
        
        # 4. Heatmap
        self._save_heatmap(save_dir, timestamp)
        
        # Text raport (PNG)
        self._save_text_report(save_dir, timestamp, stats)
        
        # Text raport (TXT)
        self._save_text_file(save_dir, timestamp, stats)

        print("Raport generation completed.")
    
    def _save_timeline(self, save_dir, timestamp):
        fig, ax = plt.subplots(figsize=(14, 6))
        
        times = [(t - self.logger.session_start).total_seconds() / 60 
                for t in self.logger.timestamps]
        
        for i in range(len(times) - 1):
            color = self.colors[self.logger.predictions[i]]
            ax.plot(times[i:i+2], self.logger.predictions[i:i+2], 
                   color=color, linewidth=4, marker='o', markersize=10)
        
        ax.set_ylabel('Posture Classification', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time (minutes)', fontsize=14)
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(self.label_names, fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_title('Posture Timeline', fontsize=16, fontweight='bold', pad=20)
        
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=self.colors[i], label=self.label_names[i])
            for i in range(3)
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
        
        plt.tight_layout()
        path = os.path.join(save_dir, f"timeline_{timestamp}.png")
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Timeline saved: {path}")
    
    def _save_pie_chart(self, save_dir, timestamp):
        fig, ax = plt.subplots(figsize=(10, 8))
        
        labels = list(self.logger.counts.keys())
        sizes = list(self.logger.counts.values())
        colors_pie = [self.colors[i] for i in range(3)]
        
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors_pie,
            autopct='%1.1f%%', startangle=90,
            textprops={'fontsize': 14, 'fontweight': 'bold'},
            explode=(0.05, 0, 0)
        )
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(16)
            autotext.set_fontweight('bold')
        
        ax.set_title('Overall Posture Distribution', fontsize=18, fontweight='bold', pad=20)
        
        plt.tight_layout()
        path = os.path.join(save_dir, f"pie_chart_{timestamp}.png")
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Pie chart saved: {path}")
    
    def _save_bar_chart(self, save_dir, timestamp):
        fig, ax = plt.subplots(figsize=(12, 7))
        
        labels = list(self.logger.counts.keys())
        sizes = list(self.logger.counts.values())
        colors_bar = [self.colors[i] for i in range(3)]
        
        bars = ax.bar(labels, sizes, color=colors_bar, edgecolor='black', linewidth=2, width=0.6)
        ax.set_ylabel('Frame Count', fontsize=14, fontweight='bold')
        ax.set_xlabel('Posture Type', fontsize=14, fontweight='bold')
        ax.set_title('Posture Frequency', fontsize=18, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontweight='bold', fontsize=14)
        
        plt.tight_layout()
        path = os.path.join(save_dir, f"bar_chart_{timestamp}.png")
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Bar chart saved: {path}")
    
    def _save_heatmap(self, save_dir, timestamp):
        if len(self.logger.all_probabilities) == 0:
            return
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        probs_array = np.array(self.logger.all_probabilities).T
        
        im = ax.imshow(probs_array, aspect='auto', cmap='RdYlGn', 
                      interpolation='nearest', vmin=0, vmax=1)
        
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(self.label_names, fontsize=12)
        ax.set_xlabel('Time Points (30s intervals)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Posture Type', fontsize=14, fontweight='bold')
        ax.set_title('Probability Heatmap Over Time', fontsize=18, fontweight='bold', pad=20)
        
        cbar = plt.colorbar(im, ax=ax, label='Probability')
        cbar.ax.tick_params(labelsize=11)
        
        plt.tight_layout()
        path = os.path.join(save_dir, f"heatmap_{timestamp}.png")
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Heatmap saved: {path}")
    
    def _save_text_report(self, save_dir, timestamp, stats):
        if stats is None:
            return
        
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.axis('off')
        
        duration_min = int(stats['duration_seconds'] // 60)
        duration_sec = int(stats['duration_seconds'] % 60)
        
        report_text = f"""
╔═══════════════════════════════════════════════════════╗
║                                                       ║
║           POSTURE ANALYSIS SESSION REPORT             ║
║                                                       ║
╚═══════════════════════════════════════════════════════╝


Session Date: {datetime.now().strftime('%Y-%m-%d')}
Session Time: {datetime.now().strftime('%H:%M:%S')}
Duration: {duration_min} minutes {duration_sec} seconds


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    OVERALL STATISTICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Total Frames Analyzed: {stats['total_frames']:,}
Sampling Interval: Every 30 seconds
Total Snapshots: {len(self.logger.timestamps)}


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                   POSTURE BREAKDOWN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CORRECT POSTURE:
   Percentage: {stats['percentages']['Correct']:.1f}%
   Frame Count: {stats['counts']['Correct']:,}
   {'█' * int(stats['percentages']['Correct'] / 2)}


BAD HEAD POSITION:
   Percentage: {stats['percentages']['Bad head position']:.1f}%
   Frame Count: {stats['counts']['Bad head position']:,}
   {'█' * int(stats['percentages']['Bad head position'] / 2)}


BAD UPPER BODY POSITION:
   Percentage: {stats['percentages']['Bad upperbody position']:.1f}%
   Frame Count: {stats['counts']['Bad upperbody position']:,}
   {'█' * int(stats['percentages']['Bad upperbody position'] / 2)}


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    RECOMMENDATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        if stats['percentages']['Correct'] >= 70:
            report_text += "\nExcellent! Maintain your good posture habits.\n"
        elif stats['percentages']['Correct'] >= 50:
            report_text += "\nGood progress. Focus on maintaining posture longer.\n"
        else:
            report_text += "\nNeeds improvement. Take regular breaks and adjust setup.\n"
        
        if stats['percentages']['Bad head position'] > 20:
            report_text += "• Adjust monitor height - eyes level with top 1/3\n"
        
        if stats['percentages']['Bad upperbody position'] > 20:
            report_text += "• Check chair back support and sitting angle\n"
        
        report_text += f"\n\n{'━'*55}\nReport generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{'━'*55}"
        
        ax.text(0.5, 0.5, report_text,
               ha='center', va='center',
               fontsize=11, fontfamily='monospace',
               bbox=dict(boxstyle='round,pad=1.5', facecolor='#E8F4F8', 
                        edgecolor='#2196F3', linewidth=3, alpha=0.9))
        
        plt.tight_layout()
        path = os.path.join(save_dir, f"summary_{timestamp}.png")
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Summary image saved: {path}")
    
    def _save_text_file(self, save_dir, timestamp, stats):
        if stats is None:
            return
        
        duration_min = int(stats['duration_seconds'] // 60)
        duration_sec = int(stats['duration_seconds'] % 60)
        
        report_lines = [
            "="*70,
            "POSTURE ANALYSIS SESSION REPORT",
            "="*70,
            "",
            f"Session Date: {datetime.now().strftime('%Y-%m-%d')}",
            f"Session Time: {datetime.now().strftime('%H:%M:%S')}",
            f"Duration: {duration_min}m {duration_sec}s",
            "",
            "="*70,
            "OVERALL STATISTICS",
            "="*70,
            f"Total Frames: {stats['total_frames']:,}",
            f"Snapshots: {len(self.logger.timestamps)}",
            "",
            "="*70,
            "POSTURE BREAKDOWN",
            "="*70,
            f"Correct Posture:        {stats['percentages']['Correct']:6.1f}% ({stats['counts']['Correct']:,} frames)",
            f"Bad Head Position:      {stats['percentages']['Bad head position']:6.1f}% ({stats['counts']['Bad head position']:,} frames)",
            f"Bad Upper Body:         {stats['percentages']['Bad upperbody position']:6.1f}% ({stats['counts']['Bad upperbody position']:,} frames)",
            "",
            "="*70,
            f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "="*70
        ]
        
        path = os.path.join(save_dir, f"summary_{timestamp}.txt")
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Summary text saved: {path}")

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
    
    def run(self, camera_id=0, enable_logging=True, log_interval=5, toast_interval=5):
        capture = cv2.VideoCapture(camera_id)
        
        print("ML POSTURE DETECTOR - Real-time Detection")
        print(f"Model: {self.label_names}")
        print(f"Features: {len(self.feature_names)}")
        print("Press 'q' to quit")
        print("Press 't' to trigger toast notification")

        if enable_logging:
            self.logger = PostureToastNotifier(
                log_interval=log_interval,
                toast_interval=toast_interval
            )
        
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

                        if enable_logging and self.logger:
                            self.logger.log_prediction(label, confidence, probabilities)
                        
                else:
                    h, w = image.shape[:2]
                    cv2.putText(image, "No person detected", (w//2 - 150, h//2),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                cv2.imshow('ML Posture Detector', image)
                
                key = cv2.waitKey(10) & 0xFF
                if key == ord('q'):
                    break
        
        capture.release()
        cv2.destroyAllWindows()

        if enable_logging and self.logger:
            print("Generating report...")

            self.logger.save_session_data()
            raport_generator = PostureRaportGenerator(self.logger)
            raport_generator.generate_raport()

if __name__ == "__main__":
    try:
        detector = MLPostureDetector()
        detector.run(camera_id=0, enable_logging=True, log_interval=30, toast_interval=600)
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