# Posture Detection System

A real-time computer vision system for monitoring and analyzing sitting posture using machine learning. The system uses MediaPipe Pose for landmark detection and Random Forest classification to identify correct posture, bad head position, and bad upper body position.

## Features

- **Personalized Training**: Build custom posture models adapted to individual body proportions
- **Real-time Detection**: 30 FPS monitoring with <50ms latency
- **Visual Feedback**: Live probability bars and pose skeleton overlay
- **Comprehensive Logging**: 30-second interval tracking throughout sessions
- **Detailed Reports**: Timeline plots, pie charts, heatmaps, and statistical summaries
- **Desktop Notifications**: Periodic reminders (Windows 10/11)
- **Privacy-Preserving**: All processing occurs locally on your device

## System Requirements

### Hardware
- Webcam (minimum 640×480 resolution)
- CPU: Intel i5 or equivalent (tested on i5-8300H)
- RAM: 8GB minimum, 12GB recommended
- OS: Windows 10/11 (notifications), Linux/macOS (core features)

### Software
- Python 3.8 or higher
- Webcam drivers properly installed

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/oskarierys/posture-detection.git
cd posture-detection
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### Required Libraries
```
opencv-python>=4.5.0
mediapipe>=0.10.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
joblib>=1.1.0
matplotlib>=3.4.0
seaborn>=0.11.0
win10toast>=0.9  # Optional, for Windows notifications
```

## Quick Start

### Step 1: Collect Training Data

Run the dataset collector to capture labeled posture samples:
```bash
python posture-dataset-collector.py
```

**Controls:**
- `1` - Capture sample as "Correct Posture"
- `2` - Capture sample as "Bad Head Position"
- `3` - Capture sample as "Bad Upper Body Position"
- `s` - Save dataset to CSV
- `q` - Quit application

**Guidelines:**
- Collect 50-100 samples per category (minimum 150 total)
- Vary your position slightly within each category
- Ensure good lighting and clear view of upper body
- Keep shoulders and head visible in frame

### Step 2: Train the Model

Process the collected dataset and train the classifier:
```bash
python posture-model-trainer.py
```

The trainer will:
- Automatically find the latest dataset in `data-set/`
- Split data (90% train, 10% test)
- Train Random Forest classifier (100 trees)
- Perform 5-fold cross-validation
- Generate confusion matrix and feature importance plots
- Save trained model to `models/`

**Expected Output:**
```
FOUND DATASET FILE: data-set/posture_dataset_20241228_143052.csv
Label 'correct': 600 samples
Label 'bad_head_position': 500 samples
Label 'bad_upper_body_position': 500 samples

Training samples: 1440, Testing samples: 160
PERFECT MODEL TRAINED!

Model saved to models/posture_model_20241228_143052.joblib
```

### Step 3: Run Real-Time Detection

Start monitoring your posture:
```bash
python posture-ml-detector.py
```

**Controls:**
- `q` - Quit and generate session report
- `t` - Trigger manual notification

**What You'll See:**
- Live video feed with pose skeleton overlay
- Three probability bars (Green/Orange/Red)
- Current posture classification
- Session statistics (cumulative percentages)

## Output Files

### Directory Structure
```
posture-detection/
├── data-set/              # Training datasets
│   ├── posture_dataset_*.csv
│   └── posture_metadata_*.json
├── models/                # Trained models
│   ├── posture_model_*.joblib
│   ├── scaler_*.joblib
│   └── model_metadata_*.json
├── plots/                 # Training visualizations
│   ├── confusion_matrix.png
│   └── feature_importance.png
├── posture_logs/          # Session data
│   └── posture_session_*.csv
└── posture-raports/       # Session reports
    ├── timeline_*.png
    ├── pie_chart_*.png
    ├── bar_chart_*.png
    ├── heatmap_*.png
    ├── summary_*.png
    └── summary_*.txt
```

## Posture Categories

### Correct Posture
- Head upright and centered
- Ears aligned above shoulders
- Shoulders level and relaxed
- Back supported

### Bad Head Position
- Forward head posture (neck strain)
- Excessive head tilt to either side
- Chin jutting forward

### Bad Upper Body Position
- Slouching or hunching
- Shoulder asymmetry (one higher than other)
- Lateral leaning to either side
- Torso rotation away from screen

## Understanding the Reports

### Timeline Plot
Shows posture classification over session duration with color-coded segments (green/orange/red).

### Pie Chart
Proportional breakdown of time in each category.

### Bar Chart
Absolute frame counts for each posture type.

### Probability Heatmap
Classification confidence evolution over time. High-intensity colors indicate stable classifications.

### Summary Report
Includes:
- Session metadata (date, time, duration)
- Total frames analyzed
- Percentage breakdown per category
- Personalized recommendations

## Troubleshooting

### "No person detected"
- Ensure adequate lighting
- Position yourself centered in frame
- Keep upper body visible (shoulders to head)
- Check webcam is not obstructed

### Low Model Accuracy
- Collect more training samples (aim for 100+ per category)
- Ensure variety within each category
- Verify consistent lighting during collection
- Retrain model with additional data

### Webcam Not Opening
```python
# Try different camera indices
python posture-ml-detector.py  # Uses camera_id=0 by default

# Modify in code if needed:
detector.run(camera_id=1)  # Try camera_id=1, 2, etc.
```

### Notifications Not Working
- Windows 10/11 required for desktop notifications
- Install win10toast: `pip install win10toast`
- Check notification settings in Windows
- Core functionality works without notifications

## Configuration

### Adjust Logging Interval
```python
# In posture-ml-detector.py, modify:
detector.run(
    camera_id=0,
    enable_logging=True,
    log_interval=30,      # Log every 30 seconds
    toast_interval=600    # Notify every 10 minutes
)
```

### Model Hyperparameters
```python
# In posture-model-trainer.py, modify RandomForestClassifier:
self.model = RandomForestClassifier(
    n_estimators=100,      # Number of trees
    max_depth=10,          # Maximum tree depth
    min_samples_split=5,   # Minimum samples to split node
    min_samples_leaf=2,    # Minimum samples in leaf
    random_state=42
)
```

## Technical Details

### Feature Extraction
Nine geometric features computed from landmarks:
1. Shoulder asymmetry (vertical difference)
2. Head tilt (ear height difference)
3. Head forward position (ear-shoulder horizontal distance)
4. Shoulder rotation (horizontal separation)
5. Torso lean forward/backward
6. Torso lean sideways
7. Shoulder width
8. Head-to-shoulder ratio
9. Average landmark visibility

### Model Architecture
- **Algorithm**: Random Forest Classifier
- **Trees**: 100
- **Max Depth**: 10
- **Normalization**: Z-score standardization
- **Validation**: 5-fold cross-validation
- **Split**: 90% train, 10% test (stratified)

### Performance
- **Frame Rate**: 30 FPS
- **Latency**: 30-50ms per frame
- **Pose Estimation**: 25-35ms (MediaPipe)
- **Classification**: <2ms (Random Forest)
- **Temporal Smoothing**: 10-frame window

## Citation

If you use this system in your research, please cite:
```bibtex
@mastersthesis{yourname2024posture,
  title={Real-Time Posture Detection System Using Machine Learning},
  author={Oskar Kierys},
  year={2025},
  school={AGH Uniersity in Cracow}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MediaPipe by Google for pose estimation
- scikit-learn for machine learning tools
- OpenCV for computer vision capabilities

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Areas for Improvement
- Multi-person detection support
- Mobile/web deployment
- Additional posture categories
- Ergonomic assessment integration (RULA/REBA)
- Historical trend analysis dashboard

## Contact

For questions or issues, please open an issue on GitHub or contact [osk.kierys@gmail.com]

## Roadmap

- [ ] Web-based interface
- [ ] Mobile app version
- [ ] Cloud synchronization (optional)
- [ ] Multi-language support
- [ ] Exercise recommendations
- [ ] Integration with productivity tools

---

**Version**: 1.0.0  
**Last Updated**: November 2025  
**Status**: Active Development