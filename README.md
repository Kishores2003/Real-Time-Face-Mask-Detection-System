# 🚀 Real-Time Face Mask Detection System

![Demo](demo.gif) *Replace with actual demo GIF*

A high-performance deep learning solution for detecting face masks in real-time video streams using **MobileNetV2**, **OpenCV**, and **TensorFlow/Keras**.

## 🌟 Key Features
- **Real-time processing**: 15-20 FPS on standard webcams (640x480)
- **High accuracy**: 95.2% test accuracy with <5% false positives
- **Two operational modes**:
  - Standalone Python script (`detect_mask.py`)
  - Web interface (`app.py`) with Flask backend
- **Pre-trained models included**:
  - `mask_detector_model.h5` (Keras)
  - `mask_detector_model.keras` (TF SavedModel format)

## 📦 Installation
### Prerequisites
- Python 3.8+
- Webcam or video input device

```bash
# Clone repository
git clone https://github.com/Kishores2003/face-mask-detection.git
cd face-mask-detection

# Install dependencies
pip install -r requirements.txt
```

## 🖥️ Usage

### 1.Real-time Detection (Terminal)
```
python detect_mask.py \
    --model mask_detector_model.h5 \
    --confidence 0.5 \
    --source 0  # 0 for webcam, or path to video file
```
Arguments:
--model: Path to trained model
--confidence: Minimum probability threshold (0-1)
--source: Input video source

### 2.Web Interface
```
python app.py
```
Access at http://localhost:5000

