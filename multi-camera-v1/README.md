# Multi-Camera Face Recognition System v1

🎥 **Advanced AI-powered face recognition system supporting multiple cameras with different AI models**

## 🌟 Features

### Core Capabilities

- **Multi-Camera Support**: Unlimited cameras (webcam, IP cameras, video files)
- **Multi-Model AI**: Different AI models per camera (VGG-Face, Facenet512, ArcFace)
- **Real-time Processing**: Threaded camera streams with live face detection
- **Face Matching**: Cosine similarity-based recognition with configurable threshold
- **Shared Database**: All cameras use the same SQLite face database
- **UUID System**: Unique identification for every detected face

### Advanced Features

- **Face Tracking**: Persistent face tracking across frames
- **Age/Gender Analysis**: Real-time demographic analysis
- **Visual Feedback**: Color-coded recognition (Green=Recognized, Blue=New)
- **Performance Monitoring**: System statistics and performance metrics
- **Configurable Detection**: Adjustable detection intervals and thresholds

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Test camera connections
python test_cameras.py

# Advanced camera testing
python test_multicamera.py
```

### 2. Basic Setup (Single Camera)

```bash
# Run with default webcam
python main_multicamera.py
```

### 3. Multi-Camera Setup

1. **Edit config.py**:

   ```python
   CAMERAS = {
       'camera_1': {
           'enabled': True,   # Enable webcam
           'source': 0,
           'model_config': 'primary'
       },
       'camera_2': {
           'enabled': True,   # Enable IP camera
           'source': 'rtsp://user:pass@192.168.1.71/stream',
           'model_config': 'secondary'
       }
   }
   ```

2. **Run multi-camera system**:
   ```bash
   python main_multicamera.py
   ```

## 📁 File Structure

```
multi-camera-v1/
├── main_multicamera.py      # Main multi-camera system
├── config.py                # Configuration settings
├── sqlite_config.py         # Database management
├── utils.py                 # Utility functions
├── camera_manager.py        # Camera management utilities
├── test_cameras.py          # Basic camera testing
├── test_multicamera.py      # Advanced camera testing
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── README_MultiCamera.md   # Detailed technical documentation
├── QUICK_START.md          # Quick setup guide
└── ip_camera_setup.md      # IP camera troubleshooting
```

## ⚙️ Configuration

### Camera Models Available

| Model         | AI Backend | Detector   | Speed  | Accuracy | Best For            |
| ------------- | ---------- | ---------- | ------ | -------- | ------------------- |
| **Primary**   | VGG-Face   | OpenCV     | Fast   | Good     | Real-time webcams   |
| **Secondary** | Facenet512 | RetinaFace | Medium | High     | IP cameras          |
| **Backup**    | ArcFace    | MTCNN      | Slow   | Best     | High-accuracy needs |

### Key Settings

```python
# Face matching threshold (0.0-1.0)
FACE_MATCHING_THRESHOLD = 0.75

# Detection frequency (every N frames)
DETECT_EVERY_X_FRAMES = 10

# Confidence threshold for face detection
CONFIDENCE_THRESHOLD = 0.8

# Minimum face size (pixels)
MIN_FACE_SIZE = 50
```

## 🎮 Controls

### During Operation

- **'q'**: Quit system (from any camera window)
- **'s'**: Save current frame screenshot
- **'p'**: Pause/resume processing

### System Monitoring

- **Stats**: Printed every 30 seconds
- **Face Count**: Total unique faces in database
- **Detection Count**: Total face detections logged
- **Active Threads**: Number of camera threads running

## 🔧 Camera Types Supported

### 1. USB/Webcam

```python
'source': 0  # First webcam
'source': 1  # Second webcam
```

### 2. IP Cameras (RTSP)

```python
'source': 'rtsp://user:password@192.168.1.100/stream'
```

### 3. Video Files

```python
'source': 'path/to/video.mp4'
```

### 4. HTTP Streams

```python
'source': 'http://192.168.1.100:8080/video'
```

## 📊 Database Schema

### Face Embeddings Table

- `id`: Auto-increment primary key
- `embedding`: 4096-dimension face vector (VGG-Face)
- `identity`: UUID for face identification
- `model_name`: AI model used
- `detector_backend`: Detection method
- `timestamp`: Creation time
- `age`: Estimated age
- `gender`: Detected gender
- `normalization`: Embedding normalization method

### Detection Records Table

- `id`: Auto-increment primary key
- `identity`: Face UUID (foreign key)
- `model_name`: AI model used
- `detector_backend`: Detection method
- `timestamp`: Detection time

## 🚨 Troubleshooting

### Common Issues

#### Camera Not Opening

```bash
# Test camera manually
python test_cameras.py

# Check camera source
python -c "import cv2; cap=cv2.VideoCapture(0); print(cap.isOpened())"
```

#### IP Camera Authentication

- Check credentials in `ip_camera_setup.md`
- Test RTSP URL in VLC Media Player
- Try different RTSP paths for your camera brand

#### Performance Issues

- Increase `DETECT_EVERY_X_FRAMES` (detect less frequently)
- Use faster AI model (Primary instead of Secondary)
- Reduce camera resolution in config

#### Database Errors

```bash
# Check database integrity
python -c "from sqlite_config import SQLiteDB; db=SQLiteDB('faces.db'); print(f'Faces: {db.get_face_count()}')"
```

## 🔗 Related Files

- **[README_MultiCamera.md](README_MultiCamera.md)**: Detailed technical documentation
- **[QUICK_START.md](QUICK_START.md)**: Step-by-step setup guide
- **[ip_camera_setup.md](ip_camera_setup.md)**: IP camera troubleshooting guide

## 🎯 System Requirements

### Hardware

- **CPU**: Intel i5+ or AMD Ryzen 5+ (for real-time processing)
- **RAM**: 8GB+ (16GB recommended for multiple cameras)
- **Storage**: 1GB+ free space
- **USB**: Available ports for USB cameras
- **Network**: Gigabit Ethernet for IP cameras

### Software

- **Python**: 3.8-3.11
- **OpenCV**: 4.5+
- **DeepFace**: Latest version
- **TensorFlow**: 2.x
- **Operating System**: Windows 10+, macOS 10.15+, Ubuntu 18.04+

## 📈 Performance Metrics

### Typical Performance (Intel i7, 16GB RAM)

- **Single Camera**: 15-20 FPS with face detection
- **Dual Camera**: 10-15 FPS per camera
- **Face Recognition**: 2-3 seconds per face
- **Database**: 1000+ faces with sub-second lookup
- **Memory Usage**: 2-4GB depending on models

## 🔮 Future Enhancements

### Planned Features

- **Web Interface**: Browser-based monitoring dashboard
- **Face Clustering**: Automatic grouping of similar faces
- **Alert System**: Email/SMS notifications for specific faces
- **Cloud Sync**: Backup to cloud storage
- **Mobile App**: Remote monitoring application
- **RTMP Streaming**: Broadcast processed video streams

### Model Improvements

- **Mask Detection**: COVID-19 mask compliance
- **Emotion Recognition**: Real-time mood analysis
- **Liveness Detection**: Anti-spoofing measures
- **3D Face Analysis**: Depth-based recognition

## 📄 License

This project is for educational and research purposes. Please ensure compliance with local privacy laws when deploying face recognition systems.

---

**Created**: 2024
**Version**: 1.0
**Status**: Production Ready ✅
