# Multi-Camera & Multi-Model Face Recognition System

## 🚀 **Overview**

This enhanced version of the face recognition system supports **multiple cameras** with **different AI models** running simultaneously. Each camera can be configured with its own model, detector, and processing parameters.

## 📁 **New Files**

- `config.py` - Multi-camera and model configuration
- `main_multicamera.py` - Main multi-camera system
- `camera_manager.py` - Camera management utilities
- `README_MultiCamera.md` - This documentation

## 🎥 **Features**

### **Multi-Camera Support**

- ✅ Support for unlimited cameras (webcam, IP cameras, video files)
- ✅ Individual camera configuration (resolution, FPS, location)
- ✅ Threaded processing for parallel camera streams
- ✅ Independent face tracking per camera

### **Multi-Model Support**

- ✅ Different AI models per camera
- ✅ VGG-Face, Facenet512, ArcFace support
- ✅ Multiple detectors: OpenCV, RetinaFace, MTCNN, etc.
- ✅ Custom tracker configurations

### **Enhanced Features**

- ✅ Real-time face recognition across all cameras
- ✅ Shared database for face matching
- ✅ Face crop saving with camera identification
- ✅ System statistics and monitoring
- ✅ UUID-based face identification

## ⚙️ **Configuration**

### **1. Camera Setup**

Edit `config.py` to configure your cameras:

```python
CAMERAS = {
    'camera_1': {
        'name': 'Main Entrance',
        'location': 'Building A - Front Door',
        'source': 0,  # Webcam
        'model_config': 'primary',
        'enabled': True,
        'resolution': (640, 480),
        'fps': 30
    },
    'camera_2': {
        'name': 'IP Camera',
        'location': 'Building B - Main Hall',
        'source': 'rtsp://admin:password@192.168.1.100/stream',
        'model_config': 'secondary',
        'enabled': True,
        'resolution': (1280, 720),
        'fps': 25
    }
}
```

### **2. Model Configuration**

Configure different AI models:

```python
MODELS = {
    'primary': {
        'name': 'VGG-Face',
        'detector': 'opencv',
        'tracker': 'KCF',
        'actions': ['age', 'gender'],
        'enabled': True
    },
    'secondary': {
        'name': 'Facenet512',
        'detector': 'retinaface',
        'tracker': 'CSRT',
        'actions': ['age', 'gender'],
        'enabled': True
    }
}
```

## 🏃 **Running the System**

### **Method 1: Multi-Camera System**

```bash
python main_multicamera.py
```

### **Method 2: Single Camera (Original)**

```bash
python main.py
```

## 📊 **Camera Sources**

### **Webcam**

```python
'source': 0,  # Primary webcam
'source': 1,  # Secondary webcam
```

### **IP Camera (RTSP)**

```python
'source': 'rtsp://username:password@ip:port/stream'
'source': 'rtsp://admin:123456@192.168.1.100/cam/realmonitor?channel=1&subtype=0'
```

### **Video File**

```python
'source': '/path/to/video.mp4'
'source': 'C:\\Users\\Videos\\test_video.avi'
```

### **USB Camera (Linux)**

```python
'source': '/dev/video0'
'source': '/dev/video1'
```

## 🤖 **Available Models**

### **Recognition Models**

- `VGG-Face` - Fast, good accuracy
- `Facenet512` - High accuracy, slower
- `ArcFace` - Best accuracy, slowest
- `OpenFace` - Lightweight
- `Dlib` - Traditional approach

### **Detector Backends**

- `opencv` - Fastest
- `retinaface` - Best accuracy
- `mtcnn` - Good balance
- `ssd` - Fast
- `mediapipe` - Mobile optimized

### **Tracker Types**

- `KCF` - Fast, reliable
- `CSRT` - High accuracy
- `MOSSE` - Fastest
- `BOOSTING` - Traditional

## 🎛️ **Controls**

### **Keyboard Commands**

- `q` - Quit system
- `s` - Save current frame

### **Window Management**

- Each camera opens in its own window
- Windows can be resized and moved independently
- System shows real-time statistics

## 📈 **Performance Optimization**

### **For Best Performance:**

1. **Use faster models for multiple cameras:**

   ```python
   'name': 'VGG-Face',
   'detector': 'opencv',
   'tracker': 'KCF'
   ```

2. **Adjust detection frequency:**

   ```python
   DETECT_EVERY_X_FRAMES = 15  # Higher = faster but less responsive
   ```

3. **Lower resolution for more cameras:**

   ```python
   'resolution': (320, 240)  # For 4+ cameras
   ```

4. **Enable only necessary cameras:**
   ```python
   'enabled': False  # Disable unused cameras
   ```

## 🗂️ **Database Structure**

The system uses a shared SQLite database:

- **face_embeddings** - Stores face vectors with UUID
- **records** - Detection history across all cameras
- **pictures** - Saved face crops with camera ID

## 🔧 **Troubleshooting**

### **Camera Won't Open**

```bash
# Check camera index
python -c "import cv2; print([cv2.VideoCapture(i).isOpened() for i in range(5)])"
```

### **IP Camera Issues**

- Verify RTSP URL
- Check network connectivity
- Ensure camera supports RTSP

### **Performance Issues**

- Reduce number of active cameras
- Use faster models (VGG-Face + OpenCV)
- Increase `DETECT_EVERY_X_FRAMES`
- Lower camera resolutions

### **Memory Issues**

- Close unnecessary windows
- Restart system periodically
- Check available RAM

## 📋 **System Requirements**

- **RAM:** 8GB+ (for multiple cameras)
- **CPU:** Multi-core recommended
- **GPU:** Optional but recommended for heavy models
- **Storage:** 1GB+ for face database

## 🔄 **Migration from Single Camera**

To migrate from the original single-camera system:

1. **Copy your existing database:**

   ```bash
   cp faces.db faces_backup.db
   ```

2. **Update configuration:**

   - Enable one camera initially
   - Test with existing data
   - Gradually add more cameras

3. **Verify face recognition:**
   - Existing faces should be recognized
   - New faces will be added to the shared database

## 🎯 **Example Use Cases**

### **Home Security**

```python
'camera_1': {'source': 0, 'location': 'Front Door'},
'camera_2': {'source': 'rtsp://192.168.1.100/stream', 'location': 'Back Yard'}
```

### **Office Building**

```python
'camera_1': {'source': 0, 'location': 'Main Entrance', 'model_config': 'primary'},
'camera_2': {'source': 1, 'location': 'Side Door', 'model_config': 'secondary'},
'camera_3': {'source': 2, 'location': 'Parking Lot', 'model_config': 'backup'}
```

### **Retail Store**

```python
'camera_1': {'source': 'rtsp://cam1/stream', 'location': 'Store Front'},
'camera_2': {'source': 'rtsp://cam2/stream', 'location': 'Checkout Area'},
'camera_3': {'source': 'rtsp://cam3/stream', 'location': 'Storage Room'}
```

## 🚀 **Getting Started**

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Configure cameras in `config.py`**

3. **Run the system:**

   ```bash
   python main_multicamera.py
   ```

4. **Monitor the logs for camera initialization**

5. **Press 'q' in any window to quit**

---

🎉 **Your multi-camera face recognition system is now ready!**

Each camera can now use different AI models while sharing the same face database for comprehensive recognition across your entire setup.
