# 🚀 Quick Start Guide - Multi-Camera Face Recognition

## ✅ **What's Been Added**

Your version-3 now supports **multiple cameras** and **multiple AI models**! Here's what's new:

### 📁 **New Files Created**

- `config.py` - Multi-camera & model configuration
- `main_multicamera.py` - Multi-camera face recognition system
- `camera_manager.py` - Camera management utilities
- `test_cameras.py` - Camera testing tool
- `README_MultiCamera.md` - Comprehensive documentation
- `QUICK_START.md` - This guide

### 🎥 **Features Added**

- ✅ **Multiple cameras** (webcam, IP cameras, video files)
- ✅ **Different AI models** per camera
- ✅ **Threaded processing** for parallel camera streams
- ✅ **Shared database** across all cameras
- ✅ **Real-time face recognition** across all streams
- ✅ **Camera-specific settings** (resolution, FPS, models)

## 🏃 **Quick Start (3 Steps)**

### **Step 1: Test Your Cameras**

```bash
python test_cameras.py
```

This will:

- Find available webcams
- Test configured cameras
- Show model configurations
- Offer live camera preview

### **Step 2: Configure Cameras (Optional)**

Edit `config.py` to add more cameras:

```python
CAMERAS = {
    'camera_1': {
        'name': 'Main Entrance',
        'location': 'Building A - Front Door',
        'source': 0,  # Your webcam
        'model_config': 'primary',
        'enabled': True,
        'resolution': (640, 480),
        'fps': 30
    },
    # Add more cameras here
}
```

### **Step 3: Run Multi-Camera System**

```bash
python main_multicamera.py
```

## 🎛️ **Controls**

- **`q`** in any window = Quit entire system
- **`s`** in any window = Save current frame
- Each camera opens its own window

## 📊 **Current Configuration**

Based on your test results:

### **Available Cameras**

- ✅ **Webcam 0** - Working (640x480, 30 FPS)

### **Configured Cameras**

- 🎥 **camera_1** - "Main Entrance" (source: 0)
  - Model: VGG-Face + OpenCV detector
  - Status: ✅ READY

### **AI Models Available**

- 🤖 **Primary**: VGG-Face + OpenCV (fast, good accuracy)
- 🤖 **Secondary**: Facenet512 + RetinaFace (high accuracy, slower)
- 🤖 **Backup**: ArcFace + MTCNN (best accuracy, slowest)

## 🔧 **Add More Cameras**

### **For Second Webcam** (if available):

```python
'camera_2': {
    'name': 'Side Entrance',
    'location': 'Building A - Side Door',
    'source': 1,  # Second webcam
    'model_config': 'secondary',
    'enabled': True,
    'resolution': (640, 480),
    'fps': 30
}
```

### **For IP Camera** (RTSP):

```python
'camera_3': {
    'name': 'IP Camera',
    'location': 'Parking Lot',
    'source': 'rtsp://admin:password@192.168.1.100/stream',
    'model_config': 'primary',
    'enabled': True,
    'resolution': (1280, 720),
    'fps': 25
}
```

### **For Video File**:

```python
'camera_4': {
    'name': 'Video File',
    'location': 'Test Video',
    'source': '/path/to/video.mp4',
    'model_config': 'primary',
    'enabled': True,
    'resolution': (640, 480),
    'fps': 30
}
```

## 📈 **Performance Tips**

### **For Multiple Cameras:**

1. **Use faster models:**

   ```python
   'model_config': 'primary'  # VGG-Face + OpenCV
   ```

2. **Lower resolution:**

   ```python
   'resolution': (320, 240)  # For 4+ cameras
   ```

3. **Reduce detection frequency:**
   ```python
   DETECT_EVERY_X_FRAMES = 15  # In config.py
   ```

## 🔄 **Migration from Original System**

Your original `main.py` still works! You can run either:

- **Single Camera**: `python main.py` (original system)
- **Multi-Camera**: `python main_multicamera.py` (new system)

Both use the same database, so faces learned in one system work in the other.

## 🆔 **UUID System**

Your face recognition still uses UUIDs:

- Each detected face gets a unique UUID
- Same person recognized across all cameras
- Database shared between all camera streams
- Face crops saved with camera ID prefix

## 🎯 **Example Scenarios**

### **Home Security Setup**

```python
CAMERAS = {
    'front_door': {'source': 0, 'location': 'Front Door'},
    'back_yard': {'source': 'rtsp://cam1/stream', 'location': 'Back Yard'},
}
```

### **Office Building Setup**

```python
CAMERAS = {
    'main_entrance': {'source': 0, 'model_config': 'primary'},
    'side_door': {'source': 1, 'model_config': 'secondary'},
    'parking': {'source': 'rtsp://cam1/stream', 'model_config': 'primary'},
}
```

## ⚡ **System Status**

✅ **Ready to use with 1 camera**
✅ **Database working** (faces.db)
✅ **Face matching active** (0.75 threshold)
✅ **UUID system operational**
✅ **Multi-model support ready**

## 🔍 **Next Steps**

1. **Test the system**: `python main_multicamera.py`
2. **Add more cameras** by editing `config.py`
3. **Test each camera**: `python test_cameras.py`
4. **Monitor performance** and adjust settings as needed

---

🎉 **Your multi-camera face recognition system is ready!**

Start with one camera and add more as needed. Each camera can use different AI models while sharing the same face database for comprehensive recognition across your entire setup.
