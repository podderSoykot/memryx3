# IP Camera Setup Guide

## 🔧 Current Status

- **IP Camera**: `rtsp://admin:qwq1234.@192.168.1.71/channel=1/subtype=0`
- **Issue**: 401 Unauthorized (Authentication failed)
- **Status**: Disabled in config.py

## 🛠️ Troubleshooting Steps

### 1. Check Camera Credentials

Your current RTSP URL uses:

- **Username**: `admin`
- **Password**: `qwq1234.`
- **IP**: `192.168.1.71`

### 2. Common Authentication Issues

#### Option A: Try Different Username/Password Combinations

```python
# In config.py, try these alternatives:
'source': 'rtsp://admin:admin@192.168.1.71/channel=1/subtype=0'
'source': 'rtsp://admin:123456@192.168.1.71/channel=1/subtype=0'
'source': 'rtsp://admin:password@192.168.1.71/channel=1/subtype=0'
'source': 'rtsp://admin:@192.168.1.71/channel=1/subtype=0'  # No password
```

#### Option B: Try Different RTSP Paths

```python
# Common RTSP paths for different camera brands:
'source': 'rtsp://admin:qwq1234.@192.168.1.71/live'
'source': 'rtsp://admin:qwq1234.@192.168.1.71/stream1'
'source': 'rtsp://admin:qwq1234.@192.168.1.71/video'
'source': 'rtsp://admin:qwq1234.@192.168.1.71/h264'
```

### 3. Test IP Camera Access

#### Using VLC Media Player

1. Open VLC Media Player
2. Go to Media → Open Network Stream
3. Enter: `rtsp://admin:qwq1234.@192.168.1.71/channel=1/subtype=0`
4. If it works in VLC, the URL is correct

#### Using FFmpeg (if installed)

```bash
ffplay rtsp://admin:qwq1234.@192.168.1.71/channel=1/subtype=0
```

### 4. Enable IP Camera in System

Once you find the working credentials:

1. **Edit config.py**:

   ```python
   'camera_2': {
       'name': 'IP Camera',
       'location': 'Building B - Main Hall',
       'source': 'rtsp://CORRECT_USER:CORRECT_PASS@192.168.1.71/channel=1/subtype=0',
       'model_config': 'primary',
       'enabled': True,  # Change this to True
       'resolution': (1280, 720),
       'fps': 30
   }
   ```

2. **Test the camera**:

   ```bash
   python test_multicamera.py
   ```

3. **Run dual-camera system**:
   ```bash
   python main_multicamera.py
   ```

## 📋 Camera Brands & Default Credentials

| Brand     | Default User | Default Password | Common RTSP Path                       |
| --------- | ------------ | ---------------- | -------------------------------------- |
| Hikvision | admin        | 12345            | `/Streaming/Channels/101`              |
| Dahua     | admin        | admin            | `/cam/realmonitor?channel=1&subtype=0` |
| Axis      | root         | pass             | `/axis-media/media.amp`                |
| Foscam    | admin        | (empty)          | `/videoMain`                           |
| Generic   | admin        | admin            | `/live` or `/stream1`                  |

## 🎯 Quick Test Commands

```bash
# Test current configuration
python test_multicamera.py

# Test specific RTSP URL manually
python -c "
import cv2
cap = cv2.VideoCapture('rtsp://admin:qwq1234.@192.168.1.71/channel=1/subtype=0')
print('Camera opened:', cap.isOpened())
if cap.isOpened():
    ret, frame = cap.read()
    print('Frame read:', ret)
    if ret:
        print('Frame shape:', frame.shape)
cap.release()
"
```

## ✅ Success Indicators

When IP camera is working correctly, you'll see:

- ✅ `IP Camera is accessible!`
- ✅ `camera_2 started successfully`
- 🪟 Two windows: "Main Entrance" + "IP Camera"
- 📊 Stats show 2 active cameras

## 🚀 Next Steps

1. Find correct IP camera credentials
2. Update config.py with working RTSP URL
3. Set `'enabled': True` for camera_2
4. Run `python main_multicamera.py`
5. Enjoy dual-camera face recognition!
