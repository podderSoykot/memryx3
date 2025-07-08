class Config:
    # Database settings
    DB_PATH = "faces.db"
    
    # Face Recognition Models - Multiple models support
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
        },
        'backup': {
            'name': 'ArcFaceMX',
            'detector': 'mtcnn', 
            'tracker': 'MOSSE',
            'actions': ['age', 'gender'],
            'enabled': False  # Can be enabled if needed
        }
    }
    
    # Multiple Cameras Configuration
    CAMERAS = {
        'camera_1': {
            'name': 'Main Entrance',
            'location': 'Building A - Front Door',
            'source': 0,  # Default webcam
            'model_config': 'primary',  # Use primary model
            'enabled': True,
            'resolution': (640, 480),
            'fps': 30
        },

        'camera_2': {
            'name': 'IP Camera',
            'location': 'Building B - Main Hall',
            'source': 'rtsp://admin:XRRZKY@192.168.1.71/channel=1/subtype=0',
            'model_config': 'primary',
            'enabled': True,  # Disabled by default
            'resolution': (1280, 720),
            'fps': 30
        }
    }
    
    # Processing Settings
    DETECT_EVERY_X_FRAMES = 10
    IOU_THRESHOLD = 0.6
    CONFIDENCE_THRESHOLD = 0.8
    MIN_FACE_SIZE = 80
    SAVE_AFTER_X_FRAMES = 2
    RESIZE_SCALE = 1
    
    # Performance Settings
    MAX_WORKERS = 4
    FACE_MATCHING_THRESHOLD = 0.75
    PROCESSING_TIMEOUT = 30  # seconds
    
    # Storage Settings
    PICS_FOLDER = "pics"
    REKOGNITION_ENABLED = True
    CLOUD_BACKUP_ENABLED = False  # Disabled for local operation
    
    # Display Settings
    SHOW_CROP_FACE = True
    DISPLAY_STATS_EVERY_N_FRAMES = 30
    
    # Available options for reference
    DETECTOR_BACKENDS = ["opencv", "ssd", "mtcnn", "dlib", "retinaface", "retinafacelite", "mediapipe", "yolov8", "yunet"]
    MODEL_NAMES = ["VGG-Face", "OpenFace", "Facenet", "Facenet512", "DeepFace", "Dlib", "ArcFace", "SFace"]
    TRACKER_TYPES = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    AVAILABLE_ACTIONS = ['age', 'gender', 'race', 'emotion'] 