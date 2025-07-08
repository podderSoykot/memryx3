import cv2
import threading
import time
import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from config import Config
from sqlite_config import SQLiteDB
import utils
from deepface import DeepFace
import numpy as np
import uuid


class CameraProcessor:
    """Individual camera processor with its own model configuration"""
    
    def __init__(self, camera_id, camera_config, db):
        self.camera_id = camera_id
        self.camera_config = camera_config
        self.db = db
        self.running = False
        self.cap = None
        self.frame_count = 0
        self.tracked_faces_dict = {}
        
        # Get model configuration
        model_config_name = camera_config['model_config']
        self.model_config = Config.MODELS[model_config_name]
        
        # Initialize model components
        self.detector_backend = self.model_config['detector']
        self.model_name = self.model_config['name']
        self.tracker_type = self.model_config['tracker']
        self.actions = self.model_config['actions']
        
        # Initialize DeepFace models
        self.models = {}
        
        print(f"🎥 Camera {camera_id} initialized:")
        print(f"   📍 Location: {camera_config['location']}")
        print(f"   🤖 Model: {self.model_name}")
        print(f"   🔍 Detector: {self.detector_backend}")
        print(f"   📊 Tracker: {self.tracker_type}")
        print(f"   🎯 Actions: {self.actions}")
    
    def start(self):
        """Start camera processing"""
        self.running = True
        
        # Initialize camera
        source = self.camera_config['source']
        self.cap = cv2.VideoCapture(source)
        
        if not self.cap.isOpened():
            print(f"❌ Failed to open camera {self.camera_id} (source: {source})")
            return False
        
        # Configure camera settings
        resolution = self.camera_config['resolution']
        fps = self.camera_config['fps']
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        print(f"✅ Camera {self.camera_id} started successfully")
        return True
    
    def stop(self):
        """Stop camera processing"""
        self.running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print(f"🛑 Camera {self.camera_id} stopped")
    
    def detect_faces(self, frame):
        """Detect faces using configured detector"""
        try:
            face_objs = DeepFace.extract_faces(
                img_path=frame,
                detector_backend=self.detector_backend,
                enforce_detection=False,
                align=True
            )
            
            extracted_faces = []
            for face_obj in face_objs:
                facial_area = face_obj.get("facial_area", {})
                confidence = face_obj.get("confidence", 1.0)
                
                extracted_faces.append({
                    'facial_area': facial_area,
                    'confidence': confidence,
                    'face': face_obj.get('face', None)
                })
            
            return extracted_faces
        
        except Exception as e:
            print(f"❌ Face detection failed for camera {self.camera_id}: {e}")
            return []
    
    def process_frame(self, frame):
        """Process a single frame"""
        modified_frame = frame.copy()
        
        # Detect faces every N frames
        if self.frame_count % Config.DETECT_EVERY_X_FRAMES == 0:
            try:
                extracted_faces = self.detect_faces(frame)
                print(f"🔍 Camera {self.camera_id}: Found {len(extracted_faces)} faces")
                
                for face in extracted_faces:
                    confidence = face.get('confidence', 1.0)
                    facial_area = face.get('facial_area', {})
                    
                    if (confidence > Config.CONFIDENCE_THRESHOLD and 
                        facial_area.get('w', 0) > Config.MIN_FACE_SIZE and 
                        facial_area.get('h', 0) > Config.MIN_FACE_SIZE):
                        
                        face_x = facial_area['x']
                        face_y = facial_area['y'] 
                        face_w = facial_area['w']
                        face_h = facial_area['h']
                        
                        position = (face_x, face_y, face_w, face_h)
                        
                        # Check if face is already being tracked
                        already_tracked = False
                        for tracked_face in self.tracked_faces_dict.values():
                            if utils.get_iou(position, tracked_face['position']) > Config.IOU_THRESHOLD:
                                already_tracked = True
                                tracked_face['last_seen'] = self.frame_count
                                break
                        
                        if not already_tracked:
                            # Create new face object
                            new_face = self.create_face_object(frame, face)
                            
                            if new_face:
                                # Save face to database
                                new_face.save()
                                
                                # Save face image if enabled
                                if Config.REKOGNITION_ENABLED:
                                    self.save_face_crop(frame, face_x, face_y, face_w, face_h)
                                
                                # Add to tracking
                                face_uuid = new_face.uuid
                                self.tracked_faces_dict[face_uuid] = {
                                    'face': new_face,
                                    'position': position,
                                    'last_seen': self.frame_count,
                                    'frames_tracked': 0
                                }
                                
                                # Draw face rectangle
                                color = (0, 255, 255) if new_face.found else (255, 0, 0)  # Yellow for recognized, Blue for new
                                cv2.rectangle(modified_frame, (face_x, face_y), 
                                            (face_x + face_w, face_y + face_h), color, 2)
                                
                                # Add face info text
                                if new_face.found:
                                    text = f"RECOGNIZED: {new_face.matched_identity[:8]}... ({new_face.similarity_score:.3f})"
                                else:
                                    text = f"NEW: {new_face.uuid[:8]}..."
                                
                                cv2.putText(modified_frame, text, (face_x, face_y-10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            except Exception as e:
                print(f"❌ Frame processing error for camera {self.camera_id}: {e}")
        
        # Update existing trackers
        self.update_trackers(modified_frame)
        
        # Add camera info overlay
        self.add_info_overlay(modified_frame)
        
        return modified_frame
    
    def create_face_object(self, frame, face_data):
        """Create Face object with camera-specific model configuration"""
        try:
            # Import Face class (assuming it's in main.py)
            from main import Face
            
            return Face(
                image=frame,
                detector_backend=self.detector_backend,
                model_name=self.model_name,
                models_dict=self.models,
                actions=self.actions,
                db_name=self.db,
                extracted_faces=face_data
            )
        except Exception as e:
            print(f"❌ Failed to create Face object for camera {self.camera_id}: {e}")
            return None
    
    def save_face_crop(self, frame, x, y, w, h):
        """Save cropped face image"""
        try:
            crop_face = frame[y:y + h, x:x + w]
            epoch = utils.create_epochtime()
            file_name = f"{self.camera_id}_{epoch}.jpg"
            file_path = os.path.join(Config.PICS_FOLDER, file_name)
            
            # Create folder if it doesn't exist
            os.makedirs(Config.PICS_FOLDER, exist_ok=True)
            
            cv2.imwrite(file_path, crop_face)
            self.db.save_picture(file_name, epoch, file_path)
            
            print(f"💾 Saved face crop from camera {self.camera_id}: {file_name}")
            
        except Exception as e:
            print(f"❌ Failed to save face crop for camera {self.camera_id}: {e}")
    
    def update_trackers(self, frame):
        """Update face trackers"""
        to_remove = []
        
        for face_uuid, tracked_data in self.tracked_faces_dict.items():
            # Remove old tracks
            if self.frame_count - tracked_data['last_seen'] > Config.DETECT_EVERY_X_FRAMES * 2:
                to_remove.append(face_uuid)
        
        for face_uuid in to_remove:
            del self.tracked_faces_dict[face_uuid]
    
    def add_info_overlay(self, frame):
        """Add information overlay to frame"""
        # Camera info
        info_text = f"Camera: {self.camera_config['name']} | Location: {self.camera_config['location']}"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Model info
        model_text = f"Model: {self.model_name} | Detector: {self.detector_backend}"
        cv2.putText(frame, model_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Tracking info
        tracking_text = f"Tracked Faces: {len(self.tracked_faces_dict)} | Frame: {self.frame_count}"
        cv2.putText(frame, tracking_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Instructions
        instructions = "Press 'q' to quit, 's' to save frame"
        cv2.putText(frame, instructions, (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


class MultiCameraManager:
    """Manager for multiple cameras with different models"""
    
    def __init__(self):
        self.db = SQLiteDB(Config.DB_PATH)
        self.camera_processors = {}
        self.running = False
        self.threads = []
        
        # Initialize enabled cameras
        for camera_id, camera_config in Config.CAMERAS.items():
            if camera_config['enabled']:
                processor = CameraProcessor(camera_id, camera_config, self.db)
                self.camera_processors[camera_id] = processor
        
        print(f"🎥 MultiCameraManager initialized with {len(self.camera_processors)} cameras")
    
    def start(self):
        """Start all camera processors"""
        self.running = True
        print(f"🚀 Starting {len(self.camera_processors)} cameras...")
        
        # Start each camera in its own thread
        for camera_id, processor in self.camera_processors.items():
            if processor.start():
                thread = threading.Thread(
                    target=self.run_camera_loop,
                    args=(camera_id, processor),
                    daemon=True
                )
                thread.start()
                self.threads.append(thread)
                print(f"✅ Started thread for camera {camera_id}")
        
        try:
            # Keep main thread alive
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n⏹️ Shutdown signal received...")
            self.stop()
    
    def run_camera_loop(self, camera_id, processor):
        """Main processing loop for a single camera"""
        window_name = f"Face Recognition - {processor.camera_config['name']}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)
        
        while self.running and processor.running:
            ret, frame = processor.cap.read()
            if not ret:
                print(f"❌ Failed to read frame from camera {camera_id}")
                time.sleep(0.1)
                continue
            
            # Basic frame display for now
            cv2.imshow(window_name, frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
                break
            
            processor.frame_count += 1
        
        processor.stop()
    
    def stop(self):
        """Stop all camera processors"""
        self.running = False
        
        # Stop all processors
        for processor in self.camera_processors.values():
            processor.stop()
        
        cv2.destroyAllWindows()
        print("🛑 All cameras stopped")


if __name__ == "__main__":
    # Run multi-camera system
    manager = MultiCameraManager()
    manager.start() 