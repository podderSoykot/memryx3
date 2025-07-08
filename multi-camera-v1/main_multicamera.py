import os
import uuid
import asyncio
from dotenv import load_dotenv

import cv2
import dlib
import time
import datetime
from deepface import DeepFace
# import backup  # Disabled for local-only operation

import utils
from sqlite_config import SQLiteDB
from config import Config
import threading
import numpy as np
from arcface_mx_client import ArcFaceMXClient

load_dotenv()  # take environment variables from .env.
REKOGNITION = False  # Disabled for local-only operation


class FaceDetection:
    def __init__(self, detector_backend):
        self.detector_backend = detector_backend

    def detect_faces(self, frame):
        return DeepFace.extract_faces(frame, enforce_detection=False, detector_backend=self.detector_backend)


class Face:
    """Enhanced Face class with multi-camera support"""
    def __init__(self, image, detector_backend, model_name, models_dict, actions, db_name=None, embedding=None,
                 extracted_faces=None, camera_id="unknown"):
        self.image = image
        self.detector_backend = detector_backend
        self.model_name = model_name
        self.db_name = db_name
        self.embedding = embedding
        self.extracted_faces = [extracted_faces] if extracted_faces else None
        self.models_dict = models_dict
        self.actions = actions
        self.camera_id = camera_id  # Track which camera detected this face

        # Attributes extracted from analysis
        self.age = None
        self.gender = None
        self.uuid = str(uuid.uuid4())  # Generate UUID immediately
        self.found = False  # Initialize as False
        self.saved = False
        self.matched_identity = None  # Store matched face identity
        self.similarity_score = 0.0  # Store similarity score

        # normalization
        self.normalization = None
        if self.model_name == "VGG-Face":
            self.normalization = "VGGFace"
        elif self.model_name == "Facenet":
            self.normalization = "Facenet"
        elif self.model_name == "Facenet512":
            self.normalization = "Facenet2018"
        elif self.model_name in ["ArcFace", "ArcFaceMX"]:
            self.normalization = "ArcFace"
        else:
            self.normalization = "base"

        # Process the face immediately
        self.find_or_create()

    def find_or_create(self):
        """Process face - analyze, create embedding, and check if it matches existing faces"""
        print(f"📷 Camera {self.camera_id}: Processing face {self.uuid[:8]}...")
        
        # Always analyze the face for age/gender
        self.analyze()
        
        # Always create embedding
        self.create_embedding()
        
        # Check if this face matches an existing one
        if self.embedding:
            self.find()
        
        if self.found:
            print(f"✅ Camera {self.camera_id}: Recognized {self.matched_identity[:8]}... (similarity: {self.similarity_score:.3f})")
        else:
            print(f"🆕 Camera {self.camera_id}: New face {self.uuid[:8]}")

    def find(self):
        """Compare current face with existing faces in database"""
        if not self.db_name or not self.embedding:
            self.found = False
            return False
            
        try:
            # Get all existing face embeddings from database
            existing_faces = self.db_name.get_all_embeddings()
            
            if not existing_faces:
                print("No existing faces in database")
                self.found = False
                return False
            
            print(f"🔍 Comparing with {len(existing_faces)} existing faces...")
            
            import numpy as np
            from scipy.spatial.distance import cosine
            
            current_embedding = np.array(self.embedding)
            best_match = None
            best_similarity = 0.0
            similarity_threshold = 0.75  # Adjust this threshold as needed
            
            for face_data in existing_faces:
                existing_embedding = np.array(face_data['embedding'])
                
                # Calculate cosine similarity (1 - cosine distance)
                similarity = 1 - cosine(current_embedding, existing_embedding)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = face_data
            
            print(f"🎯 Best match similarity: {best_similarity:.3f}")
            
            if best_similarity >= similarity_threshold:
                self.found = True
                self.matched_identity = best_match['identity']
                self.similarity_score = best_similarity
                
                # Update age/gender from existing data if current analysis failed
                if self.age == "Unknown" and best_match['age'] != "Unknown":
                    self.age = best_match['age']
                if self.gender == "Unknown" and best_match['gender'] != "Unknown":
                    self.gender = best_match['gender']
                    
                print(f"✅ Match found! Identity: {self.matched_identity[:8]}... (similarity: {best_similarity:.3f})")
                return True
            else:
                print(f"🆕 No match found (best similarity: {best_similarity:.3f} < {similarity_threshold})")
                self.found = False
                return False
                
        except Exception as e:
            print(f"❌ Camera {self.camera_id}: Face matching failed: {e}")
            self.found = False
            return False

    def analyze(self):
        print("Starting age/gender analysis...")
        
        # Skip analysis if no actions specified
        if not self.actions:
            print("No analysis actions specified - skipping age/gender analysis")
            self.age = "Unknown"
            self.gender = "Unknown"
            return
        
        try:
            # Use DeepFace analyze with current API
            analyzed_face = DeepFace.analyze(
                img_path=self.image,
                actions=self.actions,
                enforce_detection=False,
                detector_backend=self.detector_backend
            )
            
            print(f"Analysis complete: {type(analyzed_face)}")

            # Handle both single face and multiple faces results
            if isinstance(analyzed_face, list) and len(analyzed_face) > 0:
                face_data = analyzed_face[0]
            else:
                face_data = analyzed_face

            # Store the results in the object's attributes
            self.age = face_data.get("age", "Unknown")
            self.gender = face_data.get("dominant_gender", "Unknown")
            
            print(f"✅ Age: {self.age}, Gender: {self.gender}")
            
        except Exception as e:
            print(f"❌ Camera {self.camera_id}: Age/Gender analysis failed: {e}")
            self.age = "Unknown"
            self.gender = "Unknown"

    def create_embedding(self):
        """Create face embedding using DeepFace"""
        try:
            print("Creating face embedding...")
            embedding_result = DeepFace.represent(
                img_path=self.image, 
                model_name=self.model_name,
                detector_backend=self.detector_backend, 
                enforce_detection=False,
                normalization=self.normalization
            )
            
            # Handle both list and dict results
            if isinstance(embedding_result, list) and len(embedding_result) > 0:
                self.embedding = embedding_result[0]['embedding']
            else:
                self.embedding = embedding_result['embedding']
                
            print(f"✅ Embedding created: {len(self.embedding)} dimensions")
            
        except Exception as e:
            print(f"❌ Camera {self.camera_id}: Embedding creation failed: {e}")
            self.embedding = None

    def save(self):
        """Save face to database only if it's a new face"""
        if self.found:
            print(f"⏩ Face already exists in database (ID: {self.matched_identity[:8]}...), skipping save")
            # Record this detection for existing face
            if hasattr(self.db_name, 'save_detection_record'):
                self.db_name.save_detection_record(
                    self.matched_identity, self.model_name, self.detector_backend, utils.create_epochtime()
                )
                print(f"📝 Camera {self.camera_id}: Logged detection for {self.matched_identity[:8]}...")
            return True
            
        if not self.db_name:
            print("❌ No database connection")
            return False
            
        if not self.embedding:
            print("❌ No embedding to save")
            return False
            
        if self.saved:
            print("⏩ Already saved previously, skipped")
            return False

        try:
            # Save new face
            if hasattr(self.db_name, 'save_face_embedding'):
                # Using SQLiteDB object
                self.db_name.save_face_embedding(
                    self.embedding, self.uuid, self.model_name, self.detector_backend,
                    utils.create_epochtime(), self.age, self.gender, self.normalization
                )
                print(f"💾 Camera {self.camera_id}: Saved new face {self.uuid[:8]} (Age: {self.age}, Gender: {self.gender})")
                self.saved = True
                return True
            else:
                print("❌ Database object doesn't have save_face_embedding method")
                return False
                
        except Exception as e:
            print(f"❌ Camera {self.camera_id}: Database save failed: {e}")
            return False


class TrackedFace:
    def __init__(self, tracker, position, detected_frame, face):
        self.tracker = tracker
        self.position = position
        self.last_detected_frame = detected_frame  # Attribute to store the frame where the face was last detected
        self.face = face
        self.uuid = face.uuid
        self.frames_tracked = 1

    def draw_face(self, img):
        """
        Draw the bounding box around the face on the given image frame with age/gender info.
        """
        # Extract the bounding box coordinates from the position and convert them to integers
        x, y, w, h = map(int, self.position)

        # Draw different colored rectangles based on face status
        if hasattr(self.face, 'found') and self.face.found:
            # Green for recognized faces
            color = (0, 255, 0)
            status = "RECOGNIZED"
            display_id = self.face.matched_identity[:8] if self.face.matched_identity else self.uuid[:8]
            similarity = f" ({self.face.similarity_score:.2f})" if hasattr(self.face, 'similarity_score') else ""
        else:
            # Blue for new faces
            color = (255, 0, 0)
            status = "NEW"
            display_id = self.uuid[:8]
            similarity = ""
            
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

        # Display comprehensive face information
        info_lines = [
            f"{status}: {display_id}{similarity}",
            f"Age: {getattr(self.face, 'age', 'Unknown')}",
            f"Gender: {getattr(self.face, 'gender', 'Unknown')}",
            f"Frames: {self.frames_tracked}"
        ]
        
        # Draw text lines above the face box
        for i, line in enumerate(info_lines):
            text_y = y - 15 - (i * 15)
            if text_y > 0:  # Make sure text is visible
                cv2.putText(img, line, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Add a small filled circle to indicate face center
        center_x, center_y = x + w//2, y + h//2
        cv2.circle(img, (center_x, center_y), 3, color, -1)


class FaceTracker:
    def __init__(self, tracker_type):
        self.tracker_type = tracker_type
        self.tracker = None  # Initialize the tracker object as None

    def init_tracker(self, frame, position):
        self.tracker = utils.create_tracker(self.tracker_type)
        self.tracker.init(frame, position)
        return self.tracker

    def update_tracker(self, frame):
        if self.tracker is None:
            raise ValueError("Tracker has not been initialized. Call init_tracker first.")

        success, bbox = self.tracker.update(frame)
        if success:
            return bbox
        else:
            return None


def display_frame(raw_img, tracker_type, fps, resize=0, db_stats=None, processing_status=None):
    cv2.putText(raw_img, f"{tracker_type} + Age/Gender + SQLite", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(raw_img, f"FPS: {int(fps)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Display processing status
    if processing_status:
        cv2.putText(raw_img, processing_status, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        y_offset = 110
    else:
        y_offset = 90
    
    # Display SQLite database statistics
    if db_stats:
        cv2.putText(raw_img, f"Unique Faces: {db_stats['faces']}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(raw_img, f"Total Detections: {db_stats['detections']}", (10, y_offset + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    cv2.putText(raw_img, "Press 'q' to quit | 's' for stats", (10, raw_img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    if resize != 1.0 and resize != 0:
        # Get the width and height of the image
        raw_img = cv2.resize(raw_img, (int(raw_img.shape[1] * resize), int(raw_img.shape[0] * resize)))
    cv2.imshow('Face Recognition - Age/Gender + SQLite', raw_img)


class CameraProcessor:
    """Individual camera processor with its own model configuration"""
    
    def __init__(self, camera_id, camera_config, db):
        self.camera_id = camera_id
        self.camera_config = camera_config
        self.db = db
        self.running = False
        self.cap = None
        self.frame_count = 0
        
        # Get model configuration
        model_config_name = camera_config['model_config']
        self.model_config = Config.MODELS[model_config_name]
        
        # Initialize model components
        self.detector_backend = self.model_config['detector']
        self.model_name = self.model_config['name']
        self.actions = self.model_config['actions']
        self.models = {}
        
        print(f"🎥 Initializing {camera_id}:")
        print(f"   📍 Location: {camera_config['location']}")
        print(f"   🤖 Model: {self.model_name}")
        print(f"   🔍 Detector: {self.detector_backend}")
        print(f"   📐 Resolution: {camera_config['resolution']}")
    
    def start(self):
        """Start camera processing"""
        try:
            self.running = True
            source = self.camera_config['source']
            
            print(f"🔌 Starting {self.camera_id} (source: {source})...")
            
            # Initialize camera
            self.cap = cv2.VideoCapture(source)
            
            if not self.cap.isOpened():
                print(f"❌ Failed to open {self.camera_id}")
                return False
            
            # Configure camera settings
            resolution = self.camera_config['resolution']
            fps = self.camera_config['fps']
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
            self.cap.set(cv2.CAP_PROP_FPS, fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
            
            # Test frame read
            ret, frame = self.cap.read()
            if not ret:
                print(f"❌ Failed to read test frame from {self.camera_id}")
                return False
            
            print(f"✅ {self.camera_id} started successfully ({frame.shape[1]}x{frame.shape[0]})")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start {self.camera_id}: {e}")
            return False
    
    def stop(self):
        """Stop camera processing"""
        self.running = False
        if self.cap:
            self.cap.release()
        print(f"🛑 {self.camera_id} stopped")
    
    def process_frame(self, frame):
        """Process frame with face detection and recognition"""
        modified_frame = frame.copy()
        
        # Detect faces every N frames to improve performance
        if self.frame_count % Config.DETECT_EVERY_X_FRAMES == 0:
            try:
                # Extract faces
                face_objs = DeepFace.extract_faces(
                    img_path=frame,
                    detector_backend=self.detector_backend,
                    enforce_detection=False,
                    align=True
                )
                
                faces_found = len(face_objs)
                if faces_found > 0:
                    print(f"🔍 {self.camera_id}: Found {faces_found} face(s)")
                
                for face_obj in face_objs:
                    facial_area = face_obj.get("facial_area", {})
                    confidence = face_obj.get("confidence", 1.0)
                    
                    # Filter faces by confidence and size
                    if (confidence > Config.CONFIDENCE_THRESHOLD and 
                        facial_area.get('w', 0) > Config.MIN_FACE_SIZE and 
                        facial_area.get('h', 0) > Config.MIN_FACE_SIZE):
                        
                        # Get face coordinates
                        face_x = facial_area['x']
                        face_y = facial_area['y'] 
                        face_w = facial_area['w']
                        face_h = facial_area['h']
                        
                        # Create face object for processing
                        new_face = Face(
                            image=frame,
                            detector_backend=self.detector_backend,
                            model_name=self.model_name,
                            models_dict=self.models,
                            actions=self.actions,
                            db_name=self.db,
                            camera_id=self.camera_id
                        )
                        
                        # Save to database
                        new_face.save()
                        
                        # Save face crop if enabled
                        if Config.REKOGNITION_ENABLED:
                            self.save_face_crop(frame, face_x, face_y, face_w, face_h)
                        
                        # Draw face rectangle and info
                        if new_face.found:
                            color = (0, 255, 0)  # Green for recognized faces
                            label = f"RECOGNIZED: {new_face.matched_identity[:8]}... ({new_face.similarity_score:.3f})"
                        else:
                            color = (255, 0, 0)  # Blue for new faces
                            label = f"NEW: {new_face.uuid[:8]}..."
                        
                        # Draw rectangle around face
                        cv2.rectangle(modified_frame, (face_x, face_y), 
                                    (face_x + face_w, face_y + face_h), color, 2)
                        
                        # Add label
                        cv2.putText(modified_frame, label, (face_x, face_y-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
                        # Add confidence score
                        conf_text = f"Conf: {confidence:.2f}"
                        cv2.putText(modified_frame, conf_text, (face_x, face_y + face_h + 20),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            except Exception as e:
                print(f"❌ {self.camera_id}: Frame processing error: {e}")
        
        # Add camera info overlay
        self.add_overlay_info(modified_frame)
        
        return modified_frame
    
    def save_face_crop(self, frame, x, y, w, h):
        """Save cropped face image"""
        try:
            crop_face = frame[y:y + h, x:x + w]
            epoch = utils.create_epochtime()
            file_name = f"{self.camera_id}_{epoch}.jpg"
            file_path = os.path.join(Config.PICS_FOLDER, file_name)
            
            os.makedirs(Config.PICS_FOLDER, exist_ok=True)
            cv2.imwrite(file_path, crop_face)
            
            # Save picture info to database if method exists
            if hasattr(self.db, 'save_picture'):
                self.db.save_picture(file_name, epoch, file_path)
            
        except Exception as e:
            print(f"❌ {self.camera_id}: Failed to save face crop: {e}")
    
    def add_overlay_info(self, frame):
        """Add information overlay to frame"""
        # Camera info
        info_text = f"{self.camera_config['name']} | {self.camera_config['location']}"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Model info
        model_text = f"Model: {self.model_name} | Detector: {self.detector_backend}"
        cv2.putText(frame, model_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Frame counter
        frame_text = f"Frame: {self.frame_count}"
        cv2.putText(frame, frame_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


class MultiCameraManager:
    """Manager for multiple cameras with different AI models"""
    
    def __init__(self):
        print("🚀 Initializing Multi-Camera Face Recognition System")
        
        self.db = SQLiteDB(Config.DB_PATH)
        self.camera_processors = {}
        self.running = False
        self.threads = []
        
        # Initialize enabled cameras
        enabled_cameras = 0
        for camera_id, camera_config in Config.CAMERAS.items():
            if camera_config['enabled']:
                processor = CameraProcessor(camera_id, camera_config, self.db)
                self.camera_processors[camera_id] = processor
                enabled_cameras += 1
        
        print(f"📹 {enabled_cameras} camera(s) configured and ready")
        
        if enabled_cameras == 0:
            print("⚠️ No cameras enabled in config.py")
            print("💡 Enable at least one camera by setting 'enabled': True")
    
    def start(self):
        """Start all camera processors"""
        if not self.camera_processors:
            print("❌ No cameras to start")
            return
        
        self.running = True
        print("\n🔄 Starting camera processors...")
        
        # Start each camera in its own thread
        for camera_id, processor in self.camera_processors.items():
            if processor.start():
                thread = threading.Thread(
                    target=self.run_camera_loop,
                    args=(camera_id, processor),
                    daemon=True,
                    name=f"Camera_{camera_id}"
                )
                thread.start()
                self.threads.append(thread)
                print(f"🧵 Started thread for {camera_id}")
        
        if not self.threads:
            print("❌ No cameras started successfully")
            return
        
        print(f"\n✅ {len(self.threads)} camera(s) running")
        print("💡 Controls:")
        print("   • Press 'q' in any window to quit")
        print("   • Press 's' in any window to save frame")
        print("   • Press 'p' to pause/resume")
        
        try:
            # Keep main thread alive and print stats
            while self.running:
                time.sleep(1)
                
                # Print stats every 30 seconds
                if int(time.time()) % 30 == 0:
                    self.print_stats()
                    
        except KeyboardInterrupt:
            print("\n⏹️ Shutdown signal received...")
            self.stop()
    
    def run_camera_loop(self, camera_id, processor):
        """Main processing loop for a single camera"""
        window_name = f"Face Recognition - {processor.camera_config['name']}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 900, 700)
        
        print(f"🪟 Opened window: {window_name}")
        
        while self.running and processor.running:
            try:
                ret, frame = processor.cap.read()
                if not ret:
                    print(f"⚠️ {camera_id}: Failed to read frame, retrying...")
                    time.sleep(0.1)
                    continue
                
                # Process frame with face detection
                processed_frame = processor.process_frame(frame)
                
                # Display frame
                cv2.imshow(window_name, processed_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print(f"🛑 Quit command received from {camera_id}")
                    self.running = False
                    break
                elif key == ord('s'):
                    # Save current frame
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"{camera_id}_frame_{timestamp}.jpg"
                    cv2.imwrite(filename, processed_frame)
                    print(f"💾 Saved frame: {filename}")
                elif key == ord('p'):
                    print(f"⏸️ Pause/Resume from {camera_id}")
                    cv2.waitKey(0)  # Wait for any key to resume
                
                processor.frame_count += 1
                
            except Exception as e:
                print(f"❌ {camera_id}: Loop error: {e}")
                time.sleep(0.1)
        
        processor.stop()
        cv2.destroyWindow(window_name)
        print(f"🪟 Closed window: {window_name}")
    
    def stop(self):
        """Stop all camera processors"""
        print("🛑 Stopping all cameras...")
        self.running = False
        
        # Stop all processors
        for processor in self.camera_processors.values():
            processor.stop()
        
        # Wait for threads to finish
        for thread in self.threads:
            thread.join(timeout=2)
        
        cv2.destroyAllWindows()
        print("✅ All cameras stopped")
    
    def print_stats(self):
        """Print system statistics"""
        try:
            face_count = self.db.get_face_count()
            detection_count = self.db.get_detection_count()
            
            print(f"\n📊 System Stats:")
            print(f"   👤 Total unique faces: {face_count}")
            print(f"   🔍 Total detections: {detection_count}")
            print(f"   📹 Active cameras: {len(self.camera_processors)}")
            print(f"   🧵 Active threads: {len([t for t in self.threads if t.is_alive()])}")
            
        except Exception as e:
            print(f"❌ Failed to get system stats: {e}")


def main():
    """Main function to run the multi-camera system"""
    print("=" * 70)
    print("🎥 MULTI-CAMERA FACE RECOGNITION SYSTEM")
    print("=" * 70)
    print(f"🎯 Face matching threshold: {Config.FACE_MATCHING_THRESHOLD}")
    print(f"🔍 Detection interval: Every {Config.DETECT_EVERY_X_FRAMES} frames")
    print(f"📊 Confidence threshold: {Config.CONFIDENCE_THRESHOLD}")
    print("-" * 70)
    
    # Create and start manager
    manager = MultiCameraManager()
    
    if manager.camera_processors:
        manager.start()
    else:
        print("\n💡 Quick Setup Guide:")
        print("1. Edit config.py")
        print("2. Set camera_1 'enabled': True for webcam")
        print("3. Configure IP camera credentials for camera_2")
        print("4. Run: python main_multicamera.py")


if __name__ == "__main__":
    main()
