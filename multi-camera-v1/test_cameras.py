#!/usr/bin/env python3
"""
Camera Test Script for Multi-Camera Face Recognition System

This script helps you test your camera configuration before running the full system.
"""

import cv2
import time
from config import Config


def test_camera_source(source, name):
    """Test if a camera source is accessible"""
    print(f"\n🔍 Testing {name} (source: {source})...")
    
    try:
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print(f"   ❌ Failed to open camera")
            return False
        
        # Try to read a frame
        ret, frame = cap.read()
        
        if not ret:
            print(f"   ❌ Failed to read frame from camera")
            cap.release()
            return False
        
        # Get camera properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"   ✅ Camera accessible")
        print(f"   📐 Resolution: {width}x{height}")
        print(f"   🎬 FPS: {fps}")
        print(f"   📊 Frame shape: {frame.shape}")
        
        cap.release()
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def test_all_webcams():
    """Test all available webcam indices"""
    print("\n🎥 Testing available webcams...")
    available_cameras = []
    
    for i in range(10):  # Test indices 0-9
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                available_cameras.append(i)
                print(f"   ✅ Webcam {i} available")
            cap.release()
    
    if not available_cameras:
        print("   ❌ No webcams found")
    else:
        print(f"   📹 Available webcam indices: {available_cameras}")
    
    return available_cameras


def test_camera_configuration():
    """Test the configured cameras from config.py"""
    print("\n⚙️ Testing configured cameras from config.py...")
    
    working_cameras = []
    
    for camera_id, camera_config in Config.CAMERAS.items():
        if camera_config['enabled']:
            print(f"\n📷 Testing {camera_id}:")
            print(f"   Name: {camera_config['name']}")
            print(f"   Location: {camera_config['location']}")
            print(f"   Source: {camera_config['source']}")
            print(f"   Model: {camera_config['model_config']}")
            
            success = test_camera_source(camera_config['source'], camera_config['name'])
            
            if success:
                working_cameras.append(camera_id)
            
    return working_cameras


def live_camera_test(camera_source, duration=10):
    """Show live camera feed for testing"""
    print(f"\n📺 Live test for camera source: {camera_source}")
    print(f"   Duration: {duration} seconds")
    print("   Press 'q' to quit early, 's' to save frame")
    
    cap = cv2.VideoCapture(camera_source)
    
    if not cap.isOpened():
        print("   ❌ Failed to open camera for live test")
        return
    
    window_name = f"Camera Test - {camera_source}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 640, 480)
    
    start_time = time.time()
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("   ❌ Failed to read frame")
            break
        
        frame_count += 1
        elapsed_time = time.time() - start_time
        
        # Add overlay information
        info_text = f"Source: {camera_source} | Frame: {frame_count} | Time: {elapsed_time:.1f}s"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        instructions = "Press 'q' to quit, 's' to save frame"
        cv2.putText(frame, instructions, (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow(window_name, frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"test_frame_{camera_source}_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print(f"   💾 Saved frame: {filename}")
        
        # Auto-quit after duration
        if elapsed_time >= duration:
            break
    
    cap.release()
    cv2.destroyWindow(window_name)
    
    actual_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    print(f"   📊 Test completed: {frame_count} frames in {elapsed_time:.1f}s (FPS: {actual_fps:.1f})")


def print_model_configuration():
    """Print the model configuration"""
    print("\n🤖 Model Configurations:")
    
    for model_id, model_config in Config.MODELS.items():
        enabled_status = "✅ ENABLED" if model_config['enabled'] else "❌ DISABLED"
        print(f"\n   {model_id.upper()} {enabled_status}")
        print(f"      Model: {model_config['name']}")
        print(f"      Detector: {model_config['detector']}")
        print(f"      Tracker: {model_config['tracker']}")
        print(f"      Actions: {', '.join(model_config['actions'])}")


def main():
    """Main test function"""
    print("🚀 Multi-Camera Face Recognition System - Camera Test")
    print("=" * 60)
    
    # Test 1: Check available webcams
    available_webcams = test_all_webcams()
    
    # Test 2: Test configured cameras
    working_cameras = test_camera_configuration()
    
    # Test 3: Show model configuration
    print_model_configuration()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    print(f"Available webcams: {available_webcams}")
    print(f"Working configured cameras: {working_cameras}")
    
    if working_cameras:
        print(f"\n✅ {len(working_cameras)} camera(s) ready for use!")
        
        # Offer live test
        print("\n🎬 LIVE CAMERA TEST")
        print("Would you like to test any camera live? (y/n): ", end="")
        
        try:
            choice = input().lower().strip()
            if choice == 'y':
                print("\nAvailable cameras for live test:")
                for i, camera_id in enumerate(working_cameras):
                    camera_config = Config.CAMERAS[camera_id]
                    print(f"   {i+1}. {camera_config['name']} (source: {camera_config['source']})")
                
                print("Enter camera number (or 0 to skip): ", end="")
                selection = int(input())
                
                if 1 <= selection <= len(working_cameras):
                    camera_id = working_cameras[selection-1]
                    camera_source = Config.CAMERAS[camera_id]['source']
                    live_camera_test(camera_source, duration=15)
                
        except (KeyboardInterrupt, ValueError):
            print("\n🛑 Live test skipped")
    
    else:
        print("\n❌ No working cameras found!")
        print("\n🔧 TROUBLESHOOTING TIPS:")
        print("1. Check camera connections")
        print("2. Verify camera indices (try different numbers)")
        print("3. For IP cameras, check RTSP URLs")
        print("4. Ensure cameras aren't being used by other applications")
        print("5. Try running as administrator/sudo")
    
    print("\n🎯 Next steps:")
    print("1. Update config.py with working camera sources")
    print("2. Run: python main_multicamera.py")
    print("3. Press 'q' in any window to quit the main system")


if __name__ == "__main__":
    main() 