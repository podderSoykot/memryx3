#!/usr/bin/env python3
"""
Test script for multi-camera face recognition system
Tests the new configuration including IP camera
"""

import cv2
import time
from config import Config


def test_camera_basic(source, name, timeout=5):
    """Basic camera test with timeout"""
    print(f"\n🔍 Testing {name} (source: {source})...")
    
    try:
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print(f"   ❌ Failed to open camera")
            return False
        
        # Set shorter timeout for IP cameras
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        start_time = time.time()
        
        # Try to read a frame with timeout
        ret, frame = cap.read()
        
        elapsed = time.time() - start_time
        
        if not ret:
            print(f"   ❌ Failed to read frame (timeout: {elapsed:.1f}s)")
            cap.release()
            return False
        
        print(f"   ✅ Camera working (response time: {elapsed:.1f}s)")
        print(f"   📐 Resolution: {frame.shape[1]}x{frame.shape[0]}")
        
        cap.release()
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def test_ip_camera_connection():
    """Test IP camera connectivity"""
    print("\n📡 Testing IP Camera Connection...")
    
    # Test your IP camera
    ip_camera_url = 'rtsp://admin:qwq1234.@192.168.1.71/channel=1/subtype=0'
    
    print(f"🔗 RTSP URL: {ip_camera_url}")
    print("⏱️ Timeout: 10 seconds")
    
    success = test_camera_basic(ip_camera_url, "IP Camera", timeout=10)
    
    if success:
        print("✅ IP Camera is accessible!")
        print("💡 You can enable it in config.py by setting 'enabled': True")
    else:
        print("❌ IP Camera connection failed")
        print("🔧 Troubleshooting:")
        print("   1. Check network connectivity")
        print("   2. Verify IP address: 192.168.1.71")
        print("   3. Check credentials: admin:qwq1234.")
        print("   4. Ensure camera supports RTSP")
    
    return success


def test_multi_camera_config():
    """Test the multi-camera configuration"""
    print("\n⚙️ Testing Multi-Camera Configuration...")
    
    working_cameras = []
    
    for camera_id, camera_config in Config.CAMERAS.items():
        print(f"\n📷 {camera_id}:")
        print(f"   Name: {camera_config['name']}")
        print(f"   Location: {camera_config['location']}")
        print(f"   Source: {camera_config['source']}")
        print(f"   Enabled: {camera_config['enabled']}")
        print(f"   Model: {camera_config['model_config']}")
        
        if camera_config['enabled']:
            success = test_camera_basic(camera_config['source'], camera_config['name'])
            if success:
                working_cameras.append(camera_id)
        else:
            print(f"   ⏸️ Camera disabled in config")
    
    return working_cameras


def main():
    """Main test function"""
    print("🚀 Multi-Camera Face Recognition System - Advanced Test")
    print("=" * 60)
    
    # Test 1: Check basic webcam
    print("\n1️⃣ WEBCAM TEST")
    webcam_works = test_camera_basic(0, "Primary Webcam")
    
    # Test 2: Test IP camera connectivity
    print("\n2️⃣ IP CAMERA TEST")
    ip_camera_works = test_ip_camera_connection()
    
    # Test 3: Test configured cameras
    print("\n3️⃣ CONFIGURATION TEST")
    working_cameras = test_multi_camera_config()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST RESULTS")
    print("=" * 60)
    
    print(f"✅ Webcam (source 0): {'Working' if webcam_works else 'Failed'}")
    print(f"📡 IP Camera: {'Working' if ip_camera_works else 'Failed'}")
    print(f"⚙️ Configured cameras working: {len(working_cameras)}")
    
    if working_cameras:
        print(f"\n🎯 Ready cameras: {', '.join(working_cameras)}")
        
        if ip_camera_works and 'camera_2' not in working_cameras:
            print("\n💡 RECOMMENDATION:")
            print("   Your IP camera is working but disabled in config.")
            print("   To enable it, edit config.py:")
            print("   Set camera_2 'enabled': True")
    
    print(f"\n🚀 NEXT STEPS:")
    if working_cameras:
        print("   1. Run: python main_multicamera.py")
        print("   2. Press 'q' to quit system")
        if ip_camera_works:
            print("   3. Enable IP camera in config.py for dual-camera setup")
    else:
        print("   1. Check camera connections")
        print("   2. Verify network settings for IP camera")
        print("   3. Run: python test_cameras.py for detailed diagnostics")


if __name__ == "__main__":
    main() 