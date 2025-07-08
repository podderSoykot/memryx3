#!/usr/bin/env python3
"""
Quick launcher for Multi-Camera Face Recognition System v1
"""

import sys
import os
import subprocess


def print_banner():
    """Print system banner"""
    print("=" * 70)
    print("🎥 MULTI-CAMERA FACE RECOGNITION SYSTEM v1.0")
    print("=" * 70)
    print("🚀 Production Ready | 🤖 AI-Powered | 📊 Real-time Analytics")
    print("-" * 70)


def check_dependencies():
    """Check if required packages are installed"""
    try:
        import cv2
        import deepface
        import numpy
        import scipy
        print("✅ All dependencies found")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("💡 Run: pip install -r requirements.txt")
        return False


def main_menu():
    """Show main menu options"""
    print("\n📋 SELECT OPTION:")
    print("   1. 🎥 Start Multi-Camera System")
    print("   2. 🔧 Test Cameras")
    print("   3. 🧪 Advanced Camera Test")
    print("   4. 📊 Check Database Stats")
    print("   5. 📖 View Documentation")
    print("   6. ❌ Exit")
    
    while True:
        try:
            choice = input("\n➤ Enter choice (1-6): ").strip()
            
            if choice == "1":
                print("\n🚀 Starting Multi-Camera System...")
                print("💡 Press 'q' in any window to quit")
                subprocess.run([sys.executable, "main_multicamera.py"])
                
            elif choice == "2":
                print("\n🔧 Running Camera Tests...")
                subprocess.run([sys.executable, "test_cameras.py"])
                
            elif choice == "3":
                print("\n🧪 Running Advanced Camera Tests...")
                subprocess.run([sys.executable, "test_multicamera.py"])
                
            elif choice == "4":
                print("\n📊 Checking Database Statistics...")
                try:
                    from sqlite_config import SQLiteDB
                    db = SQLiteDB("faces.db")
                    faces = db.get_face_count()
                    detections = db.get_detection_count()
                    print(f"   👤 Unique faces: {faces}")
                    print(f"   🔍 Total detections: {detections}")
                    print(f"   📁 Database: faces.db")
                except Exception as e:
                    print(f"   ❌ Database error: {e}")
                
            elif choice == "5":
                print("\n📖 Available Documentation:")
                docs = [
                    "README.md - Main documentation",
                    "README_MultiCamera.md - Technical details", 
                    "QUICK_START.md - Setup guide",
                    "ip_camera_setup.md - IP camera help",
                    "VERSION_INFO.txt - Version details"
                ]
                for doc in docs:
                    print(f"   📄 {doc}")
                
            elif choice == "6":
                print("\n👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice. Please enter 1-6.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def quick_status():
    """Show quick system status"""
    print("\n🔍 SYSTEM STATUS:")
    
    # Check Python version
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"   🐍 Python: {py_version}")
    
    # Check if database exists
    if os.path.exists("faces.db"):
        print("   💾 Database: Found")
    else:
        print("   💾 Database: Not found (will be created)")
    
    # Check config file
    if os.path.exists("config.py"):
        print("   ⚙️ Config: Found")
    else:
        print("   ⚙️ Config: Missing!")
    
    # Check main file
    if os.path.exists("main_multicamera.py"):
        print("   🎥 Main System: Ready")
    else:
        print("   🎥 Main System: Missing!")


def main():
    """Main function"""
    print_banner()
    
    if not check_dependencies():
        print("\n🛑 Cannot start - missing dependencies")
        return
    
    quick_status()
    main_menu()


if __name__ == "__main__":
    main() 