#!/usr/bin/env python3
"""
Video Analysis Debug Script
This script helps debug video analysis integration issues
"""

import asyncio
import logging
import sys
import os

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.services.enhanced_video_analysis import get_enhanced_video_analyzer
from app.services.video_analysis import get_video_analyzer
from app.services.langgraph_workflow import get_pitch_workflow

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def debug_video_analysis():
    """Debug video analysis components"""
    
    print("🔍 Video Analysis Debug Report")
    print("=" * 50)
    
    # 1. Check Enhanced Video Analyzer
    print("\n1. Enhanced Video Analyzer:")
    try:
        enhanced_analyzer = get_enhanced_video_analyzer()
        if enhanced_analyzer:
            print("   ✅ Enhanced video analyzer available")
            print(f"   📊 CVZone: {enhanced_analyzer.cvzone_available}")
            print(f"   😊 FER: {enhanced_analyzer.fer_available}")
            print(f"   🧍 MediaPipe: {enhanced_analyzer.mediapipe_available}")
        else:
            print("   ❌ Enhanced video analyzer not available")
    except Exception as e:
        print(f"   ❌ Error loading enhanced analyzer: {e}")
    
    # 2. Check Basic Video Analyzer
    print("\n2. Basic Video Analyzer:")
    try:
        basic_analyzer = get_video_analyzer()
        if basic_analyzer:
            print("   ✅ Basic video analyzer available")
        else:
            print("   ❌ Basic video analyzer not available")
    except Exception as e:
        print(f"   ❌ Error loading basic analyzer: {e}")
    
    # 3. Check Workflow Integration
    print("\n3. Workflow Integration:")
    try:
        workflow = get_pitch_workflow()
        if workflow:
            print("   ✅ Pitch workflow available")
            
            # Test workflow state initialization
            test_state = {
                "conversation_id": "debug_test",
                "persona": "friendly"
            }
            
            # Initialize session to check if video fields are created
            initialized_state = workflow._initialize_session(test_state)
            
            video_fields = [
                "video_analysis_enabled",
                "video_insights", 
                "gesture_feedback",
                "posture_feedback",
                "expression_feedback"
            ]
            
            print("   📋 Video analysis fields in state:")
            for field in video_fields:
                if field in initialized_state:
                    print(f"      ✅ {field}: {initialized_state[field]}")
                else:
                    print(f"      ❌ {field}: Missing")
                    
        else:
            print("   ❌ Pitch workflow not available")
    except Exception as e:
        print(f"   ❌ Error testing workflow: {e}")
    
    # 4. Check Dependencies
    print("\n4. Dependencies Check:")
    dependencies = [
        ("opencv-python", "cv2"),
        ("numpy", "numpy"),
        ("cvzone", "cvzone"),
        ("fer", "fer"),
        ("mediapipe", "mediapipe")
    ]
    
    for dep_name, import_name in dependencies:
        try:
            __import__(import_name)
            print(f"   ✅ {dep_name}")
        except ImportError:
            print(f"   ❌ {dep_name} - Not installed")
    
    # 5. Test Video Frame Processing
    print("\n5. Video Frame Processing Test:")
    try:
        import cv2
        import numpy as np
        import base64
        
        # Create a test frame (black image)
        test_frame = np.zeros((240, 320, 3), dtype=np.uint8)
        
        # Encode as JPEG
        _, buffer = cv2.imencode('.jpg', test_frame)
        frame_data = base64.b64encode(buffer).decode('utf-8')
        
        print(f"   ✅ Test frame created: {len(frame_data)} bytes")
        
        # Test decoding
        decoded_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(decoded_bytes, np.uint8)
        decoded_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if decoded_frame is not None:
            print(f"   ✅ Frame decode successful: {decoded_frame.shape}")
        else:
            print("   ❌ Frame decode failed")
            
    except Exception as e:
        print(f"   ❌ Frame processing test failed: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 Debug Summary:")
    print("If you see ❌ errors above, those need to be fixed for video analysis to work.")
    print("If everything shows ✅, the issue might be in the frontend integration.")

if __name__ == "__main__":
    asyncio.run(debug_video_analysis())