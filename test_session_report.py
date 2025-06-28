#!/usr/bin/env python3
"""
Test script to verify session report generation with video analysis
"""

import asyncio
import sys
import os

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.services.langgraph_workflow import get_pitch_workflow

async def test_session_report():
    """Test session report generation with video analysis data"""
    
    print("🧪 Testing Session Report Generation with Video Analysis")
    print("=" * 60)
    
    try:
        # Get workflow
        workflow = get_pitch_workflow()
        
        # Create test session with video analysis data
        test_session_id = "test_video_session_123"
        config = {"configurable": {"thread_id": test_session_id}}
        
        # Initialize session
        initial_state = {
            "conversation_id": test_session_id,
            "persona": "friendly",
            "messages": []
        }
        
        print("1. Initializing test session...")
        workflow.workflow.invoke(initial_state, config)
        
        # Simulate video analysis data
        print("2. Adding simulated video analysis data...")
        video_state_update = {
            "video_analysis_enabled": True,
            "video_insights": [
                "Strong hand gestures detected during product explanation",
                "Confident facial expressions throughout presentation",
                "Good posture and professional presence maintained"
            ],
            "gesture_feedback": [
                "Effective pointing gesture when highlighting key features",
                "Open palm gestures showing transparency and openness",
                "Good hand coordination with speech rhythm"
            ],
            "posture_feedback": [
                "Upright posture conveying confidence",
                "Appropriate use of space and positioning",
                "Engaged body language throughout session"
            ],
            "expression_feedback": [
                "Genuine smile when discussing passion for product",
                "Focused expression during technical explanations",
                "Good eye contact with camera/audience"
            ]
        }
        
        workflow.workflow.update_state(config, video_state_update)
        
        # Add some conversation data
        print("3. Adding conversation data...")
        conversation_update = {
            "messages": [
                {"role": "human", "content": "Hello, I'd like to present my startup idea."},
                {"role": "ai", "content": "Great! I'm excited to hear about your startup. What's your company about?"},
                {"role": "human", "content": "We're building an AI-powered fitness app that personalizes workouts."},
                {"role": "ai", "content": "That sounds interesting! What makes your solution unique in the fitness market?"}
            ]
        }
        
        workflow.workflow.update_state(config, conversation_update)
        
        # Generate session analysis
        print("4. Generating session analysis...")
        current_state = workflow.workflow.get_state(config)
        
        if current_state.values:
            # Call the analysis generation method
            analysis_result = workflow.generate_pitch_analysis(test_session_id)
            
            print("5. Analysis Result:")
            print("-" * 40)
            
            if "error" in analysis_result:
                print(f"❌ Error: {analysis_result['error']}")
            else:
                # Check video analysis sections
                category_scores = analysis_result.get("category_scores", {})
                
                print("📊 Category Scores:")
                for category, data in category_scores.items():
                    if category in ["body_language", "presentation_presence"]:
                        print(f"\n🎥 {category.replace('_', ' ').title()}:")
                        print(f"   Score: {data.get('score', 0)}")
                        print(f"   Rating: {data.get('rating', 'N/A')}")
                        print(f"   Description: {data.get('description', 'N/A')}")
                        
                        if category == "body_language":
                            print(f"   Gesture Effectiveness: {data.get('gesture_effectiveness', 'N/A')}")
                            print(f"   Posture Assessment: {data.get('posture_assessment', 'N/A')}")
                            print(f"   Overall Presence: {data.get('overall_presence', 'N/A')}")
                        
                        if category == "presentation_presence":
                            print(f"   Facial Expressions: {data.get('facial_expressions', 'N/A')}")
                            print(f"   Eye Contact: {data.get('eye_contact', 'N/A')}")
                            print(f"   Confidence Indicators: {data.get('confidence_indicators', 'N/A')}")
                
                # Check if video analysis is properly detected
                video_enabled = current_state.values.get('video_analysis_enabled', False)
                print(f"\n🔍 Video Analysis Status:")
                print(f"   Enabled: {video_enabled}")
                print(f"   Video Insights: {len(current_state.values.get('video_insights', []))}")
                print(f"   Gesture Feedback: {len(current_state.values.get('gesture_feedback', []))}")
                print(f"   Posture Feedback: {len(current_state.values.get('posture_feedback', []))}")
                print(f"   Expression Feedback: {len(current_state.values.get('expression_feedback', []))}")
                
        else:
            print("❌ No state found for session")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("✅ Test completed!")

if __name__ == "__main__":
    asyncio.run(test_session_report())