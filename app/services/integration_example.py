# integration_example.py
"""
Example integration showing how to use the improved AI agent and LangGraph workflow
"""

import os
import logging
from typing import Dict, Any
import uuid

# Import the improved systems
from .intelligent_ai_agent_improved import (
    initialize_improved_agent,
    start_improved_conversation,
    generate_improved_response
)
from .langgraph_workflow import (
    initialize_pitch_workflow,
    start_pitch_session,
    process_pitch_message,
    get_pitch_analytics
)

logger = logging.getLogger(__name__)

class PitchPracticeManager:
    """Manager class to handle pitch practice sessions with both approaches"""
    
    def __init__(self):
        self.improved_agent = None
        self.workflow_agent = None
        self.active_sessions = {}
        
    def initialize(self):
        """Initialize both AI systems"""
        try:
            # Initialize improved agent
            self.improved_agent = initialize_improved_agent()
            logger.info("Improved AI agent initialized")
            
            # Initialize LangGraph workflow
            self.workflow_agent = initialize_pitch_workflow()
            logger.info("LangGraph workflow initialized")
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing pitch practice systems: {e}")
            return False
    
    def start_improved_session(self, persona: str = "friendly") -> Dict[str, Any]:
        """Start a session with the improved agent (single question focus)"""
        session_id = str(uuid.uuid4())
        
        try:
            # Start conversation with improved agent
            context = start_improved_conversation(session_id, persona)
            
            # Get initial greeting
            initial_message = context.chat_history[-1]['message']
            
            # Store session info
            self.active_sessions[session_id] = {
                "type": "improved",
                "persona": persona,
                "context": context
            }
            
            return {
                "session_id": session_id,
                "message": initial_message,
                "type": "improved_agent",
                "persona": persona,
                "stage": context.current_stage.value
            }
            
        except Exception as e:
            logger.error(f"Error starting improved session: {e}")
            return {"error": "Failed to start improved session"}
    
    def start_workflow_session(self, persona: str = "friendly") -> Dict[str, Any]:
        """Start a session with the LangGraph workflow"""
        session_id = str(uuid.uuid4())
        
        try:
            # Start workflow session
            result = start_pitch_session(session_id, persona)
            
            # Store session info
            self.active_sessions[session_id] = {
                "type": "workflow",
                "persona": persona
            }
            
            result["type"] = "langgraph_workflow"
            return result
            
        except Exception as e:
            logger.error(f"Error starting workflow session: {e}")
            return {"error": "Failed to start workflow session"}
    
    def process_message(self, session_id: str, message: str) -> Dict[str, Any]:
        """Process a message using the appropriate system"""
        
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}
        
        session_info = self.active_sessions[session_id]
        session_type = session_info["type"]
        
        try:
            if session_type == "improved":
                # Use improved agent
                response = generate_improved_response(session_id, message)
                context = session_info["context"]
                
                return {
                    "message": response,
                    "type": "improved_agent",
                    "stage": context.current_stage.value,
                    "session_id": session_id
                }
                
            elif session_type == "workflow":
                # Use LangGraph workflow
                result = process_pitch_message(session_id, message)
                result["type"] = "langgraph_workflow"
                return result
                
            else:
                return {"error": "Unknown session type"}
                
        except Exception as e:
            logger.error(f"Error processing message for {session_type}: {e}")
            return {"error": f"Failed to process message with {session_type}"}
    
    def get_session_analytics(self, session_id: str) -> Dict[str, Any]:
        """Get analytics for a session"""
        
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}
        
        session_info = self.active_sessions[session_id]
        session_type = session_info["type"]
        
        try:
            if session_type == "improved":
                # Get analytics from improved agent
                if self.improved_agent:
                    stats = self.improved_agent.get_conversation_stats(session_id)
                    stats["type"] = "improved_agent"
                    return stats
                else:
                    return {"error": "Improved agent not initialized"}
                    
            elif session_type == "workflow":
                # Get analytics from workflow
                analytics = get_pitch_analytics(session_id)
                analytics["type"] = "langgraph_workflow"
                return analytics
                
            else:
                return {"error": "Unknown session type"}
                
        except Exception as e:
            logger.error(f"Error getting analytics for {session_type}: {e}")
            return {"error": f"Failed to get analytics for {session_type}"}
    
    def compare_systems(self, test_conversation: list) -> Dict[str, Any]:
        """Compare both systems with the same test conversation"""
        
        results = {
            "improved_agent": {"responses": [], "analytics": {}},
            "langgraph_workflow": {"responses": [], "analytics": {}},
            "comparison": {}
        }
        
        try:
            # Test improved agent
            improved_session = self.start_improved_session("friendly")
            if "session_id" in improved_session:
                improved_id = improved_session["session_id"]
                results["improved_agent"]["responses"].append(improved_session["message"])
                
                for message in test_conversation:
                    response = self.process_message(improved_id, message)
                    if "message" in response:
                        results["improved_agent"]["responses"].append(response["message"])
                
                results["improved_agent"]["analytics"] = self.get_session_analytics(improved_id)
            
            # Test workflow
            workflow_session = self.start_workflow_session("friendly")
            if "session_id" in workflow_session:
                workflow_id = workflow_session["session_id"]
                results["langgraph_workflow"]["responses"].append(workflow_session["message"])
                
                for message in test_conversation:
                    response = self.process_message(workflow_id, message)
                    if "message" in response:
                        results["langgraph_workflow"]["responses"].append(response["message"])
                
                results["langgraph_workflow"]["analytics"] = self.get_pitch_analytics(workflow_id)
            
            # Generate comparison
            results["comparison"] = {
                "improved_questions": len(results["improved_agent"]["responses"]),
                "workflow_questions": len(results["langgraph_workflow"]["responses"]),
                "improved_stages": results["improved_agent"]["analytics"].get("completed_stages", []),
                "workflow_stages": results["langgraph_workflow"]["analytics"].get("completed_stages", [])
            }
            
        except Exception as e:
            logger.error(f"Error comparing systems: {e}")
            results["error"] = str(e)
        
        return results

# Example usage functions
def example_improved_agent_usage():
    """Example of how to use the improved agent"""
    
    manager = PitchPracticeManager()
    if not manager.initialize():
        print("Failed to initialize systems")
        return
    
    # Start session
    session = manager.start_improved_session("skeptical")
    print("Investor:", session["message"])
    
    # Simulate conversation
    founder_responses = [
        "Hi, I'm John and my company is TechSolve",
        "We help small businesses automate their accounting with AI",
        "Our target market is small businesses with 5-50 employees",
        "We charge $99/month per business"
    ]
    
    for response in founder_responses:
        print("Founder:", response)
        result = manager.process_message(session["session_id"], response)
        if "message" in result:
            print("Investor:", result["message"])
            print(f"Stage: {result['stage']}")
        print()
    
    # Get analytics
    analytics = manager.get_session_analytics(session["session_id"])
    print("Session Analytics:", analytics)

def example_workflow_usage():
    """Example of how to use the LangGraph workflow"""
    
    manager = PitchPracticeManager()
    if not manager.initialize():
        print("Failed to initialize systems")
        return
    
    # Start session
    session = manager.start_workflow_session("technical")
    print("Investor:", session["message"])
    
    # Simulate conversation
    founder_responses = [
        "I'm Sarah and I run DataFlow Analytics",
        "We solve the problem of real-time data processing for IoT devices",
        "We target manufacturing companies that need real-time monitoring",
        "Our solution uses edge computing and machine learning"
    ]
    
    for response in founder_responses:
        print("Founder:", response)
        result = manager.process_message(session["session_id"], response)
        if "message" in result:
            print("Investor:", result["message"])
            print(f"Stage: {result['stage']}")
            if "insights" in result:
                print("Key insights:", result["insights"])
        print()
    
    # Get analytics
    analytics = manager.get_session_analytics(session["session_id"])
    print("Session Analytics:", analytics)

def example_system_comparison():
    """Example comparing both systems"""
    
    manager = PitchPracticeManager()
    if not manager.initialize():
        print("Failed to initialize systems")
        return
    
    test_conversation = [
        "Hi, I'm Alex and my startup is GreenTech Solutions",
        "We're solving climate change by helping companies reduce their carbon footprint",
        "Our target market is mid-size companies that want to go carbon neutral",
        "We make money through SaaS subscriptions and consulting services"
    ]
    
    comparison = manager.compare_systems(test_conversation)
    
    print("=== SYSTEM COMPARISON ===")
    print(f"Improved Agent Questions: {comparison['comparison']['improved_questions']}")
    print(f"Workflow Questions: {comparison['comparison']['workflow_questions']}")
    print(f"Improved Stages: {comparison['comparison']['improved_stages']}")
    print(f"Workflow Stages: {comparison['comparison']['workflow_stages']}")

# Global manager instance
pitch_manager = None

def get_pitch_manager():
    """Get global pitch manager instance"""
    global pitch_manager
    if pitch_manager is None:
        pitch_manager = PitchPracticeManager()
        pitch_manager.initialize()
    return pitch_manager

# Convenience functions for FastAPI integration
def start_practice_session(session_type: str = "workflow", persona: str = "friendly") -> Dict[str, Any]:
    """Start a new pitch practice session"""
    manager = get_pitch_manager()
    
    if session_type == "improved":
        return manager.start_improved_session(persona)
    else:
        return manager.start_workflow_session(persona)

def handle_practice_message(session_id: str, message: str) -> Dict[str, Any]:
    """Handle a message in the pitch practice session"""
    manager = get_pitch_manager()
    return manager.process_message(session_id, message)

def get_practice_analytics(session_id: str) -> Dict[str, Any]:
    """Get analytics for a pitch practice session"""
    manager = get_pitch_manager()
    return manager.get_session_analytics(session_id)

if __name__ == "__main__":
    # Run examples
    print("=== IMPROVED AGENT EXAMPLE ===")
    example_improved_agent_usage()
    
    print("\n=== WORKFLOW EXAMPLE ===")
    example_workflow_usage()
    
    print("\n=== COMPARISON EXAMPLE ===")
    example_system_comparison()