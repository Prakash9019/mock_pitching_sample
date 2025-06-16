# enhanced_text_to_speech.py
import os
import logging
import json
from typing import Optional, Dict
from google.oauth2 import service_account
from google.cloud import texttospeech

# Configure logging
logger = logging.getLogger(__name__)

# Voice configurations for different personas using Google Cloud TTS
PERSONA_VOICES = {
    "skeptical": {
        "language_code": "en-US",
        "name": "en-US-Neural2-F",  # Professional, authoritative female voice
        "ssml_gender": texttospeech.SsmlVoiceGender.FEMALE,
        "speaking_rate": 0.95,  # Slightly slower
        "pitch": -2.0,  # Slightly lower pitch
        "volume_gain_db": 0.0
    },
    "technical": {
        "language_code": "en-US", 
        "name": "en-US-Neural2-D",  # Enthusiastic male voice
        "ssml_gender": texttospeech.SsmlVoiceGender.MALE,
        "speaking_rate": 1.1,  # Slightly faster
        "pitch": 1.0,  # Higher pitch for enthusiasm
        "volume_gain_db": 0.0
    },
    "friendly": {
        "language_code": "en-US",
        "name": "en-US-Neural2-C",  # Warm, friendly female voice  
        "ssml_gender": texttospeech.SsmlVoiceGender.FEMALE,
        "speaking_rate": 1.0,  # Normal speed
        "pitch": 2.0,  # Warm tone
        "volume_gain_db": 0.0
    }
}

# Fallback voice settings
FALLBACK_VOICE = {
    "language_code": "en-US",
    "name": "en-US-Studio-O",
    "ssml_gender": texttospeech.SsmlVoiceGender.FEMALE,
    "speaking_rate": 1.0,
    "pitch": 0.0,
    "volume_gain_db": 0.0
}

class EnhancedTextToSpeech:
    """Enhanced TTS service with persona-specific voices using Google Cloud TTS"""
    
    def __init__(self):
        self.client = None
        self._initialize_client()
        
    def _initialize_client(self):
        """Initialize Google Cloud TTS client"""
        try:
            # Get credentials from environment variables
            credentials_info = {
                'type': os.getenv('TYPE'),
                'project_id': os.getenv('PROJECT_ID'),
                'private_key_id': os.getenv('PRIVATE_KEY_ID'),
                'private_key': os.getenv('PRIVATE_KEY').replace('\\n', '\n') if os.getenv('PRIVATE_KEY') else None,
                'client_email': os.getenv('CLIENT_EMAIL'),
                'client_id': os.getenv('CLIENT_ID'),
                'auth_uri': os.getenv('AUTH_URI'),
                'token_uri': os.getenv('TOKEN_URI'),
                'auth_provider_x509_cert_url': os.getenv('AUTH_PROVIDER_X509_CERT_URL'),
                'client_x509_cert_url': os.getenv('CLIENT_X509_CERT_URL'),
                'universe_domain': os.getenv('UNIVERSE_DOMAIN', 'googleapis.com')
            }
            
            # Check if all required credentials are present
            if not all([credentials_info['type'], credentials_info['project_id'], 
                       credentials_info['private_key'], credentials_info['client_email']]):
                logger.error("Missing required Google Cloud credentials")
                return
            
            # Create credentials object
            credentials = service_account.Credentials.from_service_account_info(
                credentials_info,
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
            
            # Initialize the client with credentials
            self.client = texttospeech.TextToSpeechClient(credentials=credentials)
            logger.info("Google Cloud TTS client initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Google Cloud TTS client: {str(e)}")
            self.client = None
    
    def convert_text_to_speech_with_persona(self, text: str, persona: str = "friendly", output_path: str = None) -> Optional[bytes]:
        """Convert text to speech with persona-specific voice"""
        if not self.client:
            logger.error("Google Cloud TTS client not initialized")
            return None
            
        try:
            logger.info(f"Converting text to speech with {persona} persona (text length: {len(text)} chars)")
            
            # Get voice settings for persona
            voice_config = PERSONA_VOICES.get(persona, FALLBACK_VOICE)
            
            # Set the text input to be synthesized
            synthesis_input = texttospeech.SynthesisInput(text=text)

            # Build the voice request with persona-specific settings
            voice = texttospeech.VoiceSelectionParams(
                language_code=voice_config["language_code"],
                name=voice_config["name"],
                ssml_gender=voice_config["ssml_gender"]
            )

            # Select the type of audio file with persona-specific audio settings
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=voice_config["speaking_rate"],
                pitch=voice_config["pitch"],
                volume_gain_db=voice_config["volume_gain_db"]
            )

            # Perform the text-to-speech request
            response = self.client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )
            
            # The response's audio_content is binary
            audio_data = response.audio_content
            
            # Save to file if output_path is provided
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'wb') as out:
                    out.write(audio_data)
                logger.info(f"Saved audio to: {output_path}")
            
            logger.info(f"Generated {len(audio_data)} bytes of audio with {voice_config['name']} voice")
            return audio_data
            
        except Exception as e:
            logger.error(f"Error converting text to speech: {str(e)}")
            return None
    
    def get_available_voices(self) -> Dict:
        """Get list of available voices for each persona"""
        if not self.client:
            return {"error": "TTS client not initialized"}
            
        try:
            # List available voices from Google Cloud TTS
            voices = self.client.list_voices()
            
            # Filter to English voices
            english_voices = [
                {
                    "name": voice.name,
                    "language_codes": list(voice.language_codes),
                    "ssml_gender": voice.ssml_gender.name
                }
                for voice in voices.voices 
                if any(lang.startswith('en-') for lang in voice.language_codes)
            ]
            
            return {
                "available_voices": len(english_voices),
                "persona_voices": {
                    persona: {
                        "name": config["name"],
                        "language_code": config["language_code"],
                        "gender": config["ssml_gender"].name,
                        "speaking_rate": config["speaking_rate"],
                        "pitch": config["pitch"]
                    }
                    for persona, config in PERSONA_VOICES.items()
                },
                "sample_voices": english_voices[:15]  # First 15 for reference
            }
            
        except Exception as e:
            logger.error(f"Error getting available voices: {str(e)}")
            return {"error": f"Failed to get voices: {str(e)}"}
    
    def test_persona_voice(self, persona: str, test_text: str = "Hello! This is a test of the text to speech system.") -> Optional[bytes]:
        """Test a specific persona voice with sample text"""
        return self.convert_text_to_speech_with_persona(test_text, persona)

# Global TTS instance
tts_service = EnhancedTextToSpeech()

# Convenience functions for external use
def convert_text_to_speech_with_persona(text: str, persona: str = "friendly", output_path: str = None) -> Optional[bytes]:
    """Convert text to speech with persona-specific voice"""
    return tts_service.convert_text_to_speech_with_persona(text, persona, output_path)

def get_persona_voice_info(persona: str) -> Dict:
    """Get voice information for a specific persona"""
    voice_config = PERSONA_VOICES.get(persona, FALLBACK_VOICE)
    return {
        "name": voice_config["name"],
        "language_code": voice_config["language_code"],
        "gender": voice_config["ssml_gender"].name,
        "speaking_rate": voice_config["speaking_rate"],
        "pitch": voice_config["pitch"],
        "volume_gain_db": voice_config["volume_gain_db"]
    }

def list_available_voices() -> Dict:
    """List all available voices"""
    return tts_service.get_available_voices()

def test_all_personas(test_text: str = "Hello! I'm excited to hear about your business idea.") -> Dict[str, bool]:
    """Test all persona voices"""
    results = {}
    for persona in PERSONA_VOICES.keys():
        audio_data = tts_service.test_persona_voice(persona, test_text)
        results[persona] = audio_data is not None and len(audio_data) > 0
        logger.info(f"Persona '{persona}' test: {'SUCCESS' if results[persona] else 'FAILED'}")
    return results