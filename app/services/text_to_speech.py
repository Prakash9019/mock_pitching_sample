#text_to_speech.py
import os
import logging
import json
from typing import Optional
from google.oauth2 import service_account
from google.cloud import texttospeech

# Configure logging
logger = logging.getLogger(__name__)

def convert_text_to_speech(text: str, output_path: str = None) -> bytes:
    """
    Convert text to speech using Google Cloud Text-to-Speech API.
    
    Args:
        text: The text to convert to speech
        output_path: Optional path to save the audio file. If None, file won't be saved.
        
    Returns:
        The audio data as bytes in MP3 format
    """
    logger.info(f"Converting text to speech (text length: {len(text)} chars)")
    
    try:
        # Get credentials from environment variables
        credentials_info = {
            'type': os.getenv('TYPE'),
            'project_id': os.getenv('PROJECT_ID'),
            'private_key_id': os.getenv('PRIVATE_KEY_ID'),
            'private_key': os.getenv('PRIVATE_KEY').replace('\\n', '\n'),
            'client_email': os.getenv('CLIENT_EMAIL'),
            'client_id': os.getenv('CLIENT_ID'),
            'auth_uri': os.getenv('AUTH_URI'),
            'token_uri': os.getenv('TOKEN_URI'),
            'auth_provider_x509_cert_url': os.getenv('AUTH_PROVIDER_X509_CERT_URL'),
            'client_x509_cert_url': os.getenv('CLIENT_X509_CERT_URL'),
            'universe_domain': os.getenv('UNIVERSE_DOMAIN', 'googleapis.com')
        }
        
        # Create credentials object
        credentials = service_account.Credentials.from_service_account_info(
            credentials_info,
            scopes=['https://www.googleapis.com/auth/cloud-platform']
        )
        
        # Initialize the client with credentials
        client = texttospeech.TextToSpeechClient(credentials=credentials)
        # Set the text input to be synthesized
        synthesis_input = texttospeech.SynthesisInput(text=text)

        # Build the voice request
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name="en-US-Studio-O",  # A high-quality voice
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE  # Using FEMALE as it's widely supported
        )

        # Select the type of audio file you want returned
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
        # Initialize the client            speaking_rate=1.0,  # Normal speed
            pitch=0.0,  # Normal pitch
            volume_gain_db=0.0  # No volume adjustment
        )

        # Perform the text-to-speech request
        response = client.synthesize_speech(
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
            
        return audio_data
        
    except Exception as e:
        logger.error(f"Error converting text to speech: {str(e)}")
        # Return empty bytes on error
        return b''