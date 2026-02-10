import os
import requests
from typing import Optional
import hashlib
from dotenv import load_dotenv

load_dotenv()

# Audio processing - try to import, but make it optional
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

# Whisper - try to import, but make it optional
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

# OpenAI API for cloud transcription
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

OUT_DIR = "out"
AUDIO_CACHE_DIR = os.path.join(OUT_DIR, "audio_cache")


def ensure_audio_cache_dir() -> None:
    """Create the audio cache directory if it doesn't exist."""
    os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)


def get_audio_file_path(audio_url: str) -> str:
    """Generate a cache file path for an audio URL."""
    # Create a hash-based filename
    url_hash = hashlib.sha256(audio_url.encode("utf-8")).hexdigest()[:16]
    # Try to preserve extension from URL
    ext = ".mp3"  # Default
    if audio_url.endswith(".m4a"):
        ext = ".m4a"
    elif audio_url.endswith(".mp3"):
        ext = ".mp3"
    elif audio_url.endswith(".wav"):
        ext = ".wav"
    return os.path.join(AUDIO_CACHE_DIR, f"{url_hash}{ext}")


def download_audio(audio_url: str, timeout: int = 300) -> Optional[str]:
    """
    Download audio file and save to cache.
    Returns the local file path, or None on error.
    """
    ensure_audio_cache_dir()
    cache_path = get_audio_file_path(audio_url)
    
    # Return cached file if it exists
    if os.path.exists(cache_path):
        return cache_path
    
    try:
        response = requests.get(audio_url, timeout=timeout, stream=True)
        response.raise_for_status()
        
        # Write to cache
        with open(cache_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return cache_path
    except Exception as e:
        print(f"Warning: Failed to download audio from {audio_url}: {e}")
        return None


def trim_audio_to_minutes(audio_path: str, minutes: int = 10) -> Optional[str]:
    """
    Trim audio file to first N minutes.
    Returns path to trimmed file, or original path if trimming fails.
    """
    if not PYDUB_AVAILABLE:
        # If pydub not available, return original (will transcribe full file)
        return audio_path
    
    try:
        # Load audio
        audio = AudioSegment.from_file(audio_path)
        
        # Calculate duration in milliseconds
        max_duration_ms = minutes * 60 * 1000
        
        # If audio is shorter than requested, return original
        if len(audio) <= max_duration_ms:
            return audio_path
        
        # Trim to first N minutes
        trimmed = audio[:max_duration_ms]
        
        # Save trimmed version
        trimmed_path = audio_path.replace(".mp3", "_trimmed.mp3").replace(".m4a", "_trimmed.m4a").replace(".wav", "_trimmed.wav")
        trimmed.export(trimmed_path, format="mp3")
        
        return trimmed_path
    except Exception as e:
        print(f"Warning: Failed to trim audio: {e}")
        # Return original if trimming fails
        return audio_path


def transcribe_audio(audio_path: str, model_name: str = "base") -> Optional[str]:
    """
    Transcribe audio file using local Whisper.
    Returns transcribed text, or None on error.
    """
    if not WHISPER_AVAILABLE:
        return None

    try:
        # Load Whisper model (will download on first use)
        model = whisper.load_model(model_name)

        # Transcribe
        result = model.transcribe(audio_path)

        return result["text"].strip()
    except Exception as e:
        print(f"Warning: Failed to transcribe audio: {e}")
        return None


def transcribe_audio_openai(audio_path: str) -> Optional[str]:
    """
    Transcribe audio file using OpenAI Whisper API.
    Returns transcribed text, or None on error.
    """
    if not OPENAI_AVAILABLE:
        return None

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your-openai-api-key":
        return None

    try:
        client = openai.OpenAI(api_key=api_key)

        with open(audio_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )

        return transcript.strip() if isinstance(transcript, str) else transcript
    except Exception as e:
        print(f"Warning: OpenAI transcription failed: {e}")
        return None


def transcribe_episode(audio_url: str, minutes: int = 15, model_name: str = "base") -> Optional[str]:
    """
    Main function: Download, trim, and transcribe podcast episode.

    Args:
        audio_url: URL to the audio file
        minutes: Number of minutes to transcribe (default: 15)
        model_name: Whisper model to use for local transcription (default: "base")

    Returns:
        Transcribed text, or None on error
    """
    # Check if we have any transcription capability
    has_openai = OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY")

    if not WHISPER_AVAILABLE and not has_openai:
        print("Warning: No transcription available. Install whisper or set OPENAI_API_KEY")
        return None

    # Download audio
    audio_path = download_audio(audio_url)
    if audio_path is None:
        return None

    # Trim to first N minutes
    trimmed_path = trim_audio_to_minutes(audio_path, minutes)

    # Try local Whisper first (faster and free)
    if WHISPER_AVAILABLE:
        transcript = transcribe_audio(trimmed_path, model_name)
        if transcript:
            return transcript

    # Fall back to OpenAI Whisper API
    if has_openai:
        print("  Using OpenAI Whisper API for transcription...")
        transcript = transcribe_audio_openai(trimmed_path)
        if transcript:
            return transcript

    return None
