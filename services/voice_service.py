"""ElevenLabs TTS service for generating daily brief audio."""
import logging
import os
from datetime import date
from typing import Optional

from config import settings

logger = logging.getLogger(__name__)

AUDIO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "web", "static", "audio")


class VoiceService:
    """Generates spoken audio from daily brief summaries using ElevenLabs TTS."""

    def generate_audio_script(self, summary: dict) -> str:
        """Build a concise spoken script from headline + key insights (~300 chars)."""
        headline = summary.get("headline", "").strip()
        if not headline:
            return ""

        insights = summary.get("key_insights") or []
        insight_text = ". ".join(i.strip().rstrip(".") for i in insights[:3] if i.strip())

        script = f"Today's AI brief: {headline}."
        if insight_text:
            script += f" Key insights: {insight_text}."

        return script

    def generate_audio(self, script: str, output_path: str) -> str:
        """Call ElevenLabs TTS API and save MP3 to disk. Returns the file path."""
        from elevenlabs import ElevenLabs

        client = ElevenLabs(api_key=settings.elevenlabs_api_key)

        logger.info("Generating TTS audio (%d chars)", len(script))
        audio_iterator = client.text_to_speech.convert(
            voice_id=settings.elevenlabs_voice_id,
            text=script,
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
        )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            for chunk in audio_iterator:
                f.write(chunk)

        logger.info("Audio saved to %s", output_path)
        return output_path

    def get_or_generate_audio(self, summary: dict, digest_date: date, db_session=None, digest=None) -> Optional[str]:
        """Check if audio exists; generate if not. Returns URL path or None."""
        if not settings.elevenlabs_api_key:
            return None

        filename = f"brief-{digest_date.isoformat()}.mp3"
        file_path = os.path.join(AUDIO_DIR, filename)

        # Check if already generated
        if os.path.exists(file_path):
            return f"/static/audio/{filename}"

        script = self.generate_audio_script(summary)
        if not script:
            return None

        try:
            self.generate_audio(script, file_path)
        except Exception:
            logger.exception("Failed to generate TTS audio for %s", digest_date)
            return None

        # Cache the filename on the digest record
        if db_session and digest:
            try:
                digest.brief_audio_path = filename
                db_session.commit()
            except Exception:
                logger.exception("Failed to save audio path to digest")
                db_session.rollback()

        return f"/static/audio/{filename}"
