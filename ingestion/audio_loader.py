# ingestion/audio_loader.py
import logging

logger = logging.getLogger(__name__)


def load_audio(file_path):
    """Transcribe audio to text using OpenAI Whisper Python API."""
    try:
        import whisper
    except ImportError:
        logger.error("Whisper not installed. Install with: pip install openai-whisper")
        return []

    try:
        logger.info(f"Loading Whisper 'base' model...")
        model = whisper.load_model("base")
        logger.info(f"Transcribing audio: {file_path}")
        result = model.transcribe(file_path)
        transcript = result.get("text", "").strip()

    except Exception as e:
        logger.error(f"Failed to transcribe audio '{file_path}': {e}")
        return []

    if not transcript:
        logger.warning(f"No transcript generated for '{file_path}'")
        return []

    logger.info(f"Transcription complete: {len(transcript)} characters")
    return [{
        "content": transcript,
        "metadata": {
            "source": file_path,
            "type": "audio"
        }
    }]
