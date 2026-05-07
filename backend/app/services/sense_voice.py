import logging
logger = logging.getLogger(__name__)

_sense_voice_instance = None
_load_attempted = False

def get_sense_voice():
    global _sense_voice_instance, _load_attempted
    if _load_attempted:
        return _sense_voice_instance
    _load_attempted = True
    try:
        import torch
        from funasr import AutoModel          # ← now inside try, safe
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading SenseVoiceSmall on {device}...")
        _sense_voice_instance = AutoModel(
            model="iic/SenseVoiceSmall",
            trust_remote_code=True,
            device=device,
        )
        logger.info("SenseVoiceSmall loaded ✓")
    except ImportError:
        logger.info("funasr not installed — SenseVoice disabled")
    except Exception as e:
        logger.warning(f"SenseVoice load failed: {e}")
    return _sense_voice_instance  # returns None if failed — callers handle None
