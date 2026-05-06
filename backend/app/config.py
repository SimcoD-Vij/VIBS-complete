from pydantic_settings import BaseSettings
from pathlib import Path
import torch


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql+asyncpg://vibs_user:vibs_pass@postgres:5432/vibs"
    REDIS_URL: str = "redis://redis:6379"
    HF_TOKEN: str = ""
    LLM_PROVIDER: str = "groq"
    GROQ_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    AUDIO_DIR: Path = Path("/audio_files")
    WHISPER_MODEL: str = "tiny.en"
    REALTIME_CHUNK_SECONDS: float = 2.0
    SPEAKER_SIMILARITY_THRESHOLD: float = 0.72
    SILENCE_THRESHOLD_SECONDS: int = 600
    MAX_SPEAKERS: int = 10

    class Config:
        env_file = ".env"

    @property
    def device(self) -> str:
        """Auto-detect GPU. Falls back to CPU gracefully."""
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @property
    def compute_type(self) -> str:
        """Optimal compute type for the detected device."""
        dev = self.device
        if dev == "cuda":
            # float16 on GPU = fastest + accurate
            return "float16"
        # int8 on CPU = 2-3x faster than float32, minimal accuracy loss
        return "int8"

    @property
    def gpu_info(self) -> dict:
        """Returns GPU info for the /health endpoint."""
        if not torch.cuda.is_available():
            return {"available": False, "device": "cpu"}
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                g = gpus[0]
                return {
                    "available": True,
                    "device": "cuda",
                    "name": g.name,
                    "memory_total_mb": g.memoryTotal,
                    "memory_free_mb": g.memoryFree,
                    "gpu_load_pct": round(g.load * 100, 1),
                }
        except Exception:
            pass
        return {
            "available": True,
            "device": "cuda",
            "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unknown",
        }


settings = Settings()
