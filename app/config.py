from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

@dataclass(frozen=True)
class Settings:
    model_path: Path = Path(os.getenv("MODEL_PATH", ROOT/"models/model.joblib"))
    metrics_path: Path = Path(os.getenv("METRICS_PATH", ROOT/"models/metrics.json"))
    api_keys: frozenset = field(default_factory=lambda: frozenset(k.strip() for k in os.getenv("API_KEYS", "").split(",") if k.strip()))
    cors_origins: tuple = field(default_factory=lambda: tuple(o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()))
    max_batch: int = int(os.getenv("MAX_BATCH", "500"))

settings = Settings()
