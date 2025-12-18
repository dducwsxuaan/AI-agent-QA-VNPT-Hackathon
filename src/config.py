import os
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    DB_PATH: Path = DATA_DIR / "qdrant_storage"

    # LLM Large credentials
    VNPT_API_KEY_LARGE: str = Field(..., env="VNPT_API_KEY_LARGE")  # Token Key
    VNPT_TOKEN_ID_LARGE: str = Field(..., env="VNPT_TOKEN_ID_LARGE")
    VNPT_ACCESS_TOKEN_LARGE: str = Field(..., env="VNPT_ACCESS_TOKEN_LARGE")  # Authorization (Bearer ...)

    # LLM Small credentials
    VNPT_API_KEY_SMALL: str = Field(..., env="VNPT_API_KEY_SMALL")
    VNPT_TOKEN_ID_SMALL: str = Field(..., env="VNPT_TOKEN_ID_SMALL")
    VNPT_ACCESS_TOKEN_SMALL: str = Field(..., env="VNPT_ACCESS_TOKEN_SMALL")

    # Embedding credentials
    VNPT_API_KEY_EMBED: str = Field(..., env="VNPT_API_KEY_EMBED")
    VNPT_TOKEN_ID_EMBED: str = Field(..., env="VNPT_TOKEN_ID_EMBED")
    VNPT_ACCESS_TOKEN_EMBED: str = Field(..., env="VNPT_ACCESS_TOKEN_EMBED")

    class Config:
        env_file = Path(__file__).resolve().parent.parent / ".env"
        extra = "ignore"

settings = Settings()