"""Knowledge package bootstrap."""

from pathlib import Path

from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parent / ".env")

from knowledge.core.model_paths import configure_local_model_environment


configure_local_model_environment()
