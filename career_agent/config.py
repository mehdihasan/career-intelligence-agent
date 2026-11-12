from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from career_agent.schemas import CareerProfile, ScoringWeights


class ResearchSettings(BaseModel):
    max_queries_per_company: int = Field(default=5, ge=0, le=20)
    max_pages_per_company: int = Field(default=10, ge=0, le=40)
    timeout_seconds: int = Field(default=60, ge=1, le=300)
    enabled: bool = True


class ModelSettings(BaseModel):
    model_id: str = "Qwen/Qwen3-4B-Instruct-2507"
    device: str = "mps"
    dtype: str = "bfloat16"
    max_input_chars: int = Field(default=24000, ge=1000, le=100000)
    max_context_tokens: int = Field(default=16384, ge=1024, le=32768)
    max_new_tokens: int = Field(default=3000, ge=64, le=8192)
    enabled: bool = True


class Settings(BaseModel):
    data_dir: Path = Path.home() / ".career-agent"
    input_directory: Path = Path.home() / ".career-agent" / "jobs"
    database_path: Path | None = None
    profile_path: Path | None = None
    model: ModelSettings = Field(default_factory=ModelSettings)
    research: ResearchSettings = Field(default_factory=ResearchSettings)
    scoring_weights: ScoringWeights = Field(default_factory=ScoringWeights)

    def model_post_init(self, __context: object) -> None:
        if self.database_path is None:
            self.database_path = self.data_dir / "career-agent.duckdb"
        if self.profile_path is None:
            self.profile_path = self.data_dir / "career-profile.yaml"


def load_settings() -> Settings:
    data_dir = Path(os.environ.get("CAREER_AGENT_HOME", Path.home() / ".career-agent"))
    settings_path = Path(os.environ.get("CAREER_AGENT_SETTINGS", data_dir / "settings.yaml"))
    if not settings_path.exists():
        return Settings(data_dir=data_dir)
    raw = yaml.safe_load(settings_path.read_text(encoding="utf-8")) or {}
    settings = Settings.model_validate(raw)
    if "CAREER_AGENT_HOME" in os.environ:
        settings.data_dir = data_dir
        if "database_path" not in raw:
            settings.database_path = data_dir / "career-agent.duckdb"
        if "profile_path" not in raw:
            settings.profile_path = data_dir / "career-profile.yaml"
        if "input_directory" not in raw:
            settings.input_directory = data_dir / "jobs"
    return settings


def settings_file(settings: Settings) -> Path:
    return Path(
        os.environ.get("CAREER_AGENT_SETTINGS", settings.data_dir / "settings.yaml")
    ).expanduser()


def load_profile(path: Path) -> CareerProfile:
    if not path.exists():
        return CareerProfile()
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return CareerProfile.model_validate(value)


def save_profile(path: Path, profile: CareerProfile) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(profile.model_dump(mode="json"), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
