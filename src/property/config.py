"""Application configuration loader based on TOML files."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import tomllib

CONFIG_ENV_VAR = "PROPERTY_CONFIG_PATH"


@dataclass(frozen=True)
class AppSettings:
    secret_key: str


@dataclass(frozen=True)
class DatabaseSettings:
    url: str
    pool_size: int = 5
    pool_recycle: int = 1800
    pool_timeout: int = 30


@dataclass(frozen=True)
class SecuritySettings:
    token_exp_minutes: int = 60
    token_algorithm: str = "HS256"


@dataclass(frozen=True)
class Settings:
    app: AppSettings
    database: DatabaseSettings
    security: SecuritySettings


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "config" / "settings.toml"


def _resolve_config_path(path: str | os.PathLike[str] | None) -> Path:
    if path is not None:
        candidate = Path(path)
    else:
        env_value = os.getenv(CONFIG_ENV_VAR)
        candidate = Path(env_value) if env_value else _default_config_path()
    resolved = candidate.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Configuration file not found: {resolved}")
    return resolved


def load_settings(path: str | os.PathLike[str] | None = None) -> Settings:
    resolved_path = _resolve_config_path(path)
    return _load_settings_cached(str(resolved_path))


@lru_cache(maxsize=4)
def _load_settings_cached(path: str) -> Settings:
    config_path = Path(path)
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))

    app_section = _require_section(data, "app")
    database_section = _require_section(data, "database")
    security_section = data.get("security", {})

    app_settings = AppSettings(secret_key=_require_value(app_section, "secret_key", "app"))
    database_settings = DatabaseSettings(
        url=_require_value(database_section, "url", "database"),
        pool_size=int(database_section.get("pool_size", DatabaseSettings.pool_size)),
        pool_recycle=int(database_section.get("pool_recycle", DatabaseSettings.pool_recycle)),
        pool_timeout=int(database_section.get("pool_timeout", DatabaseSettings.pool_timeout)),
    )
    security_settings = SecuritySettings(
        token_exp_minutes=int(
            security_section.get("token_exp_minutes", SecuritySettings.token_exp_minutes)
        ),
        token_algorithm=str(
            security_section.get("token_algorithm", SecuritySettings.token_algorithm)
        ),
    )
    return Settings(app=app_settings, database=database_settings, security=security_settings)


def _require_section(data: Mapping[str, Any], section: str) -> Mapping[str, Any]:
    try:
        return data[section]
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Missing '{section}' section in configuration") from exc


def _require_value(section: Mapping[str, Any], key: str, section_name: str) -> str:
    value = section.get(key)
    if not value:
        raise ValueError(f"Missing '{section_name}.{key}' in configuration")
    return str(value)


__all__ = [
    "AppSettings",
    "DatabaseSettings",
    "SecuritySettings",
    "Settings",
    "load_settings",
]
