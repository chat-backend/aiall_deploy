# config.py
#!/usr/bin/env python3
"""
AIALL Gateway – Unified Configuration (Ubuntu-Friendly, Project-Local)
----------------------------------------------------------------------
- Domain chính thức: api.aiallplatform.com
- Không dùng /etc/vllm (chỉ dùng thư mục dự án)
- Tự tạo file config nếu thiếu
- Một file duy nhất cho toàn bộ hệ thống (train + inference + gateway)
"""

from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict
import os

print("[CONFIG] Loaded AIALL Gateway configuration (LOCAL-UBUNTU-MODE)")

# ============================================================
#  DOMAIN CONFIG
# ============================================================

DOMAINS: List[str] = ["api.aiallplatform.com"]
EMAIL: str = "openaimanage@gmail.com"

# ============================================================
#  BASE DIRECTORIES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

CONFIG_DIR = BASE_DIR / "config"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

PROJECT_CONFIG_FILE = CONFIG_DIR / "project.conf"
API_KEY_FILE = CONFIG_DIR / "api_key"
MODEL_TOKEN_FILE = CONFIG_DIR / "model_token"

BACKENDS_CONFIG = CONFIG_DIR / "backends.conf"
DRAIN_CONFIG = CONFIG_DIR / "backends.drain"

DEFAULT_BACKENDS = ["127.0.0.1:8001"]

UPSTREAM_FILE = CONFIG_DIR / "nginx_upstream.conf"
LOG_FILE = CONFIG_DIR / "vllm_deploy.log"

# ============================================================
#  LOGGING
# ============================================================

def log(msg: str):
    print(f"[CONFIG-LOG] {msg}")

# ============================================================
#  AUTO FIX CONFIG
# ============================================================

def ensure_config_files() -> None:
    """Tự tạo file config nếu thiếu."""

    if not PROJECT_CONFIG_FILE.exists():
        log(f"project.conf missing → creating default at {PROJECT_CONFIG_FILE}")
        PROJECT_CONFIG_FILE.write_text(
            "CONFIG_VERSION=1.0\n"
            "BASE_URL=https://api.aiallplatform.com\n"
            "API_CHAT=/v1/chat/completions\n"
            "API_COMPLETION=/v1/completions\n"
            "API_MODELS=/v1/models\n"
            "API_KEY=dev-local\n"
            "DEFAULT_MAX_TOKENS=10000\n"
            "DEFAULT_MIN_TOKENS=5000\n"
            "DEFAULT_TEMPERATURE=0.7\n"
            "DEFAULT_TOP_P=0.9\n"
        )

    if not API_KEY_FILE.exists():
        log(f"api_key missing → creating default at {API_KEY_FILE}")
        API_KEY_FILE.write_text("dev-local\n")

    if not MODEL_TOKEN_FILE.exists():
        log(f"model_token missing → creating default at {MODEL_TOKEN_FILE}")
        MODEL_TOKEN_FILE.write_text("AIALL_MODEL_TOKEN=dev-local\n")

# ============================================================
#  PROJECT CONFIG STRUCTURE
# ============================================================

@dataclass
class ProjectConfig:
    config_version: str = "1.0"

    base_url: str = os.getenv("AIALL_BASE_URL", "https://api.aiallplatform.com")

    api_chat: str = "/v1/chat/completions"
    api_completion: str = "/v1/completions"
    api_models: str = "/v1/models"

    api_key: str = ""

    default_max_tokens: int = 10000
    default_min_tokens: int = 5000
    default_temperature: float = 0.7
    default_top_p: float = 0.9

    def normalize(self, path: str) -> str:
        return path if path.startswith("/") else f"/{path}"

    def full(self, path: str) -> str:
        return self.base_url + self.normalize(path)

    @property
    def url_chat(self) -> str:
        return self.full(self.api_chat)

    @property
    def url_completion(self) -> str:
        return self.full(self.api_completion)

    @property
    def url_models(self) -> str:
        return self.full(self.api_models)

    def backend_url(self, backend: str, path: str) -> str:
        backend = backend if backend.startswith("http") else f"http://{backend}"
        return backend + self.normalize(path)

    def backend_health_url(self, backend: str) -> str:
        return self.backend_url(backend, self.api_models)

    @staticmethod
    def from_dict(data: dict) -> "ProjectConfig":
        return ProjectConfig(
            config_version=data.get("CONFIG_VERSION", "1.0"),
            base_url=data.get("BASE_URL", "https://api.aiallplatform.com"),

            api_chat=data.get("API_CHAT", "/v1/chat/completions"),
            api_completion=data.get("API_COMPLETION", "/v1/completions"),
            api_models=data.get("API_MODELS", "/v1/models"),

            api_key=data.get("API_KEY", ""),

            default_max_tokens=int(data.get("DEFAULT_MAX_TOKENS", 10000)),
            default_min_tokens=int(data.get("DEFAULT_MIN_TOKENS", 5000)),
            default_temperature=float(data.get("DEFAULT_TEMPERATURE", 0.7)),
            default_top_p=float(data.get("DEFAULT_TOP_P", 0.9)),
        )

# ============================================================
#  LOAD CONFIG
# ============================================================

def load_project_conf() -> Dict[str, str]:
    ensure_config_files()

    data: Dict[str, str] = {}
    for line in PROJECT_CONFIG_FILE.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            data[k.strip()] = v.strip()
    return data


def load_runtime_config() -> ProjectConfig:
    return ProjectConfig.from_dict(load_project_conf())


def load_model_token() -> str:
    """Đọc token mô hình từ MODEL_TOKEN_FILE."""
    ensure_config_files()
    content = MODEL_TOKEN_FILE.read_text().strip()
    if "=" in content:
        _, v = content.split("=", 1)
        return v.strip()
    return content

# ============================================================
#  INIT CONFIG SYSTEM
# ============================================================

def init_config_system():
    log("Initializing unified config system...")
    ensure_config_files()
    log("Config integrity check:")
    for f in [PROJECT_CONFIG_FILE, API_KEY_FILE, MODEL_TOKEN_FILE]:
        log(f"  {f.name}: {'OK' if f.exists() else 'MISSING'}")
