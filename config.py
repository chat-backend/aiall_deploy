# config.py
#!/usr/bin/env python3
"""
AIALL vLLM Gateway – System Configuration (V6-PRODUCTION-AIALL)
----------------------------------------------------------------------
- API chuẩn OpenAI (vLLM)
- Không prefix
- Backend normalize
- Path normalize
- Health check /v1/models
- Hỗ trợ multi-backend
- Chạy được trên Windows + Linux
- Phân biệt rõ API_KEY (client) và MODEL_TOKEN (internal)
"""

from pathlib import Path
from dataclasses import dataclass
from typing import List
import os
import platform

print("[CONFIG] Loaded AIALL vLLM Gateway configuration (V6-PRODUCTION-AIALL)")

# ============================================================
#  DOMAIN & EMAIL CONFIG
# ============================================================

DOMAINS: List[str] = ["api.aiallplatform.com"]
EMAIL: str = "openaimanage@gmail.com"

# ============================================================
#  BASE DIRECTORIES (CROSS-PLATFORM)
# ============================================================

IS_WINDOWS = platform.system().lower().startswith("win")

if IS_WINDOWS:
    CONFIG_DIR = Path("vllm_config")  # Windows local dev
else:
    CONFIG_DIR = Path("/etc/vllm")    # Linux production

CONFIG_DIR.mkdir(parents=True, exist_ok=True)

PROJECT_CONFIG_FILE = CONFIG_DIR / "project.conf"
API_KEY_FILE = CONFIG_DIR / "api_key"
MODEL_TOKEN_FILE = CONFIG_DIR / "model_token"

BACKENDS_CONFIG = CONFIG_DIR / "backends.conf"
DRAIN_CONFIG = CONFIG_DIR / "backends.drain"

DEFAULT_BACKENDS = ["127.0.0.1:8000"]

# ============================================================
#  NGINX CONFIG PATHS (Linux only)
# ============================================================

if IS_WINDOWS:
    UPSTREAM_FILE = Path("nginx_upstream.conf")
    LOG_FILE = Path("vllm_deploy.log")
else:
    UPSTREAM_FILE = Path("/etc/nginx/conf.d/vllm-upstream.conf")
    LOG_FILE = Path("/var/log/vllm-deploy.log")

# ============================================================
#  PROJECT CONFIG STRUCTURE
# ============================================================

@dataclass
class ProjectConfig:
    config_version: str = "1.0"

    # URL chính thức của dự án AIALL
    base_url: str = os.getenv("AIALL_BASE_URL", "https://api.aiallplatform.com")

    # API chuẩn OpenAI/vLLM
    api_chat: str = "/v1/chat/completions"
    api_completion: str = "/v1/completions"
    api_models: str = "/v1/models"

    # API Key dành cho người dùng
    api_key: str = ""

    # Token dành cho mô hình nội bộ (khác API Key)
    model_token: str = ""

    # Default params (nâng cấp theo yêu cầu)
    default_max_tokens: int = 10000
    default_min_tokens: int = 5000
    default_temperature: float = 0.7
    default_top_p: float = 0.9

    # Chuẩn hóa path
    def normalize(self, path: str) -> str:
        return path if path.startswith("/") else f"/{path}"

    # Tạo URL đầy đủ cho người dùng cuối
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

    # Tạo URL backend vLLM
    def backend_url(self, backend: str, path: str) -> str:
        backend = backend if backend.startswith("http") else f"http://{backend}"
        return backend + self.normalize(path)

    # URL health-check backend
    def backend_health_url(self, backend: str) -> str:
        return self.backend_url(backend, self.api_models)

    @staticmethod
    def from_dict(data: dict) -> "ProjectConfig":
        return ProjectConfig(
            config_version=data.get("CONFIG_VERSION", "1.0"),
            base_url=data.get("BASE_URL", os.getenv("AIALL_BASE_URL", "https://api.aiallplatform.com")),

            api_chat=data.get("API_CHAT", "/v1/chat/completions"),
            api_completion=data.get("API_COMPLETION", "/v1/completions"),
            api_models=data.get("API_MODELS", "/v1/models"),

            api_key=data.get("API_KEY", ""),
            model_token=data.get("MODEL_TOKEN", ""),

            default_max_tokens=int(data.get("DEFAULT_MAX_TOKENS", 10000)),
            default_min_tokens=int(data.get("DEFAULT_MIN_TOKENS", 5000)),
            default_temperature=float(data.get("DEFAULT_TEMPERATURE", 0.7)),
            default_top_p=float(data.get("DEFAULT_TOP_P", 0.9)),
        )

# ============================================================
#  SIMPLE HELPERS FOR API_KEY / MODEL_TOKEN
# ============================================================

def read_api_key() -> str:
    if not API_KEY_FILE.exists():
        return ""
    return API_KEY_FILE.read_text(encoding="utf-8").strip()


def write_api_key(value: str) -> None:
    API_KEY_FILE.write_text(value.strip() + "\n", encoding="utf-8")


def read_model_token() -> str:
    if not MODEL_TOKEN_FILE.exists():
        return ""
    return MODEL_TOKEN_FILE.read_text(encoding="utf-8").strip()


def write_model_token(value: str) -> None:
    MODEL_TOKEN_FILE.write_text(value.strip() + "\n", encoding="utf-8")


__all__ = [
    "DOMAINS",
    "EMAIL",
    "CONFIG_DIR",
    "PROJECT_CONFIG_FILE",
    "API_KEY_FILE",
    "MODEL_TOKEN_FILE",
    "BACKENDS_CONFIG",
    "DRAIN_CONFIG",
    "DEFAULT_BACKENDS",
    "UPSTREAM_FILE",
    "LOG_FILE",
    "ProjectConfig",
    "read_api_key",
    "write_api_key",
    "read_model_token",
    "write_model_token",
]




