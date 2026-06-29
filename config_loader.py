# config_loader.py
#!/usr/bin/env python3
"""
Runtime Config Loader for AIALL vLLM Gateway (V4-FULL-SYNC)
-----------------------------------------------------------
- Đọc cấu hình runtime từ project.conf + api_key + model_token
- Validate đầy đủ
- Chuẩn hóa URL cho API vLLM / OpenAI
"""

from pathlib import Path
from typing import Dict
from config import (
    PROJECT_CONFIG_FILE,
    API_KEY_FILE,
    MODEL_TOKEN_FILE,
    ProjectConfig,
)

print("DEBUG: PROJECT_CONFIG_FILE =", PROJECT_CONFIG_FILE.resolve())


# ============================================================
#  LOAD project.conf
# ============================================================

def load_project_conf() -> Dict[str, str]:
    if not PROJECT_CONFIG_FILE.exists():
        raise FileNotFoundError(f"Missing project config: {PROJECT_CONFIG_FILE}")

    data = {}
    for line in PROJECT_CONFIG_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        data[k.strip()] = v.strip()

    if not data:
        raise ValueError("project.conf is empty or invalid")

    required = [
        "BASE_URL",
        "API_CHAT",
        "API_COMPLETION",
        "API_MODELS",
    ]

    for key in required:
        if key not in data or not data[key]:
            raise ValueError(f"Missing required config key: {key}")

    return data


# ============================================================
#  LOAD api_key
# ============================================================

def load_api_key() -> str:
    if not API_KEY_FILE.exists():
        raise FileNotFoundError(f"Missing API key file: {API_KEY_FILE}")

    content = API_KEY_FILE.read_text().strip()
    if not content:
        raise ValueError("API key file is empty")

    if "=" in content:
        name, key = content.split("=", 1)
        if name.strip() != "AIALL_API_KEY":
            raise ValueError(f"Unexpected key name in api_key file: {name}")
        return key.strip()

    return content


# ============================================================
#  LOAD model_token
# ============================================================

def load_model_token() -> str:
    if not MODEL_TOKEN_FILE.exists():
        raise FileNotFoundError(f"Missing model token file: {MODEL_TOKEN_FILE}")

    content = MODEL_TOKEN_FILE.read_text().strip()
    if not content:
        raise ValueError("model_token file is empty")

    if "=" in content:
        name, key = content.split("=", 1)
        if name.strip() != "AIALL_MODEL_TOKEN":
            raise ValueError(f"Unexpected key name in model_token file: {name}")
        return key.strip()

    return content


# ============================================================
#  LOAD FULL RUNTIME CONFIG
# ============================================================

def load_runtime_config() -> ProjectConfig:
    data = load_project_conf()

    # Ghi đè API_KEY & MODEL_TOKEN từ file riêng
    data["API_KEY"] = load_api_key()
    data["MODEL_TOKEN"] = load_model_token()

    # Chuẩn hóa BASE_URL
    base = data["BASE_URL"].rstrip("/")

    # Chuẩn hóa URL đầy đủ
    data["URL_CHAT"] = base + data["API_CHAT"]
    data["URL_COMPLETION"] = base + data["API_COMPLETION"]
    data["URL_MODELS"] = base + data["API_MODELS"]

    return ProjectConfig.from_dict(data)




