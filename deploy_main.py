# deploy_main.py
#!/usr/bin/env python3
"""
AIALL vLLM Gateway – Cluster Deployer (Python V3-SYNC)
------------------------------------------------------
- BASE_URL = https://api.aiallplatform.com
- API chuẩn OpenAI (vLLM):
    /v1/chat/completions
    /v1/completions
    /v1/models
- Không dùng prefix /ollama/
- Không dùng /api/generate, /api/pull, /api/health của Ollama nữa
"""

import argparse
import os
import shutil
import subprocess
from secrets import token_hex
from typing import List

from config import (
    DOMAINS,
    CONFIG_DIR,
    PROJECT_CONFIG_FILE,
    API_KEY_FILE,
    ProjectConfig,
)

import core.backends as be
import core.nginx as ngx

from core.auto_update import auto_update_mode
from core.rolling_restart import rolling_restart
from core.monitoring import setup_monitoring
from core.firewall import setup_firewall
from core.backup import setup_backup

from core.system_services import (
    install_vllm,
    configure_vllm_service,
    install_nginx,
    install_certbot,
)

from core.health_cluster import health_check
from core.auto_drain import auto_drain


# ============================================================
#  UTILS
# ============================================================

def is_linux() -> bool:
    return os.name == "posix"


def apt_exists() -> bool:
    return shutil.which("apt") is not None


def require_root() -> None:
    if hasattr(os, "geteuid") and os.geteuid() != 0:
        raise SystemExit("Please run as root (sudo).")


def log(msg: str) -> None:
    print(msg)


def run(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
    log(f"[RUN] {' '.join(cmd)}")
    return subprocess.run(cmd, check=check)


# ============================================================
#  PROJECT CONFIG – ALWAYS REGENERATED
# ============================================================

def backup_project_config() -> None:
    if PROJECT_CONFIG_FILE.exists():
        backup_path = PROJECT_CONFIG_FILE.with_suffix(".bak")
        log(f"[INFO] Backing up old project config to {backup_path}")
        backup_path.write_text(PROJECT_CONFIG_FILE.read_text())


def init_project_config() -> ProjectConfig:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    if not DOMAINS:
        raise SystemExit("[ERROR] No domains configured in DOMAINS.")

    if len(set(DOMAINS)) != len(DOMAINS):
        raise SystemExit("[ERROR] Duplicate domains detected in DOMAINS.")

    base_url = f"https://{DOMAINS[0]}"

    api_chat = "/v1/chat/completions"
    api_completion = "/v1/completions"
    api_models = "/v1/models"

    backup_project_config()

    log("[INFO] Creating fresh vLLM project config (v1.0)...")

    api_key = token_hex(64)
    token_secret = token_hex(64)

    PROJECT_CONFIG_FILE.write_text(
        "CONFIG_VERSION=1.0\n"
        f"BASE_URL={base_url}\n"
        f"API_CHAT={api_chat}\n"
        f"API_COMPLETION={api_completion}\n"
        f"API_MODELS={api_models}\n"
        f"API_KEY={api_key}\n"
        f"TOKEN_SECRET={token_secret}\n"
        "DEFAULT_MAX_TOKENS=1024\n"
        "DEFAULT_MIN_TOKENS=1\n"
        "DEFAULT_TEMPERATURE=0.7\n"
        "DEFAULT_TOP_P=0.9\n"
    )

    data = {}
    for line in PROJECT_CONFIG_FILE.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            data[k.strip()] = v.strip()

    API_KEY_FILE.write_text(f"AIALL_API_KEY={data['API_KEY']}\n")

    cfg = ProjectConfig.from_dict(data)

    log("[INFO] Project config loaded:")
    for k, v in data.items():
        log(f"  {k} = {v}")

    return cfg


# ============================================================
#  DNS CHECK
# ============================================================

def check_dns() -> None:
    if not is_linux():
        log("[WARN] Non-Linux system — skipping DNS check.")
        return

    log("[INFO] Checking DNS...")
    for domain in DOMAINS:
        result = subprocess.run(["getent", "hosts", domain], capture_output=True)
        if result.returncode != 0:
            raise SystemExit(f"[ERROR] DNS for {domain} not resolved.")
        log(f"[OK] DNS OK for {domain}")


# ============================================================
#  SYSTEM UPDATE
# ============================================================

def update_system() -> None:
    if not is_linux():
        log("[WARN] Non-Linux system — skipping system update.")
        return

    if not apt_exists():
        log("[WARN] apt not found — skipping system update.")
        return

    log("[INFO] Updating system...")
    run(["apt", "update"])
    run(["apt", "upgrade", "-y"])


# ============================================================
#  DEPLOY STEPS
# ============================================================

def deploy_services() -> None:
    install_vllm()
    configure_vllm_service()
    install_certbot()
    install_nginx()


def configure_nginx_and_ssl() -> None:
    ngx.generate_upstream_block()

    for domain in DOMAINS:
        ngx.issue_ssl_for_domain(domain)
        ngx.configure_nginx_site_for_domain(domain)

    ngx.reload_nginx()


def finalize_security() -> None:
    setup_monitoring()
    setup_backup()
    setup_firewall()


def print_api_info(cfg: ProjectConfig) -> None:
    log("=== API ENDPOINTS (vLLM / OpenAI) ===")
    log(f"  BASE_URL       : {cfg.base_url}")
    log(f"  CHAT_URL       : {cfg.url_chat}")
    log(f"  COMPLETION_URL : {cfg.url_completion}")
    log(f"  MODELS_URL     : {cfg.url_models}")
    log(f"  API_KEY        : {cfg.api_key}")
    log(f"  TOKEN_SECRET   : {cfg.token_secret}")

    log("[INFO] Test your API (chat/completions):")
    log(
        f"curl -X POST {cfg.url_chat} "
        f"-H \"x-api-key: {cfg.api_key}\" "
        f"-H \"Content-Type: application/json\" "
        f"-d '{{\"model\":\"meta-llama/Llama-3.1-8B-Instruct\",\"messages\":[{{\"role\":\"user\",\"content\":\"hello\"}}]}}'"
    )

    log("[INFO] Test your API (models list / health):")
    log(
        f"curl -X GET {cfg.url_models} "
        f"-H \"x-api-key: {cfg.api_key}\""
    )


# ============================================================
#  FULL DEPLOY
# ============================================================

def full_deploy() -> None:
    if not is_linux():
        raise SystemExit("[ERROR] Full deploy is only supported on Linux servers.")

    require_root()
    log(f"[INFO] Starting AIALL vLLM Gateway deployment for: {', '.join(DOMAINS)}")

    cfg = init_project_config()
    backends = be.load_backends()

    if not backends:
        log("[WARN] No backends registered. API will not function until you add one.")

    check_dns()
    update_system()

    deploy_services()
    configure_nginx_and_ssl()
    finalize_security()

    try:
        health_check()
        log("[OK] Cluster health-check passed.")
    except SystemExit as e:
        log(f"[WARN] Health-check failed: {e}")

    log("[OK] Core deploy completed.")
    print_api_info(cfg)


# ============================================================
#  CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AIALL vLLM Gateway Cluster Deployer")

    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("deploy")
    sub.add_parser("update")
    sub.add_parser("health-check")
    sub.add_parser("auto-drain")
    sub.add_parser("rolling-restart")

    add_be = sub.add_parser("add-backend")
    add_be.add_argument("backend")

    rm_be = sub.add_parser("remove-backend")
    rm_be.add_argument("backend")

    dr_be = sub.add_parser("drain-backend")
    dr_be.add_argument("backend")

    undr_be = sub.add_parser("undrain-backend")
    undr_be.add_argument("backend")

    return p


def handle_backend_command(cmd: str, backend: str) -> None:
    actions = {
        "add-backend": be.add_backend,
        "remove-backend": be.remove_backend,
        "drain-backend": be.drain_backend,
        "undrain-backend": be.undrain_backend,
    }

    if cmd not in actions:
        raise SystemExit(f"[ERROR] Unknown backend command: {cmd}")

    actions[cmd](backend)

    if cmd in ("add-backend", "remove-backend"):
        ngx.generate_upstream_block()
        ngx.reload_nginx()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    cmd = args.cmd

    if cmd == "deploy":
        full_deploy()

    elif cmd == "update":
        require_root()
        auto_update_mode()

    elif cmd == "health-check":
        require_root()
        health_check()

    elif cmd == "auto-drain":
        require_root()
        auto_drain()

    elif cmd == "rolling-restart":
        require_root()
        rolling_restart()

    elif cmd in ("add-backend", "remove-backend", "drain-backend", "undrain-backend"):
        require_root()
        handle_backend_command(cmd, args.backend)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()


