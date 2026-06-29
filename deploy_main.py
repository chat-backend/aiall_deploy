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
"""

import argparse
import os
import shutil
import subprocess
import platform
from secrets import token_hex
from typing import List

from config import (
    DOMAINS,
    CONFIG_DIR,
    PROJECT_CONFIG_FILE,
    API_KEY_FILE,
    MODEL_TOKEN_FILE,
    ProjectConfig,
)

# ============================================================
#  WINDOWS DEV MODE PROTECTION
# ============================================================

IS_LINUX = platform.system().lower() == "linux"

if not IS_LINUX:
    print("[DEV] Windows mode detected — deploy_main Linux features disabled")

from core.system_services import (
    install_vllm,
    configure_vllm_service,
    install_nginx,
    install_certbot,
)

# ============================================================
#  IMPORT LINUX-ONLY MODULES SAFELY
# ============================================================

if IS_LINUX:
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
    return IS_LINUX


def apt_exists() -> bool:
    return shutil.which("apt") is not None


def require_root() -> None:
    if IS_LINUX and hasattr(os, "geteuid") and os.geteuid() != 0:
        raise SystemExit("Please run as root (sudo).")


def log(msg: str) -> None:
    print(msg)


def run(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
    log(f"[RUN] {' '.join(cmd)}")
    return subprocess.run(cmd, check=check)


# ============================================================
#  NGINX AUTO-CONFIG (LINUX ONLY)
# ============================================================

if IS_LINUX:
    ngx.configure_aiall_route()
    ngx.reload_nginx()
else:
    print("[DEV] Skipping nginx auto-config on Windows")


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
    model_token = token_hex(64)

    PROJECT_CONFIG_FILE.write_text(
        "CONFIG_VERSION=1.0\n"
        f"BASE_URL={base_url}\n"
        f"API_CHAT={api_chat}\n"
        f"API_COMPLETION={api_completion}\n"
        f"API_MODELS={api_models}\n"
        f"API_KEY={api_key}\n"
        f"MODEL_TOKEN={model_token}\n"
        "DEFAULT_MAX_TOKENS=10000\n"
        "DEFAULT_MIN_TOKENS=5000\n"
        "DEFAULT_TEMPERATURE=0.7\n"
        "DEFAULT_TOP_P=0.9\n"
    )

    API_KEY_FILE.write_text(f"AIALL_API_KEY={api_key}\n")
    MODEL_TOKEN_FILE.write_text(f"AIALL_MODEL_TOKEN={model_token}\n")

    data = {}
    for line in PROJECT_CONFIG_FILE.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            data[k.strip()] = v.strip()

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
    log(f"  MODEL_TOKEN    : {cfg.model_token}")

    log("[INFO] Test your API (chat/completions):")
    log(
        f"curl -X POST {cfg.url_chat} "
        f"-H \"x-api-key: {cfg.api_key}\" "
        f"-H \"Content-Type: application/json\" "
        f"-d '{{"
            f"\"model\": \"aiall-merged\", "
            f"\"messages\": [{{\"role\": \"user\", \"content\": \"Xin chào AIALL, bạn đang hoạt động chứ?\"}}]"
        f"}}'"
    )

    log("[INFO] Test your API (text completions):")
    log(
        f"curl -X POST {cfg.url_completion} "
        f"-H \"x-api-key: {cfg.api_key}\" "
        f"-H \"Content-Type: application/json\" "
        f"-d '{{"
            f"\"model\": \"aiall-merged\", "
            f"\"prompt\": \"Viết một câu chào thân thiện bằng tiếng Việt.\""
        f"}}'"
    )

    log("[INFO] Test your API (models list):")
    log(
        f"curl -X GET {cfg.url_models} "
        f"-H \"x-api-key: {cfg.api_key}\""
    )

    log("[INFO] Test internal AIALL backend (FastAPI routes):")
    log(
        f"curl -X POST {cfg.base_url}/aiall/chat "
        f"-H \"Content-Type: application/json\" "
        f"-d '{{\"prompt\": \"Xin chào AIALL\"}}'"
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

    print_api_info(cfg)
    log("[OK] Core deploy completed.")


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


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    cmd = args.cmd

    if cmd == "deploy":
        full_deploy()

    elif cmd == "update":
        if not IS_LINUX:
            raise SystemExit("[ERROR] update is Linux-only")
        require_root()
        auto_update_mode()

    elif cmd == "health-check":
        if not IS_LINUX:
            raise SystemExit("[ERROR] health-check is Linux-only")
        require_root()
        health_check()

    elif cmd == "auto-drain":
        if not IS_LINUX:
            raise SystemExit("[ERROR] auto-drain is Linux-only")
        require_root()
        auto_drain()

    elif cmd == "rolling-restart":
        if not IS_LINUX:
            raise SystemExit("[ERROR] rolling-restart is Linux-only")
        require_root()
        rolling_restart()

    elif cmd in ("add-backend", "remove-backend", "drain-backend", "undrain-backend"):
        if not IS_LINUX:
            raise SystemExit("[ERROR] backend operations are Linux-only")
        require_root()
        be_action = {
            "add-backend": be.add_backend,
            "remove-backend": be.remove_backend,
            "drain-backend": be.drain_backend,
            "undrain-backend": be.undrain_backend,
        }
        be_action[cmd](args.backend)
        ngx.generate_upstream_block()
        ngx.reload_nginx()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()



