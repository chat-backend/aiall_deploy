# core/deploy_aiall_url.py
#!/usr/bin/env python3
"""
AIALL vLLM Gateway – Cluster Deployer (Ubuntu Desktop / Server Friendly)
------------------------------------------------------
- BASE_URL = https://api.aiallplatform.com (hoặc domain bạn cấu hình)
- API chuẩn OpenAI (vLLM):
    /v1/chat/completions
    /v1/completions
    /v1/models

Phiên bản này:
- Vẫn giữ đầy đủ logic deploy vLLM Gateway.
- Nhưng được chỉnh để chạy được trên Ubuntu (không bắt buộc root).
- Các bước cần quyền root (apt, nginx, firewall, vLLM service...) sẽ:
    - cố gắng chạy nếu có quyền,
    - nếu không, chỉ log cảnh báo, không dừng toàn bộ script.
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

IS_LINUX = platform.system().lower() == "linux"

# ============================================================
#  SAFE IMPORTS (Linux-only modules)
# ============================================================

if IS_LINUX:
    try:
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
    except ImportError:
        be = None
        ngx = None
        auto_update_mode = None
        rolling_restart = None
        setup_monitoring = None
        setup_firewall = None
        setup_backup = None
        install_vllm = None
        configure_vllm_service = None
        install_nginx = None
        install_certbot = None
        health_check = None
        auto_drain = None
else:
    be = None
    ngx = None
    auto_update_mode = None
    rolling_restart = None
    setup_monitoring = None
    setup_firewall = None
    setup_backup = None
    install_vllm = None
    configure_vllm_service = None
    install_nginx = None
    install_certbot = None
    health_check = None
    auto_drain = None


# ============================================================
#  UTILS
# ============================================================

def is_linux() -> bool:
    return IS_LINUX


def apt_exists() -> bool:
    return shutil.which("apt") is not None


def is_root() -> bool:
    return hasattr(os, "geteuid") and os.geteuid() == 0


def require_root_soft() -> None:
    """
    Phiên bản mềm: nếu không phải root, chỉ cảnh báo.
    Không dừng script, để dùng được trên Ubuntu desktop.
    """
    if IS_LINUX and not is_root():
        print("[WARN] Not running as root. Some system-level operations may fail (apt, nginx, firewall, vLLM service).")


def log(msg: str) -> None:
    print(msg)


def run(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
    log(f"[RUN] {' '.join(cmd)}")
    return subprocess.run(cmd, check=check)


# ============================================================
#  NGINX AUTO-CONFIG (LINUX ONLY, SOFT)
# ============================================================

def initial_nginx_autoconfig() -> None:
    if not IS_LINUX or ngx is None:
        print("[DEV] Skipping nginx auto-config (non-Linux or core.nginx missing).")
        return

    try:
        ngx.configure_aiall_route()
        ngx.reload_nginx()
        log("[INFO] Initial nginx auto-config done.")
    except Exception as e:
        log(f"[WARN] Initial nginx auto-config failed: {e}")


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
#  DNS CHECK (SOFT)
# ============================================================

def check_dns() -> None:
    if not is_linux():
        log("[WARN] Non-Linux system — skipping DNS check.")
        return

    log("[INFO] Checking DNS...")
    for domain in DOMAINS:
        try:
            result = subprocess.run(["getent", "hosts", domain], capture_output=True)
            if result.returncode != 0:
                log(f"[WARN] DNS for {domain} not resolved. On desktop/dev, this may be OK.")
            else:
                log(f"[OK] DNS OK for {domain}")
        except Exception as e:
            log(f"[WARN] DNS check failed for {domain}: {e}")


# ============================================================
#  SYSTEM UPDATE (SOFT)
# ============================================================

def update_system() -> None:
    if not is_linux():
        log("[WARN] Non-Linux system — skipping system update.")
        return

    if not apt_exists():
        log("[WARN] apt not found — skipping system update.")
        return

    if not is_root():
        log("[WARN] Not root — skipping apt update/upgrade. Run manually if needed.")
        return

    log("[INFO] Updating system...")
    try:
        run(["apt", "update"])
        run(["apt", "upgrade", "-y"])
    except Exception as e:
        log(f"[WARN] System update failed: {e}")


# ============================================================
#  DEPLOY STEPS (SOFT)
# ============================================================

def deploy_services() -> None:
    if not IS_LINUX or install_vllm is None or configure_vllm_service is None or install_nginx is None or install_certbot is None:
        log("[WARN] core.system_services not available or non-Linux — skipping deploy_services.")
        return

    require_root_soft()
    try:
        install_vllm()
        configure_vllm_service()
        install_certbot()
        install_nginx()
    except Exception as e:
        log(f"[WARN] deploy_services failed: {e}")


def configure_nginx_and_ssl() -> None:
    if not IS_LINUX or ngx is None:
        log("[WARN] core.nginx not available or non-Linux — skipping nginx/SSL config.")
        return

    require_root_soft()
    try:
        ngx.generate_upstream_block()

        for domain in DOMAINS:
            try:
                ngx.issue_ssl_for_domain(domain)
                ngx.configure_nginx_site_for_domain(domain)
            except Exception as e:
                log(f"[WARN] SSL/nginx config failed for {domain}: {e}")

        ngx.reload_nginx()
    except Exception as e:
        log(f"[WARN] configure_nginx_and_ssl failed: {e}")


def finalize_security() -> None:
    if not IS_LINUX or setup_monitoring is None or setup_backup is None or setup_firewall is None:
        log("[WARN] Monitoring/backup/firewall modules not available or non-Linux — skipping finalize_security.")
        return

    require_root_soft()
    try:
        setup_monitoring()
        setup_backup()
        setup_firewall()
    except Exception as e:
        log(f"[WARN] finalize_security failed: {e}")


def print_api_info(cfg: ProjectConfig) -> None:
    log("=== API ENDPOINTS (vLLM / OpenAI) ===")
    log(f"  BASE_URL       : {cfg.base_url}")
    log(f"  CHAT_URL       : {cfg.url_chat}")
    log(f"  COMPLETION_URL : {cfg.url_completion}")
    log(f"  MODELS_URL     : {cfg.url_models}")
    log(f"  API_KEY        : {cfg.api_key}")
    log(f"  MODEL_TOKEN    : {cfg.model_token}")


# ============================================================
#  FULL DEPLOY (SOFT, UBUNTU-FRIENDLY)
# ============================================================

def full_deploy() -> None:
    if not is_linux():
        raise SystemExit("[ERROR] Full deploy is only supported on Linux (Ubuntu).")

    require_root_soft()
    log(f"[INFO] Starting AIALL vLLM Gateway deployment for: {', '.join(DOMAINS)}")

    initial_nginx_autoconfig()

    cfg = init_project_config()

    if be is not None:
        try:
            backends = be.load_backends()
        except Exception as e:
            log(f"[WARN] load_backends failed: {e}")
            backends = []
    else:
        backends = []

    if not backends:
        log("[WARN] No backends registered. API will not function until you add one (e.g., AIALL backend on port 8001).")

    check_dns()
    update_system()

    deploy_services()
    configure_nginx_and_ssl()
    finalize_security()

    if health_check is not None:
        try:
            health_check()
            log("[OK] Cluster health-check passed.")
        except SystemExit as e:
            log(f"[WARN] Health-check failed: {e}")
        except Exception as e:
            log(f"[WARN] Health-check error: {e}")
    else:
        log("[WARN] health_check not available — skipping cluster health-check.")

    print_api_info(cfg)
    log("[OK] Core deploy completed (Ubuntu-friendly, soft mode).")


# ============================================================
#  CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AIALL vLLM Gateway Cluster Deployer (Ubuntu-friendly)")

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
        if not IS_LINUX or auto_update_mode is None:
            raise SystemExit("[ERROR] update is Linux-only and requires core.auto_update.")
        require_root_soft()
        auto_update_mode()

    elif cmd == "health-check":
        if not IS_LINUX or health_check is None:
            raise SystemExit("[ERROR] health-check is Linux-only and requires core.health_cluster.")
        require_root_soft()
        health_check()

    elif cmd == "auto-drain":
        if not IS_LINUX or auto_drain is None:
            raise SystemExit("[ERROR] auto-drain is Linux-only and requires core.auto_drain.")
        require_root_soft()
        auto_drain()

    elif cmd == "rolling-restart":
        if not IS_LINUX or rolling_restart is None:
            raise SystemExit("[ERROR] rolling-restart is Linux-only and requires core.rolling_restart.")
        require_root_soft()
        rolling_restart()

    elif cmd in ("add-backend", "remove-backend", "drain-backend", "undrain-backend"):
        if not IS_LINUX or be is None or ngx is None:
            raise SystemExit("[ERROR] backend operations are Linux-only and require core.backends/core.nginx.")
        require_root_soft()
        be_action = {
            "add-backend": be.add_backend,
            "remove-backend": be.remove_backend,
            "drain-backend": be.drain_backend,
            "undrain-backend": be.undrain_backend,
        }
        be_action[cmd](args.backend)
        try:
            ngx.generate_upstream_block()
            ngx.reload_nginx()
        except Exception as e:
            log(f"[WARN] nginx reload after backend operation failed: {e}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()




