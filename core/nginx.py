# core/nginx.py

import subprocess
from pathlib import Path

from config import (
    UPSTREAM_FILE,
    PROJECT_CONFIG_FILE,
    DOMAINS,
    EMAIL,
)

from core.backends import get_active_backends


def log(msg: str) -> None:
    print(f"[NGINX] {msg}")


def run(cmd: list[str], check: bool = True) -> None:
    log(f"RUN: {' '.join(cmd)}")
    subprocess.run(cmd, check=check)


def atomic_write(path: Path, content: str) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(content)
    tmp.replace(path)


def load_project_config() -> dict:
    if not PROJECT_CONFIG_FILE.exists():
        raise RuntimeError(f"Project config not found: {PROJECT_CONFIG_FILE}")

    data = {}
    for line in PROJECT_CONFIG_FILE.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            data[k.strip()] = v.strip()
    return data


# ============================================================
#  UPSTREAM (vLLM cluster)
# ============================================================

def generate_upstream_block() -> None:
    backends = get_active_backends()

    if not backends:
        log("❌ No active backends available!")
        content = (
            "upstream vllm_cluster {\n"
            "    least_conn;\n"
            "    # No active backends\n"
            "}\n"
        )
        atomic_write(UPSTREAM_FILE, content)
        return

    lines = [
        "upstream vllm_cluster {",
        "    least_conn;",
    ]

    for be in backends:
        lines.append(f"    server {be} max_fails=3 fail_timeout=30s;")

    lines.append("}")

    atomic_write(UPSTREAM_FILE, "\n".join(lines) + "\n")
    log(f"Updated upstream with active backends: {', '.join(backends)}")


# ============================================================
#  NGINX SITE FOR DOMAIN (FULL STACK: /v1/, /aiall/, /system/)
# ============================================================

def build_nginx_site_content(domain: str) -> str:
    return f"""
server {{
    listen 80;
    server_name {domain};
    return 301 https://$host$request_uri;
}}

server {{
    listen 443 ssl http2;
    server_name {domain};

    ssl_certificate /etc/letsencrypt/live/{domain}/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/{domain}/privkey.pem;

    ssl_protocols TLSv1.2 TLSv1.3;

    client_max_body_size 100M;

    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;

    # ============================
    # OpenAI-compatible API (vLLM cluster)
    # ============================
    location /v1/ {{
        proxy_pass http://vllm_cluster;

        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        proxy_set_header Authorization $http_authorization;

        proxy_read_timeout 3600;
        proxy_send_timeout 3600;
        proxy_buffering off;
        proxy_request_buffering off;
    }}

    # ============================
    # AIALL merged model (cluster backend)
    # ============================
    location /aiall/ {{
        proxy_pass http://vllm_cluster;

        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }}

    # ============================
    # Gateway System Service (FastAPI)
    # ============================
    location /system/ {{
        proxy_pass http://127.0.0.1:6001;

        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }}
}}
""".lstrip()


def configure_nginx_site_for_domain(domain: str) -> None:
    site_file = Path(f"/etc/nginx/sites-available/{domain}.conf")
    atomic_write(site_file, build_nginx_site_content(domain))

    enabled = Path(f"/etc/nginx/sites-enabled/{domain}.conf")
    if enabled.exists() or enabled.is_symlink():
        enabled.unlink()

    enabled.symlink_to(site_file)

    log(f"Nginx site configured for {domain}")


# ============================================================
#  (OPTIONAL) LEGACY AIALL ROUTE – CÓ THỂ BỎ NẾU DÙNG SITE CHÍNH
# ============================================================

def configure_aiall_route():
    """
    Route nội bộ cho FastAPI /aiall/
    Đồng bộ với upstream vllm_cluster
    (Có thể không cần nếu đã cấu hình trong server chính)
    """

    conf = f"""
server {{
    listen 443 ssl;
    server_name api.aiallplatform.com;

    ssl_certificate /etc/letsencrypt/live/api.aiallplatform.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.aiallplatform.com/privkey.pem;

    location /aiall/ {{
        proxy_pass http://vllm_cluster;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }}
}}
"""
    Path("/etc/nginx/conf.d/aiall-route.conf").write_text(conf)
    log("AIALL route configured")


# ============================================================
#  SSL + RELOAD
# ============================================================

def issue_ssl_for_domain(domain: str) -> None:
    log(f"Requesting SSL for {domain}")

    live_dir = Path(f"/etc/letsencrypt/live/{domain}")
    if live_dir.exists():
        log(f"SSL for {domain} already exists, skipping.")
        return

    run(["systemctl", "stop", "nginx"], check=False)

    run([
        "certbot", "certonly", "--standalone",
        "-d", domain,
        "--non-interactive",
        "--agree-tos",
        "-m", EMAIL,
    ])

    log(f"SSL obtained for {domain}")


def reload_nginx() -> None:
    log("Testing nginx config...")
    run(["nginx", "-t"])

    log("Reloading nginx...")
    run(["nginx", "-s", "reload"])

