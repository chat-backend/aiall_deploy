# routes/aiall_advanced.py
# ============================================================
#  AIALL ADVANCED FEATURES (Assistant-Pro / Sandbox / Knowledge Base / SQL)
# ============================================================

import os
import sqlite3
from fastapi import APIRouter
from config_loader import load_runtime_config
from train.aiall_train import load_aiall_for_inference, chat

router = APIRouter(prefix="/aiall", tags=["AIALL Advanced"])


# ============================================================
#  ASSISTANT PRO (Deep Reasoning)
# ============================================================

@router.post("/assistant-pro")
def aiall_assistant_pro(messages: list, token: str = ""):
    cfg = load_runtime_config()
    if token != cfg.model_token:
        return {"auth": "failed", "error": "invalid model token"}

    try:
        model, tokenizer = load_aiall_for_inference()

        convo = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            convo += f"{role.upper()}: {content}\n"

        prompt = f"""
Bạn là AI reasoning giống ChatGPT-o1.
Hãy:
1) Phân tích vấn đề
2) Lập luận từng bước
3) Đưa ra kết luận rõ ràng

Cuộc hội thoại:
{convo}

ASSISTANT (reasoning chi tiết):
"""

        reasoning = chat(model, tokenizer, prompt)

        return {
            "object": "chat.completion",
            "model": "aiall-assistant-pro",
            "messages": messages,
            "reasoning": reasoning
        }

    except Exception as e:
        return {"assistant_pro": "failed", "error": str(e)}


# ============================================================
#  SANDBOX (Safe Python Execution)
# ============================================================

@router.post("/sandbox")
def aiall_sandbox(code: str, token: str = ""):
    cfg = load_runtime_config()
    if token != cfg.model_token:
        return {"auth": "failed", "error": "invalid model token"}

    try:
        import io
        import contextlib

        safe_globals = {
            "__builtins__": {
                "print": print,
                "range": range,
                "len": len,
                "sum": sum,
            }
        }

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(code, safe_globals, {})

        return {
            "object": "sandbox.run",
            "model": "aiall",
            "code": code,
            "output": buf.getvalue()
        }

    except Exception as e:
        return {"sandbox": "failed", "error": str(e)}


# ============================================================
#  KNOWLEDGE BASE (Private Data Query)
# ============================================================

@router.post("/knowledge-base")
def aiall_knowledge_base(query: str, token: str = ""):
    cfg = load_runtime_config()
    if token != cfg.model_token:
        return {"auth": "failed", "error": "invalid model token"}

    try:
        kb_file = "knowledge_base.txt"
        kb_content = (
            open(kb_file, "r", encoding="utf-8").read()
            if os.path.exists(kb_file)
            else ""
        )

        model, tokenizer = load_aiall_for_inference()

        prompt = f"""
Dưới đây là knowledge base nội bộ:

{kb_content}

Câu hỏi:
{query}

Hãy trả lời dựa trên knowledge base, nếu không có thông tin thì nói rõ.
"""

        answer = chat(model, tokenizer, prompt)

        return {
            "object": "knowledge.base",
            "model": "aiall",
            "query": query,
            "answer": answer
        }

    except Exception as e:
        return {"knowledge_base": "failed", "error": str(e)}


# ============================================================
#  SQL (Read-only)
# ============================================================

@router.post("/sql")
def aiall_sql(query: str, token: str = ""):
    cfg = load_runtime_config()
    if token != cfg.model_token:
        return {"auth": "failed", "error": "invalid model token"}

    try:
        conn = sqlite3.connect("aiall.db")
        cur = conn.cursor()

        if not query.strip().lower().startswith("select"):
            conn.close()
            return {"sql": "failed", "error": "only SELECT queries are allowed"}

        cur.execute(query)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]

        conn.close()

        return {
            "object": "sql.query",
            "model": "aiall",
            "query": query,
            "columns": cols,
            "rows": rows
        }

    except Exception as e:
        return {"sql": "failed", "error": str(e)}
