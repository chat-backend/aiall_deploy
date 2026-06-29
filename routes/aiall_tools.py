# routes/aiall_tools.py
# ============================================================
#  AIALL AGENT / TOOLS / FUNCTIONS / VISION-ADVANCED
# ============================================================

import json
from fastapi import APIRouter
from config_loader import load_runtime_config

from train.aiall_train import (
    load_aiall_for_inference,
    chat
)

router = APIRouter(prefix="/aiall", tags=["AIALL Tools & Agent"])


# ============================================================
#  AI AGENT (AutoGPT-style)
# ============================================================

@router.post("/agent")
def aiall_agent(goal: str, token: str = ""):
    cfg = load_runtime_config()
    if token != cfg.model_token:
        return {"auth": "failed", "error": "invalid model token"}

    try:
        model, tokenizer = load_aiall_for_inference()

        plan = chat(model, tokenizer, f"Tạo kế hoạch chi tiết để hoàn thành mục tiêu sau: {goal}")
        result = chat(model, tokenizer, f"Thực thi kế hoạch sau và trả về kết quả cuối cùng:\n{plan}")

        return {
            "object": "agent",
            "model": "aiall",
            "goal": goal,
            "plan": plan,
            "result": result
        }

    except Exception as e:
        return {"agent": "failed", "error": str(e)}


# ============================================================
#  TOOLS (Model calls external API)
# ============================================================

@router.post("/tools")
def aiall_tools(prompt: str, token: str = ""):
    cfg = load_runtime_config()
    if token != cfg.model_token:
        return {"auth": "failed", "error": "invalid model token"}

    try:
        import requests

        model, tokenizer = load_aiall_for_inference()

        tool_prompt = f"""
Phân tích yêu cầu sau và quyết định API nào cần gọi:
{prompt}

Nếu cần gọi API, trả về JSON:
{{
  "call_api": true,
  "url": "...",
  "method": "GET",
  "payload": {{}}
}}
Nếu không cần, trả về:
{{
  "call_api": false,
  "response": "..."
}}
"""

        decision_text = chat(model, tokenizer, tool_prompt)

        try:
            decision = json.loads(decision_text)
        except:
            return {
                "object": "tool.response",
                "model": "aiall",
                "decision_raw": decision_text,
                "error": "model did not return valid JSON"
            }

        if decision.get("call_api") is True:
            url = decision.get("url")
            method = decision.get("method", "GET").upper()
            payload = decision.get("payload", {})

            try:
                if method == "GET":
                    r = requests.get(url, params=payload)
                else:
                    r = requests.post(url, json=payload)

                return {
                    "object": "tool.call",
                    "model": "aiall",
                    "decision": decision,
                    "api_response": r.json()
                }
            except Exception as api_err:
                return {
                    "object": "tool.call",
                    "model": "aiall",
                    "decision": decision,
                    "error": str(api_err)
                }

        return {
            "object": "tool.response",
            "model": "aiall",
            "decision": decision
        }

    except Exception as e:
        return {"tools": "failed", "error": str(e)}


# ============================================================
#  FUNCTION CALLING (OpenAI-style)
# ============================================================

@router.post("/functions")
def aiall_functions(prompt: str, token: str = ""):
    cfg = load_runtime_config()
    if token != cfg.model_token:
        return {"auth": "failed", "error": "invalid model token"}

    try:
        model, tokenizer = load_aiall_for_inference()

        fc_prompt = f"""
Phân tích yêu cầu sau và trả về JSON function_call:
{prompt}

Ví dụ:
{{
  "name": "get_weather",
  "arguments": {{
      "location": "Đà Nẵng"
  }}
}}
"""

        fc_text = chat(model, tokenizer, fc_prompt)

        try:
            fc = json.loads(fc_text)
        except:
            return {
                "object": "function_call",
                "model": "aiall",
                "function_call_raw": fc_text,
                "error": "model did not return valid JSON"
            }

        name = fc.get("name")
        args = fc.get("arguments", {})

        if name == "get_weather":
            location = args.get("location", "unknown")
            result = {"location": location, "temp": "29°C", "status": "Nắng đẹp"}
        else:
            result = {"message": "Không có function phù hợp"}

        return {
            "object": "function_call",
            "model": "aiall",
            "function_call": fc,
            "result": result
        }

    except Exception as e:
        return {"functions": "failed", "error": str(e)}


# ============================================================
#  VISION ADVANCED (BLIP)
# ============================================================

BLIP_MODEL = None
BLIP_PROCESSOR = None

@router.post("/vision-advanced")
def aiall_vision_advanced(file: bytes, token: str = ""):
    cfg = load_runtime_config()
    if token != cfg.model_token:
        return {"auth": "failed", "error": "invalid model token"}

    try:
        global BLIP_MODEL, BLIP_PROCESSOR

        from PIL import Image
        import io
        from transformers import BlipProcessor, BlipForConditionalGeneration

        if BLIP_MODEL is None:
            BLIP_PROCESSOR = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            BLIP_MODEL = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

        img = Image.open(io.BytesIO(file)).convert("RGB")

        inputs = BLIP_PROCESSOR(img, return_tensors="pt")
        out = BLIP_MODEL.generate(**inputs)
        caption = BLIP_PROCESSOR.decode(out[0], skip_special_tokens=True)

        return {
            "object": "image.caption",
            "model": "aiall-vision",
            "caption": caption
        }

    except Exception as e:
        return {"vision_advanced": "failed", "error": str(e)}
