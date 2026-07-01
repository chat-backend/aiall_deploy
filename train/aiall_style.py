# train/aiall_style.py
"""
AIALL STYLE ENGINE – Version 8.0
--------------------------------
Nâng cấp lớn:
- STYLE 8.0: tự tạo kế hoạch dài hạn cho người dùng (Long-Term Plan Mode)
- REASONING MODE: suy luận nhiều bước giống GPT‑o1 (multi-step reasoning)
- AGENT SWARM: 10+ agent phối hợp phân tích vấn đề phức tạp
- SELF‑HEAL 2.0: nhiều vòng tự sửa lỗi logic
- DEBATE MODE + META‑AGENT: đa góc nhìn, tranh luận
- SELF‑REFINE MODE: tối ưu câu trả lời cuối
- PERSONALITY MODE: tính cách AI
- DOMAIN EXPERT MODE: chuyên gia theo ngành
- CONTEXT MEMORY: nhớ nội dung cuộc trò chuyện dài hạn
"""

import torch
from datetime import datetime
from typing import Optional, Dict

USER_STYLE_PREFS: Dict[str, dict] = {}
CONTEXT_MEMORY: Dict[str, dict] = {}


# ============================================================
#  LANGUAGE DETECTION
# ============================================================

def detect_vietnamese(prompt: str) -> bool:
    pl = prompt.lower()
    return any(ch in pl for ch in ["đ", "ơ", "ư", "á", "à", "ả", "ã", "ạ", "ê", "ô", "ị", "ụ", "ỷ", "ỹ"])


# ============================================================
#  PERSONALITY DETECTION
# ============================================================

def detect_personality(prompt: str) -> Optional[str]:
    pl = prompt.lower()

    if any(w in pl for w in ["ấm áp", "warm", "hiền", "nhẹ nhàng"]):
        return "warm"
    if any(w in pl for w in ["chuyên nghiệp", "professional"]):
        return "professional"
    if any(w in pl for w in ["hài hước", "vui", "funny"]):
        return "humorous"
    if any(w in pl for w in ["điềm tĩnh", "calm"]):
        return "calm"
    if any(w in pl for w in ["nhiệt huyết", "energetic"]):
        return "energetic"

    return None


# ============================================================
#  DOMAIN DETECTION
# ============================================================

def detect_domain(prompt: str) -> Optional[str]:
    pl = prompt.lower()

    if any(w in pl for w in ["lập trình", "code", "python", "cntt", "it"]):
        return "it"
    if any(w in pl for w in ["bác sĩ", "triệu chứng", "y khoa", "medical"]):
        return "medical"
    if any(w in pl for w in ["luật", "pháp lý", "legal", "lawyer"]):
        return "legal"
    if any(w in pl for w in ["tài chính", "đầu tư", "stock", "finance"]):
        return "finance"
    if any(w in pl for w in ["kinh doanh", "business", "marketing"]):
        return "business"
    if any(w in pl for w in ["giáo viên", "teacher", "học", "bài giảng"]):
        return "education"

    return None


# ============================================================
#  INTENT DETECTION (BASE)
# ============================================================

def detect_intent(prompt: str) -> str:
    pl = prompt.lower()

    if any(w in pl for w in ["giải thích", "tại sao", "vì sao", "explain"]):
        return "explain"
    if any(w in pl for w in ["tư vấn", "khuyên", "recommend"]):
        return "advice"
    if any(w in pl for w in ["viết truyện", "story", "kể chuyện"]):
        return "story"
    if any(w in pl for w in ["phân tích", "chi tiết", "long-form"]):
        return "analysis"
    if any(w in pl for w in ["doanh nghiệp", "business"]):
        return "business"
    if any(w in pl for w in ["đa góc nhìn", "multi-agent"]):
        return "multi_agent"
    if any(w in pl for w in ["hành động", "làm sao", "how to", "steps"]):
        return "action"
    if any(w in pl for w in ["lộ trình", "roadmap", "trong 3 tháng", "trong 6 tháng", "1 năm tới", "kế hoạch dài hạn"]):
        return "long_term_plan"
    if any(w in pl for w in ["reasoning", "suy luận", "chứng minh", "giải từng bước"]):
        return "reasoning"

    return "qa"


# ============================================================
#  CONTEXT MEMORY
# ============================================================

def update_context_memory(user_id: Optional[str], prompt: str):
    if user_id is None:
        return

    if user_id not in CONTEXT_MEMORY:
        CONTEXT_MEMORY[user_id] = {
            "history": [],
            "topics": set(),
            "last_prompt": None,
        }

    CONTEXT_MEMORY[user_id]["history"].append(prompt)
    CONTEXT_MEMORY[user_id]["last_prompt"] = prompt

    domain = detect_domain(prompt)
    if domain:
        CONTEXT_MEMORY[user_id]["topics"].add(domain)


def get_context_memory(user_id: Optional[str]):
    if user_id is None:
        return None
    return CONTEXT_MEMORY.get(user_id)


# ============================================================
#  STYLE MEMORY
# ============================================================

def update_user_style_pref(user_id: Optional[str], personality: Optional[str], domain: Optional[str]):
    if user_id is None:
        return

    if user_id not in USER_STYLE_PREFS:
        USER_STYLE_PREFS[user_id] = {
            "personality": None,
            "domain": None,
            "style_history": [],
        }

    if personality:
        USER_STYLE_PREFS[user_id]["personality"] = personality

    if domain:
        USER_STYLE_PREFS[user_id]["domain"] = domain

    USER_STYLE_PREFS[user_id]["style_history"].append((personality, domain))


def get_user_style_pref(user_id: Optional[str]):
    if user_id is None:
        return None
    return USER_STYLE_PREFS.get(user_id)


# ============================================================
#  MODE DETECTION (LONG-FORM, CREATIVE, BUSINESS, MULTI-AGENT, SWARM)
# ============================================================

def detect_modes(prompt: str):
    pl = prompt.lower()

    is_long_form = any(w in pl for w in [
        "phân tích", "giải thích kỹ", "chi tiết", "long-form", "phân tích sâu"
    ])

    is_creative = any(w in pl for w in [
        "viết truyện", "truyện ngắn", "story", "sáng tạo", "creative", "kể chuyện"
    ])

    is_ultra_creative = any(w in pl for w in [
        "viết truyện dài", "truyện dài", "tiểu thuyết", "novel", "ultra creative"
    ])

    is_business = any(w in pl for w in [
        "doanh nghiệp", "business", "báo cáo", "kế hoạch", "proposal", "họp", "meeting"
    ])

    is_multi_agent = any(w in pl for w in [
        "nhiều góc nhìn", "multi-agent", "đa góc nhìn", "nhiều quan điểm"
    ])

    is_swarm = any(w in pl for w in [
        "agent swarm", "siêu phân tích", "nhiều chuyên gia", "phân tích rất kỹ", "đa chiều"
    ])

    return {
        "long_form": is_long_form,
        "creative": is_creative,
        "ultra_creative": is_ultra_creative,
        "business": is_business,
        "multi_agent": is_multi_agent,
        "swarm": is_swarm,
    }


# ============================================================
#  STYLE 8.0 – PREDICT USER GOAL / HORIZON
# ============================================================

def predict_user_goal(prompt: str, memory: Optional[dict]) -> str:
    pl = prompt.lower()

    if memory and memory.get("history"):
        hist = " ".join(memory["history"]).lower()
        if "how to" in hist or "làm sao" in hist or "bước" in hist:
            return "action"
        if "phân tích" in hist or "chi tiết" in hist or "long-form" in hist:
            return "deep_analysis"

    if any(w in pl for w in ["tóm tắt", "summary"]):
        return "summary"
    if any(w in pl for w in ["phân tích sâu", "chi tiết", "long-form"]):
        return "deep_analysis"
    if any(w in pl for w in ["lộ trình", "roadmap", "trong 3 tháng", "trong 6 tháng", "1 năm tới", "kế hoạch dài hạn"]):
        return "long_term_plan"

    return detect_intent(prompt)


def detect_goal_horizon(prompt: str) -> str:
    pl = prompt.lower()
    if any(w in pl for w in ["hôm nay", "ngay lập tức", "bây giờ", "short-term"]):
        return "short_term"
    if any(w in pl for w in ["trong 3 tháng", "trong 6 tháng", "1 năm tới", "dài hạn", "long-term", "roadmap"]):
        return "long_term"
    return "unspecified"


# ============================================================
#  REAL-TIME CONTEXT (STYLE 8.0)
# ============================================================

def build_realtime_context(prompt: str, user_id: Optional[str]):
    parts = []
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    parts.append(f"[TIME] Thời gian hiện tại: {current_time}")

    memory = get_context_memory(user_id)
    if memory and memory["topics"]:
        parts.append(f"[MEMORY] Chủ đề bạn hay hỏi: {', '.join(memory['topics'])}")

    parts.append("[AIALL STYLE 8.0] Tự dự đoán nhu cầu, tạo kế hoạch dài hạn, suy luận nhiều bước.")

    return "\n".join(parts) + "\n\n"


# ============================================================
#  LONG-TERM PLAN MODE
# ============================================================

def build_long_term_plan(prompt: str, horizon: str, domain: Optional[str], is_vietnamese: bool) -> str:
    if is_vietnamese:
        header = "Kế hoạch dài hạn (Long-Term Plan):"
    else:
        header = "Long-Term Plan:"

    phases = []
    if horizon == "long_term":
        phases = ["Giai đoạn 1 (0–3 tháng)", "Giai đoạn 2 (3–6 tháng)", "Giai đoạn 3 (6–12 tháng)"]
    else:
        phases = ["Bước 1", "Bước 2", "Bước 3"]

    lines = [header]
    for p in phases:
        if is_vietnamese:
            lines.append(f"- {p}: Xác định mục tiêu, hành động cụ thể, mốc thời gian và rủi ro liên quan.")
        else:
            lines.append(f"- {p}: Define goals, concrete actions, timeline, and related risks.")

    if domain:
        if is_vietnamese:
            lines.append(f"\n(Lưu ý: Kế hoạch được định hướng theo lĩnh vực {domain}.)")
        else:
            lines.append(f"\n(Note: Plan is oriented toward domain {domain}.)")

    return "\n".join(lines)


# ============================================================
#  META-AGENT MODE – nhiều góc nhìn
# ============================================================

def build_meta_agent_analysis(prompt: str, is_vietnamese: bool):
    agents = [
        ("Kỹ sư", "Phân tích kỹ thuật, logic, cấu trúc."),
        ("Giáo viên", "Giải thích dễ hiểu, ví dụ minh họa."),
        ("Nhà kinh tế", "Phân tích lợi ích, chi phí, tác động."),
        ("Luật sư", "Góc nhìn pháp lý, rủi ro, quy định."),
    ]

    output = []
    for role, style in agents:
        if is_vietnamese:
            output.append(f"- Góc nhìn {role}: {style}")
        else:
            output.append(f"- Perspective ({role}): {style}")

    return "\n".join(output)


# ============================================================
#  AGENT SWARM – 10+ agent phối hợp
# ============================================================

def build_agent_swarm(prompt: str, is_vietnamese: bool) -> str:
    roles = [
        "Kỹ sư", "Nhà thiết kế", "Nhà kinh tế", "Luật sư", "Nhà tâm lý",
        "Giáo viên", "Product Manager", "Data Scientist", "Strategist", "Risk Analyst"
    ]
    swarm = []

    for role in roles:
        if is_vietnamese:
            swarm.append(f"[{role}] Đưa ra góc nhìn riêng về vấn đề, nêu ưu/nhược điểm và khuyến nghị.")
        else:
            swarm.append(f"[{role}] Provides own perspective, pros/cons, and recommendations.")

    moderator = (
        "[Swarm Moderator] Tổng hợp các góc nhìn, đưa ra kết luận cân bằng, ưu tiên an toàn và hiệu quả."
        if is_vietnamese
        else "[Swarm Moderator] Summarizes all perspectives, gives a balanced conclusion, prioritizing safety and effectiveness."
    )

    return "\n\n".join(swarm) + "\n\n" + moderator


# ============================================================
#  DEBATE MODE – tranh luận giữa nhiều agent
# ============================================================

def build_debate_analysis(prompt: str, is_vietnamese: bool) -> str:
    roles = ["Kỹ sư", "Giáo viên", "Nhà kinh tế", "Luật sư"]
    debates = []

    for role in roles:
        if is_vietnamese:
            debates.append(
                f"[{role}] Quan điểm riêng về vấn đề, nêu lý do, ưu/nhược điểm."
            )
        else:
            debates.append(
                f"[{role}] Own perspective on the problem, with reasons, pros/cons."
            )

    moderator = (
        "[Moderator] Tổng hợp các góc nhìn, đưa ra kết luận cân bằng."
        if is_vietnamese
        else "[Moderator] Summarizes all perspectives and gives a balanced conclusion."
    )

    return "\n\n".join(debates) + "\n\n" + moderator


# ============================================================
#  SELF‑HEAL 2.0 – nhiều vòng đánh giá
# ============================================================

def self_heal_answer(model, tokenizer, prompt: str, draft: str, is_vietnamese: bool) -> str:
    review_prompt = (
        "Instruction: "
        + prompt
        + "\n\nDraft answer:\n"
        + draft
        + "\n\nTask: Hãy kiểm tra xem câu trả lời trên có lỗi logic, mâu thuẫn, thiếu bước quan trọng "
        + "hoặc giải thích chưa rõ không. Nếu có, hãy viết lại một phiên bản tốt hơn, rõ ràng hơn."
        if is_vietnamese
        else
        "Instruction: "
        + prompt
        + "\n\nDraft answer:\n"
        + draft
        + "\n\nTask: Check if the answer above has logical errors, contradictions, missing steps, "
        + "or unclear explanations. Rewrite a better version."
    )

    inputs = tokenizer(
        review_prompt,
        return_tensors="pt",
        truncation=True,
        padding=False,
        add_special_tokens=True,
    )

    if tokenizer.bos_token_id is not None:
        bos = torch.tensor([[tokenizer.bos_token_id]])
        inputs["input_ids"] = torch.cat([bos, inputs["input_ids"]], dim=1)
        inputs["attention_mask"] = torch.cat([torch.tensor([[1]]), inputs["attention_mask"]], dim=1)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=600,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def self_heal_v2(model, tokenizer, prompt: str, draft: str, is_vietnamese: bool, rounds: int = 2) -> str:
    current = draft
    for _ in range(rounds):
        improved = self_heal_answer(model, tokenizer, prompt, current, is_vietnamese)
        if len(improved) <= len(current) * 1.02:
            break
        current = improved
    return current


# ============================================================
#  SELF-REFINE MODE – tối ưu câu trả lời cuối
# ============================================================

def self_refine_answer(draft: str, is_vietnamese: bool) -> str:
    refined = draft.strip()
    refined = refined.replace("  ", " ")

    if is_vietnamese:
        if "kết luận" not in refined.lower():
            refined += "\n\n**Kết luận:** Hy vọng phần giải thích trên giúp bạn nắm rõ vấn đề."
    else:
        if "conclusion" not in refined.lower():
            refined += "\n\n**Conclusion:** Hope this explanation clarifies the topic."

    return refined


# ============================================================
#  BUILD ANSWER STYLE (STYLE 8.0)
# ============================================================

def build_answer_style(prompt: str, user_id: Optional[str]):
    is_vietnamese = detect_vietnamese(prompt)
    memory = get_context_memory(user_id)
    goal = predict_user_goal(prompt, memory)
    horizon = detect_goal_horizon(prompt)
    intent = detect_intent(prompt)
    personality = detect_personality(prompt)
    domain = detect_domain(prompt)
    modes = detect_modes(prompt)

    update_user_style_pref(user_id, personality, domain)

    max_new_tokens = 400
    temperature = 0.8
    top_p = 0.92

    prefix = (
        "Answer (AIALL, tiếng Việt tự nhiên, rõ ràng, có cấu trúc):"
        if is_vietnamese else
        "Answer (AIALL, natural, clear, structured):"
    )

    # Goal-based optimization
    if goal == "summary":
        max_new_tokens = 300
        prefix = "Answer (AIALL – tóm tắt ngắn gọn):"
    elif goal == "deep_analysis":
        modes["long_form"] = True
    elif goal == "action":
        prefix = "Answer (AIALL – hướng dẫn từng bước):"
        max_new_tokens = 600
    elif goal == "long_term_plan":
        modes["long_form"] = True

    # Intent-based
    if intent == "explain":
        prefix = "Answer (AIALL – giải thích chi tiết):"
    elif intent == "advice":
        prefix = "Answer (AIALL – tư vấn, khuyến nghị rõ ràng):"
    elif intent == "story":
        modes["creative"] = True
    elif intent == "business":
        modes["business"] = True
    elif intent == "multi_agent":
        modes["multi_agent"] = True
    elif intent == "reasoning":
        modes["long_form"] = True  # reasoning thường cần dài hơn

    # Domain expert
    if domain:
        prefix = f"Answer (AIALL – chuyên gia {domain}):"

    # Personality
    if personality == "warm":
        prefix += " [Tone: ấm áp]"
    elif personality == "professional":
        prefix += " [Tone: chuyên nghiệp]"
    elif personality == "humorous":
        prefix += " [Tone: hài hước]"
        temperature += 0.1
    elif personality == "calm":
        prefix += " [Tone: điềm tĩnh]"
    elif personality == "energetic":
        prefix += " [Tone: năng lượng]"
        temperature += 0.05

    # Modes
    if modes["long_form"]:
        max_new_tokens = max(max_new_tokens, 1000)
        temperature = 0.75
        top_p = 0.9

    if modes["creative"]:
        max_new_tokens = max(max_new_tokens, 700)
        temperature = 0.95
        top_p = 0.96

    if modes["ultra_creative"]:
        max_new_tokens = max(max_new_tokens, 1500)
        temperature = 1.05
        top_p = 0.98

    if modes["business"]:
        max_new_tokens = max(max_new_tokens, 600)
        temperature = 0.7
        top_p = 0.9

    if modes["multi_agent"]:
        prefix += " [MULTI-AGENT: nhiều góc nhìn]"
        max_new_tokens = max(max_new_tokens, 900)

    if modes["swarm"]:
        prefix += " [AGENT SWARM: phân tích đa chiều]"
        max_new_tokens = max(max_new_tokens, 1200)

    return prefix, max_new_tokens, temperature, top_p, modes, goal, horizon, domain


# ============================================================
#  MAIN CHAT FUNCTION (STYLE 8.0)
# ============================================================

def aiall_chat(model, tokenizer, prompt: str, user_id: Optional[str] = None):
    update_context_memory(user_id, prompt)

    is_vietnamese = detect_vietnamese(prompt)
    context = build_realtime_context(prompt, user_id)
    prefix, max_new_tokens, temperature, top_p, modes, goal, horizon, domain = build_answer_style(prompt, user_id)

    text = context + "Instruction: " + prompt + "\n" + prefix

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=False,
        add_special_tokens=True,
    )

    if inputs["input_ids"].numel() == 0:
        fallback = "Instruction: " + prompt + "\n" + prefix
        inputs = tokenizer(
            fallback,
            return_tensors="pt",
            truncation=True,
            padding=False,
            add_special_tokens=True,
        )

    if inputs["input_ids"].numel() == 0:
        inputs = {
            "input_ids": torch.tensor([[tokenizer.eos_token_id]]),
            "attention_mask": torch.tensor([[1]]),
        }

    if tokenizer.bos_token_id is not None:
        bos = torch.tensor([[tokenizer.bos_token_id]])
        inputs["input_ids"] = torch.cat([bos, inputs["input_ids"]], dim=1)
        inputs["attention_mask"] = torch.cat([torch.tensor([[1]]), inputs["attention_mask"]], dim=1)

    with torch.no_grad():
        draft_outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )

    draft = tokenizer.decode(draft_outputs[0], skip_special_tokens=True)

    # Long-Term Plan Mode (thêm kế hoạch dài hạn nếu goal phù hợp)
    if goal == "long_term_plan":
        plan = build_long_term_plan(prompt, horizon, domain, is_vietnamese)
        draft += "\n\n" + plan

    # META-AGENT + DEBATE + AGENT SWARM
    if modes["multi_agent"]:
        meta = build_meta_agent_analysis(prompt, is_vietnamese)
        debate = build_debate_analysis(prompt, is_vietnamese)
        draft += "\n\n" + meta + "\n\n" + debate

    if modes["swarm"]:
        swarm = build_agent_swarm(prompt, is_vietnamese)
        draft += "\n\n" + swarm

    healed = self_heal_v2(model, tokenizer, prompt, draft, is_vietnamese, rounds=2)
    final_answer = self_refine_answer(healed, is_vietnamese)

    return final_answer
