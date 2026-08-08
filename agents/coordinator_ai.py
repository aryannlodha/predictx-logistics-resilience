# agents/coordinator_ai.py
# Drop-in replacement for the rule-based coordinator
# Uses Groq (free Llama 3 70B) to reason about every situation in plain English

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are the AI coordinator for a real-time Indian logistics system called LogiCore.
You receive live data: inventory levels, ML demand forecasts, weather, and disruption events.
Your job: analyse the situation and give ONE specific, actionable decision.

Rules:
- 2-3 sentences maximum
- Always state: what the problem is, what action you're taking, why
- Use real numbers from the data given
- Think like a startup logistics manager, not a textbook
- Prioritise the most urgent issue if multiple exist
- Output ONLY the decision. No preamble like "Based on the data..."."""


def ai_coordinator_decision(situation: dict) -> str:
    """
    Called every simulation cycle with the full live situation.

    situation dict keys:
      stock        — {"Electronics": 120, "Groceries": 450, ...}
      thresholds   — {"Electronics": 150, ...}
      last_demand  — {"city": "Mumbai", "category": "Groceries", "units": 230}
      disruption   — {"type": "road_blocked", "city": "Pune", "label": "Road blocked"} or None
      weather      — {"Mumbai": "Heavy rain 14°C", "Delhi": "Clear 28°C"} or {}
      anomaly      — {"category": "Electronics", "z_score": 3.2} or None
      cycle        — int

    Returns: plain-English decision string
    """
    low_stock = [
        f"{cat} ({qty} units, min {situation['thresholds'].get(cat, '?')})"
        for cat, qty in situation["stock"].items()
        if qty < situation["thresholds"].get(cat, 9999)
    ]

    lines = [f"CYCLE {situation.get('cycle', '?')} — LIVE SITUATION:"]

    if low_stock:
        lines.append(f"LOW STOCK ALERT: {', '.join(low_stock)}")

    d = situation.get("last_demand")
    if d:
        lines.append(
            f"ML DEMAND SIGNAL: {d.get('city')} needs {d.get('units')} units "
            f"of {d.get('category')} (RandomForest prediction)"
        )

    dis = situation.get("disruption")
    if dis:
        lines.append(
            f"DISRUPTION: {dis.get('label')} in {dis.get('city')} "
            f"affecting {dis.get('category', 'operations')}"
        )

    wx = situation.get("weather", {})
    bad = [f"{c}: {w}" for c, w in wx.items()
           if any(x in w.lower() for x in ["rain", "storm", "flood", "fog", "hail"])]
    if bad:
        lines.append(f"WEATHER: {', '.join(bad)}")

    anom = situation.get("anomaly")
    if anom:
        lines.append(
            f"ANOMALY: {anom['category']} demand is statistically unusual "
            f"(z-score {anom['z_score']:.1f} — possible surge or data error)"
        )

    if len(lines) == 1:
        lines.append("All systems operating normally. No critical alerts.")

    lines.append("\nState your decision and rationale:")

    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": "\n".join(lines)},
            ],
            max_tokens=160,
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        # Graceful fallback — system keeps running without AI
        if low_stock:
            return f"[AI offline] Auto-reorder triggered: {low_stock[0]}"
        if dis:
            return f"[AI offline] Disruption in {dis.get('city')} — rerouting"
        return "[AI offline] All systems nominal — monitoring"


def ai_chat_response(user_message: str, live_state: dict) -> str:
    """
    Powers the chatbot sidebar.
    The AI sees the full live system state and answers like a logistics analyst.
    """
    context = f"""You are the AI assistant for LogiCore, a real-time Indian logistics system.

LIVE SYSTEM STATE RIGHT NOW:
- Total stock: {live_state.get('total_stock', '?'):,} units
- Stock by category: {live_state.get('stock_by_cat', {})}
- Last ML demand forecast: {live_state.get('last_demand', 'none')}
- Last disruption: {live_state.get('last_disruption', 'none')}
- Last delivery route: {live_state.get('last_route', 'none')}
- Current weather: {live_state.get('weather', {})}
- Recent AI decisions: {[d['action'] for d in live_state.get('decisions', [])[-3:]]}
- KPIs: {live_state.get('kpis', {})}

Answer the user's question using this live data. Be specific and use real numbers.
Keep answers under 4 sentences unless a full summary is asked for.
Sound like a smart logistics analyst, not a chatbot."""

    try:
        resp = client.chat.completions.create(
           model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": context},
                {"role": "user",   "content": user_message},
            ],
            max_tokens=220,
            temperature=0.5,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"AI temporarily offline ({str(e)[:50]}). Try again in a moment."
