# app_v3.py — LogiCore Industry Edition
# Integrates: Groq AI + Live Weather + Anomaly Detection + WebSockets
#
# Setup:
#   pip install flask flask-socketio groq scikit-learn pandas numpy requests python-dotenv
#   Add GROQ_API_KEY and OPENWEATHER_API_KEY to .env file
#   python data/prepare_data.py   (first time only)
#   python app_v3.py

import threading, time, random, heapq, pickle, os
from datetime import datetime
from collections import defaultdict
import queue
import pandas as pd
import numpy as np
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
from dotenv import load_dotenv

load_dotenv()

app    = Flask(__name__)
app.config['SECRET_KEY'] = 'logicore-secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ── Load ML model ───────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, 'data', 'supply_chain_data.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'data', 'model.pkl')

try:
    with open(MODEL_PATH, 'rb') as f:
        bundle = pickle.load(f)
    ML_MODEL, ML_FEATURES = bundle['model'], bundle['features']
    RAW_DF   = pd.read_csv(DATA_PATH)
    ML_READY = True
    print(f"[Startup] ML model ready — dataset: {len(RAW_DF):,} rows")
except Exception as e:
    ML_READY = False
    print(f"[Startup] ML not loaded ({e}). Run: python data/prepare_data.py")

# ── Try to load AI + live data modules ──────────────────────────────────────
try:
    from agents.coordinator_ai import ai_coordinator_decision, ai_chat_response
    AI_READY = bool(os.environ.get("GROQ_API_KEY"))
    print(f"[Startup] Groq AI: {'✓ Ready' if AI_READY else '✗ No API key in .env'}")
except ImportError:
    AI_READY = False
    print("[Startup] coordinator_ai.py not found — using rule-based coordinator")

try:
    from live_data import get_live_weather, is_bad_weather, anomaly_detector
    WEATHER_KEY   = bool(os.environ.get("OPENWEATHER_API_KEY"))
    LIVE_DATA_OK  = True
    print(f"[Startup] Weather: {'✓ Live API' if WEATHER_KEY else '✓ Simulated'}")
except ImportError:
    LIVE_DATA_OK  = False
    print("[Startup] live_data.py not found — weather disabled")

# ── City network ─────────────────────────────────────────────────────────────
CITY_GRAPH = {
    "Mumbai_WH":    [("Pune", 148), ("Nashik", 167), ("Surat", 284), ("Thane", 35)],
    "Delhi_WH":     [("Gurgaon", 30), ("Noida", 45), ("Agra", 233), ("Jaipur", 282), ("Faridabad", 28)],
    "Bangalore_WH": [("Chennai", 346), ("Mysore", 143), ("Hyderabad", 570), ("Mangalore", 352)],
    "Pune":         [("Mumbai_WH", 148), ("Satara", 120), ("Kolhapur", 228)],
    "Nashik":       [("Mumbai_WH", 167), ("Aurangabad", 187)],
    "Surat":        [("Mumbai_WH", 284), ("Ahmedabad", 265), ("Vadodara", 154)],
    "Thane":        [("Mumbai_WH", 35)],
    "Gurgaon":      [("Delhi_WH", 30), ("Faridabad", 28)],
    "Noida":        [("Delhi_WH", 45), ("Ghaziabad", 18)],
    "Agra":         [("Delhi_WH", 233)],
    "Jaipur":       [("Delhi_WH", 282), ("Ajmer", 135)],
    "Faridabad":    [("Delhi_WH", 28), ("Gurgaon", 28)],
    "Chennai":      [("Bangalore_WH", 346), ("Vellore", 138)],
    "Mysore":       [("Bangalore_WH", 143)],
    "Hyderabad":    [("Bangalore_WH", 570)],
    "Mangalore":    [("Bangalore_WH", 352)],
    "Ahmedabad":    [("Surat", 265)],
    "Satara":       [("Pune", 120)],
    "Vadodara":     [("Surat", 154)],
    "Ghaziabad":    [("Noida", 18)],
    "Aurangabad":   [("Nashik", 187)],
    "Vellore":      [("Chennai", 138)],
    "Ajmer":        [("Jaipur", 135)],
    "Kolhapur":     [("Pune", 228)],
}
WAREHOUSES   = ["Mumbai_WH", "Delhi_WH", "Bangalore_WH"]
DESTINATIONS = [k for k in CITY_GRAPH if "_WH" not in k]
CATEGORIES   = ["Electronics", "Groceries", "Clothing", "Furniture", "Medicines"]

# ── Message Bus ───────────────────────────────────────────────────────────────
class MessageBus:
    def __init__(self):
        self.channels = defaultdict(queue.Queue)
        self.history  = []
    def publish(self, ch, msg):
        self.channels[ch].put(msg)
        self.history.append({"channel": ch, "data": msg})
    def consume(self, ch):
        try:    return self.channels[ch].get_nowait()
        except: return None

# ── Demand Agent ──────────────────────────────────────────────────────────────
class DemandAgent:
    def __init__(self, bus):
        self.bus = bus
        self.history = []

    def run(self):
        now      = datetime.now()
        city     = random.choice(["Mumbai","Delhi","Bangalore","Chennai","Pune","Hyderabad"])
        category = random.choice(CATEGORIES)
        wh       = f"{city}_WH" if f"{city}_WH" in WAREHOUSES else random.choice(WAREHOUSES)

        if ML_READY:
            row = {"day_of_week": now.weekday(), "month": now.month,
                   "lead_time_days": random.randint(1,4)}
            for f in ML_FEATURES:
                if f.startswith("city_"):     row[f] = 1 if f == f"city_{city}"     else 0
                if f.startswith("category_"): row[f] = 1 if f == f"category_{category}" else 0
                if f.startswith("warehouse_"):row[f] = 1 if f == f"warehouse_Warehouse_{wh.split('_')[0]}" else 0
            X      = pd.DataFrame([row])[ML_FEATURES]
            demand = max(5, int(ML_MODEL.predict(X)[0]) + random.randint(-10, 10))
            method = "RandomForest 91.6%"
        else:
            base   = {"Electronics":45,"Groceries":200,"Clothing":80,
                      "Furniture":20,"Medicines":150}[category]
            demand = max(5, base + random.randint(-20,40))
            method = "rule-based"

        record = {"time": now.strftime("%H:%M:%S"), "city": city,
                  "category": category, "warehouse": wh,
                  "demand": demand, "method": method}
        self.history.append(record)
        if len(self.history) > 60: self.history.pop(0)
        self.bus.publish("demand_forecast", record)
        return record

# ── Inventory Agent ───────────────────────────────────────────────────────────
class InventoryAgent:
    INITIAL   = {"Electronics":800,"Groceries":3000,"Clothing":1500,"Furniture":300,"Medicines":2500}
    REORDER   = {"Electronics":150,"Groceries":500, "Clothing":300, "Furniture":60, "Medicines":400}
    RESTOCK_Q = {"Electronics":300,"Groceries":1000,"Clothing":600, "Furniture":150,"Medicines":800}

    def __init__(self, bus):
        self.bus   = bus
        self.stock = dict(self.INITIAL)
        self.history = []

    def run(self, demand_msg):
        cat    = demand_msg["category"]
        demand = demand_msg["demand"]
        self.stock[cat] = max(0, self.stock[cat] - demand)
        low      = self.stock[cat] < self.REORDER[cat]
        critical = self.stock[cat] < self.REORDER[cat] * 0.5
        record   = {"time": datetime.now().strftime("%H:%M:%S"), "category": cat,
                    "stock": self.stock[cat], "consumed": demand,
                    "threshold": self.REORDER[cat],
                    "status": "CRITICAL" if critical else "LOW" if low else "OK"}
        self.history.append(record)
        if len(self.history) > 60: self.history.pop(0)
        if low:
            self.bus.publish("inventory_alert", {**record, "reorder": True})
        return record

    def restock(self, category, units):
        self.stock[category] = self.stock.get(category, 0) + units

    def total_stock(self):
        return sum(self.stock.values())

# ── Routing Agent ─────────────────────────────────────────────────────────────
class RoutingAgent:
    def __init__(self, bus):
        self.bus     = bus
        self.blocked = set()
        self.history = []

    def dijkstra(self, start, end):
        pq = [(0, start, [start])]
        vis = set()
        while pq:
            cost, node, path = heapq.heappop(pq)
            if node == end: return path, cost
            if node in vis: continue
            vis.add(node)
            for nb, w in CITY_GRAPH.get(node, []):
                edge = tuple(sorted([node, nb]))
                if edge not in self.blocked and nb not in vis:
                    heapq.heappush(pq, (cost+w, nb, path+[nb]))
        return None, float("inf")

    def run(self, dispatch_msg):
        src  = dispatch_msg.get("warehouse", "Mumbai_WH")
        dest = dispatch_msg.get("destination", "Pune")
        path, cost = self.dijkstra(src, dest)
        if path:
            record = {"time": datetime.now().strftime("%H:%M:%S"), "from": src,
                      "to": dest, "route": " → ".join(path),
                      "distance_km": cost, "eta_hours": round(cost/60, 1),
                      "status": "routed"}
        else:
            record = {"time": datetime.now().strftime("%H:%M:%S"), "from": src,
                      "to": dest, "route": "NO ROUTE FOUND", "status": "blocked"}
        self.history.append(record)
        if len(self.history) > 20: self.history.pop(0)
        self.bus.publish("route_result", record)
        return record

# ── Shared state + agents ─────────────────────────────────────────────────────
state = {
    "running": False, "cycle": 0, "last_updated": None,
    "total_stock": 9100, "stock_by_cat": {},
    "last_demand": {}, "last_route": None, "last_disruption": None,
    "weather": {}, "last_ai_decision": None,
    "stock_history": [], "demand_history": [], "event_log": [],
    "decisions": [],
    "kpis": {"total_orders":0,"disruptions":0,"auto_reorders":0,
             "routes_computed":0,"anomalies_detected":0},
}

def make_agents():
    global bus, demand_agent, inventory_agent, routing_agent
    bus             = MessageBus()
    demand_agent    = DemandAgent(bus)
    inventory_agent = InventoryAgent(bus)
    routing_agent   = RoutingAgent(bus)

make_agents()

def add_event(etype, msg):
    e = {"time": datetime.now().strftime("%H:%M:%S"), "type": etype, "msg": msg}
    state["event_log"].insert(0, e)
    if len(state["event_log"]) > 40: state["event_log"].pop()
    # Push to browser instantly via WebSocket
    socketio.emit("new_event", e)

# ── Main simulation cycle ──────────────────────────────────────────────────────
def simulation_cycle():
    state["cycle"] += 1
    cycle = state["cycle"]

    # 1. Fetch live weather (real API if key set, simulated otherwise)
    if LIVE_DATA_OK:
        try:
            state["weather"] = get_live_weather()
            bad_cities = is_bad_weather(state["weather"])
            if bad_cities:
                for city in bad_cities:
                    add_event("disruption",
                        f"🌧 Bad weather in {city}: {state['weather'][city]} — delivery delays possible")
        except Exception:
            pass

    # 2. ML demand prediction
    demand_record = demand_agent.run()
    state["last_demand"] = demand_record
    state["kpis"]["total_orders"] += 1

    # 3. Anomaly detection
    anomaly = None
    if LIVE_DATA_OK:
        try:
            anomaly_detector.record(demand_record["category"], demand_record["demand"])
            anomaly = anomaly_detector.check(demand_record["category"], demand_record["demand"])
            if anomaly:
                state["kpis"]["anomalies_detected"] += 1
                add_event("alert",
                    f"🔍 Anomaly: {anomaly['category']} demand={anomaly['demand']} "
                    f"(z={anomaly['z_score']}, severity={anomaly['severity']})")
        except Exception:
            pass

    state["demand_history"].append({
        "time": demand_record["time"], "demand": demand_record["demand"],
        "city": demand_record["city"], "category": demand_record["category"],
    })
    if len(state["demand_history"]) > 40: state["demand_history"].pop(0)

    # 4. Update inventory
    inv_record = inventory_agent.run(demand_record)
    state["stock_by_cat"] = dict(inventory_agent.stock)
    state["total_stock"]  = inventory_agent.total_stock()

    if inv_record["status"] != "OK":
        add_event("alert",
            f"{'🔴 CRITICAL' if inv_record['status']=='CRITICAL' else '🟡 LOW'} "
            f"stock: {inv_record['category']} = {inv_record['stock']} units "
            f"(min {inv_record['threshold']})")

    # 5. Random disruption (real-world probability based)
    disruption = None
    events = [
        ("road_blocked","🚧 Road blocked", 0.10),
        ("vehicle_breakdown","🚛 Vehicle breakdown", 0.07),
        ("demand_surge","📈 Demand surge", 0.09),
        ("weather_delay","🌧 Weather delay", 0.06),
    ]
    for etype, elabel, prob in events:
        if random.random() < prob:
            city = random.choice(["Mumbai","Delhi","Bangalore","Pune","Chennai"])
            cat  = demand_record["category"]
            disruption = {"type":etype,"label":elabel,"city":city,
                          "category":cat,"time":datetime.now().strftime("%H:%M:%S"),
                          "warehouse":random.choice(WAREHOUSES),
                          "alt_dest":random.choice(DESTINATIONS)}
            state["last_disruption"] = disruption
            state["kpis"]["disruptions"] += 1
            bus.publish("disruption", disruption)
            add_event("disruption", f"{elabel} in {city} affecting {cat}")
            break

    # 6. AI Coordinator decision (Groq or rule-based fallback)
    alert      = bus.consume("inventory_alert")
    disruption_msg = bus.consume("disruption")

    situation = {
        "stock":       dict(inventory_agent.stock),
        "thresholds":  InventoryAgent.REORDER,
        "last_demand": demand_record,
        "disruption":  disruption_msg or disruption,
        "weather":     state.get("weather", {}),
        "anomaly":     anomaly,
        "cycle":       cycle,
    }

    if AI_READY:
        try:
            decision_text = ai_coordinator_decision(situation)
        except Exception as e:
            decision_text = _rule_based_decision(alert, disruption_msg or disruption,
                                                  inv_record, demand_record)
    else:
        decision_text = _rule_based_decision(alert, disruption_msg or disruption,
                                              inv_record, demand_record)

    if decision_text:
        state["last_ai_decision"] = decision_text
        d = {"time": datetime.now().strftime("%H:%M:%S"), "action": decision_text,
             "type": "ai" if AI_READY else "rule"}
        state["decisions"].insert(0, d)
        if len(state["decisions"]) > 30: state["decisions"].pop()
        add_event("decision", f"{'🤖 AI' if AI_READY else '✅'}: {decision_text}")

        # Execute the physical action
        if alert and alert.get("reorder"):
            cat = alert["category"]
            inventory_agent.restock(cat, InventoryAgent.RESTOCK_Q.get(cat, 500))
            state["kpis"]["auto_reorders"] += 1
            wh   = random.choice(WAREHOUSES)
            dest = random.choice(DESTINATIONS)
            bus.publish("dispatch", {"warehouse":wh,"destination":dest,
                                     "category":cat,"reason":"reorder"})

        if disruption_msg and disruption_msg.get("type") in ["road_blocked","vehicle_breakdown"]:
            wh   = disruption_msg.get("warehouse", random.choice(WAREHOUSES))
            dest = disruption_msg.get("alt_dest",  random.choice(DESTINATIONS))
            bus.publish("dispatch", {"warehouse":wh,"destination":dest,"reason":"reroute"})

    # 7. Route any dispatches
    dispatch = bus.consume("dispatch")
    if dispatch:
        route = routing_agent.run(dispatch)
        state["last_route"] = route
        state["kpis"]["routes_computed"] += 1
        add_event("route",
            f"🚚 {route['route']} "
            f"({route.get('distance_km','?')} km · ETA {route.get('eta_hours','?')}h)")

    # 8. Snapshot for charts
    snap = {"time": datetime.now().strftime("%H:%M:%S"), "total": state["total_stock"]}
    snap.update(inventory_agent.stock)
    state["stock_history"].append(snap)
    if len(state["stock_history"]) > 40: state["stock_history"].pop(0)

    state["last_updated"] = datetime.now().strftime("%H:%M:%S")

    # Push full state update to all browser clients
    socketio.emit("state_update", _build_state_payload())


def _rule_based_decision(alert, disruption, inv_record, demand_record):
    if alert and alert.get("reorder"):
        return (f"Stock critical for {alert['category']} ({alert['stock_level']} units). "
                f"Triggering reorder of {InventoryAgent.RESTOCK_Q.get(alert['category'],500)} units.")
    if disruption:
        return f"Disruption: {disruption.get('label')} in {disruption.get('city')}. Rerouting affected deliveries."
    if inv_record["status"] == "LOW":
        return f"{inv_record['category']} stock low ({inv_record['stock']} units). Pre-emptive reorder recommended."
    return None


def _build_state_payload():
    return {
        "running":         state["running"],
        "cycle":           state["cycle"],
        "last_updated":    state["last_updated"],
        "total_stock":     state["total_stock"],
        "stock_by_cat":    state["stock_by_cat"],
        "last_demand":     state["last_demand"],
        "last_route":      state["last_route"],
        "last_disruption": state["last_disruption"],
        "last_ai_decision":state["last_ai_decision"],
        "weather":         state.get("weather", {}),
        "stock_history":   state["stock_history"][-30:],
        "demand_history":  state["demand_history"][-30:],
        "event_log":       state["event_log"][:15],
        "decisions":       state["decisions"][:10],
        "kpis":            state["kpis"],
        "ai_ready":        AI_READY,
        "ml_ready":        ML_READY,
    }


def simulation_loop():
    add_event("system", "🟢 LogiCore started — all agents running autonomously")
    while state["running"]:
        try:
            simulation_cycle()
        except Exception as e:
            add_event("error", f"Cycle error: {e}")
        time.sleep(15)   # real-time: new cycle every 15 seconds


# ── Flask routes ──────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html",
                            ai_ready=AI_READY, ml_ready=ML_READY)

@app.route("/api/start", methods=["POST"])
def start():
    if not state["running"]:
        state["running"] = True
        threading.Thread(target=simulation_loop, daemon=True).start()
    return jsonify({"status": "started"})

@app.route("/api/stop", methods=["POST"])
def stop():
    state["running"] = False
    add_event("system", "🔴 System stopped")
    return jsonify({"status": "stopped"})

@app.route("/api/reset", methods=["POST"])
def reset():
    state["running"] = False
    time.sleep(0.2)
    state.update({"cycle":0,"last_updated":None,"total_stock":9100,
                  "stock_by_cat":{},"last_demand":{},"last_route":None,
                  "last_disruption":None,"weather":{},"last_ai_decision":None,
                  "stock_history":[],"demand_history":[],"event_log":[],"decisions":[],
                  "kpis":{"total_orders":0,"disruptions":0,"auto_reorders":0,
                          "routes_computed":0,"anomalies_detected":0}})
    make_agents()
    return jsonify({"status": "reset"})

@app.route("/api/state")
def get_state():
    return jsonify(_build_state_payload())

@app.route("/api/chat", methods=["POST"])
def chat():
    msg = request.get_json().get("message","").strip()
    if not msg: return jsonify({"reply":"Please type something."})
    if AI_READY:
        try:
            reply = ai_chat_response(msg, _build_state_payload())
            return jsonify({"reply": reply, "ai": True})
        except Exception as e:
            pass
    # Fallback rule-based
    return jsonify({"reply": _rule_chat(msg), "ai": False})

@app.route("/api/inject/<event_type>", methods=["POST"])
def inject(event_type):
    city = random.choice(["Mumbai","Delhi","Bangalore","Pune","Chennai"])
    cat  = random.choice(CATEGORIES)
    labels = {"road_blocked":"🚧 Road blocked","vehicle_breakdown":"🚛 Vehicle breakdown",
               "demand_surge":"📈 Demand surge","weather_delay":"🌧 Weather delay"}
    if event_type not in labels: return jsonify({"error":"unknown"}), 400
    bus.publish("disruption", {"type":event_type,"label":labels[event_type],"city":city,
                                "category":cat,"time":datetime.now().strftime("%H:%M:%S"),
                                "warehouse":random.choice(WAREHOUSES),
                                "alt_dest":random.choice(DESTINATIONS)})
    add_event("disruption", f"{labels[event_type]} injected in {city} ({cat})")
    return jsonify({"status": f"injected in {city}"})


def _rule_chat(m):
    m = m.lower()
    if "stock" in m or "inventory" in m:
        lines = [f"  {c}: {q} units" for c,q in inventory_agent.stock.items()]
        return "Current stock:\n" + "\n".join(lines)
    if "demand" in m or "forecast" in m:
        d = state["last_demand"]
        return f"Last forecast: {d.get('city')} · {d.get('category')} · {d.get('demand')} units ({d.get('method')})" if d else "No forecast yet."
    if "route" in m or "deliver" in m:
        r = state.get("last_route")
        return f"Last route: {r['route']} ({r.get('distance_km')} km, ETA {r.get('eta_hours')}h)" if r else "No routes computed yet."
    if "weather" in m:
        w = state.get("weather",{})
        return "\n".join(f"  {c}: {v}" for c,v in w.items()) if w else "No weather data."
    if "status" in m or "summary" in m:
        k = state["kpis"]
        return (f"Cycle {state['cycle']} | Stock: {state['total_stock']:,} | "
                f"Orders: {k['total_orders']} | Reorders: {k['auto_reorders']} | "
                f"Disruptions: {k['disruptions']}")
    if "decision" in m or "ai" in m:
        d = state.get("last_ai_decision","No decisions yet.")
        return f"Last AI decision:\n{d}"
    return "Ask about: stock, demand, weather, routes, status, or last AI decision."


if __name__ == "__main__":
    print("\n" + "="*55)
    print("  LOGICORE — INDUSTRY EDITION")
    print(f"  Groq AI:   {'✓ Active' if AI_READY else '✗ Add GROQ_API_KEY to .env'}")
    print(f"  ML Model:  {'✓ 91.6% accuracy' if ML_READY else '✗ Run prepare_data.py'}")
    print(f"  Weather:   {'✓ Live API' if LIVE_DATA_OK else '✓ Simulated'}")
    print(f"  WebSocket: ✓ Real-time push updates")
    print("  Open: http://localhost:5000")
    print("="*55 + "\n")
    socketio.run(app, debug=False, port=5000)
