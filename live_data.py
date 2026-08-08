# live_data.py
# Two real-time data sources — both completely free:
#   1. OpenWeatherMap API  — live weather for Indian cities
#   2. Isolation Forest    — anomaly detection on demand patterns

import os
import requests
import numpy as np
from datetime import datetime
from sklearn.ensemble import IsolationForest
from dotenv import load_dotenv

load_dotenv()
OWM_KEY = os.environ.get("OPENWEATHER_API_KEY", "")

CITIES = {
    "Mumbai":    (19.0760, 72.8777),
    "Delhi":     (28.6139, 77.2090),
    "Bangalore": (12.9716, 77.5946),
    "Chennai":   (13.0827, 80.2707),
    "Pune":      (18.5204, 73.8567),
    "Hyderabad": (17.3850, 78.4867),
}

# ── WEATHER ────────────────────────────────────────────────────────────────

_weather_cache = {}
_weather_last_fetch = {}
CACHE_SECONDS = 600   # re-fetch every 10 minutes (free tier: 60 calls/min)


def get_live_weather() -> dict:
    """
    Returns dict like: {"Mumbai": "Heavy rain 22°C", "Delhi": "Clear 31°C", ...}
    Falls back to simulated weather if no API key set.
    """
    if not OWM_KEY:
        return _simulated_weather()

    now = datetime.now().timestamp()
    result = {}
    for city, (lat, lon) in CITIES.items():
        # Use cache to avoid burning API calls
        if city in _weather_cache and now - _weather_last_fetch.get(city, 0) < CACHE_SECONDS:
            result[city] = _weather_cache[city]
            continue
        try:
            url = (f"https://api.openweathermap.org/data/2.5/weather"
                   f"?lat={lat}&lon={lon}&appid={OWM_KEY}&units=metric")
            r = requests.get(url, timeout=5)
            data = r.json()
            desc = data["weather"][0]["description"].capitalize()
            temp = round(data["main"]["temp"])
            text = f"{desc} {temp}°C"
            _weather_cache[city] = text
            _weather_last_fetch[city] = now
            result[city] = text
        except Exception:
            result[city] = _weather_cache.get(city, "Unknown")

    return result


def _simulated_weather() -> dict:
    """Realistic weather simulation when no API key available."""
    import random
    conditions = [
        ("Clear", 0.35), ("Partly cloudy", 0.25), ("Overcast", 0.15),
        ("Light rain", 0.12), ("Heavy rain", 0.08), ("Thunderstorm", 0.05),
    ]
    result = {}
    for city in CITIES:
        r = random.random()
        cumulative = 0
        for cond, prob in conditions:
            cumulative += prob
            if r <= cumulative:
                temp = {"Mumbai": 28, "Delhi": 32, "Bangalore": 24,
                        "Chennai": 30, "Pune": 26, "Hyderabad": 29}[city]
                temp += random.randint(-3, 3)
                result[city] = f"{cond} {temp}°C"
                break
    return result


def is_bad_weather(weather_dict: dict) -> list:
    """Returns list of cities with weather that affects logistics."""
    bad = []
    triggers = ["rain", "storm", "flood", "fog", "hail", "thunder"]
    for city, condition in weather_dict.items():
        if any(t in condition.lower() for t in triggers):
            bad.append(city)
    return bad


# ── ANOMALY DETECTION ──────────────────────────────────────────────────────

class DemandAnomalyDetector:
    """
    Uses Isolation Forest to detect statistically unusual demand patterns.
    Learns from the last 50 demand readings per category.
    When a new reading is significantly different, it flags it.
    """

    def __init__(self):
        # Rolling history of demand per category
        self.history = {}          # {category: [demand values]}
        self.models  = {}          # {category: trained IsolationForest}
        self.MIN_SAMPLES = 20      # need at least 20 readings before detecting

    def record(self, category: str, demand: int):
        """Add a new demand reading for a category."""
        if category not in self.history:
            self.history[category] = []
        self.history[category].append(demand)

        # Keep rolling window of last 100
        if len(self.history[category]) > 100:
            self.history[category].pop(0)

        # Retrain model every 10 new readings
        if len(self.history[category]) >= self.MIN_SAMPLES and \
           len(self.history[category]) % 10 == 0:
            self._train(category)

    def _train(self, category: str):
        data = np.array(self.history[category]).reshape(-1, 1)
        model = IsolationForest(contamination=0.05, random_state=42)
        model.fit(data)
        self.models[category] = model

    def check(self, category: str, demand: int):
        """
        Returns anomaly info dict if demand is anomalous, else None.
        Dict: {"category": str, "demand": int, "z_score": float, "severity": str}
        """
        hist = self.history.get(category, [])
        if len(hist) < self.MIN_SAMPLES:
            return None   # not enough data yet

        # Z-score check (fast)
        mean = np.mean(hist)
        std  = np.std(hist)
        if std == 0:
            return None
        z = abs((demand - mean) / std)

        # Isolation Forest check
        model = self.models.get(category)
        if model:
            score = model.predict([[demand]])[0]   # -1 = anomaly, 1 = normal
            is_anomaly = score == -1 and z > 2.0
        else:
            is_anomaly = z > 2.5   # fallback: pure z-score

        if is_anomaly:
            severity = "CRITICAL" if z > 3.5 else "HIGH" if z > 2.5 else "MEDIUM"
            return {
                "category": category,
                "demand":   demand,
                "z_score":  round(z, 2),
                "mean":     round(mean, 1),
                "severity": severity,
            }
        return None


# Singleton — import and use this everywhere
anomaly_detector = DemandAnomalyDetector()
