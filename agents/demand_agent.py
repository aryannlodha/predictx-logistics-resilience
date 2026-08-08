# agents/demand_agent.py  (UPDATED with ML)
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import random

class DemandAgent:
    """
    Demand forecasting with a simple ML model.
    Falls back to random if no data file found.
    """
    def __init__(self, bus, data_path="data/demand_history.csv"):
        self.bus = bus
        self.name = "DemandAgent"
        self.model = None
        self.poly = None
        self.current_day = 90   # we start forecasting from day 90 onwards

        self._train_model(data_path)

    def _train_model(self, data_path):
        """Load CSV and train a polynomial regression model."""
        try:
            df = pd.read_csv(data_path)
            X = df[["day"]].values         # input: day number
            y = df["demand"].values        # output: demand

            # Polynomial features capture the weekly cycle better than linear
            self.poly = PolynomialFeatures(degree=2)
            X_poly = self.poly.fit_transform(X)

            self.model = LinearRegression()
            self.model.fit(X_poly, y)

            print(f"[DemandAgent] Model trained on {len(df)} days of data.")
        except FileNotFoundError:
            print("[DemandAgent] No data file found, using random fallback.")

    def predict(self, day):
        """Predict demand for a given day number."""
        if self.model is None:
            return random.randint(60, 150)  # fallback

        X = np.array([[day]])
        X_poly = self.poly.transform(X)
        prediction = self.model.predict(X_poly)[0]
        # Add small noise so each tick feels live
        noisy = int(prediction + np.random.normal(0, 10))
        return max(10, noisy)

    def run(self, tick):
        demand = self.predict(self.current_day)
        self.current_day += 1

        message = {
            "tick": tick,
            "units": demand,
            "source": self.name,
            "ml_predicted": self.model is not None
        }

        self.bus.publish("demand_forecast", message)
        print(f"[{self.name}] Tick {tick} (day {self.current_day}): "
              f"Predicted demand = {demand} units")
        return demand