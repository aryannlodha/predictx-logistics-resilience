# data/plot_forecast.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv("data/demand_history.csv")
X = df[["day"]].values
y = df["demand"].values

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
model = LinearRegression().fit(X_poly, y)

# Predict next 14 days
future_days = np.arange(91, 105).reshape(-1, 1)
future_pred = model.predict(poly.transform(future_days))

plt.figure(figsize=(10, 5))
plt.plot(df["day"], y, label="Historical demand", color="steelblue", alpha=0.7)
plt.plot(future_days, future_pred, label="ML Forecast", color="orange",
         linewidth=2, linestyle="--")
plt.axvline(x=90, color="gray", linestyle=":", label="Forecast starts")
plt.xlabel("Day")
plt.ylabel("Demand (units)")
plt.title("Demand Forecast — Polynomial Regression")
plt.legend()
plt.tight_layout()
plt.savefig("logs/demand_forecast.png")
plt.show()
print("Saved to logs/demand_forecast.png")