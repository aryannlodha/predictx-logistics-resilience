# data/prepare_data.py
# Run this ONCE before starting app.py
# It creates the dataset and trains the ML model

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pickle, os

os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

print("Step 1/3 — Generating real-world supply chain dataset...")

np.random.seed(42)
cities      = ["Mumbai","Delhi","Bangalore","Chennai","Pune","Hyderabad"]
categories  = ["Electronics","Groceries","Clothing","Furniture","Medicines"]
warehouses  = ["Warehouse_North","Warehouse_South","Warehouse_West"]

rows = []
base_date = pd.Timestamp("2024-01-01")

for day in range(365):
    date        = base_date + pd.Timedelta(days=day)
    dow         = date.dayofweek
    month       = date.month
    # Real-world demand multipliers
    weekend     = 1.4  if dow >= 5          else 1.0
    festival    = 1.8  if month in [10,11]  else 1.0   # Diwali season
    summer      = 1.2  if month in [4,5,6]  else 1.0
    monsoon     = 0.85 if month in [7,8]    else 1.0   # logistics slow down

    for city in cities:
        for cat in categories:
            base = {"Electronics":45,"Groceries":200,"Clothing":80,
                    "Furniture":20,"Medicines":150}[cat]
            city_m = {"Mumbai":1.5,"Delhi":1.4,"Bangalore":1.3,
                      "Chennai":1.1,"Pune":1.0,"Hyderabad":1.0}[city]
            demand = int(base * city_m * weekend * festival * summer * monsoon
                        + np.random.normal(0, base * 0.15))
            demand = max(5, demand)
            rows.append({
                "date":            date.strftime("%Y-%m-%d"),
                "day_of_week":     dow,
                "month":           month,
                "city":            city,
                "category":        cat,
                "demand_units":    demand,
                "unit_price":      {"Electronics":8500,"Groceries":150,
                                    "Clothing":800,"Furniture":12000,"Medicines":200}[cat],
                "warehouse":       np.random.choice(warehouses),
                "stock_available": np.random.randint(200, 1000),
                "lead_time_days":  np.random.randint(1, 5),
            })

df = pd.DataFrame(rows)
csv_path = os.path.join(OUT_DIR, "supply_chain_data.csv")
df.to_csv(csv_path, index=False)
print(f"   ✓ {len(df):,} rows saved to {csv_path}")

print("\nStep 2/3 — Training ML demand prediction model (RandomForest)...")
df_enc = pd.get_dummies(df, columns=["city","category","warehouse"])
feature_cols = ["day_of_week","month","lead_time_days"] + \
               [c for c in df_enc.columns if c.startswith(("city_","category_","warehouse_"))]

X = df_enc[feature_cols]
y = df_enc["demand_units"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
preds = model.predict(X_test)
mae   = mean_absolute_error(y_test, preds)
acc   = 100 * (1 - mae / y.mean())
print(f"   ✓ Model trained — MAE: {mae:.1f} units | Accuracy: {acc:.1f}%")

model_path = os.path.join(OUT_DIR, "model.pkl")
with open(model_path, "wb") as f:
    pickle.dump({"model": model, "features": feature_cols}, f)
print(f"   ✓ Model saved to {model_path}")

print("\nStep 3/3 — Verifying...")
with open(model_path, "rb") as f:
    bundle = pickle.load(f)
test_row = {col: 0 for col in bundle["features"]}
test_row.update({"day_of_week": 0, "month": 1, "lead_time_days": 2,
                  "city_Mumbai": 1, "category_Groceries": 1, "warehouse_Warehouse_North": 1})
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    pred = bundle["model"].predict(pd.DataFrame([test_row])[bundle["features"]])[0]
print(f"   ✓ Sample prediction: Mumbai Groceries Monday = {int(pred)} units")

print("\n" + "="*50)
print("  SETUP COMPLETE")
print("  Now run:  python app.py")
print("="*50)
