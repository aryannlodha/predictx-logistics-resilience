# data/generate_data.py
import pandas as pd
import numpy as np

np.random.seed(42)
days = 90   # 3 months of data

data = {
    "day": range(1, days + 1),
    "demand": [
        int(100                              # base demand
            + 0.5 * d                        # slight growth trend
            + 20 * np.sin(2 * np.pi * d / 7) # weekly cycle
            + np.random.normal(0, 15))        # random noise
        for d in range(1, days + 1)
    ]
}

df = pd.DataFrame(data)
df["demand"] = df["demand"].clip(lower=10)   # no negative demand
df.to_csv("data/demand_history.csv", index=False)
print(f"Generated {len(df)} days of demand data.")
print(df.tail())