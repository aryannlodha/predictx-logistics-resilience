# data/dataset_guide.md
# EXACT DATASETS TO DOWNLOAD — all free on Kaggle
# ═══════════════════════════════════════════════════════

# DATASET 1 — Real e-commerce orders (most important)
# URL: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
# Files to download: olist_orders_dataset.csv
#                    olist_order_items_dataset.csv
#                    olist_products_dataset.csv
# Why: 100,000 real orders with timestamps, geolocation, delivery times
# How to use: replaces simulated demand with actual order patterns

# DATASET 2 — Supply chain + logistics
# URL: https://www.kaggle.com/datasets/harshsingh2209/supply-chain-analysis
# File: supply_chain_data.csv
# Why: product demand, stock levels, shipping times, defect rates
# How to use: train your inventory reorder model

# DATASET 3 — Indian city distances
# URL: https://www.kaggle.com/datasets/rajkumarpandey02/distance-between-major-cities-india
# File: city_distances.csv
# Why: replace fake distances in your routing agent with real km values
# How to use: feed into Dijkstra's algorithm

# DATASET 4 — Road accident / disruption events
# URL: https://www.kaggle.com/datasets/s3separator/india-road-accidents-dataset
# File: road_accidents.csv
# Why: real disruption probability per city/route
# How to use: DisruptionSimulator uses real probabilities, not random 15%

# ══════════════════════════════════════════════════════
# HOW TO LOAD EACH ONE — paste into prepare_data.py
# ══════════════════════════════════════════════════════

import pandas as pd
import numpy as np

def load_olist_demand(orders_path, items_path, products_path):
    """
    Load Olist Brazilian e-commerce dataset.
    Returns daily demand per product category — exactly what DemandAgent needs.
    """
    orders   = pd.read_csv(orders_path, parse_dates=["order_purchase_timestamp"])
    items    = pd.read_csv(items_path)
    products = pd.read_csv(products_path)

    merged = orders.merge(items, on="order_id") \
                   .merge(products[["product_id","product_category_name"]], on="product_id")

    merged["date"]       = merged["order_purchase_timestamp"].dt.date
    merged["day_of_week"]= merged["order_purchase_timestamp"].dt.dayofweek
    merged["month"]      = merged["order_purchase_timestamp"].dt.month

    # Map Portuguese categories to English
    cat_map = {
        "eletronicos": "Electronics",
        "alimentos":   "Groceries",
        "moda_roupa":  "Clothing",
        "moveis_sala": "Furniture",
        "saude_beleza":"Medicines",
    }
    merged["category"] = merged["product_category_name"].map(
        lambda x: next((v for k,v in cat_map.items() if k in str(x)), "Other")
    )
    merged = merged[merged["category"] != "Other"]

    daily = merged.groupby(["date","day_of_week","month","category"]) \
                  .size().reset_index(name="demand_units")

    daily.to_csv("data/real_demand.csv", index=False)
    print(f"Saved {len(daily)} daily demand records from Olist dataset")
    return daily


def load_city_distances(distances_path):
    """
    Load Indian city distance matrix.
    Returns dict suitable for the RoutingAgent graph.
    """
    df = pd.read_csv(distances_path)
    # Expected columns: city1, city2, distance_km
    graph = {}
    for _, row in df.iterrows():
        c1, c2, dist = row["city1"], row["city2"], row["distance_km"]
        if c1 not in graph: graph[c1] = []
        if c2 not in graph: graph[c2] = []
        graph[c1].append((c2, int(dist)))
        graph[c2].append((c1, int(dist)))
    return graph


def load_supply_chain(sc_path):
    """
    Load supply chain analysis dataset.
    Returns dataframe with stock levels, lead times, defect rates.
    """
    df = pd.read_csv(sc_path)
    # Standardise column names (dataset has: Product type, SKU, Price,
    # Availability, Number of products sold, Lead times, etc.)
    df.columns = [c.lower().replace(" ","_") for c in df.columns]
    return df


# ══════════════════════════════════════════════════════
# STEP-BY-STEP KAGGLE DOWNLOAD INSTRUCTIONS
# ══════════════════════════════════════════════════════
# 1. Go to kaggle.com → sign up free
# 2. Go to account → API → "Create New Token" → downloads kaggle.json
# 3. Place kaggle.json in C:\Users\aryan\.kaggle\
# 4. Run: pip install kaggle
# 5. Then download:
#    kaggle datasets download -d olistbr/brazilian-ecommerce
#    kaggle datasets download -d harshsingh2209/supply-chain-analysis
#    kaggle datasets download -d rajkumarpandey02/distance-between-major-cities-india
# 6. Unzip into your data/ folder
# 7. Run: python data/prepare_data.py --real
