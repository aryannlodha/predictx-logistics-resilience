# 🚚 PredictX — Logistics Resilience & Decision Support

> An AI-powered, multi-agent logistics system for demand forecasting, inventory management, route optimization and disruption response.

## 🚀 Overview

PredictX is a resilience-driven logistics decision-support system that combines **Machine Learning, Multi-Agent Systems, optimization and AI-based decision support** to help supply chains respond to changing demand and operational disruptions.

The system coordinates demand, inventory, routing and decision-making through specialized agents and presents operational information through an interactive dashboard.

## ✨ Key Features

- 📈 **Demand Forecasting** — Random Forest-based demand prediction
- 🤖 **Multi-Agent System** — Demand, Inventory, Routing and Coordinator Agents
- 📦 **Inventory Monitoring** — Stock tracking and reorder alerts
- 🗺️ **Route Optimization** — Dijkstra-based shortest-path routing
- ⚠️ **Disruption Handling** — Demand spikes, supplier delays, weather disruptions, road closures and vehicle breakdowns
- 🧠 **AI Decision Support** — Groq-hosted Llama model for situation-aware logistics decisions
- 🌦️ **Weather Integration** — OpenWeatherMap data for weather-aware decisions
- 🔍 **Anomaly Detection** — Isolation Forest for unusual demand patterns
- 📊 **Interactive Dashboard** — KPIs, event logs, inventory status and logistics activity
- 🔄 **Agent Communication** — FIFO message-bus architecture for inter-agent coordination

## 🧠 System Flow

Demand Forecasting → Inventory Monitoring → Disruption Detection → Route Optimization → AI Decision → Logistics Action → Dashboard Update

## 📊 Dataset & Model

The prototype uses a generated logistics dataset covering **6 cities, 5 product categories and 365 days**, with seasonal demand patterns, lead-time information and warehouse features.

The Random Forest model uses calendar, city, product, warehouse and lead-time features for demand forecasting. The project reports approximately **91.6% prediction accuracy** for the developed prototype.

## 🛠️ Tech Stack

**Python · Flask · Flask-SocketIO · Pandas · NumPy · Scikit-learn · Groq · Llama 3.3 · OpenWeatherMap · Matplotlib**

## 📂 Project Structure

```text
agents/
├── coordinator.py
├── coordinator_ai.py
├── demand_agent.py
├── inventory_agent.py
└── routing_agent.py

data/
├── demand_history.csv
├── supply_chain_data.csv
├── generate_data.py
├── prepare_data.py
└── plot_forecast.py

templates/
└── index.html

app.py
app_v3.py
dashboard.py
disruption_simulator.py
live_data.py
main.py
message_bus.py
test_phase1.py
test_system.py
```

## ▶️ Run Locally

Install dependencies with `pip install -r requirements.txt`.

Create a `.env` file using `.env.example` and add your API keys.

Generate the dataset/model with `python data/prepare_data.py`.

Start the dashboard with `python app_v3.py`.

Then open the local Flask address shown in the terminal.

## 🎯 Applications

PredictX can support logistics environments such as **e-commerce, quick-commerce, warehouse operations, transportation networks, distribution systems and supply-chain control towers** where demand variability and disruptions require rapid decisions.

## 🔮 Future Scope

- Real-time operational and IoT data integration
- Larger multi-city logistics networks
- Advanced MILP-based allocation and recovery optimization
- Continuous inventory and vehicle monitoring
- Integrated logistics control tower
- More autonomous agent-to-agent decision making

## 📌 Project Status

**Functional prototype** with demand forecasting, multi-agent coordination, routing, disruption handling, AI-assisted decisions and an interactive dashboard.

---

⭐ If you find this project useful, consider giving the repository a star.
