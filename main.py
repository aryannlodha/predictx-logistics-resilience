# In main.py — add near the top:
import matplotlib
matplotlib.use("TkAgg")  # or "Agg" if no display
import matplotlib.pyplot as plt
from dashboard import Dashboard

# Inside run_simulation(), add:
dash = Dashboard()
plt.ion()   # interactive mode

# Inside the loop, after each tick:
dash.update(tick, inventory_agent.stock, last_demand, disruption_happened)# main.py
import time
from message_bus import MessageBus
from disruption_simulator import DisruptionSimulator
from agents.demand_agent import DemandAgent
from agents.inventory_agent import InventoryAgent
from agents.routing_agent import RoutingAgent
from agents.coordinator import CoordinatorAgent

def run_simulation(num_ticks=20, delay=0.3):
    print("=" * 50)
    print("  predictX - SIMULATION")
    print("=" * 50)

    # 1. Create the shared message bus
    bus = MessageBus()

    # 2. Create all agents
    demand_agent    = DemandAgent(bus)
    inventory_agent = InventoryAgent(bus, initial_stock=500)
    routing_agent   = RoutingAgent(bus)
    coordinator     = CoordinatorAgent(bus, inventory_agent)
    disruptions     = DisruptionSimulator(bus, probability=0.2)

    # 3. Main simulation loop
    for tick in range(1, num_ticks + 1):
        print(f"\n{'─'*40}")
        print(f"  TICK {tick}")
        print(f"{'─'*40}")

        # Order matters! Each agent builds on the previous one's output.
        disruptions.run(tick)       # Step A: inject disruptions (if any)
        demand_agent.run(tick)      # Step B: forecast demand
        inventory_agent.run(tick)   # Step C: update stock
        coordinator.run(tick)       # Step D: make decisions
        routing_agent.run(tick)     # Step E: route dispatches

        time.sleep(delay)           # slow down so you can read the output

    print("\n" + "=" * 50)
    print("  SIMULATION COMPLETE")
    print(f"  Final stock: {inventory_agent.stock} units")
    print(f"  Coordinator decisions: {len(coordinator.decisions)}")
    print("=" * 50)

if __name__ == "__main__":
    run_simulation(num_ticks=20)