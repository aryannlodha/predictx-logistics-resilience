# agents/coordinator.py

class CoordinatorAgent:
    """
    Reads alerts from Inventory and Disruption channels.
    Makes decisions: trigger reorders, reroute deliveries.
    """
    def __init__(self, bus, inventory_agent):
        self.bus = bus
        self.inventory = inventory_agent   # direct reference for restocking
        self.name = "Coordinator"
        self.decisions = []

    def run(self, tick):
        # Check for inventory alerts
        alert = self.bus.consume("inventory_alert")
        if alert and alert.get("reorder"):
            restock_amount = 200   # fixed restock for simplicity
            self.inventory.restock(restock_amount)

            # Also dispatch a new delivery
            self.bus.publish("dispatch", {
                "tick": tick,
                "destination": "city_A",   # default destination
                "priority": "high",
                "reason": "low_stock"
            })
            decision = f"Tick {tick}: Reorder {restock_amount} units, dispatch to city_A"
            self.decisions.append(decision)
            print(f"[{self.name}] {decision}")

        # Check for disruptions
        disruption = self.bus.consume("disruption")
        if disruption:
            dtype = disruption.get("type")
            print(f"[{self.name}] Tick {tick}: Handling disruption '{dtype}'")

            if dtype == "road_closed":
                # Reroute to alternate destination
                self.bus.publish("dispatch", {
                    "tick": tick,
                    "destination": "city_B",
                    "priority": "urgent",
                    "reason": "road_closed"
                })
            elif dtype == "demand_spike":
                # Increase stock urgently
                self.inventory.restock(300)
                print(f"[{self.name}] Emergency restock due to demand spike")