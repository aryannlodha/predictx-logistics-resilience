# agents/inventory_agent.py

class InventoryAgent:
    """
    Tracks stock levels.
    Reads demand forecasts from the bus.
    Publishes alerts when stock is low.
    """
    def __init__(self, bus, initial_stock=500):
        self.bus = bus
        self.stock = initial_stock
        self.reorder_threshold = 100   # alert if below this
        self.name = "InventoryAgent"
        self.history = []

    def run(self, tick):
        # Read the demand forecast published by DemandAgent
        msg = self.bus.consume("demand_forecast")

        if msg is None:
            print(f"[{self.name}] Tick {tick}: No demand message yet.")
            return

        units_needed = msg["units"]
        self.stock -= units_needed
        self.stock = max(0, self.stock)   # stock can't go negative

        self.history.append({"tick": tick, "stock": self.stock})

        status = "OK"
        if self.stock < self.reorder_threshold:
            status = "LOW - reorder triggered"
            self.bus.publish("inventory_alert", {
                "tick": tick,
                "stock_level": self.stock,
                "reorder": True,
                "units_needed": units_needed
            })

        print(f"[{self.name}] Tick {tick}: Stock = {self.stock} [{status}]")

    def restock(self, units):
        """Called by Coordinator when a reorder arrives."""
        self.stock += units
        print(f"[{self.name}] Restocked +{units}. New stock = {self.stock}")