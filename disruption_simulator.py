# disruption_simulator.py
import random

class DisruptionSimulator:
    """
    Randomly fires disruption events into the system.
    Each tick has a small chance (15%) of generating a disruption.
    """
    EVENTS = [
        "road_closed",
        "supplier_delay",
        "demand_spike",
        "weather_delay",
        "vehicle_breakdown"
    ]

    def __init__(self, bus, probability=0.15):
        self.bus = bus
        self.probability = probability   # chance per tick
        self.name = "DisruptionSimulator"

    def run(self, tick):
        if random.random() < self.probability:
            event = random.choice(self.EVENTS)
            self.bus.publish("disruption", {
                "tick": tick,
                "type": event
            })
            print(f"[{self.name}] *** DISRUPTION at tick {tick}: {event} ***")
            return event
        return None