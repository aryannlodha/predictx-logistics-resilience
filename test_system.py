# test_system.py
from message_bus import MessageBus
from agents.demand_agent import DemandAgent
from agents.inventory_agent import InventoryAgent
from agents.routing_agent import RoutingAgent
from agents.coordinator import CoordinatorAgent

def test_message_bus():
    bus = MessageBus()
    bus.publish("test_channel", {"value": 42})
    msg = bus.consume("test_channel")
    assert msg["value"] == 42
    assert bus.consume("test_channel") is None  # empty now
    print("PASS: MessageBus")

def test_demand_agent():
    bus = MessageBus()
    agent = DemandAgent(bus)
    demand = agent.run(tick=1)
    assert demand > 0
    msg = bus.consume("demand_forecast")
    assert msg is not None
    assert msg["units"] == demand
    print("PASS: DemandAgent")

def test_inventory_agent():
    bus = MessageBus()
    # Manually publish a demand message
    bus.publish("demand_forecast", {"tick": 1, "units": 50, "source": "test"})
    inv = InventoryAgent(bus, initial_stock=200)
    inv.run(tick=1)
    assert inv.stock == 150
    print("PASS: InventoryAgent")

def test_routing_agent():
    bus = MessageBus()
    bus.publish("dispatch", {"tick": 1, "destination": "city_A"})
    router = RoutingAgent(bus)
    router.run(tick=1)
    result = bus.consume("route_result")
    assert result is not None
    assert "warehouse" in result["route"]
    print("PASS: RoutingAgent")

def test_low_stock_triggers_reorder():
    bus = MessageBus()
    inv = InventoryAgent(bus, initial_stock=80)   # below threshold
    coord = CoordinatorAgent(bus, inv)
    # Simulate an alert already in the bus
    bus.publish("inventory_alert", {"tick": 1, "stock_level": 80, "reorder": True, "units_needed": 50})
    old_stock = inv.stock
    coord.run(tick=1)
    assert inv.stock > old_stock   # should have restocked
    print("PASS: Low stock triggers reorder")

if __name__ == "__main__":
    print("Running tests...\n")
    test_message_bus()
    test_demand_agent()
    test_inventory_agent()
    test_routing_agent()
    test_low_stock_triggers_reorder()
    print("\nAll tests passed!")