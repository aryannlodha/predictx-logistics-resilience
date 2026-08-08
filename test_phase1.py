# test_phase1.py
#
# Run this with:  python test_phase1.py
#
# All tests should print PASS. If any print FAIL,
# read the message below it to find out what went wrong.

from message_bus import MessageBus
from agents.demand_agent import DemandAgent


# ─────────────────────────────────────────
# TEST 1: MessageBus — basic publish/consume
# ─────────────────────────────────────────
def test_publish_and_consume():
    bus = MessageBus()
    bus.publish("test_channel", {"value": 42})
    msg = bus.consume("test_channel")

    assert msg is not None,      "FAIL: consume() returned None — message was not published"
    assert msg["value"] == 42,   "FAIL: message content is wrong"
    print("PASS: MessageBus publish and consume")


# ─────────────────────────────────────────
# TEST 2: MessageBus — empty channel returns None
# ─────────────────────────────────────────
def test_empty_channel_returns_none():
    bus = MessageBus()
    msg = bus.consume("nonexistent_channel")

    assert msg is None, "FAIL: expected None from empty channel, got something else"
    print("PASS: MessageBus returns None on empty channel")


# ─────────────────────────────────────────
# TEST 3: MessageBus — FIFO order
# ─────────────────────────────────────────
def test_fifo_order():
    bus = MessageBus()
    bus.publish("ordered", {"n": 1})
    bus.publish("ordered", {"n": 2})
    bus.publish("ordered", {"n": 3})

    assert bus.consume("ordered")["n"] == 1, "FAIL: first message should be 1"
    assert bus.consume("ordered")["n"] == 2, "FAIL: second message should be 2"
    assert bus.consume("ordered")["n"] == 3, "FAIL: third message should be 3"
    print("PASS: MessageBus preserves FIFO order")


# ─────────────────────────────────────────
# TEST 4: MessageBus — has_messages()
# ─────────────────────────────────────────
def test_has_messages():
    bus = MessageBus()
    assert bus.has_messages("ch") == False, "FAIL: empty channel should return False"
    bus.publish("ch", {"x": 1})
    assert bus.has_messages("ch") == True,  "FAIL: channel with message should return True"
    print("PASS: MessageBus has_messages()")


# ─────────────────────────────────────────
# TEST 5: MessageBus — history log
# ─────────────────────────────────────────
def test_history_log():
    bus = MessageBus()
    bus.publish("ch_a", {"tick": 1})
    bus.publish("ch_b", {"tick": 2})
    bus.publish("ch_a", {"tick": 3})

    full   = bus.get_history()
    ch_a   = bus.get_history("ch_a")
    ch_b   = bus.get_history("ch_b")

    assert len(full) == 3,  f"FAIL: expected 3 total history entries, got {len(full)}"
    assert len(ch_a) == 2,  f"FAIL: expected 2 entries for ch_a, got {len(ch_a)}"
    assert len(ch_b) == 1,  f"FAIL: expected 1 entry for ch_b, got {len(ch_b)}"
    print("PASS: MessageBus history log and filtering")


# ─────────────────────────────────────────
# TEST 6: DemandAgent — run() publishes a message
# ─────────────────────────────────────────
def test_demand_agent_publishes():
    bus = MessageBus()
    agent = DemandAgent(bus)
    returned_demand = agent.run(tick=1)

    msg = bus.consume("demand_forecast")

    assert msg is not None,                      "FAIL: no message on demand_forecast channel"
    assert msg["tick"] == 1,                     "FAIL: tick should be 1"
    assert msg["units"] == returned_demand,      "FAIL: message units don't match return value"
    assert msg["source"] == "DemandAgent",       "FAIL: source field is wrong"
    assert msg["ml_used"] == False,              "FAIL: ml_used should be False in Phase 1"
    print("PASS: DemandAgent publishes correct message")


# ─────────────────────────────────────────
# TEST 7: DemandAgent — demand is always positive
# ─────────────────────────────────────────
def test_demand_always_positive():
    bus = MessageBus()
    agent = DemandAgent(bus)

    for tick in range(1, 51):   # run 50 ticks
        demand = agent.run(tick)
        assert demand > 0, f"FAIL: demand was {demand} at tick {tick} — must be positive"

    print("PASS: DemandAgent always produces positive demand (50 ticks)")


# ─────────────────────────────────────────
# TEST 8: DemandAgent — history tracking
# ─────────────────────────────────────────
def test_demand_agent_history():
    bus = MessageBus()
    agent = DemandAgent(bus, seed=99)   # fixed seed = reproducible

    for tick in range(1, 6):
        agent.run(tick)

    assert len(agent.forecast_history) == 5, \
        f"FAIL: expected 5 history entries, got {len(agent.forecast_history)}"

    avg = agent.get_average_demand()
    assert avg > 0, "FAIL: average demand should be positive"

    print(f"PASS: DemandAgent history tracking (5 ticks, avg demand = {avg})")


# ─────────────────────────────────────────
# TEST 9: Full Phase 1 mini-simulation
# ─────────────────────────────────────────
def test_mini_simulation():
    print("\n--- Mini simulation (5 ticks) ---")
    bus = MessageBus()
    agent = DemandAgent(bus, base_demand=120, seed=42)

    for tick in range(1, 6):
        agent.run(tick)

    # Consume all messages and verify them
    messages = []
    while bus.has_messages("demand_forecast"):
        messages.append(bus.consume("demand_forecast"))

    assert len(messages) == 5, f"FAIL: expected 5 messages, got {len(messages)}"

    for i, msg in enumerate(messages, start=1):
        assert msg["tick"] == i,    f"FAIL: tick mismatch at message {i}"
        assert msg["units"] > 0,    f"FAIL: units must be positive at tick {i}"

    bus.summary()
    print("PASS: Mini simulation — all 5 messages verified")


# ─────────────────────────────────────────
# Run all tests
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 45)
    print("  PHASE 1 TEST SUITE")
    print("=" * 45 + "\n")

    test_publish_and_consume()
    test_empty_channel_returns_none()
    test_fifo_order()
    test_has_messages()
    test_history_log()
    test_demand_agent_publishes()
    test_demand_always_positive()
    test_demand_agent_history()
    test_mini_simulation()

    print("\n" + "=" * 45)
    print("  ALL PHASE 1 TESTS PASSED")
    print("=" * 45)