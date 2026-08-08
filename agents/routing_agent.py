# agents/routing_agent.py
import heapq

class RoutingAgent:
    """
    Finds shortest delivery routes using Dijkstra's algorithm.
    The graph is a dict: node -> list of (neighbor, cost) tuples.
    """
    def __init__(self, bus):
        self.bus = bus
        self.name = "RoutingAgent"

        # Our delivery network (you can extend this!)
        # Format: "location": [(connected_location, travel_cost), ...]
        self.graph = {
            "warehouse":  [("hub_north", 10), ("hub_south", 15)],
            "hub_north":  [("warehouse", 10), ("city_A", 5), ("city_B", 8)],
            "hub_south":  [("warehouse", 15), ("city_C", 6), ("city_D", 9)],
            "city_A":     [("hub_north", 5)],
            "city_B":     [("hub_north", 8)],
            "city_C":     [("hub_south", 6)],
            "city_D":     [("hub_south", 9)],
        }

    def dijkstra(self, start, end):
        """
        Classic Dijkstra's shortest path.
        Returns (path_as_list, total_cost).
        """
        # Priority queue: (cost, current_node, path_so_far)
        pq = [(0, start, [start])]
        visited = set()

        while pq:
            cost, node, path = heapq.heappop(pq)

            if node == end:
                return path, cost         # found it!

            if node in visited:
                continue
            visited.add(node)

            for neighbor, weight in self.graph.get(node, []):
                if neighbor not in visited:
                    heapq.heappush(pq, (cost + weight, neighbor, path + [neighbor]))

        return None, float("inf")         # no path found

    def run(self, tick):
        msg = self.bus.consume("dispatch")

        if msg is None:
            return   # nothing to route this tick

        destination = msg.get("destination", "city_A")
        path, cost = self.dijkstra("warehouse", destination)

        if path:
            result = {
                "tick": tick,
                "destination": destination,
                "route": " -> ".join(path),
                "cost": cost
            }
            self.bus.publish("route_result", result)
            print(f"[{self.name}] Tick {tick}: Route to {destination}: {' -> '.join(path)} (cost={cost})")
        else:
            print(f"[{self.name}] Tick {tick}: No route found to {destination}!")