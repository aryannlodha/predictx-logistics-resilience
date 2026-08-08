# dashboard.py
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Dashboard:
    """
    Simple live chart that updates each tick.
    Shows stock level and demand over time.
    """
    def __init__(self):
        self.ticks = []
        self.stock_levels = []
        self.demand_values = []
        self.disruption_ticks = []

        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 6))
        self.fig.suptitle("predictX — Live Dashboard", fontsize=13)

    def update(self, tick, stock, demand, disruption=False):
        self.ticks.append(tick)
        self.stock_levels.append(stock)
        self.demand_values.append(demand)
        if disruption:
            self.disruption_ticks.append(tick)

        # Plot stock
        self.ax1.clear()
        self.ax1.plot(self.ticks, self.stock_levels, color="steelblue", linewidth=2)
        self.ax1.axhline(y=100, color="red", linestyle="--", alpha=0.5, label="Reorder threshold")
        for dt in self.disruption_ticks:
            self.ax1.axvline(x=dt, color="orange", alpha=0.4)
        self.ax1.set_ylabel("Stock level (units)")
        self.ax1.set_title("Inventory over time")
        self.ax1.legend(loc="upper right")

        # Plot demand
        self.ax2.clear()
        self.ax2.bar(self.ticks, self.demand_values, color="teal", alpha=0.7)
        self.ax2.set_ylabel("Demand (units)")
        self.ax2.set_xlabel("Simulation tick")
        self.ax2.set_title("Demand per tick")

        plt.tight_layout()
        plt.pause(0.05)

    def save(self, path="logs/dashboard.png"):
        self.fig.savefig(path)
        print(f"Dashboard saved to {path}")