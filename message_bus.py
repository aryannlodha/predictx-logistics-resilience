# message_bus.py

from collections import defaultdict
import queue


class MessageBus:
    """
    The shared communication layer for all agents.

    Agents never talk to each other directly.
    Instead, they publish messages to named channels
    and consume messages from named channels.

    Think of it like a group of labeled mailboxes.
    Anyone can drop a letter in. Anyone can pick one up.
    """

    def __init__(self):
        # Each channel is an independent FIFO queue.
        # defaultdict means a new queue is auto-created
        # the first time a channel name is used.
        self.channels = defaultdict(queue.Queue)

        # A full history log — useful for the dashboard later.
        self.history = []

    def publish(self, channel, message):
        """
        Drop a message into a named channel.

        Args:
            channel (str): e.g. "demand_forecast", "inventory_alert"
            message (dict): any data you want to send
        """
        self.channels[channel].put(message)
        self.history.append({
            "channel": channel,
            "data": message
        })

    def consume(self, channel):
        """
        Pick up the next waiting message from a channel.
        Returns None if nothing is waiting — agents should
        always handle the None case gracefully.

        Args:
            channel (str): the channel to read from

        Returns:
            dict or None
        """
        try:
            return self.channels[channel].get_nowait()
        except queue.Empty:
            return None

    def has_messages(self, channel):
        """
        Check if a channel has any waiting messages
        without consuming them.

        Returns:
            bool
        """
        return not self.channels[channel].empty()

    def get_history(self, channel=None):
        """
        Return the full log of published messages.
        Pass a channel name to filter by that channel only.

        Args:
            channel (str, optional): filter by channel name

        Returns:
            list of dicts
        """
        if channel:
            return [h for h in self.history if h["channel"] == channel]
        return self.history

    def clear_channel(self, channel):
        """
        Empty all messages from a channel.
        Useful for resetting state between tests.
        """
        while not self.channels[channel].empty():
            try:
                self.channels[channel].get_nowait()
            except queue.Empty:
                break

    def summary(self):
        """
        Print a quick overview of message counts per channel.
        Handy for debugging.
        """
        print("\n--- Message Bus Summary ---")
        if not self.history:
            print("  No messages published yet.")
            return
        counts = defaultdict(int)
        for h in self.history:
            counts[h["channel"]] += 1
        for ch, count in sorted(counts.items()):
            print(f"  [{ch}]: {count} message(s) published total")
        print("---------------------------\n")