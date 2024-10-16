import simpy
from collections import deque


class DataTracker:
    def __init__(self, env, window_size=500):
        self.env = env
        self.total_classifications = 0
        self.correct_classifications = 0
        self.window_size = window_size
        self.accuracy_window = deque(maxlen=window_size)
        self.latency_window = deque(maxlen=window_size)
        self.exit_window = deque(maxlen=window_size)
        self.action_window = deque(maxlen=window_size)
        self.cumulative_exit_numbers = {f'block{i}': 0 for i in range(4)}
        self.cumulative_action_numbers = {i: 0 for i in range(3)}  # 0: continue, 1: exit without transmit, 2: exit and transmit
        self.env.process(self.run())

    def run(self):
        while True:
            yield self.env.timeout(500)
            print(f"Rolling accuracy: {self.get_rolling_accuracy():.2f}")
            if self.latency_window:
                print(f"Rolling average latency): {self.get_rolling_latency():.2f}")
            print(f"Rolling exit numbers): {self.get_rolling_exit_numbers()}")
            # print(f"Cumulative exit numbers: {self.cumulative_exit_numbers}")

    def update(self, accuracy, start_time, exit_number, latency, action):
        self.total_classifications += 1
        self.correct_classifications += accuracy
        self.accuracy_window.append(accuracy)
        self.latency_window.append(latency)
        self.exit_window.append(exit_number)
        self.action_window.append(action)
        self.cumulative_exit_numbers[f'block{exit_number}'] += 1
        self.cumulative_action_numbers[action] += 1

    def get_rolling_accuracy(self):
        if not self.accuracy_window:
            return 0
        return sum(self.accuracy_window) / len(self.accuracy_window)

    def get_rolling_latency(self):
        if not self.latency_window:
            return 0
        return sum(self.latency_window) / len(self.latency_window)

    def get_rolling_exit_numbers(self):
        if not self.exit_window:
            return {f'block{i}': 0 for i in range(4)}
        exit_counts = {f'block{i}': self.exit_window.count(i) for i in range(4)}
        return exit_counts

    def get_rolling_action_numbers(self):
        if not self.action_window:
            return {i: 0 for i in range(3)}
        action_counts = {i: self.action_window.count(i) for i in range(3)}
        return action_counts

    def get_current_accuracy(self):
        if self.total_classifications == 0:
            return 0
        return self.correct_classifications / self.total_classifications

    def get_accuracy_history(self):
        return list(self.accuracy_window)

    def get_latency_history(self):
        return list(self.latency_window)

    def get_exit_history(self):
        return list(self.exit_window)

    def get_cumulative_exit_numbers(self):
        return self.cumulative_exit_numbers

    def get_cumulative_action_numbers(self):
        return self.cumulative_action_numbers
