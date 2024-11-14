import simpy
from collections import deque
import numpy as np


class DataTracker:
    def __init__(self, env, window_size=500, track_stats=False):
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
        self.data_sent_window = deque(maxlen=window_size)
        self.cumulative_data_sent = 0
        self.flops_window = deque(maxlen=window_size)
        self.cumulative_flops = 0
        self.battery_soc_window = deque(maxlen=window_size)
        self.exit_logits = []
        self.exit_confidences = []
        if track_stats:
            self.env.process(self.run())

    def run(self):
        while True:
            yield self.env.timeout(500)
            self.print_stats()

            # print(f"Rolling battery voltage: {self.get_rolling_battery_voltage():.2f}")
            # print(f"Rolling battery temperature: {self.get_rolling_battery_temperature():.2f}")

    def print_stats(self):
        print(f"Rolling accuracy: {self.get_rolling_accuracy():.2f}")
        print(f"Rolling latency: {self.get_rolling_latency():.2f}")
        print(f"Rolling exit numbers: {self.get_rolling_exit_numbers()}")
        print(f"Rolling average data sent: {self.get_rolling_data_sent():.2f} bytes")
        print(f"Rolling battery SoC: {self.get_rolling_battery_soc():.2f}")

    def update(self, accuracy, start_time, block_num, latency, action, total_flops, battery_state, confidence=None):
        self.total_classifications += 1
        self.correct_classifications += int(accuracy)
        self.accuracy_window.append(accuracy)
        self.latency_window.append(latency)
        self.exit_window.append(block_num)
        self.action_window.append(action)
        self.cumulative_exit_numbers[f'block{block_num}'] += 1
        self.cumulative_action_numbers[action] += 1
        self.flops_window.append(total_flops)
        self.cumulative_flops += total_flops
        # Update battery state (assuming battery_state is a 1D tensor with [soc, voltage, temperature])
        self.battery_soc_window.append(battery_state[0].item())
        # self.battery_voltage_window.append(battery_state[1].item())
        # self.battery_temperature_window.append(battery_state[2].item())
        if confidence is not None:
            self.exit_confidences.append({
                'block': block_num,
                'confidence': confidence,
                'accuracy': accuracy
            })

    def update_data_sent(self, data_sent):
        self.data_sent_window.append(data_sent)
        self.cumulative_data_sent += data_sent

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

    def get_rolling_data_sent(self):
        if not self.data_sent_window:
            return 0
        return sum(self.data_sent_window) / len(self.data_sent_window)

    def get_cumulative_data_sent(self):
        return self.cumulative_data_sent

    def get_data_sent_history(self):
        return list(self.data_sent_window)

    def get_rolling_flops(self):
        if not self.flops_window:
            return 0
        return sum(self.flops_window) / len(self.flops_window)

    def get_cumulative_flops(self):
        return self.cumulative_flops

    def get_flops_history(self):
        return list(self.flops_window)

    def update_battery(self, soc, voltage, temperature):
        self.battery_soc_window.append(soc)
        self.battery_voltage_window.append(voltage)
        self.battery_temperature_window.append(temperature)

    def get_rolling_battery_soc(self):
        if not self.battery_soc_window:
            return 0
        return sum(self.battery_soc_window) / len(self.battery_soc_window)
    def get_final_battery_soc(self):
        if not self.battery_soc_window:
            return 0
        return self.battery_soc_window[-1]

    def get_exit_logits(self):
        return self.exit_confidences

def generate_threshold_combinations(start=0.6, end=0.9, step=0.1):
    thresholds = np.arange(start, end + step, step)
    combinations = []
    for t1 in thresholds:
        for t2 in thresholds:
            for t3 in thresholds:
                for t4 in thresholds:
                    combinations.append([t1, t2, t3, t4])
    return combinations