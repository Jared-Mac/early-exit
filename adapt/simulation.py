import simpy
from functools import partial
from components import Camera, Head, Tail, EarlyExitHead, EarlyExitTail, AlwaysTransmitHead, AlwaysTransmitTail, block_cpu_flops
from utils import DataTracker
from dql import DQLAgent
from dataset import CacheDataset
from ns.port.wire import Wire
import random
import torch
import numpy as np

class Simulation:
    def __init__(self, strategy='dql', cached_data_file=None, load_model=None):
        self.env = simpy.Environment()
        self.debug = False
        self.cache_dataset = CacheDataset(cached_data_file=cached_data_file, compute_logits=False)
        self.camera = Camera(self.env, self.cache_dataset)
        image_shape = self.cache_dataset.image_shape
        self.data_tracker = DataTracker(self.env)
        
        # Define the number of battery features to use
        battery_features = 1  # e.g., SoC, voltage, temperature
        
        if strategy == 'dql':
            self.agent = DQLAgent(self.env, num_classes=200, num_exit=4, image_shape=image_shape, battery_features=battery_features)
            if load_model:
                self.agent.load_model(load_model)
            self.head = Head(self.env, "head", self.agent, self.data_tracker, debug=self.debug)
            self.tail = Tail(self.env, self.agent, self.data_tracker, debug=self.debug)
        elif strategy == 'always_transmit':
            self.head = AlwaysTransmitHead(self.env, "head", self.data_tracker, debug=self.debug)
            self.tail = AlwaysTransmitTail(self.env, self.data_tracker, debug=self.debug)
        elif strategy == 'early_exit_conservative':
            self.head = EarlyExitHead(self.env, "head", self.data_tracker, confidence_threshold=0.9, debug=self.debug)
            self.tail = EarlyExitTail(self.env, self.data_tracker, debug=self.debug)
        elif strategy == 'early_exit_balanced':
            self.head = EarlyExitHead(self.env, "head", self.data_tracker, confidence_threshold=0.8, debug=self.debug)
            self.tail = EarlyExitTail(self.env, self.data_tracker, debug=self.debug)
        elif strategy == 'early_exit_aggressive':
            self.head = EarlyExitHead(self.env, "head", self.data_tracker, confidence_threshold=0.6, debug=self.debug)
            self.tail = EarlyExitTail(self.env, self.data_tracker, debug=self.debug)
        elif strategy == 'early_exit_very_aggressive':
            self.head = EarlyExitHead(self.env, "head", self.data_tracker, confidence_threshold=0.4, debug=self.debug)
            self.tail = EarlyExitTail(self.env, self.data_tracker, debug=self.debug)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Wire setup
        self.wire1 = Wire(self.env, partial(random.gauss, 0.15, 0.02), wire_id=1, debug=self.debug)

        self.camera.out = self.head
        self.head.out = self.wire1
        self.wire1.out = self.tail
        
        self.stability_window = 2000
        self.stability_threshold = 0.03
        self.strategy = strategy
        self.evaluation_time = 1000  # Time to run evaluation after training
    
    def is_training_stable(self):
        accuracy_history = self.data_tracker.get_accuracy_history()
        if len(accuracy_history) < self.stability_window:
            print(f"Not enough data for stability check. Current window: {len(accuracy_history)}")
            return False
        
        recent_accuracy = accuracy_history[-self.stability_window:]
        stability = np.std(recent_accuracy) < self.stability_threshold
        print(f"Stability check: std={np.std(recent_accuracy):.4f}, threshold={self.stability_threshold}")
        return stability

    def train(self, max_sim_time=5000):
        if self.strategy != 'dql':
            raise ValueError("Training is only applicable for DQL strategy")
        
        self.env.run(until=max_sim_time)
        
        print("\nTraining completed")
        print(f"Final rolling accuracy: {self.data_tracker.get_rolling_accuracy():.2f}")
        print(f"Final rolling latency: {self.data_tracker.get_rolling_latency():.2f}")
        
        return self.agent

    def evaluate(self, max_sim_time=1000):
        # Set DQNet to evaluation mode if using DQL strategy
        if self.strategy == 'dql':
            self.agent.model.eval()
        
        self.env.run(until=max_sim_time)
        print(f"Final rolling accuracy: {self.data_tracker.get_rolling_accuracy():.2f}")
        final_accuracy = self.data_tracker.get_current_accuracy()
        print(f"\nEvaluation results for {self.strategy.upper()} strategy:")
        print(f"Final accuracy: {final_accuracy:.2f}")
        print(f"Final rolling accuracy: {self.data_tracker.get_rolling_accuracy():.2f}")
        print(f"Final rolling latency: {self.data_tracker.get_rolling_latency():.2f}")
        print(f"Final rolling exit numbers: {self.data_tracker.get_rolling_exit_numbers()}")
        print(f"Cumulative exit numbers: {self.data_tracker.get_cumulative_exit_numbers()}")
        print(f"Cumulative action numbers: {self.data_tracker.get_cumulative_action_numbers()}")
        print(f"Average data sent: {self.data_tracker.get_rolling_data_sent():.2f} bytes")
        print(f"Cumulative data sent: {self.data_tracker.get_cumulative_data_sent()} bytes")
        print(f"Average CPU FLOPS: {self.data_tracker.get_rolling_flops():.2f}")
        print(f"Cumulative CPU FLOPS: {self.data_tracker.get_cumulative_flops()}")
        print(f"Final Battery SoC: {self.data_tracker.get_final_battery_soc():.2f}")
        return {
            'accuracy': final_accuracy,
            'avg_latency': self.data_tracker.get_rolling_latency(),
            'exit_numbers': self.data_tracker.get_cumulative_exit_numbers(),
            'action_numbers': self.data_tracker.get_cumulative_action_numbers(),
            'avg_data_sent': self.data_tracker.get_rolling_data_sent(),
            'cumulative_data_sent': self.data_tracker.get_cumulative_data_sent(),
            'avg_flops': self.data_tracker.get_rolling_flops(),
            'cumulative_flops': self.data_tracker.get_cumulative_flops(),
            'final_battery_soc': self.data_tracker.get_final_battery_soc()
        }
