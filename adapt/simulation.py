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
    def __init__(self, strategy='dql', cached_data_file=None, load_model=None, confidence_thresholds=None, num_classes=10):
        self.env = simpy.Environment()
        self.debug = False
        self.cache_dataset = CacheDataset(cached_data_file=cached_data_file, compute_logits=False)
        self.camera = Camera(self.env, self.cache_dataset)
        image_shape = self.cache_dataset.image_shape
        self.data_tracker = DataTracker(self.env)
        
        # Define the number of battery features to use
        battery_features = 1  # e.g., SoC, voltage, temperature
        
        if strategy == 'dql':
            self.agent = DQLAgent(self.env, num_classes=num_classes, num_exit=4, image_shape=image_shape, battery_features=battery_features)
            if load_model:
                self.agent.load_model(load_model)
            self.head = Head(self.env, "head", self.agent, self.data_tracker, debug=self.debug)
            self.tail = Tail(self.env, self.agent, self.data_tracker, debug=self.debug)
        elif strategy == 'always_transmit':
            self.head = AlwaysTransmitHead(self.env, "head", self.data_tracker, debug=self.debug)
            self.tail = AlwaysTransmitTail(self.env, self.data_tracker, debug=self.debug)
        elif strategy.startswith('early_exit_custom'):
            if confidence_thresholds is None:
                raise ValueError("confidence_thresholds must be provided for early_exit_custom strategy")
            if len(confidence_thresholds) != 4:  # Assuming 4 exit points
                raise ValueError("confidence_thresholds must contain exactly 4 values")
            self.head = EarlyExitHead(self.env, "head", self.data_tracker, 
                                    confidence_thresholds=confidence_thresholds, 
                                    debug=self.debug)
            self.tail = EarlyExitTail(self.env, self.data_tracker, debug=self.debug)
        elif strategy == 'early_exit_conservative':
            self.head = EarlyExitHead(self.env, "head", self.data_tracker, confidence_thresholds=[0.9,0.9,0.9,0.9], debug=self.debug)
            self.tail = EarlyExitTail(self.env, self.data_tracker, debug=self.debug)
        elif strategy == 'early_exit_balanced':
            self.head = EarlyExitHead(self.env, "head", self.data_tracker, confidence_thresholds=[0.8,0.8,0.8,0.8], debug=self.debug)
            self.tail = EarlyExitTail(self.env, self.data_tracker, debug=self.debug)
        elif strategy == 'early_exit_aggressive':
            self.head = EarlyExitHead(self.env, "head", self.data_tracker, confidence_thresholds=[0.6,0.6,0.6,0.6], debug=self.debug)
            self.tail = EarlyExitTail(self.env, self.data_tracker, debug=self.debug)
        elif strategy == 'early_exit_very_aggressive':
            self.head = EarlyExitHead(self.env, "head", self.data_tracker, confidence_thresholds=[0.4,0.4,0.4,0.4], debug=self.debug)
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
        
        # Add new parameters for convergence checking and loss printing
        self.print_interval = 100  # Print stats every 100 steps
        self.convergence_check_interval = 500  # Check convergence every 500 steps
        self.min_training_steps = 1000  # Minimum steps before checking convergence
    
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
        
        start_time = self.env.now
        last_print_time = start_time
        last_convergence_check = start_time
        avg_change = float('inf')  # Initialize avg_change
        
        def should_continue():
            current_time = self.env.now
            training_duration = current_time - start_time
            
            nonlocal avg_change  # Add nonlocal declaration
            
            # Check for convergence periodically
            nonlocal last_convergence_check
            if (current_time - start_time >= self.min_training_steps and 
                current_time - last_convergence_check >= self.convergence_check_interval):
                avg_change, threshold = self.agent.check_convergence()
                if avg_change < threshold:
                    print("\nTraining converged!")
                    return False
                last_convergence_check = current_time
            
            # Print statistics periodically
            nonlocal last_print_time
            if current_time - last_print_time >= self.print_interval:
                avg_loss = np.mean(list(self.agent.loss_history)) if self.agent.loss_history else float('nan')
                print(f"\nTime step {current_time}:")
                print(f"Average loss: {avg_loss:.4f}")
                print(f"Rolling accuracy: {self.data_tracker.get_rolling_accuracy():.2f}")
                print(f"Rolling latency: {self.data_tracker.get_rolling_latency():.2f}")
                print(f"Average change: {avg_change:.4f}")
                last_print_time = current_time
            
            return current_time - start_time < max_sim_time

        # Run simulation until convergence or max_sim_time
        while should_continue():
            self.env.step()
        
        print("\nTraining completed")
        return self.agent

    def evaluate(self, max_sim_time=1000):
        if self.strategy == 'dql':
            self.agent.model.eval()
        
        self.env.run(until=max_sim_time)
        
        # Get exit confidences in the format expected by visualization
        exit_confidences = self.data_tracker.get_exit_logits()
        logits_list = []
        if exit_confidences:
            # Create a list of lists where each sublist represents confidences for each block
            # Initialize with zeros
            current_logits = [0] * 4
            for conf in exit_confidences:
                current_logits = [0] * 4  # Reset for each decision
                current_logits[conf['block']] = conf['confidence']
                logits_list.append(current_logits)
        
        return {
            'accuracy': self.data_tracker.get_current_accuracy(),
            'avg_latency': self.data_tracker.get_rolling_latency(),
            'exit_numbers': self.data_tracker.get_cumulative_exit_numbers(),
            'exit_logits': logits_list,
            'accuracies': [conf['accuracy'] for conf in exit_confidences] if exit_confidences else []
        }
