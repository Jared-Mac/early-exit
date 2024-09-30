
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import simpy
from ns.packet.packet import Packet
from collections import defaultdict as dd
import time
# import ee_gym as ee_gym
import dql_agent as dql_agent
import numpy as np

class Item:
    def __init__(self, logit, label, yhat, elapsed,exit_num):
        self.logit = logit
        self.yhat  = yhat
        self.label = label
        self.elapsed = elapsed
        self.exit_num = exit_num
    def __str__(self):
        return f"logit:{self.logit},yhat:{self.yhat},label:{self.elapsed},logit:{self.elapsed},exit:{self.exit_num}"
class Camera:
    def __init__(self, env, test_loader, interval=1,out=None, agent_store=None):
        """
        Initialize the Camera class.

        Parameters:
        - env: simpy.Environment
            The simulation environment.
        - destination_queue: simpy.Store
            The destination queue where the loaded images will be sent.
        - load_interval: int or float
            The time interval (in simulation time units) between image loads.
        """
        self.env = env
        self.out = out
        self.interval = interval
        self.test_loader = test_loader
        self.agent_store = agent_store
        self.env.process(self.run())
    def run(self):
        """
        Simulate loading an image. This function can be modified to actually load images
        if required, but here it will just return a placeholder.
        """
        while True:
            # for image, label in self.test_loader:
            #     self.out.put((image,label.numpy()[0]))
            self.agent_store.put(np.random.randint(0,100))
            yield self.env.timeout(self.interval)
    

class Head:
    def __init__(self, env, model1, model2, model1_time, model2_time,conf1,conf2,data,trainer_store, device="cuda", out = None):
        self.store = simpy.Store(env)        
        self.env = env
        self.model1 = model1
        self.model2 = model2
        self.model1_time = model1_time
        self.model2_time = model2_time
        self.data = data
        self.device = device
        self.out = out
        self.action = env.process(self.run())
        self.conf1 = conf1
        self.conf2 = conf2
        self.trainer_store = trainer_store

    def predict(self, model, images):
        with torch.no_grad():
            outputs, predictions = model(images.to(self.device))
            m = torch.nn.Softmax(dim=1)
            predictions = m(predictions)
            predicted = torch.max(predictions, 1)
            return outputs, predicted

    def run(self):
        packet_id = 0  # Initialize packet ID
        yield self.env.timeout(1)
        while True:
            image,label = yield self.store.get()
            init_time = self.env.now

            # Add loop for each exit
            # Inference on exit 1
            output, prediction = self.predict(self.model1, image)
            yield self.env.timeout(self.model1_time())

            logit, yhat = prediction
            logit, yhat = logit.cpu().numpy()[0], yhat.cpu().numpy()[0]
            
            # Policy Agent
            if logit >= self.conf1:
                item = Item(logit,label,yhat,(self.env.now-init_time),1)
                self.data.append(item)
                self.trainer_store.put(item)
                continue


            # Create a Packet instance
            packet = Packet(
                time=init_time,
                size=256,  # Assuming a size of 1 byte for simplicity; adjust as needed
                packet_id=packet_id,
                src="Head",
                dst="Tail",
                flow_id=1,  # Assuming all packets belong to the same flow; adjust as needed
                payload=(output.cpu(),label)  # Payload contains the predictions
            )
            packet_id += 1  # Increment packet ID for the next packet

            # Pass the Packet instance to the tail through the wire
            self.out.put(packet)
            # Wait for the next batch step
    def put(self,image):
        return self.store.put(image)

class Tail:
    def __init__(self, env, model, model_time,data, device="cuda"):
        self.store = simpy.Store(env)
        self.env = env
        self.model = model
        self.device = device
        self.model_time = model_time
        self.action = env.process(self.run())
        self.data_storage = data
    def predict(self, input):
        with torch.no_grad():
            input = input.to(self.device)
            predictions = self.model(input.to(self.device))
            m = torch.nn.Softmax(dim=1)
            predictions = m(predictions)
            return predictions.argmax(dim=1)

    def run(self):
        while True:
            packet = yield self.store.get()
            input,label = packet.payload
            yhat = self.predict(input)
            logit=0.5
            model_time = self.model_time()
            yield self.env.timeout( model_time if model_time > 0 else 0)
            # logit = logit.cpu().numpy()[0]
            yhat = yhat.cpu().numpy()
            time_elapsed = self.env.now - packet.time
            # print(f"End-to-inference Latency: {time_elapsed} ")
            self.data_storage.append(Item(logit,label,yhat,time_elapsed,2))
            # print(logit[0], yhat[0])
    def put(self, packet):
        self.store.put(packet)

class Agent:
    def __init__(self, env, agent, model, optimizer, criterion, device="cuda"):
        self.store = simpy.Store(env)
        self.env = env
        if agent == 'dql':
            state_dim = 1
            action_dim = 2
            self.agent = dql_agent.DQLAgent(state_dim,action_dim)
        # self.gym = ee_gym.EarlyExitGym(env)
        self.action = env.process(self.learn())
    def learn(self):
        while True:
            # Wait for data item
            item = yield self.store.get()
            print(item)
            obs, reward, done, info = self.gym.step(item)

            # Add to memory store, Train Agent
            # agent.replay_buffer.push(obs, action, reward, next_obs, done)
            # agent.train()
    def action(self, state):
        return self.agent.select_action(state)

if __name__ == "__main__":
    trainer = Agent(None,None,None,None,None)
