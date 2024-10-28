from simulation import Simulation
import os

def train_dqn(max_sim_time=50000):
    print("Starting DQN training...")
    sim = Simulation(strategy='dql')
    trained_agent = sim.train(max_sim_time=max_sim_time)
    
    # Create directory if it doesn't exist
    model_dir = '/home/coder/early-exit/models/dql/cifar10'
    os.makedirs(model_dir, exist_ok=True)
    
    # Save the trained model
    model_path = os.path.join(model_dir, 'trained_dqn_model.pth')
    trained_agent.save_model(model_path)
    print(f"DQN training completed. Model saved as '{model_path}'")

if __name__ == "__main__":
    train_dqn()
