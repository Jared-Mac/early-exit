from simulation import Simulation
import os
import argparse

def train_dqn(max_sim_time=5000, model_dir='/home/coder/early-exit/models/cifar10'):
    print("Starting DQN training...")
    sim = Simulation(strategy='dql', cached_data_file=model_dir+'/blocks/cached_logits.pkl')
    trained_agent = sim.train(max_sim_time=max_sim_time)
    
    # Create directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Save the trained model
    model_path = os.path.join(model_dir, 'trained_dqn_model.pth')
    trained_agent.save_model(model_path)
    print(f"DQN training completed. Model saved as '{model_path}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DQN agent')
    parser.add_argument('--model-dir', type=str, default='/home/coder/early-exit/models/dql/cifar10',
                        help='Directory to save the trained DQN model')
    args = parser.parse_args()
    
    train_dqn(model_dir=args.model_dir)