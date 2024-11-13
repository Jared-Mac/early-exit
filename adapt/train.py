from simulation import Simulation
import os
import argparse

def train_dqn(dataset='cifar10', model_type='resnet50', max_sim_time=50000):
    # Define number of classes for each dataset
    dataset_classes = {
        'cifar10': 10,
        'cifar100': 100,
        'tiny-imagenet': 200,
        'visualwakewords': 2,
    }
    
    num_classes = dataset_classes.get(dataset, 10)  # Default to 10 if dataset not found
    
    # Construct model directory path
    model_dir = f'models/{dataset}/{model_type}'
    print(f"Starting DQN training for {dataset} dataset with {model_type}...")
    
    sim = Simulation(
        strategy='dql',
        cached_data_file=os.path.join(model_dir, 'blocks/cached_logits.pkl'),
        num_classes=num_classes
    )
    trained_agent = sim.train(max_sim_time=max_sim_time)
    
    # Create directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Save the trained model
    model_path = os.path.join(model_dir, 'trained_dqn_model.pth')
    trained_agent.save_model(model_path)
    print(f"DQN training completed. Model saved as '{model_path}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DQN agent')
    parser.add_argument('--dataset', type=str, default='cifar10',
                       help='Dataset name (default: cifar10)')
    parser.add_argument('--model-type', type=str, default='resnet50',
                       help='Model type (default: resnet50)')
    parser.add_argument('--max-sim-time', type=int, default=50000,
                       help='Maximum simulation time (default: 50000)')
    
    args = parser.parse_args()
    train_dqn(
        dataset=args.dataset,
        model_type=args.model_type,
        max_sim_time=args.max_sim_time
    )