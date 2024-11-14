import cProfile
import pstats
import line_profiler
import os
from train import train_dqn
from functools import wraps

def profile_with_cprofile(output_file):
    """Decorator for cProfile profiling"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            profiler = cProfile.Profile()
            result = profiler.runcall(func, *args, **kwargs)
            profiler.dump_stats(output_file)
            
            # Print sorted statistics
            stats = pstats.Stats(output_file)
            stats.sort_stats('cumulative').print_stats(30)
            return result
        return wrapper
    return decorator

# Create line profiler
line_prof = line_profiler.LineProfiler()

@profile_with_cprofile('train_profile_stats.prof')
@line_prof
def profile_training():
    """Profile the training function with sample parameters"""
    train_dqn(
        dataset='visualwakewords',
        model_type='mobilenetv2',
        max_sim_time=2000  # Reduced for profiling
    )

if __name__ == "__main__":
    print("Starting profiling of train.py...")
    
    # Run profiling
    profile_training()
    
    # Output line profiler results
    line_prof.print_stats()
    
    print("\nProfiling completed. Check 'train_profile_stats.prof' for detailed results")
    print("To view the results in a more readable format, you can use:")
    print("python -m snakeviz train_profile_stats.prof") 