import json
import numpy as np
import os
from libero.libero import benchmark

def generate_indices(seed=42, output_path="libero_eval_indices.json"):
    # Define task suites
    task_suites = [
        "libero_spatial",
        "libero_object",
        "libero_goal",
        "libero_10",
        "libero_90"
    ]
    
    total_samples = 50 # LIBERO standard
    rng = np.random.default_rng(seed)
    
    indices_data = {
        "meta": {
            "seed": seed,
            "total_samples_per_task": total_samples
        },
        "suites": {}
    }
    
    print(f"Generating random indices with seed {seed}...")
    
    benchmark_dict = benchmark.get_benchmark_dict()
    
    for suite_name in task_suites:
        print(f"Processing {suite_name}...")
        try:
            task_suite = benchmark_dict[suite_name]()
            num_tasks = task_suite.n_tasks
            
            suite_indices = {}
            
            # Generate a shuffled list for EACH task
            # Or do we want the SAME shuffle for all tasks? 
            # Usually, different shuffle for each task is more robust to "lucky" task orderings,
            # but same shuffle is easier to debug. Let's do unique shuffle per task for better randomness.
            
            for task_id in range(num_tasks):
                indices = np.arange(total_samples)
                rng.shuffle(indices)
                suite_indices[str(task_id)] = indices.tolist()
                
            indices_data["suites"][suite_name] = suite_indices
            
        except Exception as e:
            print(f"Skipping {suite_name}: {e}")
            
    # Save to file
    with open(output_path, "w") as f:
        json.dump(indices_data, f, indent=2)
        
    print(f"\nIndices saved to {output_path}")
    print("Example (libero_spatial task 0):", indices_data["suites"]["libero_spatial"]["0"][:10], "...")

if __name__ == "__main__":
    generate_indices()
