
import os
import sys
from libero.libero import benchmark

# Define Task Suites to check
task_suites = [
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
    "libero_90"
]

print("Checking number of initial states for LIBERO task suites...")
benchmark_dict = benchmark.get_benchmark_dict()

for suite_name in task_suites:
    try:
        task_suite = benchmark_dict[suite_name]()
        # Get the first task to check its initial states
        # Assuming all tasks in a suite have the same number of initial states
        if task_suite.n_tasks > 0:
            initial_states = task_suite.get_task_init_states(0)
            print(f"{suite_name}: {len(initial_states)} initial states available per task (Total tasks: {task_suite.n_tasks})")
        else:
             print(f"{suite_name}: No tasks found.")
    except Exception as e:
        print(f"Error checking {suite_name}: {e}")
