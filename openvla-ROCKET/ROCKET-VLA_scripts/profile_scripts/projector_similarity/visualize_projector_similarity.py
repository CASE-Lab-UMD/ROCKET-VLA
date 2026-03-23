import os
import json
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuration
RESULTS_DIR = "profiling_results/projector_similarity"
OUTPUT_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# VLA Layers mapping (for axis labels)
# Based on script: vla_layers_align "2,4,6,8,10,12,14,16,18,-1" (where -1 is 31)
LAYER_LABELS = ["L2", "L4", "L6", "L8", "L10", "L12", "L14", "L16", "L18", "L31"]

def load_json(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data.get("average", {})

def parse_filename(filename):
    # Example: profiling_results_naive10_10000.json
    # Regex to capture Experiment Name (e.g., naive10, v12, share10_cka10) and Step
    match = re.match(r"profiling_results_(.+)_(\d+)\.json", filename)
    if match:
        return match.group(1), int(match.group(2))
    return None, None

def plot_heatmap(matrix, title, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1,
                xticklabels=LAYER_LABELS, yticklabels=LAYER_LABELS)
    plt.title(title)
    plt.xlabel("Layer Index")
    plt.ylabel("Layer Index")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_teacher_sim_comparison(all_data, metric_key="projector_teacher_similarity"):
    """
    Plots lines comparing different experiments across steps.
    """
    # 1. Group by Experiment Type (showing evolution over steps)
    experiments = list(all_data.keys())
    
    for exp_name in experiments:
        plt.figure(figsize=(12, 6))
        steps = sorted(all_data[exp_name].keys())
        
        for step in steps:
            values = all_data[exp_name][step].get(metric_key, [])
            if values:
                plt.plot(LAYER_LABELS, values, marker='o', label=f"Step {step}")
        
        plt.title(f"Projector-Teacher Similarity Evolution: {exp_name}")
        plt.xlabel("Aligned Layer")
        plt.ylabel("Cosine Similarity")
        plt.ylim(0, 1.05)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"evolution_{exp_name}.png"))
        plt.close()

    # 2. Compare all experiments at the FINAL step (usually 50000)
    final_step = 50000
    plt.figure(figsize=(12, 6))
    has_data = False
    
    colors = sns.color_palette("husl", len(experiments))
    
    for idx, exp_name in enumerate(experiments):
        if final_step in all_data[exp_name]:
            values = all_data[exp_name][final_step].get(metric_key, [])
            if values:
                plt.plot(LAYER_LABELS, values, marker='s', linewidth=2, 
                         label=f"{exp_name}", color=colors[idx])
                has_data = True
    
    if has_data:
        plt.title(f"Projector-Teacher Similarity Comparison (Step {final_step})")
        plt.xlabel("Aligned Layer")
        plt.ylabel("Cosine Similarity")
        plt.ylim(0, 1.05)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"comparison_all_step_{final_step}.png"))
        plt.close()

def main():
    json_files = glob.glob(os.path.join(RESULTS_DIR, "*.json"))
    
    # Structure: data[exp_name][step] = avg_dict
    all_data = {}

    print(f"Found {len(json_files)} result files.")

    for filepath in json_files:
        filename = os.path.basename(filepath)
        exp_name, step = parse_filename(filename)
        
        if exp_name is None:
            print(f"Skipping unknown file format: {filename}")
            continue
            
        print(f"Processing: {exp_name} @ Step {step}")
        
        avg_data = load_json(filepath)
        
        if exp_name not in all_data:
            all_data[exp_name] = {}
        all_data[exp_name][step] = avg_data

        # 1. Plot Individual Heatmaps for this file
        # Input Similarity
        if "input_layer_similarity" in avg_data:
            plot_heatmap(
                avg_data["input_layer_similarity"], 
                f"Input Layer Similarity\n{exp_name} (Step {step})",
                os.path.join(OUTPUT_DIR, f"heatmap_input_{exp_name}_{step}.png")
            )
        
        # Output Similarity
        if "output_layer_similarity" in avg_data:
            plot_heatmap(
                avg_data["output_layer_similarity"], 
                f"Projector Output Layer Similarity\n{exp_name} (Step {step})",
                os.path.join(OUTPUT_DIR, f"heatmap_output_{exp_name}_{step}.png")
            )

        # VGGT Similarity
        if "vggt_layer_similarity" in avg_data:
            plot_heatmap(
                avg_data["vggt_layer_similarity"], 
                f"VGGT Layer Similarity\n{exp_name} (Step {step})",
                os.path.join(OUTPUT_DIR, f"heatmap_vggt_{exp_name}_{step}.png")
            )

    # 2. Plot Comparison Curves
    plot_teacher_sim_comparison(all_data)
    
    print(f"\nVisualization complete! Check output directory: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

