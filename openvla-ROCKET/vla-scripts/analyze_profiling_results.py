import json
import numpy as np
import argparse
from pathlib import Path
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError:
    sns = None

def main():
    parser = argparse.ArgumentParser(description="Analyze profiling results and compute optimal layer alignment.")
    parser.add_argument("json_path", type=str, help="Path to the profiling_results.json file")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save visualization (optional)")
    args = parser.parse_args()

    json_path = Path(args.json_path)
    if not json_path.exists():
        print(f"Error: File not found at {json_path}")
        return

    print(f"Loading results from: {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)

    raw_similarities = data.get("raw_similarities", {})
    if not raw_similarities:
        print("Error: 'raw_similarities' not found in JSON data.")
        return

    # Extract layer indices
    # Convert string keys to integers and sort
    vla_layers = sorted([int(k) for k in raw_similarities.keys()])
    
    # Assume all VLA layers have the same set of VGGT layers
    first_vla_layer_key = str(vla_layers[0])
    vggt_layers = sorted([int(k) for k in raw_similarities[first_vla_layer_key].keys()])

    print(f"Found {len(vla_layers)} VLA layers: {vla_layers}")
    print(f"Found {len(vggt_layers)} VGGT layers: {vggt_layers}")

    # Build similarity matrix (Rows: VLA, Cols: VGGT)
    sim_matrix = np.zeros((len(vla_layers), len(vggt_layers)))
    for i, vla_l in enumerate(vla_layers):
        for j, vggt_l in enumerate(vggt_layers):
            # Access using string keys as JSON stores keys as strings
            sim_matrix[i, j] = raw_similarities[str(vla_l)].get(str(vggt_l), 0.0)

    # --- Hungarian Algorithm ---
    # linear_sum_assignment finds the minimum cost.
    # To find maximum similarity, we define Cost = Max_Sim - Sim
    cost_matrix = sim_matrix.max() - sim_matrix
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    print("\n" + "="*60)
    print("OPTIMAL ONE-TO-ONE ALIGNMENT (Hungarian Algorithm)")
    print("="*60)
    
    total_similarity = 0
    matched_pairs = []

    # Print header
    print(f"{'VLA Layer':<12} {'->':<4} {'VGGT Layer':<12} {'Similarity':<10}")
    print("-" * 40)

    for r, c in zip(row_ind, col_ind):
        vla_idx = vla_layers[r]
        vggt_idx = vggt_layers[c]
        score = sim_matrix[r, c]
        total_similarity += score
        matched_pairs.append((vla_idx, vggt_idx, score))
        
    # Sort by VLA layer index for cleaner output
    matched_pairs.sort(key=lambda x: x[0])
    
    for vla_l, vggt_l, score in matched_pairs:
        print(f"{vla_l:<12} {'->':<4} {vggt_l:<12} {score:.4f}")

    print("-" * 40)
    print(f"Average Similarity of Matched Pairs: {total_similarity / len(matched_pairs):.4f}")
    
    # Save the matched pairs to a simple text file if output dir is provided or in the same dir
    save_dir = Path(args.output_dir) if args.output_dir else json_path.parent
    save_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = save_dir / "optimal_alignment.txt"
    with open(output_file, "w") as f:
        f.write("VLA_Layer,VGGT_Layer,Similarity\n")
        for vla_l, vggt_l, score in matched_pairs:
            f.write(f"{vla_l},{vggt_l},{score:.6f}\n")
    print(f"\nSaved alignment list to: {output_file}")

    # Generate comma-separated lists for config usage
    matched_vla = [str(p[0]) for p in matched_pairs]
    matched_vggt = [str(p[1]) for p in matched_pairs]
    
    print("\n--- Configuration Strings ---")
    print(f"vla_layers_align = \"{','.join(matched_vla)}\"")
    print(f"vggt_layers_align = \"{','.join(matched_vggt)}\"")

    # --- Visualization ---
    plt.figure(figsize=(14, 12))
    
    # If we have many layers, seaborn heatmap is nice
    if sns:
        ax = sns.heatmap(sim_matrix, annot=False, cmap="viridis",
                         xticklabels=vggt_layers, yticklabels=vla_layers)
        
        # Highlight the optimal path
        # Note: Heatmap coordinates: x is column (VGGT), y is row (VLA)
        # We can add rectangles or dots to show the selected pairs
        # But since indices might be sparse or not 1:1 visually if sorted differently,
        # we plot simple scatters on top.
        
        # Get coordinates for scatter plot
        # col_ind corresponds to x-axis (VGGT indices in matrix), row_ind to y-axis (VLA indices in matrix)
        # +0.5 to center in the cell
        plt.scatter(col_ind + 0.5, row_ind + 0.5, color='red', marker='x', s=50, label='Optimal Match')
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        
    else:
        plt.imshow(sim_matrix, cmap='viridis', aspect='auto')
        plt.colorbar()
        plt.scatter(col_ind, row_ind, color='red', marker='x')

    plt.xlabel("VGGT Layers")
    plt.ylabel("VLA Layers")
    plt.title("CKA Similarity Matrix with Optimal Assignment (Red X)")
    
    vis_path = save_dir / "optimal_alignment_heatmap.png"
    plt.savefig(vis_path)
    print(f"Saved visualization to: {vis_path}")

if __name__ == "__main__":
    main()

