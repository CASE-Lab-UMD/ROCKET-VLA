import re
import sys
import os
import csv

def parse_logs(file_path):
    # Dictionary to store similarities: vla_layer -> {vggt_layer: similarity}
    data = {}
    
    # Regex to match the line format
    # VLA layer   2 vs VGGT layer   0: Avg. Cosine Similarity = 0.9563
    pattern = re.compile(r"VLA layer\s+(\d+)\s+vs\s+VGGT layer\s+(\d+):\s+Avg\.\s+Cosine\s+Similarity\s+=\s+([0-9.]+)")
    
    count = 0
    try:
        with open(file_path, 'r') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    vla_idx = int(match.group(1))
                    vggt_idx = int(match.group(2))
                    similarity = float(match.group(3))
                    
                    if vla_idx not in data:
                        data[vla_idx] = {}
                    data[vla_idx][vggt_idx] = similarity
                    count += 1
                    
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    
    print(f"Successfully parsed {count} data points.")
    return data

def save_to_csv(data, output_file):
    try:
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['VLA Layer', 'VGGT Layer', 'Similarity'])
            
            sorted_vla_layers = sorted(data.keys())
            for vla_idx in sorted_vla_layers:
                vggt_map = data[vla_idx]
                # Sort by VGGT layer for cleaner output
                sorted_vggt_layers = sorted(vggt_map.keys())
                
                for vggt_idx in sorted_vggt_layers:
                    writer.writerow([vla_idx, vggt_idx, vggt_map[vggt_idx]])
        
        print(f"All data saved to: {output_file}")
    except IOError as e:
        print(f"Error writing to file {output_file}: {e}")

def analyze_correspondence(data):
    print(f"\n{'VLA Layer':<10} | {'Best Match VGGT Layer':<25} | {'Max Similarity':<15}")
    print("-" * 60)
    
    # Sort by VLA layer index
    sorted_vla_layers = sorted(data.keys())
    
    for vla_idx in sorted_vla_layers:
        vggt_map = data[vla_idx]
        # Find VGGT layer with max similarity
        best_vggt = max(vggt_map.items(), key=lambda x: x[1])
        
        print(f"{vla_idx:<10} | {best_vggt[0]:<25} | {best_vggt[1]:.4f}")

def main():
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    else:
        # Default to the known location if no argument provided
        log_file = "profiling_results/profiling_logs.log"
        
    if not os.path.exists(log_file):
        # Fallback to current dir
        log_file = "profiling_logs.log"
            
    print(f"Parsing file: {log_file}")
    data = parse_logs(log_file)
    
    if not data:
        print("No matching data found in the log file.")
        return
    
    # Output file path (same directory as log file)
    output_dir = os.path.dirname(os.path.abspath(log_file))
    csv_file = os.path.join(output_dir, "layer_similarities.csv")
    
    # Save all data to CSV
    save_to_csv(data, csv_file)
    
    # Still print the summary
    analyze_correspondence(data)

if __name__ == "__main__":
    main()
