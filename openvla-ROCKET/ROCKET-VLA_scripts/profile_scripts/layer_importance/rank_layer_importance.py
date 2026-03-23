import re

# Raw data: per-layer cosine similarity scores
raw_data = """
Layer 0: 0.8727
Layer 1: 0.9457
Layer 2: 0.9562
Layer 3: 0.9809
Layer 4: 0.9826
Layer 5: 0.9849
Layer 6: 0.9810
Layer 7: 0.9729
Layer 8: 0.9596
Layer 9: 0.9575
Layer 10: 0.9451
Layer 11: 0.9536
Layer 12: 0.9573
Layer 13: 0.9550
Layer 14: 0.9502
Layer 15: 0.9454
Layer 16: 0.9554
Layer 17: 0.9680
Layer 18: 0.9644
Layer 19: 0.9746
Layer 20: 0.9761
Layer 21: 0.9787
Layer 22: 0.9743
Layer 23: 0.9855
Layer 24: 0.9795
Layer 25: 0.9862
Layer 26: 0.9852
Layer 27: 0.9896
Layer 28: 0.9901
Layer 29: 0.9841
Layer 30: 0.9797
Layer 31: 0.9532
"""

def parse_and_sort(data):
    # Parse each line
    layers = []
    lines = data.strip().split('\n')
    for line in lines:
        match = re.search(r'Layer (\d+): ([\d\.]+)', line)
        if match:
            idx = int(match.group(1))
            score = float(match.group(2))
            layers.append((idx, score))
    
    # Sort by score ascending
    sorted_layers = sorted(layers, key=lambda x: x[1])
    
    return sorted_layers

def get_indices_string(layer_list, offset=1):
    return ",".join([str(x[0] + offset) for x in layer_list])

if __name__ == "__main__":
    sorted_layers = parse_and_sort(raw_data)
    
    counts = [8, 10, 16, 24]
    
    print("Total layers:", len(sorted_layers))
    
    for k in counts:
        # Get bottom-k layers (lowest similarity)
        smallest_k = sorted_layers[:k]

        # Get top-k layers (highest similarity)
        largest_k = sorted_layers[-k:]

        print(f"\n=== K = {k} ===")
        print(f"[Bottom {k} Layer Indices (sorted by index, offset=1)]:")
        print(get_indices_string(sorted(smallest_k, key=lambda x: x[0])))

        print(f"[Top {k} Layer Indices (sorted by index, offset=1)]:")
        print(get_indices_string(sorted(largest_k, key=lambda x: x[0])))

# Total layers: 32

# === K = 8 ===
# [Bottom 8 Layer Indices (sorted by index, offset=1)]:
# 1,2,11,12,14,15,16,32
# [Top 8 Layer Indices (sorted by index, offset=1)]:
# 5,6,24,26,27,28,29,30

# === K = 10 ===
# [Bottom 10 Layer Indices (sorted by index, offset=1)]:
# 1,2,3,11,12,14,15,16,17,32
# [Top 10 Layer Indices (sorted by index, offset=1)]:
# 4,5,6,7,24,26,27,28,29,30

# === K = 16 ===
# [Bottom 16 Layer Indices (sorted by index, offset=1)]:
# 1,2,3,8,9,10,11,12,13,14,15,16,17,18,19,32
# [Top 16 Layer Indices (sorted by index, offset=1)]:
# 4,5,6,7,20,21,22,23,24,25,26,27,28,29,30,31

# === K = 24 ===
# [Bottom 24 Layer Indices (sorted by index, offset=1)]:
# 1,2,3,4,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,25,31,32
# [Top 24 Layer Indices (sorted by index, offset=1)]:
# 3,4,5,6,7,8,9,10,13,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31