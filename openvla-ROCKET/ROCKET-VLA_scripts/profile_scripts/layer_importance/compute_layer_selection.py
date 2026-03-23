import numpy as np

def get_balanced_layers(n_layers, n_keep):
    """
    Selects n_keep layers from n_layers, prioritizing deep layers and uniform spacing.
    
    Logic:
    1. If n_keep >= n_layers, return all.
    2. If perfect integer stride selection exists (n_layers % n_keep == 0):
       Select keeping the last layer. 
       Formula: indices = [total - 1 - i * stride]
    3. If perfect integer drop stride exists (n_layers % (n_layers - n_keep) == 0):
       Drop layers uniformly starting from index 0.
       This prioritizes keeping deep layers (by dropping shallow ones).
    4. Fallback: Interpolation anchored at the last layer.
    """
    if n_keep >= n_layers:
        return list(range(n_layers))
    
    # Case 1: Perfect Selection Stride (e.g. 32 -> 8, 24 -> 8)
    if n_layers % n_keep == 0:
        stride = n_layers // n_keep
        # Anchor at end: 31, 31-stride, ...
        indices = [n_layers - 1 - i * stride for i in range(n_keep)]
        return sorted(indices)
        
    # Case 2: Perfect Drop Stride (e.g. 32 -> 24, 24 -> 16)
    n_drop = n_layers - n_keep
    if n_drop > 0 and n_layers % n_drop == 0:
        stride_drop = n_layers // n_drop
        # Drop 0, stride, 2*stride... (Drop shallowest in each block)
        drop_indices = {i * stride_drop for i in range(n_drop)}
        indices = [i for i in range(n_layers) if i not in drop_indices]
        return indices
        
    # Case 3: Fallback (Float interpolation, End-Anchored)
    step = n_layers / n_keep
    indices = [int((n_layers - 1) - i * step + 0.5) for i in range(n_keep)]
    return sorted(indices)

def main():
    configs = [
        {"name": "vla", "layers": 32, "targets": [8, 16, 24]},
        {"name": "vggt", "layers": 24, "targets": [8, 16, 24]}
    ]
    
    print(f"{'Model':<10} | {'Keep':<5} | {'Indices'}")
    print("-" * 80)
    
    results = {}
    
    for config in configs:
        model_name = config["name"]
        total = config["layers"]
        for k in config["targets"]:
            indices = get_balanced_layers(total, k)
            if model_name == "vla":  ################ offset of vla is 1 ################  cause we have 33 feature, last 32 belong to vla, first one after projector
                indices = [i + 1 for i in indices]

            indices_str = ",".join(map(str, indices))
            print(f"{model_name:<10} | {k:<5} | {indices_str}")
            
            # Store for command generation
            key = f"{model_name}_{k}"
            results[key] = indices_str

    print("\nCommand Line Arguments Example:")
    print("-" * 80)
    
    for config in configs:
        model_name = config["name"]
        for k in config["targets"]:
            key = f"{model_name}_{k}"
            flag = f"--{model_name}_layers_align"
            print(f"# {model_name.upper()} Select {k}")
            print(f'{flag} "{results[key]}" \\')
            print()

if __name__ == "__main__":
    main()




#  WO offset


# Model      | Keep  | Indices
# --------------------------------------------------------------------------------
# vla        | 8     | 3,7,11,15,19,23,27,31
# vla        | 16    | 1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31
# vla        | 24    | 1,2,3,5,6,7,9,10,11,13,14,15,17,18,19,21,22,23,25,26,27,29,30,31
# vggt       | 8     | 2,5,8,11,14,17,20,23
# vggt       | 16    | 1,2,4,5,7,8,10,11,13,14,16,17,19,20,22,23
# vggt       | 24    | 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23

# Command Line Arguments Example:
# --------------------------------------------------------------------------------
# # VLA Select 8
# --vla_layers_align "3,7,11,15,19,23,27,31" \

# # VLA Select 16
# --vla_layers_align "1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31" \

# # VLA Select 24
# --vla_layers_align "1,2,3,5,6,7,9,10,11,13,14,15,17,18,19,21,22,23,25,26,27,29,30,31" \

# # VGGT Select 8
# --vggt_layers_align "2,5,8,11,14,17,20,23" \

# # VGGT Select 16
# --vggt_layers_align "1,2,4,5,7,8,10,11,13,14,16,17,19,20,22,23" \

# # VGGT Select 24
# --vggt_layers_align "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23" \





# offset 1

# Model      | Keep  | Indices
# --------------------------------------------------------------------------------
# vla        | 8     | 4,8,12,16,20,24,28,32
# vla        | 16    | 2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32
# vla        | 24    | 2,3,4,6,7,8,10,11,12,14,15,16,18,19,20,22,23,24,26,27,28,30,31,32
# vggt       | 8     | 2,5,8,11,14,17,20,23
# vggt       | 16    | 1,2,4,5,7,8,10,11,13,14,16,17,19,20,22,23
# vggt       | 24    | 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23

# Command Line Arguments Example:
# --------------------------------------------------------------------------------
# # VLA Select 8
# --vla_layers_align "4,8,12,16,20,24,28,32" \

# # VLA Select 16
# --vla_layers_align "2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32" \

# # VLA Select 24
# --vla_layers_align "2,3,4,6,7,8,10,11,12,14,15,16,18,19,20,22,23,24,26,27,28,30,31,32" \

# # VGGT Select 8
# --vggt_layers_align "2,5,8,11,14,17,20,23" \

# # VGGT Select 16
# --vggt_layers_align "1,2,4,5,7,8,10,11,13,14,16,17,19,20,22,23" \

# # VGGT Select 24
# --vggt_layers_align "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23" \