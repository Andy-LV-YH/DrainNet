import os
import ujson
import numpy as np

'''split data into train, val, and test sets'''
def random_split(data_dir, split_ratio=[0.7, 0.1, 0.2], seed=0):
    data_map = []
    curve_dir = os.path.join(data_dir, 'curve')
    dem_dir = os.path.join(data_dir, 'dem')
    mask_dir = os.path.join(data_dir, 'mask')
    
    curves = sorted([file for file in os.listdir(curve_dir) if file.endswith('.tif')])  # Ensure file order consistency
    dems = sorted([file for file in os.listdir(dem_dir) if file.endswith('.tif')]) 
    masks = sorted([file for file in os.listdir(mask_dir) if file.endswith('.tif')])
    
    if len(curves) != len(dems) or len(curves) != len(masks):
        raise ValueError(f"Shape mismatch: curve count {len(curves)}, dem count {len(dems)}, and mask count {len(masks)} must be the same")
    
    keys = sorted([os.path.splitext(file)[0] for file in os.listdir(curve_dir) if file.endswith('.tif')])
    for index, (key, curve, dem, mask) in enumerate(zip(keys, curves, dems, masks)):
        data_map.append({
            "index": index,
            "key": key,
            "curve": os.path.join(curve_dir, curve),
            "dem": os.path.join(dem_dir, dem) ,
            "mask": os.path.join(mask_dir, mask)
        })
    
    rng = np.random.default_rng(seed)
    split_ratio = np.array(split_ratio)/np.sum(split_ratio) # split_ratio becomes [0.7, 0.1, 0.2] (already sums to 1)
    split_ratio = np.cumsum(split_ratio) # split_ratio becomes [0.7, 0.8, 1.0]

    sample_split = (split_ratio * len(data_map)).astype(int) # sample_split is the number of samples in each split
    permuted_indices = rng.permutation(len(data_map)) # permuted_indices is a shuffled list of indices

    train_indices = permuted_indices[:sample_split[0]]
    val_indices = permuted_indices[sample_split[0]:sample_split[1]]
    test_indices = permuted_indices[sample_split[1]:]

    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")

    # Save data_map to a json file with train, val, and test splits
    split_data_map = {
        "train": [data_map[i] for i in train_indices],
        "val": [data_map[i] for i in val_indices],
        "test": [data_map[i] for i in test_indices]
    }

    with open(os.path.join(data_dir, 'Data_map.json'), 'w') as json_file:
        ujson.dump(split_data_map, json_file, indent=4)

    print("Data map saved successfully.")
    
    return split_data_map

if __name__ == "__main__":
    data_dir = r'E:\Data'
    random_split(data_dir, split_ratio=[0.7, 0.1, 0.2], seed=0)
