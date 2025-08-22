import ujson
import torch
from torchvision.transforms import transforms as tfs
import os
import rasterio
from torch.utils.data import Dataset

'''dataset definition for 3 modes: train, val, test'''
class ChannelDataset(Dataset):
    def __init__(self, data_map_dir, mode='val', included_dem=True, augmentation=False):
        self.mode = mode
        with open(data_map_dir, 'r') as json_file:
            self.split_data_map = ujson.load(json_file)
        self.data_map = self.split_data_map[mode]
        self.Norm = tfs.Normalize([0.4997], [0.0760])  # Default normalization for curve only
        self.length = len(self.data_map)
        self.augmentation = augmentation
        self.included_dem = included_dem

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        curve_path = self.data_map[idx]['curve']
        if self.included_dem:
            dem_path = self.data_map[idx]['dem']
        else:
            dem_path = None
        mask_path = self.data_map[idx]['mask']

        with rasterio.open(curve_path) as src:
            curve = src.read(1)
        
        curve = torch.tensor(curve, dtype=torch.float32).unsqueeze(0)  # (1, H, W)
        curve = (curve - curve.min()) / (curve.max() - curve.min())

        if dem_path is not None:
            with rasterio.open(dem_path) as src:
                dem = src.read(1)
            dem = torch.tensor(dem, dtype=torch.float32).unsqueeze(0)  # (1, H, W)
            dem = (dem - dem.min()) / (dem.max() - dem.min())
            inp = torch.cat([curve, dem], dim=0)  # (2, H, W)
            self.Norm = tfs.Normalize([0.4997, 0.1764], [0.0760, 0.3549])  # Update normalization for curve and dem
        else:
            inp = curve  # (1, H, W)
            self.Norm = tfs.Normalize([0.4997], [0.0760])  # Normalization for curve only
        
        inp = self.Norm(inp)

        with rasterio.open(mask_path) as src:
            mask = src.read(1)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # (1, H, W)
        mask = self.preprocess_mask(mask)
        return inp, mask

    def preprocess_mask(self, mask):
        mask[mask == 255] = 1
        return mask
    
    def augment(self, seed=0):
        augmented_data = []
        if self.augmentation:
            torch.manual_seed(seed)
            for idx in range(self.length):
                inp, mask = self.__getitem__(idx)
                # Horizontal Flip
                aug_inp = tfs.RandomHorizontalFlip(p=1)(inp)
                aug_mask = tfs.RandomHorizontalFlip(p=1)(mask)
                augmented_data.append((aug_inp, aug_mask))
                # Vertical Flip
                aug_inp = tfs.RandomVerticalFlip(p=1)(inp)
                aug_mask = tfs.RandomVerticalFlip(p=1)(mask)
                augmented_data.append((aug_inp, aug_mask))
                # Rotations
                aug_inp = tfs.RandomRotation([0, 360])(inp)
                aug_mask = tfs.RandomRotation([0, 360])(mask)
                augmented_data.append((aug_inp, aug_mask))
            print("Augmentation data: ", len(augmented_data))  
        else:
            print("Augmentation is disabled.")
        return augmented_data  

'''combine original and augmented dataset'''
class CombineDataset(Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset
        self.augmented_dataset = original_dataset.augment()
        self.length = len(original_dataset) + len(self.augmented_dataset)

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if idx < len(self.original_dataset):
            return self.original_dataset[idx]
        else:
            return self.augmented_dataset[idx - len(self.original_dataset)]



'''dataset definition for prediction'''
class PredictionDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.Norm = tfs.Normalize([0.4997], [0.0760])  

        "modified for curve only"
        "if dem is needed, uncomment the related lines"

        # self.curve_dir = os.path.join(data_dir, 'curve')
        # self.dem_dir = os.path.join(data_dir, 'dem')
        self.curves = sorted([file for file in os.listdir(self.data_dir) if file.endswith('.tif')])
        # self.dems = sorted([file for file in os.listdir(self.dem_dir) if file.endswith('.tif')])
        self.length = len(self.curves)  # Assuming the number of curve files and dem files are the same

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        curve_path = os.path.join(self.data_dir, self.curves[idx])
        # dem_path = os.path.join(self.dem_dir, self.dems[idx])

        with rasterio.open(curve_path) as src:
            curve = src.read(1)
        # with rasterio.open(dem_path) as src:
        #     dem = src.read(1)

        curve = torch.tensor(curve, dtype=torch.float32).unsqueeze(0)  # (1, H, W)
        # dem = torch.tensor(dem, dtype=torch.float32).unsqueeze(0)      # (1, H, W)
        curve = (curve - curve.min()) / (curve.max() - curve.min())
        # dem = (dem - dem.min()) / (dem.max() - dem.min())
        # inp = torch.cat([curve, dem], dim=0)  # (2, H, W)
        inp = curve
        inp = self.Norm(inp)

        return inp
