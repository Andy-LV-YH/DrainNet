from dataset import ChannelDataset, CombineDataset, PredictionDataset
from torch.utils.data import DataLoader


 
'''load data into DataLoader'''
def train_data_loaders(data_map_dir, mode='train', included_dem=False, augmentation=True, batch_size=4, num_workers=4, pin_memory=True):
    train_original_ds = ChannelDataset(data_map_dir, mode, included_dem, augmentation)
    train_ds = CombineDataset(train_original_ds)
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True)
    print(f"Train data loader: {len(train_dataloader.dataset)} samples")
    print(f"Train data loader: {len(train_dataloader)} batches")

    return train_dataloader


def val_data_loaders(data_map_dir, mode='val', included_dem=False, augmentation=False, batch_size=4, num_workers=4, pin_memory=True):
    val_ds = ChannelDataset(data_map_dir, mode, included_dem, augmentation)
    val_dataloader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=False)
    print(f"Val data loader: {len(val_dataloader.dataset)} samples")
    print(f"Val data loader: {len(val_dataloader)} batches")

    return val_dataloader


def test_data_loaders(data_map_dir, mode='test', included_dem=False, augmentation=False, batch_size=1, num_workers=0, pin_memory=True):
    test_ds = ChannelDataset(data_map_dir, mode, included_dem, augmentation)
    test_dataloader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=False)
    print(f"Test data loader: {len(test_dataloader.dataset)} samples")

    return test_dataloader


def predict_data_loaders(data_dir, batch_size=1, num_workers=0, pin_memory=True):
    predict_ds = PredictionDataset(data_dir)
    predict_dataloader = DataLoader(predict_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=False)
    print(f"Predict data loader: {len(predict_dataloader.dataset)} samples")

    return predict_dataloader
