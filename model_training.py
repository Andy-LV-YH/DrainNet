import os
from datetime import datetime
from utils import train_data_loaders, val_data_loaders
from DrainNet import DrainNet
import ujson
from logger import Logger
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
import time



def model_training(configuration):
    # Load configuration
    root_dir = configuration['root_dir']
    result_dir = os.path.join(root_dir, configuration['result_dir'], configuration['case_name'])

    data_map_dir = configuration['data_map_dir']
    cuda_device = configuration['cuda_device']
    batch_size = configuration['batch_size']
    num_epochs = configuration['num_epochs']
    accumulation_steps = configuration['accumulation_steps']
    start_epoch = configuration['start_epoch']
    learning_rate = configuration['learning_rate']
    checkpoint_saving_frequency = configuration['checkpoint_saving_frequency']
    augmentation = configuration['train_augmentation']
    include_dem = configuration['include_dem']

    # Save configuration
    configuration_path = os.path.join(result_dir, 'configuration.json')
    os.makedirs(result_dir, exist_ok=True)
    with open(configuration_path, 'w') as json_file:
        ujson.dump(configuration, json_file, indent=4)

    # Create a Logger for training log
    logger = Logger(os.path.join(result_dir, "Training_log.csv"))
    logger.write(f"Time,Epoch,Train_loss,Validate_loss,lr")
    
    device = torch.device(f"cuda:{cuda_device}" if torch.cuda.is_available() else "cpu")

    case_name = configuration['case_name']
    model_name = case_name.split('_')[0]
    model_classes = {
        'DrainNet': DrainNet,
    }
    if model_name not in model_classes:
        raise ValueError(f"Model {model_name} not found.")
    model_class = model_classes[model_name]

    if include_dem:
        model = model_class(2,1).to(device)

    else:
        model = model_class(1,1).to(device)

    # Define loss function and optimizer
    # class DiceLoss(torch.nn.Module):
    #     def __init__(self):
    #         super(DiceLoss, self).__init__()

    #     def forward(self, inputs, targets, smooth=1):
    #         inputs = torch.sigmoid(inputs)       
    #         inputs = inputs.view(-1)
    #         targets = targets.view(-1)
    #         intersection = (inputs * targets).sum()
    #         dice = (2.*intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
    #         return 1 - dice

    # criterion = DiceLoss()
    criterion = torch.nn.BCEWithLogitsLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)


    # Load data
    train_dataloader = train_data_loaders(data_map_dir, mode='train', included_dem=include_dem,augmentation=augmentation, batch_size=batch_size)
    val_dataloader = val_data_loaders(data_map_dir, mode='val', included_dem=include_dem, batch_size=batch_size)

    if start_epoch > 0:
        try:
            checkpoint = torch.load(os.path.join(result_dir, f'checkpoint_epoch{start_epoch}.pth'), map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except:
            print(f"Failed to load checkpoint_epoch{start_epoch}.pth")
            start_epoch = 0
    
    start_time = time.time()

    loss_epoch_train = []
    loss_epoch_validate = []

    for epoch in range(start_epoch+1, num_epochs+1):

        running_loss = 0.0
        running_loss_val = 0.0
        model.train()
        accumulation_count = 0  # for gradient accumulation
        train_loop = tqdm(train_dataloader, position=0, leave=True)
        optimizer.zero_grad()

        for inp, mask in train_loop:
            inp, mask = inp.to(device), mask.to(device)
            
            out = model(inp)
            loss = criterion(out, mask) / accumulation_steps
            loss.backward()
            running_loss += loss.item() * inp.shape[0] * accumulation_steps  # inp.shape[0] is the batch size
            accumulation_count += 1
            if accumulation_count % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                accumulation_count = 0
        loss_epoch_train.append(running_loss / len(train_dataloader.dataset))

        with torch.no_grad():
            model.eval()
            for inp, mask in val_dataloader:
                inp, mask = inp.to(device), mask.to(device)
                out = model(inp)
                loss = criterion(out, mask)
                running_loss_val += loss.item() * inp.shape[0]  # without accumulation
            loss_epoch_validate.append(running_loss_val / len(val_dataloader.dataset))

        if configuration['use_scheduler']:
            scheduler.step()

        time_stamp = datetime.now().isoformat(timespec='seconds') 

        logger.write(f"{time_stamp},{epoch},{loss_epoch_train[-1]},{loss_epoch_validate[-1]},{scheduler.get_last_lr()[0]}")   
        if epoch % checkpoint_saving_frequency == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss_epoch_train': loss_epoch_train,
                'loss_epoch_validate': loss_epoch_validate,
            }
            torch.save(checkpoint, os.path.join(result_dir, f'checkpoint_epoch{epoch}.pth'))
    
    end_time = time.time()
    training_time = end_time - start_time
    logger.write(f"Training time: {training_time} s")
    print(f"Training time: {training_time} s")


if __name__ == '__main__':
    """
    Trains a deep learning model based on the provided configuration.
    Args:
        configuration (dict): A dictionary containing the following keys:
            - root_dir (str): Root directory for the project.
            - result_dir (str): Directory to save the results.
            - case_name (str): Name of the case for saving results.
            - data_map_dir (str): Directory containing the data maps.
            - cuda_device (int): CUDA device number to use for training.
            - batch_size (int): Batch size for training.
            - num_epochs (int): Number of epochs to train the model.
            - accumulation_steps (int): Number of steps for gradient accumulation.
            - start_epoch (int): Epoch to start training from (for resuming training).
            - learning_rate (float): Learning rate for the optimizer.
            - checkpoint_saving_frequency (int): Frequency (in epochs) to save checkpoints.
            - use_scheduler (bool): Whether to use a learning rate scheduler.
    Returns:
        None
    """
    
    configuration = {
        'case_name': f'DrainNet_1c_{datetime.now().strftime("%Y_%m%d_%H%M")}',
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'cuda_device': 0,
        'root_dir': r'E:',
        'data_map_dir': r'E:\Data_map.json',
        'batch_size': 16,
        'accumulation_steps': 1,
        'num_epochs': 60,
        'start_epoch': 0,
        'learning_rate': 1e-3 ,
        'train_augmentation': False,
        'include_dem': False,
        'use_scheduler': False,
        'result_dir': r'Runs',
        'checkpoint_saving_frequency': 20,
        'note': 'note',
    }

    model_training(configuration)
