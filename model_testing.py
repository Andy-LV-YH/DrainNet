import os
import torch
import pandas as pd
import numpy as np
from utils import test_data_loaders, val_data_loaders
from DrainNet import DrainNet
from sklearn.metrics import mean_absolute_error, jaccard_score, f1_score
from ptflops import get_model_complexity_info


def calculate_metrics(pred, target):
    pred = pred.cpu().numpy().flatten()
    target = target.cpu().numpy().flatten() 
    print(pred, target)
    # Apply threshold to predictions
    pred_binary = (pred > 0.5).astype(int)
    mae = mean_absolute_error(target, pred) 
    iou = jaccard_score(target, pred_binary, average='macro')
    dice = f1_score(target, pred_binary, average='macro')

    return mae, iou, dice


def model_testing(root_dir, data_map_dir, case_name, include_dem=True, mode='test'):
    check_point_dir = os.path.join(root_dir, r'Runs', case_name)
    if mode == 'test':
        data_dataloader = test_data_loaders(data_map_dir, mode='test', included_dem=include_dem, batch_size=1)
    if mode == 'val':
        data_dataloader = val_data_loaders(data_map_dir, mode='val', included_dem=include_dem, batch_size=1)
    checkpoint = torch.load(os.path.join(check_point_dir, f'checkpoint.pth'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = case_name.split('_')[0]

    model_classes = {
        'DrainNet': DrainNet,
    }

    if model_name not in model_classes:
        raise ValueError(f"Model {model_name} not found.")
    
    model_class = model_classes[model_name]

    if include_dem:
        model= model_class(2, 1).to(device)

    else:
        model = model_class(1, 1).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    metrics = {'MAE': [], 'Mean IoU': [], 'Mean Dice': [], 'MACs': [], 'Params': []}
    with torch.no_grad():
        for inp, mask in data_dataloader:
            inp, mask = inp.to(device), mask.to(device)
            out = model(inp)
            out = torch.sigmoid(out)
            mae, iou, dice = calculate_metrics(out, mask)
            if include_dem:
                macs, params = get_model_complexity_info(model, (2, 448, 448), as_strings=True, print_per_layer_stat=True, verbose=True, backend='pytorch')
            else:
                macs, params = get_model_complexity_info(model, (1, 448, 448), as_strings=True, print_per_layer_stat=True, verbose=True, backend='pytorch')

            metrics['MAE'].append(mae)
            metrics['Mean IoU'].append(iou)
            metrics['Mean Dice'].append(dice)

    metrics['MAE'] = np.mean(metrics['MAE'])
    metrics['Mean IoU'] = np.mean(metrics['Mean IoU'])
    metrics['Mean Dice'] = np.mean(metrics['Mean Dice'])
    metrics['MACs'] = macs
    metrics['Params'] = params

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(check_point_dir, f'{mode}_Test_metrics.csv'), index=False)

if __name__ == '__main__':
    root_dir =  r'E:'
    data_map_dir = r'E:\Data_map.json'
    case_name = 'DrainNet_1c_2025_0117_2226'
    model_testing(root_dir, data_map_dir, case_name, include_dem=False, mode='test')