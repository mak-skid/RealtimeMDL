import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from us101dataset import US101Dataset


def compute_errors(preds, y_true):
    pred_mean = preds[:, 0:2]
    diff = y_true - pred_mean

    mse = np.mean(diff ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(diff))

    return mse, mae, rmse

def get_validation_loss(model, val_generator, criterion, device, history_len, predict_len):
    model.eval()
    mean_loss = []
    for i, sample in enumerate(val_generator):
        X_batch = sample["x_data"].type(torch.FloatTensor).to(device)
        Y_batch = sample["y_data"].type(torch.FloatTensor).to(device)

        outputs, _ = model(X_batch)
        mse = criterion(outputs[:, history_len-predict_len:history_len, :, :, :], Y_batch).item()
        mean_loss.append(mse)

    mean_loss = np.mean(mean_loss)
    return mean_loss

### below are for Distributed Training ###

def get_distributed_train_loader(batch_size):
    train_dataset = US101Dataset()
    train_sampler = DistributedSampler(shuffle=False)
    data_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
    )
    return data_loader