import numpy as np
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
from conv_lstm import ConvLSTM
from us101dataset import US101Dataset

LOAD_INITIAL = False

def createModelAndTrain(
    full_dataset: US101Dataset,
    test_ratio: float = 0.2,
    validation_ratio: float = 0.1,
    learning_rate: float = 0.0002,
    num_features: int = 3,
    num_epochs: int = 100,
    batch_size: int = 11,
    ) -> None:

    params = { # parameters for the DataLoader
        'batch_size': batch_size, 
        'shuffle': False, 
        'drop_last': False, 
        'num_workers': 0
    }

    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    initial_checkpoint = model_dir + '/model.best.pth'

    dataset_size = full_dataset.num_samples
    history_len = full_dataset.history_len

    indices = list(range(dataset_size))
    val_split = int(np.floor((1 - (validation_ratio + test_ratio)) * dataset_size))
    test_split = int(np.floor((1 - test_ratio) * dataset_size))
    train_indices, val_indices, test_indices = indices[:val_split], indices[val_split:test_split], indices[test_split:]
    
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    training_generator = DataLoader(full_dataset, **params, sampler=train_sampler)
    val_generator = DataLoader(full_dataset, **params, sampler=valid_sampler)
    test_generator = DataLoader(full_dataset, **params, sampler=test_sampler)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = ConvLSTM(input_dim=num_features, hidden_dim=[64, 64, 3], num_layers=3)

    if LOAD_INITIAL:
        model.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)
    loss_fn.to(device)

    min_val_loss = None
    for e in range(num_epochs):
        for i, sample in enumerate(training_generator):
            X_batch = sample["x_data"].type(torch.FloatTensor).to(device)
            Y_batch = sample["y_data"].type(torch.FloatTensor).to(device)

            # Forward pass
            outputs, _ = model(X_batch) # shape of history_data: (num_samples, history/predict_len, num_features, num_lane, num_section)
            loss = loss_fn(outputs[:, history_len - 1:history_len, :, :, :], Y_batch)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch [{}/{}], Loss: {:.4f}'.format(e + 1, num_epochs, loss.item()))

        val_loss = get_validation_loss(model, val_generator, loss_fn, device, history_len)
        print('Mean validation loss:', val_loss)

        if min_val_loss == None or val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(), initial_checkpoint)
            print('best model saved!')

    model.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))
    model.eval()
    rmse_list = []
    mse_list = []
    mae_list = []
    for i, sample in enumerate(test_generator):
        X_batch = sample["x_data"].type(torch.FloatTensor).to(device)
        Y_batch = sample["y_data"].type(torch.FloatTensor).to(device)

        outputs, _ = model(X_batch)
        mse, mae, rmse = compute_errors(outputs[:, history_len - 1:history_len, :, :, :].cpu().data.numpy(),
                                        Y_batch.cpu().data.numpy())

        rmse_list.append(rmse)
        mse_list.append(mse)
        mae_list.append(mae)

    rmse = np.mean(rmse_list)
    mse = np.mean(mse_list)
    mae = np.mean(mae_list)

    print("\n************************")
    print("Test ConvLSTM model with US101 Dataset:")
    print('Test mse: %.6f mae: %.6f rmse (norm): %.6f' % (mse, mae, rmse))


def compute_errors(preds, y_true):
    pred_mean = preds[:, 0:2]
    diff = y_true - pred_mean

    mse = np.mean(diff ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(diff))

    return mse, mae, rmse


def get_validation_loss(model, val_generator, criterion, device, history_len):
    model.eval()
    mean_loss = []
    for i, sample in enumerate(val_generator):
        X_batch = sample["x_data"].type(torch.FloatTensor).to(device)
        Y_batch = sample["y_data"].type(torch.FloatTensor).to(device)

        outputs, _ = model(X_batch)
        mse = criterion(outputs[:, history_len-1:history_len, :, :, :], Y_batch).item()
        mean_loss.append(mse)

    mean_loss = np.mean(mean_loss)
    return mean_loss
