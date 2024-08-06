import pickle
import time
import numpy as np
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
from torchinfo import summary
from train.training_models.conv_lstm import ConvLSTM
from train.training_models.lstm5d import LSTM5D
from train.trainlog import Logger
from us101dataset import US101Dataset
from utils.train_utils import *


LOAD_INITIAL = False

def createModelAndTrain(
    full_dataset: US101Dataset,
    num_features: int = 1,
    test_ratio: float = 0.2,
    validation_ratio: float = 0.1,
    learning_rate: float = 0.0002,
    num_epochs: int = 50,
    batch_size: int = 16,
    num_skip: int = 0,
    model: str = 'lstm5d'
    ) -> None:

    params = { # parameters for the DataLoader
        'batch_size': batch_size, 
        'shuffle': False, 
        'drop_last': False, 
        'num_workers': 0
    }
    hyperparams = { 
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'num_layers': 1, # should be 1 because 5 lanes shrink to 3 lanes after one conv and we don't want it to shrink further
        'hidden_dim': 64,
        'num_skip': num_skip
    }

    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)

    with_ramp_sign = "w" if full_dataset.with_ramp else "wo"
    model_name = f"{model}/model_{with_ramp_sign}_{full_dataset.timewindow}_{full_dataset.num_sections}_{full_dataset.history_len}_{num_features}.best.pth"
    logger = Logger(model_name, hyperparams)

    initial_checkpoint = model_dir + "/" + model_name

    dataset_size = full_dataset.num_samples
    history_len = full_dataset.history_len
    predict_len = full_dataset.predict_len
    num_skip = full_dataset.num_skip

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


    # Uncomment to save the test dataset for inference 
    with open(f"test_data/testdata_{with_ramp_sign}_{full_dataset.timewindow}_{full_dataset.num_sections}_{full_dataset.history_len}_{num_features}.pkl", "wb") as f:
        pickle.dump(test_generator, f)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if model == "conv_lstm":
        model = ConvLSTM(input_dim=num_features, hidden_dim=hyperparams['hidden_dim'], num_layers=hyperparams['num_layers'])
    else:
        model = LSTM5D(input_dim=num_features, hidden_dim=hyperparams['hidden_dim'], num_layers=hyperparams['num_layers'])

    if LOAD_INITIAL:
        model.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)
    loss_fn.to(device)

    min_val_loss = float('inf')
    counter = 0
    patience = 3

    timer_start = time.time()
    print("Training Started")
    for e in range(num_epochs):
        for _, sample in enumerate(training_generator):
   
            X_batch = sample["x_data"].type(torch.FloatTensor).to(device)
            Y_batch = sample["y_data"].type(torch.FloatTensor).to(device)

            # Forward pass
            outputs, _ = model(X_batch) # shape of history_data: (batch_size, history/predict_len, num_features, num_lane, num_section)
            loss = loss_fn(outputs[:, history_len-predict_len:history_len, :, :, :], Y_batch)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        log = 'Epoch [{}/{}], Loss: {:.4f}'.format(e + 1, num_epochs, loss.item())
        print(log)
    
        val_loss = get_validation_loss(model, val_generator, loss_fn, device, history_len, predict_len)
        train_rmse = round(np.sqrt(loss.item()), 4)
        print('Mean validation loss: {:.4f}, Train RMSE: {:.4f}'.format(val_loss, train_rmse))
        elapsed_time = time.time() - timer_start
        logger.write("{},{},{:.2f},{:.4f},{:.4f},{:.4f}".format(e + 1, num_epochs, elapsed_time, loss.item(), val_loss, train_rmse))

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), initial_checkpoint)
            print('best model saved!')
        else:
            counter += 1
            if counter > patience:
                print('Early stopping at epoch:', e+1)
                break
    timer_end = time.time()

    model.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))
    model.eval()
    rmse_list = []
    mse_list = []
    mae_list = []
    for i, sample in enumerate(test_generator):
        X_batch = sample["x_data"].type(torch.FloatTensor).to(device)
        Y_batch = sample["y_data"].type(torch.FloatTensor).to(device)

        outputs, _ = model(X_batch)
        mse, mae, rmse = compute_errors(outputs[:, history_len-predict_len:history_len, :, :, :].cpu().data.numpy(),
                                        Y_batch.cpu().data.numpy())

        rmse_list.append(rmse)
        mse_list.append(mse)
        mae_list.append(mae)

    rmse = np.mean(rmse_list)
    mse = np.mean(mse_list)
    mae = np.mean(mae_list)

    training_time = timer_end - timer_start

    print("\n************************")
    print(f"Training time: {int(training_time)} sec")
    print(f"Test ConvLSTM model with US101 Dataset {with_ramp_sign} ramp:")
    final_res = 'Test (Scaled) mse: %.6f mae: %.6f rmse: %.6f' % (mse, mae, rmse)
    print(final_res)
    logger.write(final_res)
