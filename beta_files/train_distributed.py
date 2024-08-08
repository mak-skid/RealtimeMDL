import time
import numpy as np
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
from lstms.conv_lstm.conv_lstm import ConvLSTM
from train.trainlog import Logger
from us101dataset import US101Dataset
from utils.lstm_train_utils import *

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def createDistributedModelAndTrain(
    full_dataset: US101Dataset,
    test_ratio: float = 0.2,
    validation_ratio: float = 0.1,
    learning_rate: float = 0.0002,
    num_features: int = 3,
    num_epochs: int = 50,
    batch_size: int = 64,
    ) -> None:

    print("Running distributed training")
    dist.init_process_group("nccl")

    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    #### Adjust batch size per worker ####
    adjusted_batch_size = batch_size // world_size
    print(f"{adjusted_batch_size=}")

    params = {
        "batch_size": adjusted_batch_size if global_rank == 0 else batch_size,
        "epochs": num_epochs,
        "trainer": "TorchDistributor",
    }

    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    with_ramp_sign = "w" if full_dataset.with_ramp else "wo"
    model_name = f"distributed/model_{with_ramp_sign}_{full_dataset.timewindow}_{full_dataset.num_sections}_{full_dataset.history_len}.best.pth"
    initial_checkpoint = model_dir + "/" + model_name
    logger = Logger(model_name)

    
    dataset_size = full_dataset.num_samples
    history_len = full_dataset.history_len
    predict_len = full_dataset.predict_len

    indices = list(range(dataset_size))
    val_split = int(np.floor((1 - (validation_ratio + test_ratio)) * dataset_size))
    test_split = int(np.floor((1 - test_ratio) * dataset_size))
    train_indices, val_indices, test_indices = indices[:val_split], indices[val_split:test_split], indices[test_split:]
    
    train_dataset = Subset(full_dataset, train_indices)
    valid_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)

    train_sampler = DistributedSampler(train_dataset)
    valid_sampler = DistributedSampler(valid_dataset)
    test_sampler = DistributedSampler(test_dataset)

    training_generator = DataLoader(train_dataset, **params, sampler=train_sampler)
    val_generator = DataLoader(valid_dataset, **params, sampler=valid_sampler)
    test_generator = DataLoader(test_dataset, **params, sampler=test_sampler)
    

    model = ConvLSTM(input_dim=num_features, hidden_dim=[64, 64, 3], num_layers=3)

    model.to(local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    
    min_val_loss = float('inf')
    counter = 0
    patience = 3
    timer_start = time.time()

    print("Training Started")
    for e in range(num_epochs):
        for _, sample in enumerate(training_generator):
            X_batch = sample["x_data"].type(torch.FloatTensor).to(local_rank)
            Y_batch = sample["y_data"].type(torch.FloatTensor).to(local_rank)

            # Forward pass
            outputs, _ = model(X_batch) # shape of history_data: (num_samples, history/predict_len, num_features, num_lane, num_section)
            loss = loss_fn(outputs[:, history_len-predict_len:history_len, :, :, :], Y_batch)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        log = 'Epoch [{}/{}], Loss: {:.4f}'.format(e + 1, num_epochs, loss.item())
        print(log)
    
        val_loss = get_validation_loss(model, val_generator, loss_fn, local_rank, history_len, predict_len)
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    final_res = 'Test mse: %.6f mae: %.6f rmse (norm): %.6f' % (mse, mae, rmse)
    print(final_res)
    logger.write(final_res)

