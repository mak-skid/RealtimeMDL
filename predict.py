import pickle
import time
from matplotlib import pyplot as plt
import numpy as np
import torch

from conv_lstm import ConvLSTM
from utils.datapreprocessing_utils import tensor_to_np_matrices


def predict() -> None:
    params = 'wo_0.5_21_20_3'
    test_generator = pickle.load(open(f"test_data/testdata_{params}.pkl", "rb"))
    model = ConvLSTM(input_dim=3, hidden_dim=[64, 64, 3], num_layers=3)
    model.load_state_dict(torch.load(f"models/model_{params}.best.pth"))
    model.eval()
    device = torch.device("cpu")
    for sample in test_generator:
        start = time.time()
        X_batch = sample["x_data"].type(torch.FloatTensor).to(device)
        Y_batch = sample["y_data"].type(torch.FloatTensor).to(device)
        print(X_batch.shape)
        output, _ = model(X_batch)
        end = time.time()
        print(f"Time taken to predict: {end-start} seconds")
        break

def visualise_output(outputT: torch.Tensor) -> None:
    vel_matrix, dens_matrix, acc_matrix = tensor_to_np_matrices(outputT)
    
    lanes = ["1", "2", "3", "4", "5", "6 (Ramp)"] if with_ramp else ["1", "2", "3", "4", "5"]
    sections = [i for i in range(num_section_split+1)]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(35, 10))
    im1 = ax1.imshow(vel_matrix, cmap='turbo_r', norm=Normalize(vmin=0, vmax=60))
    im2 = ax2.imshow(dens_matrix, cmap='turbo', norm=Normalize(vmin=100, vmax=300))
    im3 = ax3.imshow(acc_matrix, cmap='turbo_r', norm=Normalize(vmin=-3, vmax=3))

    # Show all ticks and label them with the respective list entries
    ax1.set_yticks(np.arange(len(lanes)), labels=lanes)
    ax1.set_xticks(np.arange(len(sections)), labels=sections)

    ax2.set_yticks(np.arange(len(lanes)), labels=lanes)
    ax2.set_xticks(np.arange(len(sections)), labels=sections)

    ax3.set_yticks(np.arange(len(lanes)), labels=lanes)
    ax3.set_xticks(np.arange(len(sections)), labels=sections)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax1.get_yticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")
    plt.setp(ax2.get_yticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")        
    plt.setp(ax3.get_yticklabels(), rotation=45, ha="right",
            rotation_mode="anchor") 

    # Loop over data dimensions and create text annotations.
    for i in range(len(lanes)): # y axis
        for j in range(len(sections)): # x axis
            text = ax1.text(j, i, vel_matrix[i][j],
                        ha="center", va="center", color="w")
            text = ax2.text(j, i, int(dens_matrix[i][j]),
                            ha="center", va="center", color="w")
            text = ax3.text(j, i, acc_matrix[i][j],
                            ha="center", va="center", color="w")

    ax1.set_title(f"Average Velocity (mph) by Section and Lane at t={timestamp}")
    ax2.set_title(f"Density by Section and Lane at t={timestamp}")
    ax3.set_title(f"Average acceleration by Section and Lane at t={timestamp}")

    plt.subplots_adjust(hspace=0.4)
    plt.show()

predict()