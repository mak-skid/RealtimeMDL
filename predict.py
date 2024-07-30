import pickle
import time
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import torch

from conv_lstm import ConvLSTM
from utils.datapreprocessing_utils import tensor_to_np_matrices
from utils.train_utils import compute_errors


def predict() -> None:
    history_len = 20
    predict_len = 1
    timewindow = 0.5
    num_skip = 20
    num_features = 1
    params = f'wo_0.5_21_20_{num_features}'
    test_generator = pickle.load(open(f"test_data/testdata_{params}.pkl", "rb"))
    model = ConvLSTM(input_dim=num_features, hidden_dim=64, num_layers=1)
    model.eval()
    device = torch.device("cpu")

    scale = {
        'min(avg(v_Vel))': 0.0,
        'max(avg(v_Vel))': 95.3, 
        'min(avg(v_Acc))': -11.2, 
        'max(avg(v_Acc))': 11.2, 
        'min(count)': 0, 
        'max(count)': 5
    }
    def descaler(data, col):
        return (data + scale[f"min({col})"]) * (scale[f"max({col})"] - scale[f"min({col})"])

    for i, sample in enumerate(test_generator):
        start = time.time()
        X_batch = sample["x_data"].type(torch.FloatTensor).to(device)
        Y_batch = sample["y_data"].type(torch.FloatTensor).to(device)
        outputs, _ = model(X_batch)
        descaled_outputs = outputs[:, history_len-predict_len:history_len]
        for i, f in enumerate(["avg(v_Vel)", "count", "avg(v_Acc)"]):
            break
            descaled_outputs[:, :, i] = descaler(descaled_outputs[:, :, i], f)
            #Y_batch[:, :, i] = descaler(Y_batch[:, :, i], f)
        end = time.time()
        print(f"Time taken to predict: {end-start} seconds") 
        mse, mae, rmse = compute_errors(descaled_outputs.cpu().data.numpy(),
                                        Y_batch.cpu().data.numpy())
        print(f"Real mse: {mse}, mae: {mae}, rmse: {rmse}")

        visualise_output(descaled_outputs[0][0], history_len*timewindow, 20, "Predicted")
        visualise_output(Y_batch[0][0], (history_len+num_skip)*timewindow, 20, "Actual")
        """
        for j in batch_size:
            visualise_output(outputs[j][history_len], j+history_len*i, 20, "Predicted")
            visualise_output(Y_batch[j][0], i, 20, "Actual")
            break
        """
        break
        

def visualise_output(outputT, timestamp: int, num_section_split: int, mat_type: str) -> None:
    vel_matrix, dens_matrix, acc_matrix = tensor_to_np_matrices(outputT)
    
    lanes = ["1", "2", "3", "4", "5", "6 (Ramp)"] if len(vel_matrix) else ["1", "2", "3", "4", "5"]
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

    ax1.set_title(f"{mat_type} Average Velocity (mph) by Section and Lane at t={timestamp}")
    ax2.set_title(f"{mat_type} Density by Section and Lane at t={timestamp}")
    ax3.set_title(f"{mat_type} Average acceleration by Section and Lane at t={timestamp}")

    plt.subplots_adjust(hspace=0.4)
    plt.savefig(f"predict_output_figs/{mat_type}_output_{timestamp}.png")
    plt.show()

predict()