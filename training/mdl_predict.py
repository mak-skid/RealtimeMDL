"""
Static prediction using the MDL model and visualise the output

Author: Makoto Ono (Based on Lu. et al (2022))
"""
import csv
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

from training.mdl_model import get_MDL_model
from us101dataset import US101Dataset


def mdl_predict(model_name: str, model, x_test, y_test, history_len, num_skip, num_lanes, num_sections):
    model.load_weights(f"training/models/{model_name}.weights.h5")
    y_pred = model.predict(x_test)
    print(y_pred.shape, y_test.shape)
    num_test_samples = y_test.shape[0]

    y_test = y_test.cpu().data.numpy()
    y_test = np.reshape(y_test, (num_test_samples, num_lanes*num_sections))
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    maape = np.average(np.arctan(np.abs(y_test - y_pred) / np.abs(y_test)))
    rmse = np.sqrt(mse)

    print(f"Model: {model_name}, MAE: {mae}, MAPE: {mape}, MAAPE:{maape}, RMSE: {rmse}")
    
    with open(f"training/mdl_results.csv", "a") as f:
        fields = ["model_name", "mae", "mape", "maape", "rmse"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writerow({"model_name": model_name, "mae": mae, "mape": mape, "maape": maape, "rmse": rmse})

    y_pred = np.reshape(y_pred, (num_test_samples, num_lanes, num_sections))
    y_test = np.reshape(y_test, (num_test_samples, num_lanes, num_sections))

    start = 2229.5
    timewindow = 0.5
    timestamp = start + (history_len + num_skip) * timewindow
    visualise_mdl_output(y_pred[0], y_test[0], timestamp, model_name, "velocity", num_lanes, num_sections)

def visualise_mdl_output(pred, real, timestamp: int, model_name: str, feature_name: str, num_lanes: int = 5, num_sections: int = 10):
    lanes = [str(i+1) for i in range(num_lanes)]
    sections = [i for i in range(num_sections)]

    #fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(35, 10))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    im1 = ax1.imshow(pred, cmap='turbo_r', norm=Normalize(vmin=0, vmax=60))
    im2 = ax2.imshow(real, cmap='turbo_r', norm=Normalize(vmin=0, vmax=60))
    #im2 = ax2.imshow(dens_matrix, cmap='turbo', norm=Normalize(vmin=100, vmax=300))
    #im3 = ax3.imshow(acc_matrix, cmap='turbo_r', norm=Normalize(vmin=-3, vmax=3))

    ax1.set_yticks(np.arange(len(lanes)), labels=lanes)
    ax1.set_xticks(np.arange(len(sections)), labels=sections)

    ax2.set_yticks(np.arange(len(lanes)), labels=lanes)
    ax2.set_xticks(np.arange(len(sections)), labels=sections)

    #ax3.set_yticks(np.arange(len(lanes)), labels=lanes)
    #ax3.set_xticks(np.arange(len(sections)), labels=sections)

    plt.setp(ax1.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax2.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")        
    #plt.setp(ax3.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor") 

    for i in range(len(lanes)): # y axis
        for j in range(len(sections)): # x axis
            text = ax1.text(j, i, round(pred[i][j], 2), ha="center", va="center", color="w")
            text = ax2.text(j, i, f"{real[i][j]:.2f}", ha="center", va="center", color="w")
            #text = ax3.text(j, i, acc_matrix[i][j], ha="center", va="center", color="w")

    ax1.set_title(f"Predicted Average {feature_name} by Section and Lane at {timestamp} seconds")
    ax2.set_title(f"Real Average {feature_name} by Section and Lane at {timestamp} seconds")
    #ax3.set_title(f"{mat_type} Average acceleration by Section and Lane at t={timestamp}")

    plt.subplots_adjust(hspace=0.4)
    plt.savefig(f"training/predict_output_figs/{model_name}_{feature_name}_output_{timestamp}.png")
    plt.show()