
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from sklearn.metrics import mean_squared_error


def mdl_predict(model_name: str, model, full_dataset, history_len, num_skip):
    y_pred = model.predict(x_vel_test)
    print(y_pred.shape, y_test.shape)
    num_test_samples = y_test.shape[0]

    model = model.save(f'models/mdl/{model_name}.keras')
    print('Model saved')
    y_test = y_test.cpu().data.numpy()
    y_test = np.reshape(y_test, (num_test_samples, 105))
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f"RMSE: {rmse}")
    y_pred = np.reshape(y_pred, (num_test_samples, 5, 21))
    y_test = np.reshape(y_test, (num_test_samples, 5, 21))
    visualise_mdl_output(y_pred[0], y_test[0], full_dataset.start+history_len+num_skip, 20, model_name, "velocity")

def visualise_mdl_output(pred, real, timestamp: int, num_section_split: int, model_name: str, feature_name: str):
    lanes = ["1", "2", "3", "4", "5"]
    sections = [i for i in range(num_section_split+1)]

    #fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(35, 10))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10))
    im1 = ax1.imshow(pred, cmap='turbo_r', norm=Normalize(vmin=0, vmax=60))
    im2 = ax2.imshow(real, cmap='turbo_r', norm=Normalize(vmin=0, vmax=60))
    #im2 = ax2.imshow(dens_matrix, cmap='turbo', norm=Normalize(vmin=100, vmax=300))
    #im3 = ax3.imshow(acc_matrix, cmap='turbo_r', norm=Normalize(vmin=-3, vmax=3))

    # Show all ticks and label them with the respective list entries
    ax1.set_yticks(np.arange(len(lanes)), labels=lanes)
    ax1.set_xticks(np.arange(len(sections)), labels=sections)

    ax2.set_yticks(np.arange(len(lanes)), labels=lanes)
    ax2.set_xticks(np.arange(len(sections)), labels=sections)

    #ax3.set_yticks(np.arange(len(lanes)), labels=lanes)
    #ax3.set_xticks(np.arange(len(sections)), labels=sections)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax1.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax2.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")        
    #plt.setp(ax3.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor") 

    # Loop over data dimensions and create text annotations.
    for i in range(len(lanes)): # y axis
        for j in range(len(sections)): # x axis
            text = ax1.text(j, i, round(pred[i][j], 2), ha="center", va="center", color="w")
            text = ax2.text(j, i, f"{real[i][j]:.2f}", ha="center", va="center", color="w")
            #text = ax3.text(j, i, acc_matrix[i][j], ha="center", va="center", color="w")

    ax1.set_title(f"Predicted Average {feature_name} by Section and Lane at t={timestamp}")
    ax2.set_title(f"Real Average {feature_name} by Section and Lane at t={timestamp}")
    #ax3.set_title(f"{mat_type} Average acceleration by Section and Lane at t={timestamp}")

    plt.subplots_adjust(hspace=0.4)
    plt.savefig(f"predict_output_figs/{model_name}_{feature_name}_output_{timestamp}.png")
    plt.show()