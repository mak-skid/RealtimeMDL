import csv
import time
import keras
from keras.layers import Conv2D, ConvLSTM2D, Dense, Flatten, BatchNormalization, Input
from keras.models import Model
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from us101dataset import US101Dataset


def get_ConvLSTM_model(history_len, num_lanes, num_sections):
    speed_input = Input(shape=(history_len, num_lanes, num_sections, 1)) # history, lanes, sections, features
    speed_input1 = BatchNormalization()(speed_input)
    layer1 = ConvLSTM2D(
        filters=5,
        kernel_size=(3, 3),
        padding='same',
        kernel_regularizer = keras.regularizers.l2(0.01),
        bias_regularizer = keras.regularizers.l2(0.01),
        data_format='channels_last', 
        return_sequences=False)(speed_input1)
    flat1 = Flatten()(layer1)

    out = Dense(num_lanes*num_sections)(flat1) # lanes, sections, pred_len
    return Model(inputs=[speed_input], outputs=out)


def visualise_convlstm_output(pred, real, timestamp: int, model_name: str, feature_name: str, num_lanes: int = 5, num_sections: int = 10):
    lanes = [str(i+1) for i in range(num_lanes)]
    sections = [i for i in range(num_sections)]

    #fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(35, 10))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
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

    ax1.set_title(f"Predicted Average {feature_name} by Section and Lane at {timestamp} seconds")
    ax2.set_title(f"Real Average {feature_name} by Section and Lane at {timestamp} seconds")
    #ax3.set_title(f"{mat_type} Average acceleration by Section and Lane at t={timestamp}")

    plt.subplots_adjust(hspace=0.4)
    plt.savefig(f"training/predict_output_figs/{model_name}_{feature_name}_output_{timestamp}.png")
    plt.show()


def createConvLSTMModelAndTrain(
    train_dataset: US101Dataset,
    num_features: int = 1,
    learning_rate: float = 0.001, # 0.0002
    num_epochs: int = 500,
    batch_size: int = 16,
    num_skip: int = 20,
    realtime_mode: bool = True
):
    dataset_size = train_dataset.num_samples
    history_len = train_dataset.history_len
    predict_len = train_dataset.predict_len
    num_skip = train_dataset.num_skip
    with_ramp_sign = "w" if train_dataset.with_ramp else "wo"
    num_lanes = train_dataset.num_lanes
    num_sections = train_dataset.num_sections

    optimizer = keras.optimizers.Adamax(learning_rate=learning_rate)
    model = get_ConvLSTM_model(history_len, num_lanes, num_sections)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    model.summary()
    model_name = f'convlstm_model_{with_ramp_sign}_{train_dataset.timewindow}_{num_sections}_{train_dataset.history_len}_{num_features}_{num_skip}'
    csv_logger = keras.callbacks.CSVLogger(f"training/log/{model_name}.log")
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)

    if not realtime_mode:
        test_ratio = 0.2
        validation_ratio = 0.1
        val_split = int(np.floor((1 - (validation_ratio + test_ratio)) * dataset_size))
        test_split = int(np.floor((1 - test_ratio) * dataset_size))

        x_vel_train = train_dataset.X_data[:val_split, :, :, :, 0]
        x_vel_val = train_dataset.X_data[val_split:test_split, :, :, :, 0]
        x_vel_test = train_dataset.X_data[test_split:, :, :, :, 0]
        
        x_dens_train = train_dataset.X_data[:val_split, :, :, :, 1]
        x_dens_val = train_dataset.X_data[val_split:test_split, :, :, :, 1]
        x_dens_test = train_dataset.X_data[test_split:, :, :, :, 1]
        
        x_acc_train = train_dataset.X_data[:val_split, :, :, :, 2]
        x_acc_val = train_dataset.X_data[val_split:test_split, :, :, :, 2]
        x_acc_test = train_dataset.X_data[test_split:, :, :, :, 2]

        y_vel_train = np.reshape(train_dataset.Y_data[:val_split, :, :, :, 0], (val_split, predict_len, num_lanes*num_sections))
        y_vel_val = np.reshape(train_dataset.Y_data[val_split:test_split, :, :, :, 0], (test_split - val_split, predict_len, num_lanes*num_sections))
        y_vel_test = np.reshape(train_dataset.Y_data[test_split:, :, :, :, 0], (dataset_size - test_split, predict_len, num_lanes*num_sections))

    else:
        validation_ratio: float = 0.125
        val_split = int(np.floor((1 - validation_ratio) * dataset_size))
        
        x_vel_train = train_dataset.X_data[:val_split, :, :, :, 0]
        x_vel_val = train_dataset.X_data[val_split:, :, :, :, 0]
        x_dens_train = train_dataset.X_data[:val_split, :, :, :, 1]
        x_dens_val = train_dataset.X_data[val_split:, :, :, :, 1]
        x_acc_train = train_dataset.X_data[:val_split, :, :, :, 2]
        x_acc_val = train_dataset.X_data[val_split:, :, :, :, 2]
        
        y_vel_val = np.reshape(train_dataset.Y_data[val_split:, :, :, :, 0], (dataset_size - val_split, predict_len, num_lanes*num_sections))
        y_vel_train = np.reshape(train_dataset.Y_data[:val_split, :, :, :, 0], (val_split, predict_len, num_lanes*num_sections))

    start = time.time()
    train_history = model.fit(x_vel_train, y_vel_train, epochs=num_epochs, batch_size=batch_size, verbose=2, validation_data=(x_vel_val, y_vel_val), callbacks=[early_stop, csv_logger])
    loss = train_history.history['loss']
    val_loss = train_history.history['val_loss']
    #model.save(f"mdl/models/{model_name}.keras", overwrite=True)
    model.save_weights(f"training/models/{model_name}.weights.h5", overwrite=True)
    end = time.time()

    print(f"Time taken to train: {end-start} seconds")
    plt.plot(train_history.history['loss'], label='train')
    plt.plot(train_history.history['val_loss'], label='test')
    plt.legend()
    plt.show()

    if not realtime_mode:
        y_pred = model.predict(x_vel_test)
        print(y_pred.shape, y_vel_test.shape)
        num_test_samples = y_vel_test.shape[0]

        y_vel_test = y_vel_test.cpu().data.numpy()
        y_vel_test = np.reshape(y_vel_test, (num_test_samples, num_lanes*num_sections))
        mse = mean_squared_error(y_vel_test, y_pred)
        mae = mean_absolute_error(y_vel_test, y_pred)
        mape = mean_absolute_percentage_error(y_vel_test, y_pred)
        maape = np.average(np.arctan(np.abs(y_vel_test - y_pred) / np.abs(y_vel_test)))
        rmse = np.sqrt(mse)

        print(f"Model: {model_name}, MAE: {mae}, MAPE: {mape}, MAAPE:{maape}, RMSE: {rmse}")
        
        with open(f"training/convlstm_results.csv", "a") as f:
            fields = ["model_name", "mae", "mape", "maape", "rmse"]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writerow({"model_name": model_name, "mae": mae, "mape": mape, "maape": maape, "rmse": rmse})

        y_pred = np.reshape(y_pred, (num_test_samples, num_lanes, num_sections))
        y_vel_test = np.reshape(y_vel_test, (num_test_samples, num_lanes, num_sections))

        start = 2229.5
        timestamp = start + (history_len + num_skip) * train_dataset.timewindow
        visualise_convlstm_output(y_pred[0], y_vel_test[0], timestamp, model_name, "velocity", num_lanes, num_sections)
    

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()