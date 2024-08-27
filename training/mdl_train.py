import time
import keras
from keras.layers import Conv2D, ConvLSTM2D, Dense, Flatten, BatchNormalization, Input
from keras.models import Model
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from sklearn.metrics import mean_squared_error
from training.mdl_predict import mdl_predict
from us101dataset import US101Dataset
from training.mdl_model import get_MDL_model


def createMDLModelAndTrain(
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
    model = get_MDL_model(history_len, num_lanes, num_sections)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    model.summary()
    model_name = f'mdl_model_{with_ramp_sign}_{train_dataset.timewindow}_{num_sections}_{train_dataset.history_len}_{num_features}_{num_skip}'
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
        model = get_MDL_model(history_len, num_lanes, num_sections)
        mdl_predict(model_name, model, x_vel_test, y_vel_test, history_len, num_skip, num_lanes, num_sections)
    

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
