import time
import keras
from keras.layers import Conv2D, ConvLSTM2D, Dense, Flatten, BatchNormalization, Input
from keras.models import Model
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from sklearn.metrics import mean_squared_error

from us101dataset import US101Dataset

def createMDLModelAndTrain(
    full_dataset: US101Dataset,
    num_features: int = 1,
    test_ratio: float = 0.2,
    validation_ratio: float = 0.1,
    learning_rate: float = 0.0002,
    num_epochs: int = 50,
    batch_size: int = 16,
    num_skip: int = 0
):
    dataset_size = full_dataset.num_samples
    history_len = full_dataset.history_len
    predict_len = full_dataset.predict_len
    num_skip = full_dataset.num_skip

    val_split = int(np.floor((1 - (validation_ratio + test_ratio)) * dataset_size))
    test_split = int(np.floor((1 - test_ratio) * dataset_size))
    
    x_train, x_val, x_test = full_dataset.X_data[:val_split], full_dataset.X_data[val_split:test_split], full_dataset.X_data[test_split:]
    y_train, y_val, y_test = full_dataset.Y_data[:val_split], full_dataset.Y_data[val_split:test_split], full_dataset.Y_data[test_split:]

    speed_input = Input(shape=(20, 1, 5, 21)) # history, features, lanes, sections
    speed_input1 = BatchNormalization()(speed_input)
    layer1 = ConvLSTM2D(
        filters=1,
        kernel_size=(3, 3),
        padding='same',
        kernel_regularizer = keras.regularizers.l2(0.01),
        bias_regularizer = keras.regularizers.l2(0.01),
        data_format='channels_first', 
        return_sequences=False)(speed_input1)
    layer1 = BatchNormalization()(layer1)
    layer2 = Conv2D(
        filters=1,
        kernel_size=(3, 3),
        data_format='channels_first',
        kernel_regularizer = keras.regularizers.l2(0.01),
        bias_regularizer = keras.regularizers.l2(0.01),
        activation='relu',
        padding='same')(layer1)
    flat1 = Flatten()(layer2)
    # merged_output = keras.layers.concatenate([flat1, flat2])
    out = Dense(5*21*1)(flat1) # 5 lanes, 21 sections, 1 feature
    model = Model(inputs=[speed_input], outputs=out)
    model.compile(loss='mean_squared_error', optimizer='Adamax')
    
    start = time.time()
    train_history = model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size, verbose=2, validation_data=(x_val, y_val))
    loss = train_history.history['loss']
    val_loss = train_history.history['val_loss']
    end = time.time()

    print(f"Time taken to train: {end-start} seconds")
    plt.plot(train_history.history['loss'], label='train')
    plt.plot(train_history.history['val_loss'], label='test')
    plt.legend()
    plt.show()

    y_pred = model.predict(x_test)
    print(y_pred.shape, y_test.shape)
    model = model.save('models/mdl/mdl_model.keras')
    print('Model saved')
    print('MSE: ', mean_squared_error(y_test, y_pred))
    visualise_mdl_output(y_pred, y_test, 20)


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

def visualise_mdl_output(pred, real, timestamp: int, num_section_split: int):
    lanes = ["1", "2", "3", "4", "5"]
    sections = [i for i in range(num_section_split+1)]

    #fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(35, 10))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(35, 10))
    im1 = ax1.imshow(pred, cmap='turbo_r', norm=Normalize(vmin=0, vmax=60))
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
            text = ax1.text(j, i, pred[i][j],ha="center", va="center", color="w")
            text = ax2.text(j, i, real[i][j], ha="center", va="center", color="w")
            #text = ax3.text(j, i, acc_matrix[i][j], ha="center", va="center", color="w")

    ax1.set_title(f"Predicted Average Velocity (mph) by Section and Lane at t={timestamp}")
    ax2.set_title(f"Real Average Velocity by Section and Lane at t={timestamp}")
    #ax3.set_title(f"{mat_type} Average acceleration by Section and Lane at t={timestamp}")

    plt.subplots_adjust(hspace=0.4)
    plt.savefig(f"predict_output_figs/mdl_output_{timestamp}.png")
    plt.show()
