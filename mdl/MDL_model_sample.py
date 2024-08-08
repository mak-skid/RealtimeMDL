import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
import pandas as pd
import numpy as np
import keras
from keras.layers import Input, Dense, Dropout, Concatenate,Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, ConvLSTM2D, Conv3D
from keras.models import Sequential, Model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from keras import regularizers
import math
import time

#-------Set the prediction horizon and input time window--------------------------------------------
look_back=10
look_ahead=1
#--------------Create the input data set------------------------------------------------------------
train_speed= create_dataset_input(speed, look_back, look_ahead)
train_volume= create_dataset_input(volume, look_back, look_ahead)
test_speed=create_dataset_output(speed, look_back, look_ahead)
#the dataset was divided into two parts: the training dataset and the testing dataset
train_size = int(len(train_speed) * 0.75)
print(train_speed.shape)
print(test_speed.shape)
X1=train_speed[0:train_size,:]
X2=train_volume[0:train_size,:]
Y=test_speed[0:train_size,:]
#y = np.reshape(Y,(test_size-look_back, 15))
y=Y
X1_test=train_speed[train_size:,:]
X2_test=train_volume[train_size:,:]
Y_test=test_speed[train_size:,:]
#y_test=np.reshape(Y_test,(test_size-look_back,15))
y_test=Y_test
#------------learn spatio-temporal feature from the speed data-----------------------------------------
speed_input = Input(shape=(look_back, 9, 4, 1)) # 9 sections, 4 lanes, 1 feature
speed_input1 = BatchNormalization()(speed_input)
layer1 = ConvLSTM2D(filters=10, kernel_size=(3, 3),padding='same',kernel_regularizer = regularizers.l2(0.01), bias_regularizer = regularizers.l2(0.01),data_format='channels_last', return_sequences=False)(speed_input1)
layer1 = BatchNormalization()(layer1)
#--------------Optional multilayer structure for parameter analysis------------------------------------
#layer3 = ConvLSTM2D(filters=15, kernel_size=(3, 3),padding='same',data_format='channels_last',kernel_regularizer = regularizers.l2(0.01), bias_regularizer = regularizers.l2(0.01), return_sequences=False)(layer1)
#layer3 = BatchNormalization()(layer3)
#layer2 = ConvLSTM2D(filters=12, kernel_size=(2, 2), padding='same', return_sequences=False)(layer3)
#layer2 = BatchNormalization()(layer1)
layer2 = Conv2D(filters=5, kernel_size=(3, 3), data_format='channels_last',kernel_regularizer = regularizers.l2(0.01), bias_regularizer = regularizers.l2(0.01),activation='relu', padding='same')(layer1)
flat1 = Flatten()(layer2)
#hidden2 = Dense(36*look_ahead*2, activation='relu')(flat1)
#hidden_1 = Dropout(0.15)(hidden_1)
#hidden_1 = BatchNormalization()(hidden_1)
#hidden_1 = Dropout(0.2)(hidden_1)
#X1_output = Dense(36*look_ahead, activation='relu')(hidden2)
#------------learn spatio-temporal feature from the volume data-----------------------------------------
volume_input = Input(shape=(look_back, 9, 4, 1))
volume_input1 = BatchNormalization()(volume_input)
layer4 = ConvLSTM2D(filters=10, kernel_size=(3, 3),data_format='channels_last',kernel_regularizer = regularizers.l2(0.01), bias_regularizer = regularizers.l2(0.01), padding='same', return_sequences=False)(volume_input1)
layer4= BatchNormalization()(layer4)
#--------------Optional multilayer structure for parameter analysis------------------------------------
#layer6 = ConvLSTM2D(filters=15, kernel_size=(3, 3),data_format='channels_last', padding='same',kernel_regularizer = regularizers.l2(0.01), bias_regularizer = regularizers.l2(0.01), return_sequences=False)(layer4)
#layer6 = BatchNormalization()(layer6)
#layer2 = ConvLSTM2D(filters=12, kernel_size=(2, 2), padding='same', return_sequences=False)(layer3)
#layer5 = BatchNormalization()(layer4)
layer6 = Conv2D(filters=5, kernel_size=(3, 3), data_format='channels_last',kernel_regularizer = regularizers.l2(0.01), bias_regularizer = regularizers.l2(0.01),activation='relu', padding='same')(layer4)
flat2 = Flatten()(layer6)
#hidden_3 = Dense(36*look_ahead*2, activation='relu')(flat2)
#hidden_1 = Dropout(0.15)(hidden_1)
#hidden_1 = BatchNormalization()(hidden_1)
#hidden_1 = Dropout(0.2)(hidden_1)
#X2_output = Dense(36*look_ahead, activation='relu')(hidden_3)

#------------Combining the spatio-temporal information using a fusion layer----------------------------------
merged_output = keras.layers.concatenate([flat1, flat2])
#out = keras.layers.Dense(128)(merged_output)
out = keras.layers.Dense(36*look_ahead)(merged_output)
model = Model(inputs=[speed_input,volume_input], outputs=out)
model.compile(loss='mean_squared_error', optimizer='Adamax')
start = time.time()
#-----------------------Record training history---------------------------------------------------------------
train_history = model.fit([X1,X2], y, epochs=50, batch_size=64, verbose=2, validation_data=([X1_test,X2_test], y_test))
#callbacks=[history]
#history.loss_plot('epoch')
loss = train_history.history['loss']
val_loss=train_history.history['val_loss']
end = time.time()
print (end-start)
plt.plot(train_history.history['loss'], label='train')
plt.plot(train_history.history['val_loss'], label='test')
plt.legend()
plt.show()
#--------------------------------Make prediction----------------------------------------------------------------
y_pre = model.predict([X1_test,X2_test])
# Reverse normalization of data
print(y_test.shape)
print(y_pre.shape)

y_test1 = scaler1.inverse_transform(y_test[:,0:36])
y_pre1 = scaler1.inverse_transform(y_pre[:,0:36])
y_pre1=abs(y_pre1)
# save the weights of the model
model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights('422modelcovlstm1.h5')
print('Saved model to disk')

MSE=mean_squared_error(y_pre1,y_test1)
MAE=mean_absolute_error(y_pre1,y_test1)
# save the prediction values and the real values
np.savetxt( 'test.txt',y_test1)
# save the prediction values and the real values
np.savetxt( 'pre.txt',y_pre1 )
#--------------------------------Calculate evaluation index-----------------------------------------------------
mape= np.mean((abs(y_test1- y_pre1)) /y_test1)
rmse=(y_test1- y_pre1)*(y_test1- y_pre1)
rm=np.sum(rmse)
RMSE=math.sqrt(rm/(rmse.size))
ape2=(abs(y_test1- y_pre1)) /y_test1
ape22=ape2*ape2
summape2=np.sum(ape2)
summape22=np.sum(ape22)
len2=ape2.size
vape=math.sqrt((len2*summape22-summape2*summape2)/(len2*(len2-1)))
ec=(math.sqrt((np.sum((y_test1- y_pre1)**2))/len(y_test1)))/(math.sqrt((np.sum(y_test1**2))/len(y_test1))+math.sqrt((np.sum(y_pre1**2))/len(y_test1)))
tic = (math.sqrt( (np.sum((y_test1- y_pre1)**2)) / len(y_test1) )) / (math.sqrt((np.sum((y_test1)**2)) / len(y_test1) ) + math.sqrt((np.sum((y_test1)**2)) / len(y_test1)))
print('MSE:', MSE)
print('MAE:', MAE)
print('RMSE:', RMSE)
print('MAPE' , mape)
print('EC' , ec)
print('TIC' , tic)
print('Train Score: %.4f VAPE' % (vape))