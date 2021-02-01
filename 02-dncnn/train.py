import tensorflow.keras.backend as K
from tensorflow import keras
import math
import os
import matplotlib.pyplot as plt

import data_prepare
import network
from train_callback import TrainCallback

model_path = './models/model.h5'
def mean_squared_error(y_true, y_pred):
    print(y_true,y_pred)
    return K.mean((y_true-y_pred)**2)
def peak_sifnal_to_noise(y_true, y_pred):
    return 10*keras.backend.log(1/mean_squared_error(y_true, y_pred))/math.log(10)

def get_model_from_load():
    return keras.models.load_model(model_path,compile=False)

def get_model_from_network():
    return network.dncnn()


if __name__=='__main__':
    if os.path.exists(model_path,):
        print("get_model_from_load")
        model = get_model_from_load()
    else:
        print("get_model_from_network")
        model = get_model_from_network()
    model.summary()
    checkpointer = keras.callbacks.ModelCheckpoint('./models/model_{epoch:03d}.hdf5',
                                                   verbose=1, save_weights_only=False, period=1)
    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss=keras.losses.mean_squared_error,
                  metrics=[mean_squared_error, peak_sifnal_to_noise])
    y_train,x_train = data_prepare.get_train_data()
    y_val,x_val = data_prepare.get_validation_data()
    # train_callback = TrainCallback()
    history = model.fit(
      y_train,
      x_train,
      batch_size= 128,
      epochs=5,
      validation_data=(y_val,x_val),
        callbacks=[checkpointer]
    )
    epochs = range(len(history.history['acc']))
    plt.figure()
    plt.plot(epochs, history.history['acc'], 'b', label='Training acc')
    plt.plot(epochs, history.history['val_acc'], 'r', label='Validation acc')
    model.save("./models/model.h5")