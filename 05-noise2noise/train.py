import math
import os

import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras.optimizers import Adam

import data_prepare
from network import unet

model_dir = './models'
model_path = './models/model111.h5'


def mean_squared_error(y_true, y_pred):
    print(y_true, y_pred)
    return K.mean((y_true - y_pred) ** 2)


def peak_sifnal_to_noise(y_true, y_pred):
    return 10 * keras.backend.log(1 / mean_squared_error(y_true, y_pred)) / math.log(10)


def get_model_from_load():
    return keras.models.load_model(model_path, compile=False)


def get_model_from_network(channel):
    return unet(channel)


if __name__ == '__main__':
    test_files = data_prepare.get_train_files()
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if os.path.exists(model_path):
        print("get_model_from_load")
        model = get_model_from_load()
    else:
        print("get_model_from_network")
        model = get_model_from_network(1)

    # compile the model
    model.compile(optimizer=Adam(), loss=['mse'], metrics=[mean_squared_error, peak_sifnal_to_noise])
    checkpointer = keras.callbacks.ModelCheckpoint('./models/model_{epoch:03d}.hdf5',
                                                   verbose=1, save_weights_only=False)
    file_size = 3
    files_len = len(test_files)
    test_len = int(files_len * 0.95)
    history = model.fit(
        data_prepare.train_datagen(test_files[:test_len], file_size),
        steps_per_epoch=test_len // file_size,
        epochs=10,
        validation_data=data_prepare.get_val_data(test_files[test_len:]),
        callbacks=[checkpointer])
    plt.plot(history.history['loss'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.show()
    model.save("./models/model.h5")
