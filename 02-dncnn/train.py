import tensorflow.keras.backend as K
from tensorflow import keras
import math

import data_prepare
import model
from train_callback import TrainCallback


def mean_squared_error(y_true, y_pred):
    print(y_true,y_pred)
    return K.mean((y_true-y_pred)**2)
def peak_sifnal_to_noise(y_true, y_pred):
    return 10*keras.backend.log(1/mean_squared_error(y_true, y_pred))/math.log(10)


if __name__=='__main__':
    model = model.dncnn()
    model.compile(optimizer= keras.optimizers.Adam(0.001),
                  loss=keras.losses.mean_squared_error,
                  metrics=[mean_squared_error,peak_sifnal_to_noise])
    model.summary()
    y_train,x_train = data_prepare.get_train_data()
    y_val,x_val = data_prepare.get_validation_data()
    train_callback = TrainCallback()
    model.fit(
      y_train,
      x_train,
      batch_size= 128,
      epochs=4,
      validation_data=(y_val,x_val),
        callbacks=[train_callback]
    )
    model.save("./models/models.h5")