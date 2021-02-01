import os

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

import data_prepare

model_path = "./models/model.h5"

if __name__ == "__main__":
    if os.path.exists(model_path):
        model = keras.models.load_model(model_path,compile=False)
        y,x = data_prepare.get_test_data()
        for i in range(len(y)):
            x_ = model.predict(y[i][np.newaxis,...])
            plt.subplot(121), plt.imshow(y[i]), plt.title('input')
            plt.subplot(122), plt.imshow(x_[0]), plt.title('output')
            plt.show()