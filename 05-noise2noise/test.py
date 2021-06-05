import os

import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import compare_psnr
from tensorflow import keras

import data_prepare

model_path = "./models/model.h5"

if __name__ == "__main__":
    if os.path.exists(model_path):
        model = keras.models.load_model(model_path, compile=False)
        y, x = data_prepare.get_test_data()
        sum_psnr = 0
        for i in range(len(y)):
            x_ = model.predict(y[i][np.newaxis, ..., np.newaxis])
            plt.subplot(131), plt.imshow(y[i]), plt.title('input')
            plt.subplot(132), plt.imshow(x_.reshape(y[i].shape)), plt.title('output')
            plt.subplot(133), plt.imshow(x[i]), plt.title('origin')
            psnr_x_ = compare_psnr(x[i], x_.reshape(y[i].shape))
            sum_psnr += psnr_x_
            print(psnr_x_)
            # plt.show()
        print(sum_psnr / len(y))
