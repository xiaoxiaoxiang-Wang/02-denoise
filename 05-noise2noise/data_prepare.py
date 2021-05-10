import os
import random

import cv2
import numpy as np

train_dir = './data/train'
test_dir = './data/test'

width = 256
height = 256

sigma = 25


def get_files(dir):
    files_list = []
    files = os.listdir(dir)
    for file in files:
        file_path = os.path.join(dir, file)
        files_list.append(file_path)
    return files_list


def get_train_files():
    return get_files(train_dir)


def get_test_files():
    return get_files(test_dir)


def get_val_data(val_files):
    batch_x = []
    batch_y = []
    for i in range(len(val_files)):
        img = cv2.imread(filename=val_files[i], flags=cv2.IMREAD_GRAYSCALE) / 255.0
        for i in range(0, img.shape[0] - height + 1, 30):
            for j in range(0, img.shape[1] - width + 1, 30):
                noise1 = np.random.normal(0, sigma / 255.0, (height, width))
                noise2 = np.random.normal(0, sigma / 255.0, (height, width))
                batch_x.append(img[i:i + height, j:j + width] + noise1)
                batch_y.append(img[i:i + height, j:j + width] + noise2)
    print(np.array(batch_x).shape)
    return (np.array(batch_x), np.array(batch_y))


def train_datagen(train_files,
                  file_size=8):
    while (True):
        idx = list(range(len(train_files)))
        random.shuffle(idx)
        cnt = 0
        batch_x = []
        batch_y = []
        for i in idx:
            cnt += 1
            clipImg(cv2.imread(filename=train_files[i], flags=cv2.IMREAD_GRAYSCALE) / 255.0, batch_x, batch_y)
            if cnt % file_size == 0:
                batch_x = np.array(batch_x)
                batch_y = np.array(batch_y)
                yield batch_x, batch_y
                batch_x = []
                batch_y = []


def clipImg(img, batch_x, batch_y):
    for i in range(0, img.shape[0] - height + 1, 30):
        for j in range(0, img.shape[1] - width + 1, 30):
            noise = np.random.normal(0, sigma / 255.0, (height, width))
            batch_x.append(img[i:i + height, j:j + width] + noise)
            noise = np.random.normal(0, sigma / 255.0, (height, width))
            batch_y.append(img[i:i + height, j:j + width] + noise)


if __name__ == '__main__':
    pass
