import os

import cv2
import numpy as np

train_data_path = "./data/train"
train_val_data_path = "./data/train_val"
test_data_path = "./data/test"
val_data_path = "./data/val"
save_path = "./data/npy"
name_y_npy = "noise_patches.npy"
name_x_npy = "origin_patches.npy"
pitch_height = 40
pitch_width = 40
stride = 10
scales = (1, 0.9, 0.8, 0.7)
sigma = 25
transform = {
    "origin": lambda img: img,
    # "rot90_1":lambda img:np.rot90(img),
    # "rot90_2":lambda img:np.rot90(img,2),
    # "rot90_3":lambda img:np.rot90(img,3),
    # "flip":lambda img:np.flipud(img),
    # "flip_rot90_1":lambda img:np.flipud(np.rot90(img)),
    # "flip_rot90_2": lambda img: np.flipud(np.rot90(img,2)),
    # "flip_rot90_3": lambda img: np.flipud(np.rot90(img,3)),
}


def generate_data(data_path):
    files = os.listdir(data_path)
    x = []
    y = []
    for file in files:
        if file.endswith(".db"):
            continue
        img = cv2.imread(filename=os.path.join(data_path, file), flags=cv2.IMREAD_GRAYSCALE)
        # img = cv2.imread(filename=os.path.join(data_path,file),flags=cv2.IMREAD_COLOR)
        img = img / 255.0
        h, w = img.shape[0:2]
        for scale in scales:
            # 切片左闭右开
            h_scale, w_scale = int(h * scale), int(w * scale)
            # img_scale = cv2.resize(src=img, dsize=(w_scale, h_scale), interpolation=cv2.IMREAD_GRAYSCALE)
            img_scale = cv2.resize(src=img, dsize=(w_scale, h_scale), interpolation=cv2.INTER_CUBIC)
            noise = np.random.normal(0, sigma / 255.0, img_scale.shape)
            img_scale_noise = img_scale + noise
            for k, v in transform.items():
                for i in range(0, h_scale - pitch_height - 1, stride):
                    for j in range(0, w_scale - pitch_width - 1, stride):
                        y.append(v(img_scale_noise[i:i + pitch_height, j:j + pitch_width])[..., np.newaxis])
                        x.append(v(img_scale[i:i + pitch_height, j:j + pitch_width])[..., np.newaxis])

                        # y.append(v(img_scale_noise[i:i+pitch_height,j:j+pitch_width,:]))
                        # x.append(v(img_scale[i:i+pitch_height,j:j+pitch_width,:]))
    return np.array(y), np.array(x)


def generate_test_data(data_path):
    files = os.listdir(data_path)
    x = []
    y = []
    for file in files:
        if file.endswith(".db"):
            continue
        img = cv2.imread(filename=os.path.join(data_path, file), flags=cv2.IMREAD_GRAYSCALE)
        # img = cv2.imread(filename=os.path.join(data_path,file),flags=cv2.IMREAD_COLOR)
        img = img / 255.0
        noise = np.random.normal(0, sigma / 255.0, img.shape)
        img_scale_noise = img + noise
        y.append(img_scale_noise)
        x.append(img)
    return y, x


def get_train_data():
    return generate_data(train_data_path)


def get_train_val_data():
    if not os.path.exists(os.path.join(save_path, name_y_npy)) or not os.path.exists(
            os.path.join(save_path, name_x_npy)):
        save_as_npy()
    return np.load(os.path.join(save_path, name_y_npy)), np.load(os.path.join(save_path, name_x_npy))


def get_validation_data():
    return generate_data(val_data_path)


def get_test_data():
    return generate_test_data(test_data_path)


def save_as_npy():
    y, x = generate_data(train_val_data_path)
    print('Shape of result = ' + str(y.shape))
    print('Saving data...')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    np.save(os.path.join(save_path, name_y_npy), y)
    np.save(os.path.join(save_path, name_x_npy), x)
    print('Done.')


if __name__ == '__main__':
    save_as_npy()
