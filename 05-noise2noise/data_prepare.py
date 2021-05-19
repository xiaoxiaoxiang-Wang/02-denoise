import os

import cv2
import numpy as np

train_data_path = "./data/train"
test_data_path = "./data/test"
save_path = "./data/npy"
y_npy = "y_patches.npy"
x_npy = "x_patches.npy"
y_val_npy = "y_val_patches.npy"
x_val_npy = "x_val_patches.npy"
pitch_height = 256
pitch_width = 256
stride = 64
scales = (1, 0.8)
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


def generate_data(files, is_train=True):
    x = []
    y = []
    for file in files:
        img = cv2.imread(filename=os.path.join(train_data_path, file), flags=cv2.IMREAD_GRAYSCALE)
        img = img / 255.0
        h, w = img.shape[0:2]
        for scale in scales:
            # 切片左闭右开
            h_scale, w_scale = int(h * scale), int(w * scale)
            img_scale = cv2.resize(src=img, dsize=(w_scale, h_scale), interpolation=cv2.IMREAD_GRAYSCALE)
            for m in range(noise_nums):
                noise_list.append(np.random.normal(0, sigma / 255.0, img_scale.shape))
            for k, v in transform.items():
                for i in range(0, h_scale - pitch_height - 1, stride):
                    for j in range(0, w_scale - pitch_width - 1, stride):
                        noise_list = []
                        noise_nums = 4
                        for m in range(noise_nums):
                            for n in range(noise_nums):
                                if m == n:
                                    continue
                                img_scale_noise1 = img_scale + noise_list[m]
                                img_scale_noise2 = img_scale + noise_list[n]
                                if is_train:
                                    y.append(
                                        v(img_scale_noise2[i:i + pitch_height, j:j + pitch_width])[..., np.newaxis])
                                else:
                                    y.append(v(img_scale[i:i + pitch_height, j:j + pitch_width])[..., np.newaxis])
                                x.append(v(img_scale_noise1[i:i + pitch_height, j:j + pitch_width])[..., np.newaxis])
    return np.array(x), np.array(y)


def generate_test_data(data_path):
    files = os.listdir(data_path)
    x = []
    y = []
    for file in files:
        img = cv2.imread(filename=os.path.join(data_path, file), flags=cv2.IMREAD_GRAYSCALE)
        # img = cv2.imread(filename=os.path.join(data_path,file),flags=cv2.IMREAD_COLOR)
        img = img / 255.0
        noise = np.random.normal(0, sigma / 255.0, img.shape)
        img_scale_noise = img + noise
        y.append(img_scale_noise)
        x.append(img)
    return y, x


def get_train_data():
    if not os.path.exists(os.path.join(save_path, y_npy)) or not os.path.exists(
            os.path.join(save_path, x_npy)):
        save_as_npy()
    return np.load(os.path.join(save_path, x_npy)), np.load(os.path.join(save_path, y_npy))


def get_validation_data():
    if not os.path.exists(os.path.join(save_path, y_val_npy)) or not os.path.exists(
            os.path.join(save_path, x_val_npy)):
        save_as_npy()
    return np.load(os.path.join(save_path, x_val_npy)), np.load(os.path.join(save_path, y_val_npy))


def get_test_data():
    return generate_test_data(test_data_path)


def save_as_npy():
    files = os.listdir(train_data_path)
    train_size = int(len(files) * 0.95)
    x, y = generate_data(files[0:train_size])
    x_val, y_val = generate_data(files[train_size:len(files)], False)
    print('Shape of result = ' + str(y.shape))
    print('Saving data...')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    np.save(os.path.join(save_path, x_npy), x)
    np.save(os.path.join(save_path, y_npy), y)
    np.save(os.path.join(save_path, x_val_npy), x_val)
    np.save(os.path.join(save_path, y_val_npy), y_val)
    print('Done.')


if __name__ == '__main__':
    save_as_npy()
