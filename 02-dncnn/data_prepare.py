import os

import cv2
import numpy as np

train_data_path = "./data/train"
test_data_path="./data/test"
val_data_path="./data/val"
pitch_height = 40
pitch_width = 40
stride = 10
scales = (1,0.9,0.8,0.7)
sigma = 25

def get_data(get_data):
    files = os.listdir(get_data)
    for file in files:
        if file.endswith(".db"):
            continue
        img = cv2.imread(filename=os.path.join(get_data,file),flags=cv2.IMREAD_COLOR)
        img = img/255.0
        h, w = img.shape[0:2]
        x =  []
        y =  []
        for scale in scales:
            # 切片左闭右开
            h_scale,w_scale = int(h*scale),int(w*scale)
            img_scale = cv2.resize(src=img,dsize=(w_scale,h_scale),interpolation=cv2.INTER_CUBIC)
            noise = np.random.normal(0, sigma/255.0, img_scale.shape)
            img_scale_noise = img_scale+noise
            for i in range(0,h_scale-pitch_height-1,stride):
                for j in range(0, w_scale-pitch_width-1, stride):
                    y.append(img_scale_noise[i:i+pitch_height,j:j+pitch_width,:])
                    x.append(img_scale[i:i+pitch_height,j:j+pitch_width,:])
    return np.array(y),np.array(x)

def get_train_data():
    return get_data(train_data_path)

def get_validation_data():
    return get_data(val_data_path)

if __name__=='__main__':
    data = get_train_data()