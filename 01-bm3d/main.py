import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import dct, idct

# 1 2D变换
# 2 3D变换
# 3 分配权值
block_size = 8
slid_size = 3
search_size = 39 - block_size
dct_threshold1 = 2500
lambda_2d = 2
lambda_3d = 2.7
sigma = 25  # variance of the noise
max_match_blocks1 = 16
kaiser_window_beta = 2.0

max_match_blocks2 = 32
dct_threshold2 = 400

#######################################step 1 #########################################
def first_step(img):
    dct_blocks = get_all_dct_block(img)
    basic_img = get_basic_img(img, dct_blocks)
    return basic_img, dct_blocks


def get_basic_img(img, dct_blocks):
    basic_img, basic_weight, basic_kaiser = init(img)
    h, w = img.shape
    # 以i和j为起点
    i = 0
    while i < h:
        j = 0
        while j < w:
            group_positions = basic_group_block(i, j, h, w, dct_blocks,max_match_blocks1,dct_threshold1,lambda_2d)
            group_blocks, non_zero_cnt = basic_block_3d_filter(group_positions, dct_blocks)
            # print("non_zero_cnt", non_zero_cnt)
            basic_assign_weight(group_positions, group_blocks, non_zero_cnt, basic_kaiser, basic_img, basic_weight)
            if j == min(j + slid_size, w - block_size):
                break
            j = min(j + slid_size, w - block_size)
        if i == min(i + slid_size, h - block_size):
            break
        i = min(i + slid_size, h - block_size)
    basic_weight = np.where(basic_weight == 0, 1, basic_weight)
    basic_img /= basic_weight
    return basic_img


def basic_group_block(i, j, h, w, dct_blocks,max_match_blocks,dct_threshold,lambda_d):
    group_positions = []
    for m in range(min(search_size, h - block_size + 1 - i)):
        for n in range(min(search_size, w - block_size + 1 - j)):
            dist = get_distance(dct_blocks[i][j], dct_blocks[i + m][j + n],lambda_d)
            if dist < dct_threshold:
                group_positions.append((i + m, j + n, dist))
    # 匹配的块过多按dist截取
    if len(group_positions) > max_match_blocks:
        group_positions = sorted(group_positions, key=lambda x: x[2])
        group_positions = group_positions[0:max_match_blocks]
    return group_positions


def basic_block_3d_filter(group_positions, dct_blocks):
    group_blocks = np.zeros((len(group_positions), block_size, block_size))
    for k in range(len(group_positions)):
        group_blocks[k] = dct_blocks[group_positions[k][0], group_positions[k][1]]
    non_zero_cnt = 0
    for i in range(block_size):
        for j in range(block_size):
            dct_3d = dct(group_blocks[:, i, j], norm='ortho')
            dct_3d[abs(dct_3d) < lambda_3d * sigma] = 0
            idct_3d = idct(dct_3d, norm='ortho')
            non_zero_cnt += np.nonzero(dct_3d)[0].size
            group_blocks[:, i, j] = idct_3d
    return group_blocks, non_zero_cnt


def basic_assign_weight(group_positions, group_blocks, non_zero_cnt, basic_kaiser, basic_img, basic_weight):
    if non_zero_cnt < 1:
        block_weight = basic_kaiser
    else:
        # 此系数的意义，非零数越多，表示方差越大，因此所占比例越小
        block_weight = (1. / (sigma * sigma * non_zero_cnt)) * basic_kaiser
    for idx, val in enumerate(group_positions):
        basic_img[val[0]:val[0] + block_size,
        val[1]:val[1] + block_size] += block_weight * idct2D(group_blocks[idx])
        basic_weight[val[0]:val[0] + block_size,
        val[1]:val[1] + block_size] += block_weight


def init(img):
    init_img = np.zeros(img.shape, dtype=float)
    init_weight = np.zeros(img.shape, dtype=float)
    window = np.matrix(np.kaiser(block_size, kaiser_window_beta))
    init_kaiser = np.array(window.T * window)
    return init_img, init_weight, init_kaiser


def get_distance(block1, block2, lambda_d):
    if sigma > 40:
        block1 = np.where(abs(block1) < lambda_d * sigma, 0, block1)
        block2 = np.where(abs(block2) < lambda_d * sigma, 0, block2)
    return np.linalg.norm(block1 - block2) ** 2 / (block_size ** 2)


#######################################step 2 #########################################
def second_step(basic_img, dct_blocks):
    basic_dct_blocks = get_all_dct_block(basic_img)
    final_img = get_final_img(basic_img, basic_dct_blocks, dct_blocks)
    return final_img


def get_final_img(basic_img, basic_dct_blocks, dct_blocks):
    final_img,final_weight , final_kaiser = init(basic_img)
    h, w = img.shape
    # 以i和j为起点
    i = 0
    while i < h:
        j = 0
        while j < w:
            group_positions = basic_group_block(i, j, h, w, basic_dct_blocks,max_match_blocks2,dct_threshold2,lambda_3d)
            group_blocks, weight = final_block_3d_filter(group_positions, dct_blocks, basic_dct_blocks)
            final_assign_weight(group_blocks, group_positions, weight, final_kaiser, final_img, final_weight)
            if j == min(j + slid_size, w - block_size):
                break
            j = min(j + slid_size, w - block_size)
        if i == min(i + slid_size, h - block_size):
            break
        i = min(i + slid_size, h - block_size)
    final_weight = np.where(final_weight == 0, 1, final_weight)
    final_img /= final_weight
    return final_img

def final_block_3d_filter(group_positions, dct_blocks, basic_dct_blocks):
    group_blocks = np.zeros((len(group_positions), block_size, block_size))
    basic_group_blocks = np.zeros((len(group_positions), block_size, block_size))
    for k in range(len(group_positions)):
        group_blocks[k] = dct_blocks[group_positions[k][0], group_positions[k][1]]
        basic_group_blocks[k] = basic_dct_blocks[group_positions[k][0], group_positions[k][1]]
    weight = 0
    for i in range(block_size):
        for j in range(block_size):
            # wiener filter
            dct_3d = dct(group_blocks[:, i, j], norm='ortho')
            basic_dct_3d = dct(basic_group_blocks[:, i, j], norm='ortho')
            wiener_coef = basic_dct_3d ** 2 / block_size
            wiener_coef /= (wiener_coef + sigma ** 2)
            dct_3d *= wiener_coef
            weight += np.sum(wiener_coef)
            group_blocks[:, i, j] = idct(dct_3d, norm="ortho")
    if weight == 0:
        weight = 1
    else:
        weight = 1. / (sigma ** 2 * weight)
    return group_blocks, weight

def final_assign_weight(group_blocks,group_positions, weight,final_kaiser, final_img, final_weight):
    block_weight = weight * final_kaiser
    for idx, val in enumerate(group_positions):
        final_img[val[0]:val[0] + block_size,
        val[1]:val[1] + block_size] += block_weight * idct2D(group_blocks[idx])
        final_weight[val[0]:val[0] + block_size,
        val[1]:val[1] + block_size] += block_weight


#######################################utils #########################################

# dct block 按step=1采样，得到(h - block_size + 1, w - block_size + 1)个block
def get_all_dct_block(img):
    h, w = img.shape
    dct_blocks = np.zeros((h - block_size + 1, w - block_size + 1, block_size, block_size))
    for i in range(0, h - block_size + 1):
        for j in range(0, w - block_size + 1):
            dct_blocks[i][j] = dct2D(img[i:i + block_size, j:j + block_size].astype(np.float64))
    return dct_blocks


def dct2D(img):
    return dct(dct(img, axis=0, norm='ortho'), axis=1, norm='ortho')


def idct2D(img):
    return idct(idct(img, axis=0, norm='ortho'), axis=1, norm='ortho')


def psnr(origin_img, img):
    mse = np.sum((img - origin_img) ** 2) / origin_img.size
    return 10 * np.log10(255 ** 2 / mse)


if __name__ == '__main__':
    img = cv2.imread("./data/dog.png", cv2.IMREAD_GRAYSCALE)
    noise = np.random.normal(0, sigma, img.shape)
    img_noise = img + noise

    start_time = time.time()
    basic_img, dct_blocks = first_step(img_noise)
    cv2.imwrite("./data/basic_img.png", basic_img)
    print("first step time cost:", time.time() - start_time)
    start_time = time.time()
    final_img = second_step(basic_img, dct_blocks)
    cv2.imwrite("./data/final_img.png", final_img)
    print("second step time cost:", time.time() - start_time)

    plt.subplot(221), plt.imshow(img_noise, cmap=plt.cm.gray), plt.title('img_noise')
    plt.subplot(222), plt.imshow(basic_img, cmap=plt.cm.gray), plt.title('basic_img')
    plt.subplot(223), plt.imshow(img, cmap=plt.cm.gray), plt.title('img')
    plt.subplot(224), plt.imshow(final_img, cmap=plt.cm.gray), plt.title('final_img')
    print("psnr of img and first step img:", psnr(img, basic_img))
    print("psnr of img and second step img:", psnr(img, final_img))
    plt.show()
