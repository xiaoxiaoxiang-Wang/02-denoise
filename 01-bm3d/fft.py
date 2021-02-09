import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.fftpack import dct, idct,fft
def low_pass_filter(img,radius=100):
    r = radius
    rows, cols = img.shape
    center = int(rows / 2), int(cols / 2)
    mask = np.zeros((rows, cols, 2), np.uint8)
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
    mask[mask_area] = 1
    return mask

def high_pass_filter(img,radius=100):
    return 1-low_pass_filter(img,radius)

def threshold_filter(fft_shift,threshold=200):
    magnitude_spectrum = log_magnitude(fft_shift)
    rows, cols = fft_shift.shape[0],fft_shift.shape[1]
    mask = np.zeros((rows, cols, 2), np.uint8)
    mask_area = magnitude_spectrum>threshold
    mask[mask_area] = 1;
    print(magnitude_spectrum)
    return mask

def fft(grayImg):
    dft = cv2.dft(np.float32(grayImg), flags=cv2.DFT_REAL_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    return dft_shift

def fft2(grayImg):
    return dct(dct(grayImg, axis=0, norm='ortho'), axis=1, norm='ortho')

def fft3(grayImg):
    return dct(dct(grayImg, axis=0, norm='ortho'), axis=1, norm='ortho')

def log_magnitude(dft_shift):
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    return magnitude_spectrum


def fft_demo(img):
    print(type(img),img.shape)
    plt.subplot(231), plt.imshow(img,cmap=plt.cm.gray), plt.title('picture')
    fft_shift = fft(img)
    print(fft(img))
    print(fft2(img))
    # mask = threshold_filter(fft_shift)
    # fshift = fft_shift*mask
    # f_ishift = np.fft.ifftshift(fshift)
    # img_after_filtering = cv2.idft(f_ishift,flags=cv2.DFT_COMPLEX_OUTPUT)
    # img_after_filtering = cv2.magnitude(img_after_filtering[:, :, 0], img_after_filtering[:, :, 1])
    # plt.subplot(232), plt.imshow(img_after_filtering,cmap=plt.cm.gray), plt.title('picture')
    #
    # plt.subplot(234), plt.imshow(img-img_after_filtering,cmap=plt.cm.gray), plt.title('picture')

    # magnitude_spectrum = log_magnitude(fft_shift)
    # print(magnitude_spectrum)
    # plt.subplot(232), plt.imshow(fft2(img), cmap=plt.cm.gray), plt.title('picture')
    # plt.subplot(233), plt.imshow(magnitude_spectrum,cmap=plt.cm.gray), plt.title('picture')
    # plt.show()

if __name__ == '__main__':
    img = cv2.imread('./data/noisydog.png',0)
    a = np.zeros(shape=(5, 5), dtype=np.float)
    for i in range(5):
        print(i)
        for j in range(5):
            if(i+j)%2==1:
                a[i][j] = 10
    fft_demo(a)