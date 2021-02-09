# 噪声的特点
频率的高低无法作为区分噪声的特征，因为噪音的频率范围介于图像频率范围之间，
频率的幅值是一个比较好的区分噪声的特征，因为噪音的频率幅值不会太大，即使做硬阈值处理，也不会丢失太多信息
# bm3d算法流程
1.图像分块，依次做傅里叶变换
2.根据频域 向右下方向搜索相似图像块，滑窗 相似度用矩阵二范数衡量
3.3D变换硬阈值过滤后 计算频域非零数量(即频域更广的权重更大) 反变换，分配阈值

#fft dct dft
fft(快速傅里叶变换) 本质是dft，一种快速计算dft的算法
dft(离散傅里叶变换)
dct(离散余弦变换) 离散余弦变换 系数是为了矩阵运算时方便运算