# platform

python --version 3.7.3  
tensorflow.__version__ 2.4.1  
keras.__version__ 2.4.0

# dataset

https://github.com/csjunxu/PolyU-Real-World-Noisy-Images-Dataset

# 01-bm3d

dog.png  
first step time cost: 203.54992508888245  
second step time cost: 239.55721855163574  
psnr of img and first step img: 31.75813141725017  
psnr of img and second step img: 31.790544509372616

[Image denoising by sparse 3D transform-domain collaborative ltering](https://webpages.tuni.fi/foi/GCF-BM3D/BM3D_TIP_2007.pdf)<br/>
[code ref](https://github.com/ChihaoZhang/BM3D.git)<br/>

# 02-dncnn

unet结构，最后一层残差值得参考

# 03-brdnet

Batch ReNormalization结构 + 膨胀卷积

# 04-neighbor2neighbor

不需要干净图像，不需要两张pix2pix的噪声图像，只需要一张噪声图像

# 05-noise2noise

不需要干净图像，只需要两张pix2pix的噪声图像，即可得到清晰图像 noise2noise和neighbor2neighbor这类算法都必须设计好损失函数，了解噪声的分布，使得干净图像是最优解，因此这类算法无法应用在去模糊增强等场景
只适用于和训练集相同的噪声等级，测试集噪声分布不一致的话效果很差

# 图像的噪声如何评估 Noise Estimation

均匀区域法 分快法 Filter-Based Approach Using Arithmetic Averaging ——Filter-Base Filter-Based Approach Using Statistical
Averaging —— Block-Base
