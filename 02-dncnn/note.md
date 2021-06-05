# dncnn

dncnn是一个全卷积网络，网络结构为u-net，值得参考的点是最后一层的残差

# keras save

save函数model.add_loss() 和 model.add_metric() 添加的外部损失和指标不会被保存，需要将load_model的compile置为False，compile定义了loss
function损失函数、optimizer优化器和metrics度量

# 训练过程 psnr一直稳定在21左右，无法进一步提升

网络结构有误，batch_normalize层在relu之后 数据量不足，只取了一张图片

# 训练过程 使用adam反向传播 重新加载模型训练波动很多

adam梯度值受到惯性的影响，重新加载后，惯性值未保留

# 梯度下降算法总结

# Momentum

动量法，上一状态影响当前状态，可以减少震荡

# adagrad

每个元素都有独立的学习率，这个学习率和梯度累计的平方和根成反比，即学习率一定越来越小

# RMSprop

对Adagrad算法的改进，解决学习率过快衰减的问题

# adam

结合了Momentum和RMSprop算法

# 学习速率

可以通过tensorflow自定义指数衰减或者一定的epoch减半，eg.tf.train.exponential_decay