1.learning rate
2.keras save
save函数model.add_loss() 和 model.add_metric() 添加的外部损失和指标不会被保存，需要将load_model的compile置为False，compile定义了loss function损失函数、optimizer优化器和metrics度量
