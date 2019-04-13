# Colab_Cifar10_Image-recognition
 ## 16124278 王浩 week 3  
 ## Cifar_10_图像识别 
 ### 基于Inception网络
 
由于Cifar10数据集较大，且本文中的训练模型的总参数多达142万，
即使在本地使用GPU（MX150）训练，每次运行仍需接6-8小时，不利于程序的后续调整，
故本程序借助Google Colab（约30min-1h）利用GPU加速在云端运行。
最终模型在（最佳的一次参数：batch=256,factor=0.1,patience=5,62s, 35epoch）
训练集上的准确率为：99.78%
验证集上的准确率为：97.15%
测试集上的准确率为：97.07%
在几大经典图像识别数据集（MNIST / CIFAR10 / CIFAR100 / STL-10 / SVHN / ImageNet）中，
对于 CIFAR10 数据集而言，目前业内 State-of-Art 级别的模型所能达到的最高准确率是 96.53%。
注：由于暂时无法在Colab中引用本地图片，本文中所有图片均已上传至GitHub，用网络链接的形式进行展示。
