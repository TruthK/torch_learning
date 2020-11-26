# 传统方法
* 填充缺失区域的最典型传统方法是复制和粘贴
- 主要思想是从图像本身或包含数百万个图像的大型数据集中搜索最相似的图像补丁，然后将其粘贴到缺失的区域中

# EdgeConnect
 首先预测缺失区域的骨架（即边缘/线条），然后根据生成的骨架填充颜色。这是“行优先，颜色接下来”的方法
* 作者将图像修复的任务简化为两个简单的步骤，即边缘预测和图像完成。第一生成器仅负责预测缺失区域的边缘，以便获得整体图像结构。预测边缘图也是显示图像骨架的二进制图。第二个生成器以预测的边缘贴图为条件，并负责以更好的纹理细节填充缺失区域。
![EdgeConnect](https://raw.githubusercontent.com/TruthK/vpn/master/md_imge/EdgeConnect.png)
上图显示了所提出方法EdgeConnect的整体网络结构。如您所见，我们熟悉用于图像修补任务的这种网络体系结构。第一生成器G<sub>1</sub>将掩模图像，被掩模的边缘图像和被掩模的灰度图像作为输入，并给出预测的边缘图。使用标准对抗损失和特征匹配损失训练该生成器。
第二生成器G<sub>2</sub>将预测的边缘图和掩蔽的RGB图像作为输入，并输出完整的RGB图像。使用样式损失，知觉损失，L<sub>1</sub>重建损失和标准对抗损失来训练此生成器。
* 损失函数  
对于第一个边缘生成器，有两个损失项，即对抗损失和特征匹配损失。对抗损失是最基本的损失，特征匹配损失类似于VGG的感知损失。对于第二个图像生成器，有四个损失项，即样式损失，感知损失，L1重建损失和对抗损失。

# Context Encoders
这些卷积层用于提取特征，从简单的结构特征到高级语义特征
* 如果网络中只有卷积层，则无法利用特征图中遥远空间位置处的特征。为了解决此问题，我们可以使用完全连接的层，以使当前层中每个神经元的值取决于上一层中所有神经元的值。
    - 通道级全连接层 : 我们只是完全独立地连接每个通道，而不是所有通道。举例来说，我们有m个特征图，大小为n * n。如果使用标准的完全连接的层，我们将有m<sup>2</sup>n<sup>4</sup> 参数不包括偏项。对于通道级全连接层，我们有mn<sup>4</sup>个参数。因此，我们可以在不添加太多额外参数的情况下从遥远的空间位置捕获要素
![Context_Encoders](https://raw.githubusercontent.com/TruthK/vpn/master/md_imge/Context%20Encoders.png)

# High-Resolution Image Inpainting using Multi-Scale Neural Patch Synthesis
本文的作者使用 U-Net like network with skip connections，其中所有标准卷积层都被提议的部分卷积层代替
* 在卷积期间将丢失的像素与有效像素分开，以便卷积的结果仅取决于有效像素。这就是为什么建议的卷积被称为部分卷积的原因。基于可自动更新的二进制掩码图像，对输入部分执行卷积。
* 部分卷积层  
让我们将W和b定义为卷积滤波器的权重和偏差。X代表卷积的像素值（或特征激活值），M代表相应的二进制掩码，该二进制掩码指示每个像素/特征值的有效性（缺失像素为0，有效像素为1）。计算出建议的部分卷积,
其中⦿表示逐元素乘法，而1是形状与M相同的矩阵。从这个公式，你可以看到，部分卷积的结果只依赖于有效的输入值（如X ⦿ M）。sum（1）/ sum（M）是一个缩放因子，用于随着每个卷积的有效输入值数量的变化来调整结果。

# Image Inpainting via Generative Multi-column Convolutional Neural Neworks
在特征提取方面，提出了一种生成多列的CNN结构，因为多列结构可以将图像分解成具有不同感受野和特征分辨率的分量。在寻找相似块方面，提出了一种隐式多样化马尔可夫随机场（ID-MRF）项，但只将其作为正则化项。在综合辅助信息方面，设计了一种新的置信驱动的重建损失，根据空间位置约束生成内容。
![生成多列卷积神经网络](https://raw.githubusercontent.com/TruthK/torch_learning/master/note_image/%E7%94%9F%E6%88%90%E5%A4%9A%E5%88%97%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C.png)  
首先，作者采用具有扩展卷积的多分支CNN，而不是单分支。在三个不同的分支中使用了三个不同的内核大小，以实现各种接收场并以不同的分辨率提取特征。
其次，引入了两个新的损失项来训练网络，即置信驱动的重建损失和隐式多样化马尔可夫随机场（ID-MRF）损失。置信驱动的重建损失是加权的L 1损失，而ID-MRF损失与预训练的VGG网络计算的特征补丁比较有关。

# Generative Image Inpainting with Contextual Attention
CNN的结构无法有效地建模缺失区域与遥远空间位置所给出的信息之间的长期相关性。如果您熟悉CNN，则应该知道内核大小和膨胀率控制卷积层的接收场，并且网络必须越来越深入，才能看到整个输入图像。这意味着，如果要捕获图像的上下文，则必须依赖更深的图层，但是由于较深的图层始终具有较小的特征空间尺寸，因此我们会丢失空间信息。  
对于该体系结构，所提出的框架由两个生成器网络和两个鉴别器网络组成。这两个生成器遵循具有扩展卷积的完全卷积网络。一台生成器用于粗略重建，另一台生成器用于细化。这称为标准的从粗到细网络结构。这两个鉴别器还会全局和局部查看完成的图像。全局鉴别器将整个图像作为输入，而局部鉴别器将填充区域作为输入。  
![生成多列卷积神经网络](https://raw.githubusercontent.com/TruthK/torch_learning/master/note_image/%E5%85%B7%E6%9C%89%E4%B8%8A%E4%B8%8B%E6%96%87%E6%B3%A8%E6%84%8F%E7%9A%84%E7%94%9F%E6%88%90%E5%9B%BE%E5%83%8F%E4%BF%AE%E5%A4%8D.png)  
* solution  
this paper 提出了一种情境注意机制，可以有效地从遥远的空间位置借用情境信息来重建缺失像素。上下文关注被应用于第二细化网络。第一粗略重构网络负责对缺失区域的粗略估计。与以前相同，全局和局部区分符用于鼓励生成的像素更好的局部纹理细节。  


# Shift-Net
我们必须填写图像，并且必须保持其上下文。精细的细节纹理意味着生成的像素应看起来逼真，并尽可能清晰。  
因此，本文引入了移位连接层，以在其网络内部使用“复制和粘贴”的概念来实现深度特征重排。图1（d）显示了他们提出的方法提供的修复结果。  
* solution  
提出了一种指导损失，以鼓励他们的网络（Shift-Net）在解码过程中学习填充丢失的部分。除此之外，建议使用移位连接层将缺失区域内的已解码特征与缺失区域外的已编码特征进行匹配，然后将编码特征在缺失区域外的每个匹配位置均移至匹配区域内的相应位置。缺少区域。这将捕获有关在缺失区域外找到的最相似局部图像补丁的信息，并将此信息连接到解码特征以进行进一步重建。
