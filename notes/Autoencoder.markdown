#Autoencoders
之前，我们讨论的都是一个supervised learning，我们都有labeled training examples来做为训练样本。
现在，假设我们有一堆unlabeled training examples $\textstyle \{x^{(1)}, x^{(2)}, x^{(3)}, \ldots\}$，$\textstyle x^{(i)} \in \Re^{n}$.

而autoencoder的目的很让人匪夷所思，它要用无监督的方式，使用的也是backpropagation，但是设置target values==inputs。所以有$\textstyle y^{(i)} = x^{(i)}$

下图是一个autoencoder:
![Autoencoder](Autoencoder636.png)

autoencoder试图学习一个函数使得：$\textstyle h_{W,b}(x) \approx x$ ,类似于identity function(恒等函数)。output $\textstyle \hat{x}$近似等于$x$。

这个恒等函数看似很好学习，但是假如说我们在此network上面加了限制，你就会发现一些很有趣的现象了。

举个例子，假如说输入是10*10的image的pixel intensity values，100个像素值(n=100)。在layer $L_2$,也就是我们的hidden layer，一共有$s_2=50$个hidden units.最后呢，我们要求$\textstyle y \in \Re^{100}$。

因为hidden units只有个50个，所以network的第二层只是学习到了原始数据的一个compressed的版本。hidden unit的activations $\textstyle a^{(2)} \in \Re^{50}$，它必须试图"重构"(reconstruct)那100维的输入$x$（二向箔降维后再次恢复）.

如果输入的特征之间完全随机，比如说每个特征$x_i$来自于独立同分布的高斯分布(IID Gaussian),和其它特征之间相互独立-那么这样的compression任务会非常困难。

如果data里面有一些结构关系在里面，或者说，input features之间相互关联，那么算法就会发现这些correlations。事实上，这个简单的autoencoder的作用最终学到的low-dimensional representation非常类似于PCA（主成分分析）.

所以说，autoencoder可以发现输入数据里面一些有趣的关系.这一论断依赖一个假设，那就是hidden units $s_2$的个数必须比较小。但是如果hidden units的$s_2$非常大(可能比input pixels的维度还要大)，那该怎么办那，我们可以通过对network引入一些constraints来解决这个问题。


具体来说，我们会在hidden units上面引入一个"sparsity"（稀疏）的限制，即使hidden units的数量特别大，autoencoder也会照常工作，在数据上发现一些有趣的结构。

下面说一些我们自己定义的规矩，我们会认为neuron是active（或者说是firing）的，只要它的输出值接近于1，或者说是inactive的,如果他的输出接近于0.在大多数情况我们想要限制neuron是inactive的状态。上面的取值(1或者0),只是在激活函数是sigmoid的前提下说的。如果你正在使用的是tanh激活函数，那么我们可以把值1(active)和-1(inactive)替换上去了。

回想一下之前的定义:$\textstyle a^{(2)}_j$表示hidden层的 unit j的activation。然而，这个表示不能说明是输入的哪一个$x$导致了这个activation。因此，我们重新定义了$\textstyle a^{(2)}_j(x)$来表明是一个特定的input $x$导致了hidden层的这个unit的activation。


然后，我们利用这个定义:
$$
\begin{align}
\hat\rho_j = \frac{1}{m} \sum_{i=1}^m \left[ a^{(2)}_j(x^{(i)}) \right]
\end{align}
$$

$\hat\rho_j$是hidden层中unit j(对于整个training set)的平均的activation。我们可以近似地加上下面的constraint：

$$
\begin{align}
\hat\rho_j = \rho,
\end{align}
$$

$\rho$就是我们所说的sparsity parameter.一般来说是一个接近0的比较小的一个值($p=0.05$)。换句话说，我们希望hidden layer中的每个unit j的平均activation接近于0.05(比如说)。为了满足这个constraint,hidden layer中的unit的activation必须接近于0。


为了达到这个目标,我们需要为我们的optimization objective加上一个惩罚项，如果$\hat\rho_j$和$\rho_j$差别很大的话，这个惩罚项就很大。有很多有意义的惩罚项写法，我们会挑选下面的一种:

$$
\begin{align}
\sum_{j=1}^{s_2} \rho \log \frac{\rho}{\hat\rho_j} + (1-\rho) \log \frac{1-\rho}{1-\hat\rho_j}.
\end{align}
$$
在此，$s_2$就是hidden layer中的neurons的数量,这里把该层所有的hidden units都加了个遍。如果你对KL divergence的概念很熟悉，这个惩罚项就是基于此的。它也可以表示为

$$
\begin{align}
\sum_{j=1}^{s_2} {\rm KL}(\rho || \hat\rho_j),
\end{align}
$$

$$
\textstyle {\rm KL}(\rho || \hat\rho_j)
 = \rho \log \frac{\rho}{\hat\rho_j} + (1-\rho) \log \frac{1-\rho}{1-\hat\rho_j}
$$

学术的解释一下，上面的第二个式子$\textstyle {\rm KL}(\rho || \hat\rho_j)$称之为mean为$\rho$的Bernoulli random variable和mean为$\hat\rho_j$的Bernoulli random variable之间的Kullback-Leibler(KL) divergence（交叉熵）。一句话解释它的意义：KL-divergence is a standard function for measuring how different two different distributions are. 

这个penalty function有一种属性，如果$\textstyle \hat\rho_j = \rho$,那么$\textstyle {\rm KL}(\rho || \hat\rho_j) = 0$.它随着$\textstyle \rho$离$\textstyle \rho$的距离的增长而增长。如下图，我们将$\rho$设置为0.2，画出来$\textstyle {\rm KL}(\rho || \hat\rho_j)$随着$\hat\rho_j$的变化图如下所示:

![KLPenalty](KLPenaltyExample.png)

可以看到，KL-divergence在
$$
\textstyle \hat\rho_j = \rho
$$
的情况下达到了最小值0.当$\hat\rho_j$接近于0或者1时，这个KL项可真是指数级的增长啊。因此，最小化penalty term带来的结果就是使得$\hat\rho_j$接近于$\rho$;
我们的整体的cost function 现在变为:

$$
\begin{align}
J_{\rm sparse}(W,b) = J(W,b) + \beta \sum_{j=1}^{s_2} {\rm KL}(\rho || \hat\rho_j),
\end{align}
$$

$J(W,b)$之前定义过，$\textstyle \beta$控制了sparsity penalty term的权值。sparsity penalty term中的$\textstyle \hat\rho_j$依赖于$W,b$.原因是：它其实是hidden层中的unit j的平均的activation.这个hidden layer的unit j的activation可是依赖于$W,b$的。

为了将KL-divergence term整合到derivative计算中(偏导)。

$$
\begin{align}
\delta^{(2)}_i = \left( \sum_{j=1}^{s_{2}} W^{(2)}_{ji} \delta^{(3)}_j \right) f'(z^{(2)}_i),
\end{align}
$$


$$
\begin{align}
\delta^{(2)}_i =
  \left( \left( \sum_{j=1}^{s_{2}} W^{(2)}_{ji} \delta^{(3)}_j \right)
+ \beta \left( - \frac{\rho}{\hat\rho_i} + \frac{1-\rho}{1-\hat\rho_i} \right) \right) f'(z^{(2)}_i) .
\end{align}
$$

和以前不同的一点是，为了计算偏导，另外多需要了$\textstyle \hat\rho_i$来计算这个term.在进行backpropagation的计算之前， 为了计算$\textstyle \hat\rho_i$,需要对所有的training examples都计算一遍forward pass(才能计算出平均的activation).如果你的training set足够小，可以放入计算机的内存里面，我们就可以将所有的例子都计算一遍forward,然后将activation都放入内存中，用之计算出$\textstyle \hat\rho_i$,然后对用所有的已经计算好的activation来对所有的train example来执行backpropagation.

相反，如果数据集太大，内存不足，你就必须先为每一个hidden layer保持一个累加项，然后scan through 所有的train example来做forward pass，计算出来$\textstyle \hat\rho_i$,这样要抛弃每次forward pass产生的activation $\textstyle a^{(2)}_i$（在累加之后）.所以，对于每个training example,我们都要执行forward pass两遍，这样的话计算也会显得略微低效。

训练完（稀疏）自编码器，我们还想把这自编码器学到的函数可视化出来，好弄明白它到底学到了什么。我们以在10×10图像（即n=100）上训练自编码器为例。在该自编码器中，每个隐藏单元（隐层的单元）i对如下关于输入的函数进行计算：

$$
\begin{align}
a^{(2)}_i = f\left(\sum_{j=1}^{100} W^{(1)}_{ij} x_j  + b^{(1)}_i \right).
\end{align}
$$

我们将要可视化的函数，就是上面这个以2D图像为输入、并由隐藏单元i计算出来的函数。它是依赖于参数$\textstyle W^{(1)}_{ij}$的（暂时忽略偏置项bi）。需要注意的是，$\textstyle a^{(2)}_i$可看作输入$\textstyle x$的非线性特征。不过还有个问题：什么样的输入图像$\textstyle x$可让$\textstyle a^{(2)}_i$得到最大程度的激励？（通俗一点说，隐藏单元$\textstyle i$要找个什么样的特征？）。这里我们必须给$\textstyle x$加约束，否则会得到平凡解。若假设输入有范数约束$\textstyle ||x||^2 = \sum_{i=1}^{100} x_i^2 \leq 1$，则可证（请读者自行推导）令隐藏单元$\textstyle i$得到最大激励的输入应由下面公式计算的像素$\textstyle x_j$给出（共需计算100个像素，j=1,…,100）：

$$
\begin{align}
x_j = \frac{W^{(1)}_{ij}}{\sqrt{\sum_{j=1}^{100} (W^{(1)}_{ij})^2}}.
\end{align}
$$

当我们用上式算出各像素的值、把它们组成一幅图像、并将图像呈现在我们面前之时，隐藏单元$\textstyle i$所追寻特征的真正含义也渐渐明朗起来。
假如我们训练的自编码器有100个隐藏单元，可视化结果就会包含100幅这样的图像——每个隐藏单元都对应一幅图像。审视这100幅图像，我们可以试着体会这些隐藏单元学出来的整体效果是什么样的。

当我们对稀疏自编码器（100个隐藏单元，在10X10像素的输入上训练 ）进行上述可视化处理之后，结果如下所示：

![ExampleSparseAutoencoderWeights](ExampleSparseAutoencoderWeights.png)

上图的每个小方块都给出了一个（带有有界范数 的）输入图像\textstyle x，它可使这100个隐藏单元中的某一个获得最大激励。我们可以看到，不同的隐藏单元学会了在图像的不同位置和方向进行边缘检测。
显而易见，这些特征对物体识别等计算机视觉任务是十分有用的。若将其用于其他输入域（如音频），该算法也可学到对这些输入域有用的表示或特征。
