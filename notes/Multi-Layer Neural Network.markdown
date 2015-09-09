##Multi-Layer Neural Network
神经网络是一类supervised learning problem.我们已经有了标注好的训练的样本$(x^{(i)}, y^{(i)})$,神经网络给了一种方式：定义一个复杂，non-linear格式的hypotheses$h_{W,b}(x)$，参数是$W,b$,来fit我们的数据。

神经网络的一个最小的计算单元：single neuron，举个栗子，如下图所示：
![Single neuron](SingleNeuron.png)
这个神经元(计算单元)，input是：$x_1, x_2, x_3$（以及一个intercept term: +1）.output是：$\textstyle h_{W,b}(x) = f(W^Tx) = f(\sum_{i=1}^3 W_{i}x_i +b)$.

其中$f : \Re \mapsto \Re$被称作是activation function.在本节中，我们选择sigmoid function作为激活函数：

$$
f(z) = \frac{1}{1+\exp(-z)}
$$

f也有其它的选择，例如 hyperbolic tangent或tanh(双曲正切)函数：

$$
f(z) = \tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}.
$$

最近发现的另外一种激活函数：the rectified linear function,在deep neural networks的实践更好。这个action function和sigmod以及tanh不同点在于，它没有bound,并且不是continuously differentiable. rectified linear activation 如下所示:
$$
f(z) = \max(0,x).
$$
下面是三个激活函数的对比

![Activation functions](Activation_functions.png)

tanh(z)就像是rescaled版本的sigmoid.它的output范围变成了[0,1]，而不是[-1,1].

在本节中，我们不像是其它的教程那样,不使用$x_0=1$的convention.相反，我们用b来作为intercept term.

以上三个activation函数的导数为:

$$
f(z) = 1/(1+\exp(-z))	f'(z) = f(z) (1-f(z))
$$
$$
tanh function		f'(z) = 1- (f(z))^2
$$
$$
rectified linear function:   z<0时是0，z=0处不可导，其它情况是1.
$$

神经网络的核心在于:output of a neuron can be the input of another,下图是一个栗子：

![Network 331](Network331.png)

一些概念我整理成如下图表：

| 一些概念  | 
|:------------- |
| +1的circle代表是一个bias units,对应的是一个intercept term     |
| 最左边的layer被称之为input layer      |
| 最右边的layer被称之为output layer（本例中只有一个节点） |
| 中间层的节点layer被称之为hidden layer|


上图中，我们的神经网络一共有3个input units;3个hidden units;1个output unit;

下面约定一些标示方法

| 一些约定  | 
|:------------- |
| $n_l$表示神经网络的层数，本例中$n_l=3$    |
| $L_1$表示input layer,$L_{n_l}$表示output layer.      |
| 本神经网络的参数表示为(W,b)=(W^{(1)},b^{(1)},W^{(2)},b^{(2)}) |
| $W_{ij}^{(l)}$表示l层的unit j和l+1层的unit i之间的weight|
| $b_i^{(l)}表示l+1层的unit i的相关的bias$|
| bias units没有输入，且没有相关的输入的connections，它们总是输出+1|
| $s_l表示第l层的node数量(不计入bias unit)$|
| $a_i^{(l)}$来表示第l层的unit i的activation（output value）.对于l=1，$a^{(1)}_i = x_i$表示i-th input|


举个栗子，对于上例的神经网络，给定了参数W,b，我们的神经网络定义了一个hypothesis函数$h_{W,b}(x)$输出一个real number.神经网络的计算过程如下:

$$
\begin{align}
a_1^{(2)} &= f(W_{11}^{(1)}x_1 + W_{12}^{(1)} x_2 + W_{13}^{(1)} x_3 + b_1^{(1)})  \\
a_2^{(2)} &= f(W_{21}^{(1)}x_1 + W_{22}^{(1)} x_2 + W_{23}^{(1)} x_3 + b_2^{(1)})  \\
a_3^{(2)} &= f(W_{31}^{(1)}x_1 + W_{32}^{(1)} x_2 + W_{33}^{(1)} x_3 + b_3^{(1)})  \\
h_{W,b}(x) &= a_1^{(3)} =  f(W_{11}^{(2)}a_1^{(2)} + W_{12}^{(2)} a_2^{(2)} + W_{13}^{(2)} a_3^{(2)} + b_1^{(2)}) 
\end{align}
$$

最后，我们用$z_(i)^{(l)}$来表示第l层的unit i的输入(包含了bias term)，举个栗子：

$$
\textstyle z_i^{(2)} = \sum_{j=1}^n W^{(1)}_{ij} x_j + b^{(1)}_i
$$
所以有：
$$
a^{(l)}_i = f(z^{(l)}_i)
$$

我们扩展函数$f(\cdot)$来支持向量中element-wise的写法($f([z_1, z_2, z_3]) = [f(z_1), f(z_2), f(z_3)]$),我们就可以将上面的式子表达的更紧凑:

$$
\begin{align}
z^{(2)} &= W^{(1)} x + b^{(1)} \\
a^{(2)} &= f(z^{(2)}) \\
z^{(3)} &= W^{(2)} a^{(2)} + b^{(2)} \\
h_{W,b}(x) &= a^{(3)} = f(z^{(3)})
\end{align}
$$

这种计算过程我们称之为forward propagation.推广开来，记得我们用$a^{(1)} = x$来表示input layer的values.那么给定第l层的activations $a^{(l)}$,我们这样计算l+1层的activations $a^{(l+1)}$:

$$
\begin{align}
z^{(l+1)} &= W^{(l)} a^{(l)} + b^{(l)}   \\
a^{(l+1)} &= f(z^{(l+1)})
\end{align}
$$

也可以用其他的architectures来建造神经网络.

神经网络可以有多个output units.下图的神经网络就有两个Hidden layer,$L_2$和$L_3$,并且在$L_4$中有两个output units.

![Network 3322](Network3322.png)

为了训练这个神经网络，我们需要训练样本$(x^{(i)}, y^{(i)})$,$y^{(i)} \in \Re^2$.