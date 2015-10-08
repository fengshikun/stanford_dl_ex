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

##Backpropagation Algorithm

假设我们有了一个固定的trainning set $\{ (x^{(1)}, y^{(1)}), \ldots, (x^{(m)}, y^{(m)}) \}$.我们可以用batch gradient descent方法来训练它。对于其中一个样本$(x,y)$,cost function的格式：
$$
\begin{align}
J(W,b; x,y) = \frac{1}{2} \left\| h_{W,b}(x) - y \right\|^2.
\end{align}
$$

这是一个(one-half)squared-error cost function.给定了m个examples,我们可以定义overall cost function如下所示：
$$
\begin{align}
J(W,b)
&= \left[ \frac{1}{m} \sum_{i=1}^m J(W,b;x^{(i)},y^{(i)}) \right]
                       + \frac{\lambda}{2} \sum_{l=1}^{n_l-1} \; \sum_{i=1}^{s_l} \; \sum_{j=1}^{s_{l+1}} \left( W^{(l)}_{ji} \right)^2
 \\
&= \left[ \frac{1}{m} \sum_{i=1}^m \left( \frac{1}{2} \left\| h_{W,b}(x^{(i)}) - y^{(i)} \right\|^2 \right) \right]
                       + \frac{\lambda}{2} \sum_{l=1}^{n_l-1} \; \sum_{i=1}^{s_l} \; \sum_{j=1}^{s_{l+1}} \left( W^{(l)}_{ji} \right)^2
\end{align}
$$

$J(W,b)$是一个平均的sum-of-squares的error项。第二项是一个regularization term（也被称之为weight decay term）。会减少weights的magnitude,帮助阻止过拟合(overfitting).


注意到weight decay parameter $\lambda$控制着两个terms的相对权重。同时也要注意符号$J$的重载，函数$J(W,b;x,y)$表示一个single example的squared error cost.$J(W,b)$表示同时包含了weight decay term的整体的cost function。

以上的cost function可以用于classification和regression问题:对于分类问题，我们可以用y=0或者y=1来表示两个class的label.如果我们使用的是tanh函数，那么可以用-1和+1来表示label了。对于regression问题，我们首先要scale我们的outputs，使得它们处于[0,1]之间(如果是tanh activation函数，那么应该是[-1,1]的范围之内)。

我们的目标是最小化W和b的函数J(W,b)。初始时，我们将$W^{(l)}_{ij}$和$b^{(l)}_i$初始化为random value near zero（比如说分布:${Normal}(0,\epsilon^2)$，其中$\epsilon$可以是:0.01）。

然后使用batch gradient descent.因为$J(W,b)$是non-convex函数，所以gradient descent可能会受到local optima的影响；然而，事实表明,gradient descent工作的效果还不错。

对parameters随机初始化很重要。如果对于任意的i,$W_{ij}^{(1)}$都一样，那么对于任意的输入$x$，输出的结果$a^{(2)}_1 = a^{(2)}_2 = a^{(2)}_3 = \ldots$。这种random initialization的目的就是为了_symmetry breaking_。

梯度下降的一次参数W,b的迭代公式为:
$$
\begin{align}
W_{ij}^{(l)} &= W_{ij}^{(l)} - \alpha \frac{\partial}{\partial W_{ij}^{(l)}} J(W,b) \\
b_{i}^{(l)} &= b_{i}^{(l)} - \alpha \frac{\partial}{\partial b_{i}^{(l)}} J(W,b)
\end{align}
$$

$\alpha$是我们所说的学习率，计算partial derivatives是上面式子的关键。我们现在描述一种_backpropagation_的算法，它可以很方便的求解partial derivatives.

我们首先描述backpropagation怎么样应用于计算$\textstyle \frac{\partial}{\partial W_{ij}^{(l)}} J(W,b; x, y)$和$\textstyle \frac{\partial}{\partial b_{i}^{(l)}} J(W,b; x, y)$。这个偏导数是cost function $J(W,b;x,y)$相对于一个训练样本$(x,y)$来说的.一旦我们计算出来这两个式子.我们可以看到overall cost function $J(W,b)$可以计算如下:
$$
\begin{align}
\frac{\partial}{\partial W_{ij}^{(l)}} J(W,b) &=
\left[ \frac{1}{m} \sum_{i=1}^m \frac{\partial}{\partial W_{ij}^{(l)}} J(W,b; x^{(i)}, y^{(i)}) \right] + \lambda W_{ij}^{(l)} \\
\frac{\partial}{\partial b_{i}^{(l)}} J(W,b) &=
\frac{1}{m}\sum_{i=1}^m \frac{\partial}{\partial b_{i}^{(l)}} J(W,b; x^{(i)}, y^{(i)})
\end{align}
$$

以上两个式子稍微有些不同，因为weight decay仅应用到了W，并没有应用到b上面。

backpropagation算法的直观解释：对于一个training example (x,y)，我们会首先运行一个“forward pass”来计算网络所有节点的activation，包括最终的输出$h_{W,b}(x)$.然后，对于l层的每一个节点i,我们都会计算出一个"error term"：$\delta^{(l)}_i$，它主要用来衡量每个node对于产生的错误的responsible程度。

对于一个ouput node。我们可以直接计算出来true target value和network的activation的差值，并且用它来定义$\delta^{(n_l)}_i$（$n_l$就是output layer）。

对于hidden units.比如说第l层的i节点，我们对于使用$a^{(l)}_i$作为输入的节点的error terms， 计算出一个加权平均的error term，基于这个平均值来计算$\delta^{(l)}_i$。

backpropagation algorithm的细节步骤如下：

 1. 执行一个feedforward pass，计算层$L_2$,$L_3$的activation,一直到$L_{n_l}$.
 2. 对于output layer(layer $n_l$)的unit i,设置
$$
\begin{align}
\delta^{(n_l)}_i
= \frac{\partial}{\partial z^{(n_l)}_i} \;\;
\frac{1}{2} \left\|y - h_{W,b}(x)\right\|^2 = - (y_i - a^{(n_l)}_i) \cdot f'(z^{(n_l)}_i)
\end{align}
$$
 3. 对于$l = n_l-1, n_l-2, n_l-3, \ldots, 2$
 对于l层的每个节点i,设置
$$
\delta^{(l)}_i = \left( \sum_{j=1}^{s_{l+1}} W^{(l)}_{ji} \delta^{(l+1)}_j \right) f'(z^{(l)}_i)
$$
 4. 计算出想要的partial derivatives，如下所示:

$$
\begin{align}
\frac{\partial}{\partial W_{ij}^{(l)}} J(W,b; x, y) &= a^{(l)}_j \delta_i^{(l+1)} \\
\frac{\partial}{\partial b_{i}^{(l)}} J(W,b; x, y) &= \delta_i^{(l+1)}.
\end{align}
$$

最后，我们用matrix-vectorial的名词结构来重新描述一遍算法。我们使用”∙”来表示element-wise乘操作(在Matlab或者Octave中，这被称之为Hadamard product)，有$\textstyle a = b \bullet c$就有$\textstyle a_i = b_ic_i$。同样的道理我们来扩展$\textstyle f(\cdot)$的定义来支持对向量的element-wise操作，对$\textstyle f'(\cdot)$.所以有$\textstyle f'([z_1, z_2, z_3]) = [f'(z_1), f'(z_2), f'(z_3)]$.

 1.同上
 2.对于output layer（layer $n_{l}$）,使：
 $$
 \begin{align} \delta^{(n_l)} = - (y - a^{(n_l)}) \bullet f'(z^{(n_l)}) \end{align}
 $$
 3.对于$\textstyle l = n_l-1, n_l-2, n_l-3, \ldots, 2$,使：
 $$
 \begin{align} \delta^{(l)} = \left((W^{(l)})^T \delta^{(l+1)}\right) \bullet f'(z^{(l)}) \end{align}
 $$
 4.计算出最终的所要偏导数:
 $$
 \begin{align}
 \nabla_{W^{(l)}} J(W,b;x,y) &= \delta^{(l+1)} (a^{(l)})^T, \\
 \nabla_{b^{(l)}} J(W,b;x,y) &= \delta^{(l+1)}.
 \end{align}
 $$

_实现笔记_ 

在step2和step3，我们需要对每一个i计算$\textstyle f'(z^{(l)}_i)$，假设$f(z)$是一个sigmoid activation function.而且我们在一个网络的forward pass中已经计算并且存储了$\textstyle a^{(l)}_i$，因此，我们可以使用它来计算出$\textstyle f'(z^{(l)}_i) = a^{(l)}_i (1- a^{(l)}_i)$.

最后，我们开始写出整个full gradient descent算法的伪代码，$\textstyle \Delta W^{(l)}$是一个矩阵(same dimension as $\textstyle W^{(l)}$),$\textstyle \Delta b^{(l)}$是一个向量(same dimension as $\textstyle b^{(l)}$).batch gradient descent的一次迭代如下所示:

 1. 对于所有的l,设置$\textstyle \Delta W^{(l)} := 0,\textstyle \Delta b^{(l)} := 0$
 2. for i=1到m，
  1. 使用backpropagation来计算$\textstyle \nabla_{W^{(l)}} J(W,b;x,y)$和$\textstyle \nabla_{b^{(l)}} J(W,b;x,y)$.
  2.设置$\textstyle \Delta W^{(l)} := \Delta W^{(l)} + \nabla_{W^{(l)}} J(W,b;x,y)$
  3.设置$\textstyle \Delta b^{(l)} := \Delta b^{(l)} + \nabla_{b^{(l)}} J(W,b;x,y)$
 3. 更新参数:

 $$
 \begin{align}
W^{(l)} &= W^{(l)} - \alpha \left[ \left(\frac{1}{m} \Delta W^{(l)} \right) + \lambda W^{(l)}\right] \\
b^{(l)} &= b^{(l)} - \alpha \left[\frac{1}{m} \Delta b^{(l)}\right]
\end{align}
 $$

为了训练我们的神经网络,我们现在可以重复上面的步骤来reduce我们的cost function $J(W,b)$
