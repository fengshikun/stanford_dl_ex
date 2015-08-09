##Softmax Regression

Softmax regression是一种多类别分类的logistic regression（相当于对两分类问题进行了泛化）。

原来我们处理的是binary的情况：$y^{(i)}\in\{0,1\}$.
多分类问题解决如下:$y^{(i)}\in\{1,...,K\}$.K表示类别的个数.


在多分类问题中，对于我们的训练集${(x^{(1)},y^{(1)}),...,(x^{(m)},y^{(m)})}$来说，现在$y^{(i)}\in\{1,...,K\}$

对于一个输入$x$,我们估算y取每个值 $k=1,...,K$的概率$P(y=k|x)$。所以我们的 hypothesis $h_{\theta}(x)$取值的格式如下所示：

$$
        h_{\theta}(x) = \begin{bmatrix}
        P(y=1|x;\theta)\\
        P(y=2|x;\theta)\\
        {\vdots}\\
        P(y=K|x;\theta)
        \end{bmatrix} = \frac{1}{\sum_{j=1}^{K}exp(\theta^{(j)T}x)} \begin{bmatrix}
        exp(\theta^{(1)T}x)\\
        exp(\theta^{(2)T}x)\\
        {\vdots}\\
        exp(\theta^{(K)T}x)
        \end{bmatrix}
$$

在此处$\theta^{(1)},\theta^{(2)},...,\theta^{(K)}\in \mathfrak{R}^n$都是我们的模型里面的参数。$\frac{1}{\sum_{j=1}^{K}exp(\theta^{(j)T}x)}$是为了normalization.为了让矩阵内的项加起来和为1.

在此，我们将表征模型的$\theta$写成一个由$\theta^{(1)},\theta^{(2)},...,\theta^{(K)}$ n*K的矩阵，

$$
	\theta = \begin{bmatrix}
        | & | & | & | \\
        \theta^{(1)}  & \theta^{(2)} &  \cdots & \theta^{(K)}\\
        | & | & | & | \\
        \end{bmatrix}
$$

下面来一个"indicator function"的定义: 1{a true statement}=1, 1{a false statement}=0.
Softmax Regression的cost function定义如下：

$$
        J(\theta) = -\left[\sum_{i=1}^{m}\sum_{k=1}^{K}1\{y^{(i)}=k\}log\frac{exp(\theta^{(k)T}x^{(i)})}{\sum_{j=1}^{K}exp(\theta^{(j)T}x^{(i)})}\right]
$$

回忆一下之前二分类的写法
$$
        J(\theta) = -\left[\sum_{i=1}^{m}(1-y^{(i)})log(1-h_{\theta}(x^{(i)})) + y^{(i)}logh_{\theta}(x^{(i)})\right]
$$

泛化之后的格式和现在多分类的情形也差不多

$$
        J(\theta) = -\left[\sum_{i=1}^{m}\sum_{k=1}^{K}1\{y^{(i)}=k\}logP(y^{(i)}=k|x^{(i)};\theta)\right]
$$

在softmax中,
$$
P(y^{(i)} = k | x^{(i)} ; \theta) = \frac{\exp(\theta^{(k)\top} x^{(i)})}{\sum_{j=1}^K \exp(\theta^{(j)\top} x^{(i)}) }
$$

为了最小化$J(\theta)$,我们需要给出一个求偏导的公式:
$$
\begin{align}
\nabla_{\theta^{(k)}} J(\theta) = - \sum_{i=1}^{m}{ \left[ x^{(i)} \left( 1\{ y^{(i)} = k\}  - P(y^{(i)} = k | x^{(i)}; \theta) \right) \right]  }
\end{align}
$$
注意到$\nabla_{\theta^{(k)}} J(\theta)$本身是一个向量。所以$J(\theta)$相对于$\theta^{(k)}$的第j个元素的偏导就是该向量的第j个元素。

softmax regression的要求的参数有一个性质是：redundant. 看一下下面公式的推导：

$$
\begin{align}
P(y^{(i)} = k | x^{(i)} ; \theta)
&= \frac{\exp((\theta^{(k)}-\psi)^\top x^{(i)})}{\sum_{j=1}^K \exp( (\theta^{(j)}-\psi)^\top x^{(i)})}  \\
&= \frac{\exp(\theta^{(k)\top} x^{(i)}) \exp(-\psi^\top x^{(i)})}{\sum_{j=1}^K \exp(\theta^{(j)\top} x^{(i)}) \exp(-\psi^\top x^{(i)})} \\
&= \frac{\exp(\theta^{(k)\top} x^{(i)})}{\sum_{j=1}^K \exp(\theta^{(j)\top} x^{(i)})}.
\end{align}
$$

一句话来说明就是：subtracting ψ from every θ(j) does not affect our hypothesis’ predictions at all! 
造成的后果是:多个$\theta$可以对应同一个hypothesis prediction.

这一段留空吧 暂时没看懂

##和Logistic Regression的关系
logistic regression是softmax regression的$K=2$的特殊情况。

$$
\begin{align}
h_\theta(x) &=
\frac{1}{ \exp(\theta^{(1)\top}x)  + \exp( \theta^{(2)\top} x^{(i)} ) }
\begin{bmatrix}
\exp( \theta^{(1)\top} x ) \\
\exp( \theta^{(2)\top} x )
\end{bmatrix}
\end{align}
$$

同样 我们意识到这个hypothesis function是overparameterized的，所以我们设置$\psi = \theta^{(2)}$。并且让每一个参数都减去$\theta^{(2)}$,所以有下式:

\begin{equation}
\begin{align}
h(x) &=

\frac{1}{ \exp( (\theta^{(1)}-\theta^{(2)})^\top x^{(i)} ) + \exp(\vec{0}^\top x) }
\begin{bmatrix}
\exp( (\theta^{(1)}-\theta^{(2)})^\top x )
\exp( \vec{0}^\top x ) \\
\end{bmatrix} \\

&=
\begin{bmatrix}
\frac{1}{ 1 + \exp( (\theta^{(1)}-\theta^{(2)})^\top x^{(i)} ) } \\
\frac{\exp( (\theta^{(1)}-\theta^{(2)})^\top x )}{ 1 + \exp( (\theta^{(1)}-\theta^{(2)})^\top x^{(i)} ) }
\end{bmatrix} \\

&=
\begin{bmatrix}
\frac{1}{ 1  + \exp( (\theta^{(1)}-\theta^{(2)})^\top x^{(i)} ) } \\
1 - \frac{1}{ 1  + \exp( (\theta^{(1)}-\theta^{(2)})^\top x^{(i)} ) } \\
\end{bmatrix}
\end{align}
\end{equation}

只剩下$\theta^{(2)}-\theta^{(1)}$了，我们将它替换为$\theta'$，这样就将要求解的两个参数变成了一个参数。

