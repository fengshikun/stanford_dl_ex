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


