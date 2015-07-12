##1.一些前言
_Logistic Regression_ 解决结果是离散情况下的问题，例如$y^{(i)}\in \{0,1\}$。

属于1的概率相对于属于0的概率的定义( _In logistic regression we use a different hypothesis class to try to predict the probability that a given example belongs to the “1” class versus the probability that it belongs to the “0” class._ )

\begin{equation}
P(y=1|x) = h_{\theta}(x) = \frac{1}{1+exp(-{\theta}^Tx)} \equiv \sigma(\theta^Tx),
P(y=0|x) = 1-P(y=1|x) = 1-h_{\theta}(x).
\end{equation}

$\sigma(z)\equiv 1+exp(-z)$被称之为sigmoid函数：it is an S-shaped function that “squashes” the value of ${\theta}^Tx$ into the range [0,1]。下列cost function衡量了$h_{\theta}$的好坏:

\begin{equation}
J(\theta) = - \sum_{i}(y^{(i)}log(h_{\theta}(x^{(i)})) + (1-y^{(i)})log(1-h_{\theta}(x^{(i)}))).
\end{equation}

下面给出$J(\theta)$相对于$\theta_j$的偏导数

\begin{equation}
\frac{\partial J(\theta)}{\partial \theta_j} = \sum_{i}x_{j}^{i}(h_{\theta}(x^{i})-y^{(i)})
\end{equation}

整个偏导可以写作向量形式

\begin{equation}
\triangledown_{\theta}J(\theta) = \sum_{i}x^{(i)}(h_{\theta}(x^{i})-y^{(i)})
\end{equation}

This is essentially the same as the gradient for linear regression except that now $h_{\theta}(x)=\sigma(\theta^{T}x)$.


