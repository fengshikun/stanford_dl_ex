这章的主题就是提供了一种求导数的一种近似校验方式 。

下面说一个多维的情形，假设$g_i(\theta)$是用来计算$\frac{\partial J(\theta)}{\partial \theta_i}$的函数。我们可以用如下方式来得出一个估计值。

\begin{equation}
\theta^{(i+)} = \theta + EPSILON \times \vec{e_i} 
\end{equation}

$$
\vec{e_i} = \begin{bmatrix}
        0\\
        0\\
        {\vdots}\\
        1\\
        {\vdots}\\
        0\\
        \end{bmatrix} 
$$
$\vec{e_i}$是一个$i$-th basis vector,和$\theta$是一样的维度，除了它的第i个元素是1之外(其它的元素都是0),这就意味着:
 $theta^{(i+)}$ is the same as θ, except its i-th element has been incremented by EPSILON. 

同样的定义：
\begin{equation}
\theta^{(i-)} = \theta - EPSILON \times \vec{e_i} 
\end{equation}

则可以以下式来估算$g_i(\theta)$的准确性
\begin{equation}
g_i(\theta) \approx \frac{J(\theta^{(i+)}) - J(\theta^{(i-)})}{2 \times EPSILON} 
\end{equation}
