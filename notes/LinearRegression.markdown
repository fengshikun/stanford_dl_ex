线性回归：to start out we will use linear functions: 

\begin{equation}
h_\theta(x) = \sum_{j}\theta_j x_j = \theta x.
\end{equation}

面对纷繁复杂的算法，我们值需要计算两个关于$J(\theta)$的信息即可：$J(\theta)$和$\triangledown_{\theta}J(\theta)$,$\triangledown_{\theta}J(\theta)$的信息含义教程里说的很明确：_Recall that the gradient $\triangledown_{\theta}J(\theta)$ of a differentiable function $J$ is a vector that points in the direction of steepest increase as a function of $\theta$ — so it is easy to see how an optimization algorithm could use this to make a small change to $\theta$ that decreases (or increase) $J(\theta)$_

$$
        \triangledown_{\theta}J(\theta) = \begin{bmatrix}
        {\frac{\partial J(\theta)}{\partial \theta_1}}\\
        {\frac{\partial J(\theta)}{\partial \theta_2}}\\
        {\vdots}\\
        {\frac{\partial J(\theta)}{\partial \theta_n}}
        \end{bmatrix}
$$