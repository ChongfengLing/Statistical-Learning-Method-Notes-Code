# 感知机

感知机（perceptron）是**二类线性分类**模型。




## 	1. Model

> 定义 2.1 (感知机) $\quad$假设输入空间（特征空间 ) 是 $\mathcal{X} \subseteq \mathbf{R}^{n},$ 输出空间是 $\mathcal{Y}=\{+1,-1\}_{\circ}$ 输入 $x \in \mathcal{X}$ 表示实例的特征向量，对应于输入空间 ( 特征空间 $)$ 的点; 输出 $y \in \mathcal{Y}$ 表示实例的类别。由输入空间到输出空间的如下函数:
> $$
> f(x)=\operatorname{sign}(w \cdot x+b)
> $$
> 称为感知机。其中， $w$ 和 $b$ 为感知机模型参数, $w \in \mathbf{R}^{n}$ 叫作权值（weight ) 或权值向量（weight vector) $, b \in \mathbf{R}$ 叫作偏置 ( bias $), w \cdot x$ 表示 $w$ 和 $x$ 的内积。sign 是符号 函数，即
> $$
> \operatorname{sign}(x)=\left\{\begin{array}{ll}
> +1, & x \geqslant 0 \\
> -1, & x<0
> \end{array}\right.
> $$
>

- 假设空间：特征空间中的所有线性分类模型，即函数集合$\{f|f(x)=w \cdot x+b\}$

- 几何解释：线性方程$w \cdot x+b=0$是特征空间$\mathbf{R}^{n}$的一个超平面$\mathbf{S}$，把特征空间分成2个部分，使得特征向量分别划入正负两类。$\mathbf{S}$也叫分离超平面（separating hyperplane)
  - 超平面$\mathbf{S}\subseteq \mathbf{R}^{n-1}$并且截距为0。在这需要把b当成特征而不是截距才说的通。
- 例子：
  - 通过$\{房屋面积，房龄，...\}$来判断房子总价是否高于价格$a$
    - 假设数据集线性可分



## 2. Strategy

### 2.1 数据可分性

> 定义 2.2 (数据集的线性可分性) $\quad$ 给定一个数据集
> $$
> T=\left\{\left(x_{1}, y_{1}\right),\left(x_{2}, y_{2}\right), \cdots,\left(x_{N}, y_{N}\right)\right\}
> $$
> 其中， $x_{i} \in \mathcal{X}=\mathbf{R}^{n}, y_{i} \in \mathcal{Y}=\{+1,-1\}, i=1,2, \cdots, N,$ 如果存在某个超平面 $S$
> $$
> w \cdot x+b=0
> $$
> 能够将数据集的正实例点和负实例点完全正确地划分到超平面的两侧，即对所有
> $y_{i}=+1$ 的实例 $i,$ 有 $w \cdot x_{i}+b>0,$ 对所有 $y_{i}=-1$ 的实例 $i,$ 有 $w \cdot x_{i}+b<0,$ 则 称数据集 $T$ 为线性可分数据集（linearly separable data set ) ; 否则，称数据集 $T$ 线性
> 不可分。

- 存在超平面$\mathbf{S}\Longrightarrow $数据集线性可分

### 2.2 学习策略

#### 2.2.1 误分类点总数

把误分类点总数当成损失函数，不是参数$w,b$的连续可导函数，不易优化

#### 2.2.2 误分类点到超平面的总距离

> 定理（2.01) $\quad$空间  $\mathbf{R}^n$中任意一点$x_0$到超平面  $\mathbf{S}=\{x|w\cdot x+b=0\}$ 的距离为
> $$
> \frac{1}{\|w\|}\left|w \cdot x_{0}+b\right|
> $$

对于误分类数据$(x_i,y_i)$，M为误分类点集合，误分类点到超平面的总距离为
$$
\begin{align}
D&=\frac{1}{\|w\|} \sum_{x_{i} \in M} |y_{i}\left(w \cdot x_{i}+b\right)|\nonumber\\
&=-\frac{1}{\|w\|} \sum_{x_{i} \in M} y_{i}\left(w \cdot x_{i}+b\right)\\
L(w,b)&=-\sum_{x_{i} \in M} y_{i}\left(w \cdot x_{i}+b\right)
\end{align}
$$

- 对误分类点$(x_i,y_i)$，$-y_{i}\left(w \cdot x_{i}+b\right) \geq0$
- (6)是给定数据集$T=\left\{\left(x_{1}, y_{1}\right),\left(x_{2}, y_{2}\right), \cdots,\left(x_{N}, y_{N}\right)\right\}$，其中$x_{i} \in \mathcal{X}=\mathbf{R}^{n}, y_{i} \in \mathcal{Y}=\{+1,-1\}, i=1,2, \cdots, N$
- (7)非负函数，且对于$w,b$连续可导
- 我们的策略是**极小化损失函数**，即$\min _{w, b} L(w, b)=-\sum_{x_{i} \in M} y_{i}\left(w \cdot x_{i}+b\right)$



## 3. Algorithm

求解损失函数(7)的最优化问题，通过随机梯度下降(stochastic gradient descent)

###  3.1 算法原始形式

> 算法 2.1（感知机学习算法的原始形式）
> 输入: 训练数据集 $T=\left\{\left(x_{1}, y_{1}\right),\left(x_{2}, y_{2}\right), \cdots,\left(x_{N}, y_{N}\right)\right\},$ 其中 $x_{i} \in \mathcal{X}=\mathbf{R}^{n}, y_{i} \in$$\mathcal{Y}=\{-1,+1\}, i=1,2, \cdots, N ;$ 学习率 $\eta(0<\eta \leqslant 1)$;
> 输出 $: w, b ;$ 感知机模型 $f(x)=\operatorname{sign}(w \cdot x+b)$ 。
> (1) 选取初值 $w_{0}, b_{0}$;
> (2)在训练集中随机选取数据$\left(x_{i}, y_{i}\right)$;
> (3) 如果 $y_{i}\left(w \cdot x_{i}+b\right) \leqslant 0$,
> $$
> \begin{align}
> w& \leftarrow w+\eta y_{i} x_{i} \\
> b &\leftarrow b+\eta y_{i}
> \end{align}
> $$
> (4）转至 (2)，直至训练集中没有误分类点。

- 一次随机选取一个误分类点使其梯度下降

- **随机**选取一个误分类点 $\left(x_{i}, y_{i}\right),$ 对 $w, b$ 进行更新 :
  $$
  \begin{aligned}
  w &\leftarrow w+\eta y_{i} x_{i} \\
  b &\leftarrow b+\eta y_{i}
  \end{aligned}
  $$

- 不同初值，不同分类点选取会影响最后的解

### 3.2 原始算法的收敛性

证明对一个线性可分的数据集，感知机学习算法的原始形式在有限次迭代后能得到正确的分离超平面与感知机模型

> **定理 2.1 (Novikoff)** 设训练数据集 $T=\left\{\left(x_{1}, y_{1}\right),\left(x_{2}, y_{2}\right), \cdots,\left(x_{N}, y_{N}\right)\right\}$ 是线 性可分的，其中 $x_{i} \in \mathcal{X}=\mathbf{R}^{n}, y_{i} \in \mathcal{Y}=\{-1,+1\}, i=1,2, \cdots, N,$ 则
> (1) 存在满足条件 $\left\|\hat{w}_{\mathrm{opt}}\right\|=1$ 的超平面 $\hat{w}_{\mathrm{opt}} \cdot \hat{x}=w_{\mathrm{opt}} \cdot x+b_{\mathrm{opt}}=0$ 将训练数据集完全正确分开; 且存在 $\gamma>0,$ 对所有 $i=1,2, \cdots, N$
> $$
> y_{i}\left(\hat{w}_{\mathrm{opt}} \cdot \hat{x}_{i}\right)=y_{i}\left(w_{\mathrm{opt}} \cdot x_{i}+b_{\mathrm{opt}}\right) \geqslant \gamma
> $$
> (2) 令 $R=\max _{1 \leqslant i \leqslant N}\left\|\hat{x}_{i}\right\|,$ 则感知机算法 2.1 在训练数据集上的误分类次数 $k$ 满足不等式
> $$
> k \leqslant\left(\frac{R}{\gamma}\right)^{2}
> $$

- $\hat{w}=\left(w^{\mathrm{T}}, b\right)^{\mathrm{T}}$，$\hat{x}=\left(x^{\mathrm{T}}, 1\right)^{\mathrm{T}}$，$\hat{x} \in \mathbf{R}^{n+1}, \hat{w} \in \mathbf{R}^{n+1}$，$\hat{w} \cdot \hat{x}=w \cdot x+b$
- 对于线性可分数据集，感知机的解存在但不唯一。
- 在线性支持向量机中，添加超平面约束条件，从而得到唯一超平面。

### 3.3 算法对偶形式

我们假设$w,b=0$，在原始算法中，对于一个误分类点$(x_i,y_i)$，我们进行了$n_i$次更新，对于所有$N$个点，我们有
$$
\begin{aligned}
w &=\sum_{i=1}^{N} \alpha_{i} y_{i} x_{i} =\sum_{i=1}^{N} n_i(\eta  y_{i} x_{i})\\
b &=\sum_{i=1}^{N} \alpha_{i} y_{i}=\sum_{i=1}^{N} n_i(\eta y_i)
\end{aligned}
$$
我们从解w，b转化为$n_i$即第$i$个是实例点由于误分而进行更新的次数。

实例点更新越多，说明距离分离超平面越近，越难分类，对结果影响大，很可能就是支持向量。

> 算法 2.2 (感知机学习算法的对偶形式)
> 输入: 线性可分的数据集 $T=\left\{\left(x_{1}, y_{1}\right),\left(x_{2}, y_{2}\right), \cdots,\left(x_{N}, y_{N}\right)\right\},$ 其中 $x_{i} \in \mathbf{R}^{n}, y_{i} \in$
> $\{-1,+1\}, i=1,2, \cdots, N ;$ 学习率 $\eta(0<\eta \leqslant 1) ;$
> 输 出: $\alpha, b ;$ 感 知机 模 型 $f(x)=\operatorname{sign}\left(\sum_{j=1}^{N} \alpha_{j} y_{j} x_{j} \cdot x+b\right),$ 其中 $\alpha=\left(\alpha_{1}, \alpha_{2}, \cdots, \alpha_{N}\right)^{\mathrm{T}}$
> (1) $\alpha \leftarrow 0, b \leftarrow 0 ;$
> (2)在训练集中选取数据 $\left(x_{i}, y_{i}\right)$;
> (3)如果 $y_{i}\left(\sum_{j=1}^{N} \alpha_{j} y_{j} x_{j} \cdot x_{i}+b\right) \leqslant 0$
> $$
> \begin{array}{l}
> \alpha_{i} \leftarrow \alpha_{i}+\eta \\
> b \leftarrow b+\eta y_{i}
> \end{array}
> $$
> (4)转至 (2)直到没有误分类数据。

- (12)当$\eta=1$，即$n_i=n_i+1$，对点更新次数加一
- 对偶算法只涉及到矩阵的内积$x_i\cdot x_j$计算，我们可以将内积预先计算存储，即我们计算Gram矩阵$\mathbf{G}=[x_i \cdot x_j]_{\mathbf{N}\times\mathbf{N}}$

- 和原始形式一样，感知机学习算法的对偶形式迭代是收敛的，且存在多个解。



## 4. Question

1. 超平面的定义是什么？

2. 这么判断数据集可不可分？

   - 证明以下定理： 样本集线性可分的充分必要条件是正实例点集所构成的凸壳CD 与负实例点集所构成的凸壳互不相交。

   - 设集合 $S \subset \mathbf{R}^{n}$ 是由 $\mathbf{R}^{n}$ 中的 $k$ 个点所组成的集合, 即 $S=\left\{x_{1}, x_{2}, \cdots, x_{k}\right\} .$ 定义 $S$ 的凸亮$\operatorname{conv}(S)$ 为
     $$
     \begin{aligned}
     \operatorname{conv}(S)=\left\{x=\sum_{i=1}^{k} \lambda_{i} x_{i} \mid \sum_{i=1}^{k} \lambda_{i}=1, \lambda_{i} \geqslant 0, i=1,2, \cdots, k\right\}
     \end{aligned}
     $$



## 5. Code

```python
class perceptron:
    def __init__(self, X_train, Y_train, learning_rate=0.0001, tol=0):
        '''
        Parameters
        ----------
        
        X_train:Array, default=None
            dataset for training
        
        Y_train:Array, default=None
            labelset for training
        
        learning_rate:float,default=0.0001
        
        tol:int, default=0
            the stopping criterion. When the number of misclassification 
            points small or equal to tol, stop training.
        '''
        self.X_train=np.mat(X_train)
        self.Y_train=np.mat(Y_train).T
        self.m,self.n=np.shape(self.X_train)
        self.w=np.zeros((1,np.shape(self.X_train)[1]))
        self.b=0
        self.learning_rate=learning_rate
        self.tol=tol
        
    def calculate(self, x, w, b):
        return np.dot(x,w.T)+b
    
    def train(self):
        '''
        returns
        -------
        
        w: np.mat
            weights
        b: np.mat
            bias        
        '''
        is_wrong=False
        while not is_wrong:
            wrong_count=0
            for s in range(self.m):
                xi=self.X_train[s]
                yi=self.Y_train[s]
                if self.calculate(xi,self.w,self.b)*yi <= 0:
                    self.w=self.w+self.learning_rate*yi*xi
                    self.b=self.b+self.learning_rate*yi
                    wrong_count+=1
            if wrong_count <=self.tol:
                is_wrong=True
        return self.w,self.b
    
    def score(self):
        pass
    
    def plot(self):
        pass
    
```

- [More details and examples](https://github.com/ChongfengLing/Statistical-Learning-Method-Notes-Code)



## 6. Proof

### 6.1 Theorem 2.01

<img src="https://raw.githubusercontent.com/ChongfengLing/typora-picBed/main/img/SLM-2%20thm2.01%20distance" alt="image-20201123163513240" style="zoom:50%;" />

### 6.2 Theorem 2.1

<img src="https://raw.githubusercontent.com/ChongfengLing/typora-picBed/main/img/SLM-2%20thm2.1%20Novikoff.png" alt="SLM-2 thm2.1 Novikoff" style="zoom:50%;" />

## 7. Reference

[统计学习方法（第2版）](https://book.douban.com/subject/33437381/)

[统计学习方法|感知机原理剖析及实现](https://www.pkudodo.com/2018/11/18/1-4/)

[如何理解感知机学习算法的对偶形式？ - Zongrong Zheng的回答 - 知乎]( https://www.zhihu.com/question/26526858/answer/136577337)

[fengdu78/lihang-code](https://github.com/fengdu78/lihang-code)

