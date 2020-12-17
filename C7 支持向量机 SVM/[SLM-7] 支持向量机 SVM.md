# 支持向量机 Support Vector Machines

支持向量机(Support Vector Machines)是一种**二元分类**模型。**基本想法是求解能够正确划分训练数据集并且几何间隔最大的分离超平面。**基本模型是**特征空间上的间隔最大**的线性分类器，区别于感知机。学习策略为间隔最大化，可化为求解凸二次规划convex quadratic programming。学习算法为求解凸二次规划的最优算法。



## 1. 线性可分支持向量机与硬间隔最大化

### 1.1 Model

> 定义 7.1 (线性可分支持向量机 ) $\quad$ 给定**线性可分**训练数据集
> $$
> T=\left\{\left(x_{1}, y_{1}\right),\left(x_{2}, y_{2}\right), \cdots,\left(x_{N}, y_{N}\right)\right\},\nonumber\\x_{i} \in \mathcal{X}=\mathbf{R}^{n}, y_{i} \in \mathcal{Y}=\{+1,-1\}, i=1,2, \cdots, N
> $$
> 通过**间隔最大化或等价地求解相应的凸二次规划问题**学习得到的分离超平面为
> $$
> w^{*} \cdot x+b^{*}=0
> $$
> 以及相应的分类决策函数
> $$
> f(x)=\operatorname{sign}\left(w^{*} \cdot x+b^{*}\right)
> $$
> 称为线性可分支持向量机。

- 对线性可分数据集，存在无穷超平面分离数据。感知机用误分类最小策略，得到的解无穷多个。SVM用间隔最大化策略，得到的解唯一。

### 1.2 间隔

1. 点到分离超平面的远近表示对分类预测的确信程度。

2. $y_{pre}$与$\hat{y}$的符号一致与否表示分类预测的准确性。

#### 1.2.1 函数间隔 Functional Margin

> 定义 7.2 (函数间隔) $\quad$ 对于给定的训练数据集 $T$ 和超平面 $(w, b),$ 定义超平面$(w, b)$ **关于样本点 $\left(x_{i}, y_{i}\right)$ 的函数间隔**为
> $$
> \hat{\gamma}_{i}=y_{i}\left(w \cdot x_{i}+b\right)
> $$
> 定义超平面 $(w, b)$ **关于训练数据集 $T$ 的函数间隔**为超平面 $(w, b)$ 对于 $T$ 中所有样本点 $\left(x_{i}, y_{i}\right)$ 的函数间隔之最小值，即
> $$
> \hat{\gamma}=\min _{i=1, \cdots, N} \hat{\gamma}_{i}
> $$

- (3)表示分类的准确性与准确程度

- 对超平面$w \cdot x+b=0$，成倍的改变$w,\;b$不会改变该平面，但是会成倍的改变函数间隔，且**倍数相等**

#### 1.2.2 几何间隔 Geometric Margin

对分离超平面$w \cdot x+b=0$的法向量$w$进行正规化，使得$||w'||=\frac{w}{\|w\|}$（同时对$b$也是）。

> 定义 7.3 (几何间隔) $\quad$ 对于给定的训练数据集 $T$ 和超平面 $(w, b),$ 定义超平面$(w, b)$ 关于样本,点 $\left(x_{i}, y_{i}\right)$ 的几何间隔为
> $$
> \gamma_{i}=y_{i}\left(\frac{w}{\|w\|} \cdot x_{i}+\frac{b}{\|w\|}\right)
> $$
> 定义超平面 $(w, b)$ 对于训练数据集 $T$ 的几何间隔为超平面 $(w, b)$ 关千 $T$ 中所有样本点 $\left(x_{i}, y_{i}\right)$ 的几何间隔之最小值，即
> $$
> \gamma=\min _{i=1, \cdots, N} \gamma_{i}
> $$

- 几何间隔不随参数的改变而改变

### 1.3 间隔最大化

- 对线性可分数据集，线性可分分离超平面有无穷多个（即感知机），但几何间隔最大的分离超平面唯一。
- 对线性可分训练集，间隔最大化又称硬间隔最大化。
- 间隔最大化，即以充分大的确信度，对训练数据进行分类；也就是说，在正负实例分开的同时，对离超平面最近的点也能有足够大的确信度。

#### 1.3.1 Algorithm

由几何间隔的定义可知，硬间隔最大化可以表示为
$$
\begin{array}{ll}
\max _{w, b} & \gamma \\
\text { s.t. } & y_{i}\left(\frac{w}{\|w\|} \cdot x_{i}+\frac{b}{\|w\|}\right) \geqslant \gamma>0, \quad i=1,2, \cdots, N
\end{array}
$$
由几何间隔和函数间隔的定义，(7)转化成
$$
\begin{array}{ll}
\max _{u \cdot b}& \frac{\hat{\gamma}}{\|w\|} \\
\text { s.t. } & y_{i}\left(w \cdot x_{i}+b\right) \geqslant \hat{\gamma}, \quad i=1,2, \cdots, N
\end{array}
$$
我们要求$w,\;b$，使得$\frac{\hat{\gamma}}{\|w\|}$最大化，在这个目标函数与约束条件中，$\hat{y}$的取值对最后的超平面没有影响。于是我们设$\hat{y}=1$，当成单位1。同时最大化$\frac{1}{\|w\|}$ 和最小化 $\frac{1}{2}\|w\|^{2}$等价。(8)转化成
$$
\begin{array}{ll}
\min _{w, b} & \frac{1}{2}\|w\|^{2} \\
\text { s.t. } & y_{i}\left(w \cdot x_{i}+b\right)-1 \geqslant 0, \quad i=1,2, \cdots, N
\end{array}
$$

- (9)为凸二次规划问题convex quadratic programming

求出(9)的解$w^*,\;b^*$，我们可以得出最大间隔分离超平面$w^{*} \cdot x+b^{*}=0$ 及分类决策函数$f(x)=\operatorname{sign}\left(w^{*} \cdot x+b^{*}\right)$，即线性可分向量机模型。

> 算法 7.1 (线性可分支持向量机学习算法————最大间隔法)
> 输入: 线性可分训练数据集 $T=\left\{\left(x_{1}, y_{1}\right),\left(x_{2}, y_{2}\right), \cdots,\left(x_{N}, y_{N}\right)\right\}$，$x_{i}\in \mathcal{X}=\mathbf{R}^{n}, y_{i} \in \mathcal{Y}=\{-1,+1\}, i=1,2, \cdots, N$
> 输出：最大间隔分离超平面和分类决策函数。
>
> （1）构造并求解约束最优化问题:
> $$
> \begin{array}{ll}
> \min _{w, b} & \frac{1}{2}\|w\|^{2} \\
> \text { s.t. } & y_{i}\left(w \cdot x_{i}+b\right)-1 \geqslant 0, \quad i=1,2, \cdots, N
> \end{array}
> $$
> 求得最优解 $w^{*}, b^{*}$ 。
> （2）由此得到分离超平面:
> $$
> w^{*} \cdot x+b^{*}=0
> $$
> 分类决策函数
> $$
> f(x)=\operatorname{sign}\left(w^{*} \cdot x+b^{*}\right)
> $$

#### 1.3.2 解的存在性与唯一性

> 定理 7.1 (最大间隔分离超平面的存在唯一性) $\quad$ 若训练数据集 $T$ 线性可分，则可将训练数据集中的样本点完全正确分开的最大间隔分离超平面存在且唯一。

#### 1.3.3 支持向量 Support Vector & 间隔边界

![image-20201217151120228](https://raw.githubusercontent.com/ChongfengLing/typora-picBed/main/img/SLM-7%20%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F.png)

对线性可分数据集：

- 支持向量：训练样本点中与分离超平面距离最近的**实例们$(x_i,\;b_i)$**，使得约束调节(10)取等号。即位于$H_{1,2}: w \cdot x+b=\pm1$上的两（或更多）点。
- 间隔边界：$H_1$和$H_2$

**此模型的结果只由这些少数个支持向量决定**，故此得名。

### 1.4 对偶算法 Dual Problem

求解最优化问题(10)，我们可以把其当成原始最优化问题，应用拉格朗日对偶性，通过求解对偶问题得到原始问题的最优解。

1. 对偶问题更容易求解。高维甚至无限维的最优化问题难求解。
2. 引入核函数

#### 1.4.1 对偶问题的导出

对于拘束最优化问题(10)
$$
\begin{array}{ll}
\min _{w, b} & \frac{1}{2}\|w\|^{2} \\
\text { s.t. } & y_{i}\left(w \cdot x_{i}+b\right)-1 \geqslant 0, \quad i=1,2, \cdots, N\nonumber \tag{10}
\end{array}
$$
的每一个约束条件，我们引入一个拉格朗日乘子 Lagrange multiplier $\alpha_{i} \geqslant 0$, $i=1,2, \cdots, N$，定义拉格朗日函数 Generalized Lagrange Function，$\alpha=\left(\alpha_{1}, \alpha_{2}, \cdots, \alpha_{N}\right)^{\mathrm{T}}$ 为拉格朗日乘子向量。
$$
\begin{align}
L(w, b, \alpha)&=\frac{1}{2}\|w\|^{2}-\sum_{i=1}^{N} \alpha_{i}(y_{i}\left(w \cdot x_{i}+b\right)-1)\nonumber
\\&=\frac{1}{2}\|w\|^{2}-\sum_{i=1}^{N} \alpha_{i} y_{i}\left(w \cdot x_{i}+b\right)+\sum_{i=1}^{N} \alpha_{i}
\end{align}
$$
带拘束最优化问题(10)可以变成无拘束优化问题(14)
$$
\begin{array}{l}
\min _{w, b} \max _{\lambda} \mathcal{L}(w, b, \alpha) \\
\text { s.t. } \lambda_{i} \geqslant 0
\end{array}
$$

由强对偶关系，拘束问题(14)可以化为无拘束优化问题
$$
\begin{array}{l}
\max _{\lambda} \min _{w, b} \mathcal{L}(w, b, \alpha) \\
\text { s.t. } \lambda_{i} \geqslant 0
\end{array}
$$

由上面的拉格朗日（强）对偶性可得，原始问题(10)的对偶问题是极大极小问题(15)

- 详细推导见：【拉格朗日对偶，等价对偶以及KKT条件】（文件没保存，有缘再续）

#### 1.4.2 对偶问题的计算

**1.4.2.1: 求$\min _{w, b} L(w, b, \alpha)=\frac{1}{2}\|w\|^{2}-\sum_{i=1}^{N} \alpha_{i} y_{i}\left(w \cdot x_{i}+b\right)+\sum_{i=1}^{N} \alpha_{i}$**

计算$\min _{w, b} L(w, b, \alpha)$对变量$w,\;b$的偏导为0
$$
\begin{array}{l}
\nabla_{w} L(w, b, \alpha)=w-\sum_{i=1}^{N} \alpha_{i} y_{i} x_{i}=0 \\
\nabla_{b} L(w, b, \alpha)=-\sum_{i=1}^{N} \alpha_{i} y_{i}=0\nonumber\\
\end{array}
$$
得
$$
\begin{array}{l}
w&=\sum_{i=1}^{N} \alpha_{i} y_{i} x_{i}\\
\sum_{i=1}^{N} \alpha_{i} y_{i}&=0
\end{array}
$$
把(16)带入到(13)，得
$$
\begin{aligned}
L(w, b, \alpha) &=\frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_{i} \alpha_{j} y_{i} y_{j}\left(x_{i} \cdot x_{j}\right)-\sum_{i=1}^{N} \alpha_{i} y_{i}\left(\left(\sum_{j=1}^{N} \alpha_{j} y_{j} x_{j}\right) \cdot x_{i}+b\right)+\sum_{i=1}^{N} \alpha_{i} \\
&=-\frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_{i} \alpha_{j} y_{i} y_{j}\left(x_{i} \cdot x_{j}\right)+\sum_{i=1}^{N} \alpha_{i}
\end{aligned}
$$
得到结果
$$
\min _{w, b} L(w, b, \alpha)=-\frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_{i} \alpha_{j} y_{i} y_{j}\left(x_{i} \cdot x_{j}\right)+\sum_{i=1}^{N} \alpha_{i}
$$
**求$\min _{w, b} L(w, b, \alpha) \text { 对 } \alpha \text { 的极大 }$**
$$
\begin{aligned}
\max _{\alpha} & -\frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_{i} \alpha_{j} y_{i} y_{j}\left(x_{i} \cdot x_{j}\right)+\sum_{i=1}^{N} \alpha_{i} \\
\text { s.t. } & \sum_{i=1}^{N} \alpha_{i} y_{i}=0 \\
& \alpha_{i} \geqslant 0, \quad i=1,2, \cdots, N
\end{aligned}
$$
调整符号，得与原始问题等价的对偶最优化问题
$$
\begin{array}{1}
\min _{\alpha} & \frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_{i} \alpha_{j} y_{i} y_{j}\left(x_{i} \cdot x_{j}\right)-\sum_{i=1}^{N} \alpha_{i} \\
\text { s.t. } & \sum_{i=1}^{N} \alpha_{i} y_{i}=0 \\
& \alpha_{i} \geqslant 0, \quad i=1,2, \cdots, N
\end{array}
$$

- $ \sum_{i=1}^{N} \alpha_{i} y_{i}=0$意味着很大部分的$\alpha=0$，即决定超平面的只是很小一部分的支持向量。

#### 1.4.3 对偶问题的解

> 定理 7.2 设 $\alpha^{*}=\left(\alpha_{1}^{*}, \alpha_{2}^{*}, \cdots, \alpha_{l}^{*}\right)^{\mathrm{T}}$ 是对偶最优化问题 (7.22)$\sim(7.24)$ 的解，则存在下标 $j,$ 使得 $\alpha_{j}^{*}>0,$ 并可按下式求得原始最优化问题 (7.13)$\sim(7.14)$ 的解 $w^{*}, b^{*}$ :
> $$
> \begin{array}{c}
> w^{*}=\sum_{i=1}^{N} \alpha_{i}^{*} y_{i} x_{i} \\
> b^{*}=y_{j}-\sum_{i=1}^{N} \alpha_{i}^{*} y_{i}\left(x_{i} \cdot x_{j}\right)
> \end{array}
> $$

分离超平面为$\sum_{i=1}^{N} \alpha_{i}^{*} y_{i}\left(x \cdot x_{i}\right)+b^{*}=0$

分类决策函数为$f(x)=\operatorname{sign}\left(\sum_{i=1}^{N} \alpha_{i}^{*} y_{i}\left(x \cdot x_{i}\right)+b^{*}\right)$

#### 1.4.4 Algorithm

> 算法 7.2 (线性可分支持向量机学习算法)
> 输入: 线性可分训练集 $T=\left\{\left(x_{1}, y_{1}\right),\left(x_{2}, y_{2}\right), \cdots,\left(x_{N}, y_{N}\right)\right\},$ 其中 $x_{i} \in \mathcal{X}=\mathbf{R}^{n}$, $y_{i} \in \mathcal{Y}=\{-1,+1\}, i=1,2, \cdots, N$
> 输出：分离超平面和分类决策函数。
> （1）构造并求解约束最优化问题
> $$
> \begin{array}{ll}
> \min _{\alpha} & \frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_{i} \alpha_{j} y_{i} y_{j}\left(x_{i} \cdot x_{j}\right)-\sum_{i=1}^{N} \alpha_{i} \\
> \text { s.t. } & \sum_{i=1}^{N} \alpha_{i} y_{i}=0 \\
> & \alpha_{i} \geqslant 0, \quad i=1,2, \cdots, N
> \end{array}
> $$
> 求得最优解 $\alpha^{*}=\left(\alpha_{1}^{*}, \alpha_{2}^{*}, \cdots, \alpha_{N}^{*}\right)^{\mathrm{T}}$ 。
> (2) 计算
> $$
> w^{*}=\sum_{i=1}^{N} \alpha_{i}^{*} y_{i} x_{i}
> $$
> 并选择 $\alpha^{*}$ 的一个正分量 $\alpha_{j}^{*}>0,$ 计算
> $$
> b^{*}=y_{j}-\sum_{i=1}^{N} \alpha_{i}^{*} y_{i}\left(x_{i} \cdot x_{j}\right)
> $$
> （3）求得分离超平面
> $$
> w^{*} \cdot x+b^{*}=0
> $$
> 分类决策函数:
> $$
> f(x)=\operatorname{sign}\left(w^{*} \cdot x+b^{*}\right)
> $$



## 2. 线性支持向量机与软间隔最大化

对线性不可分数据集，存在一些特异点（outlier）使得不等式不能全部满足，即函数间隔大于1。

因此我们引入松弛变量$\xi_i \geq0$，使得函数间隔大于等于$1-\xi_i$。同时对松弛变量付出一个惩罚参数。

- 约束条件
  $$
  y_{i}\left(w \cdot x_{i}+b\right) \geqslant 1-\xi_{i}
  $$
  
- 目标函数
  $$
  \frac{1}{2}\|w\|^{2}+C \sum_{i=1}^{N} \xi_{i}
  $$

  - $C$为惩罚参数。$C$越大，对误分类的惩罚越大
  - 最小化目标函数，即使$\frac{1}{2}\|w\|^{2}$尽可能小，间隔尽可能大

### 2.1 原始问题

$$
\begin{array}{ll}
\min _{w, b, \xi} & \frac{1}{2}\|w\|^{2}+C \sum_{i=1}^{N} \xi_{i} \\
\text { s.t. } & y_{i}\left(w \cdot x_{i}+b\right) \geqslant 1-\xi_{i}, \quad i=1,2, \cdots, N \\
& \xi_{i} \geqslant 0, \quad i=1,2, \cdots, N
\end{array}
$$

- 凸二次规划问题
- $w$的解唯一
- $b$的解可能不唯一，而是一个区间

### 2.2 拉格朗日函数

$$
L(w, b, \xi, \alpha, \mu) \equiv \frac{1}{2}\|w\|^{2}+C \sum_{i=1}^{N} \xi_{i}-\sum_{i=1}^{N} \alpha_{i}\left(y_{i}\left(w \cdot x_{i}+b\right)-1+\xi_{i}\right)-\sum_{i=1}^{N} \mu_{i} \xi_{i}\\
\alpha_i\geq 0,\;\mu_i\geq0
$$

### 2.3 对偶问题

原始问题(27)的对偶问题使拉格朗日函数的极大极小问题。
$$
\begin{array}{ll}
\min _{\alpha} & \frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_{i} \alpha_{j} y_{i} y_{j}\left(x_{i} \cdot x_{j}\right)-\sum_{i=1}^{N} \alpha_{i} \\
\text { s.t. } & \sum_{i=1}^{N} \alpha_{i} y_{i}=0 \\
& 0 \leqslant \alpha_{i} \leqslant C, \quad i=1,2, \cdots, N
\end{array}
$$

### 2.4 解的等价性

> 定理 7.3 设 $\alpha^{*}=\left(\alpha_{1}^{*}, \alpha_{2}^{*}, \cdots, \alpha_{N}^{*}\right)^{\mathrm{T}}$ 是对偶问题的一个解，若存在 $\alpha^{*}$ 的一个分量 $\alpha_{j}^{*}, 0<\alpha_{j}^{*}<C,$ 则原始问题的解 $w^{*}, b^{*}$ 可按下式 求得:
> $$
> \begin{array}{c}
> w^{*}=\sum_{i=1}^{N} \alpha_{i}^{*} y_{i} x_{i} \\
> b^{*}=y_{j}-\sum_{i=1}^{N} y_{i} \alpha_{i}^{*}\left(x_{i} \cdot x_{j}\right)
> \end{array}
> $$

> 算法 7.3 (线性支持向量机学习算法)
> 输入: 训练数据集 $T=\left\{\left(x_{1}, y_{1}\right),\left(x_{2}, y_{2}\right), \cdots,\left(x_{N}, y_{N}\right)\right\},$ 其中， $x_{i} \in \mathcal{X}=\mathbf{R}^{n}$, $y_{i} \in \mathcal{Y}=\{-1,+1\}, i=1,2, \cdots, N$
> 输出：分离超平面和分类决策函数。
> （1）选择惩罚参数 $C>0,$ 构造并求解凸二次规划问题
> $$
> \begin{array}{ll}
> \min _{\alpha} & \frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_{i} \alpha_{j} y_{i} y_{j}\left(x_{i} \cdot x_{j}\right)-\sum_{i=1}^{N} \alpha_{i} \\
> \text { s.t. } & \sum_{i=1}^{N} \alpha_{i} y_{i}=0 \\
> & 0 \leqslant \alpha_{i} \leqslant C, \quad i=1,2, \cdots, N
> \end{array}
> $$
> 求得最优解 $\alpha^{*}=\left(\alpha_{1}^{*}, \alpha_{2}^{*}, \cdots, \alpha_{N}^{*}\right)^{\mathrm{T}}$ 
> （2） 计算 $w^{*}=\sum_{i=1}^{N} \alpha_{i}^{*} y_{i} x_{i}$
> 选择 $\alpha^{*}$ 的一个分量 $\alpha_{j}^{*}$ 适合条件 $0<\alpha_{j}^{*}<C$ ，计算
> $$
> b^{*}=y_{j}-\sum_{i=1}^{N} y_{i} \alpha_{i}^{*}\left(x_{i} \cdot x_{j}\right)
> $$
> （3）求得分离超平面
> $$
> w^{*} \cdot x+b^{*}=0
> $$
> 分类决策函数:
> $$
> f(x)=\operatorname{sign}\left(w^{*} \cdot x+b^{*}\right)
> $$

- (32)中，每一个合适的$\alpha$都会得出一个$b$，所以$b$不唯一

### 2.4 支持向量

<img src="https://raw.githubusercontent.com/ChongfengLing/typora-picBed/main/img/SLM-7%20%E8%BD%AF%E9%97%B4%E9%9A%94%E5%88%86%E7%A6%BB%E5%90%91%E9%87%8F.png" alt="image-20201216151045483" style="zoom:50%;" />

虚线是间隔边界，实线是分离超平面。对每一个实例点$(x_i,y_i)$和他们的$\alpha _i^*,\;\xi_i$，我们有

- $\alpha_{i}^{*}<C$，则 $\xi_{i}=0$
- $\alpha_{i}^{*}=C, 0<\xi_{i}<1$，则
- $\alpha_{i}^{*}=C, \xi_{i}=1$，则
- $\alpha_{i}^{*}=C, \xi_{i}>1$，则
- 没有$\alpha_{i}^{*}=C, 0<\xi_{i}<1$

### 2.5 合页损失函数 Hinge Loss Function

> 定理 7.4 线性支持向量机原始最优化问题:
> $$
> \begin{array}{ll}
> \min _{w, b, \xi} & \frac{1}{2}\|w\|^{2}+C \sum_{i=1}^{N} \xi_{i} \\
> \text { s.t. } & y_{i}\left(w \cdot x_{i}+b\right) \geqslant 1-\xi_{i}, \quad i=1,2, \cdots, N \\
> & \xi_{i} \geqslant 0, \quad i=1,2, \cdots, N
> \end{array}
> $$
> 等价于最优化问题
> $$
> \min _{w, b} \sum_{i=1}^{N}\left[1-y_{i}\left(w \cdot x_{i}+b\right)\right]_{+}+\lambda\|w\|^{2}
> $$

- $[z]_{+}=\left\{\begin{array}{ll}z, & z>0 \\ 0, & z \leqslant 0\end{array}\right.$表示取正值的函数。
- $L(y(w \cdot x+b))=[1-y(w \cdot x+b)]_{+}$ 经验损失。但实例点被正确分类且函数间隔大于1，损失才是0。位于间隔边界上的实例点损失不为0。
- $\lambda\|w\|^{2}$ 正则化项

#### 2.5.1 损失函数对比

<img src="https://raw.githubusercontent.com/ChongfengLing/typora-picBed/main/img/SLM-7%20%E5%90%88%E9%A1%B5%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0.png" alt="image-20201216155233841" style="zoom:50%;" />

虚线为感知机的损失函数$\left[-y_{i}\left(w \cdot x_{i}+b\right)\right]_{+}$

- 0-1损失函数为二元分类真正的损失函数。**不过因其不连续可导，无法直接优化它所构成的函数**。
- 合页损失函数是0-1损失函数的上界。**线性支持向量机是优化由0-1损失函数的上界构成的目标函数**。
- 合页损失函数对学习有更高要求。不但需要分类正确，还要确信度（距离）足够高才为0损失。



## 3. 非线性支持向量机与核函数

对于一个非线性数据集，首先使用一个变换将原空间的数据映射到新空间；然后在新空间里用线性分类学习方法从训练数据中学习分类模型。

由于特征空间是高维甚至无限维，加上对偶问题中多组高维向量计算内积。因此引入核函数减少计算量。

### 3.1 核技巧 Kernel Trick

核技巧应用到支持向量机，其基本想法就是通过一个非线性变换将输入空间 (欧氏空间 $\mathbf{R}^{n}$ 或离散集合) 对应于一个特征空间 (希尔伯特空间 $\mathcal{H}$ )，使得在输入空间 $\mathbf{R}^{n}$ 中的超曲面模型对应于特征空间 $\mathcal{H}$ 中的超平面模型支持向量机）。这样，分类问题的学习任务通过在特征空间中求解线性支持向量机就可以完成。

> 定义 7.6 (核函数) $\quad$ 设 $\mathcal{X}$ 是输入空间 ( 欧氏空间 $\mathbf{R}^{n}$ 的子集或离散集合 $),$ 又设 $\mathcal{H}$ 为特征空间 $($ 希 尔伯特空间 $),$ 如果存在一个从 $\mathcal{X}$ 到 $\mathcal{H}$ 的映射
> $$
> \phi(x): \mathcal{X} \rightarrow \mathcal{H}
> $$
> 使得对所有 $x, z \in \mathcal{X},$ 函数 $K(x, z)$ 满足条件
> $$
> K(x, z)=\phi(x) \cdot \phi(z)
> $$
> 则称 **$K(x, z)$ 为核函数， $\phi(x)$ 为映射函数**，式中 $\phi(x) \cdot \phi(z)$ 为 $\phi(x)$ 和 $\phi(z)$ 的内积。

- **在使用过程中只定义核函数$K(x,z)$，不显式定义映射函数$\phi$**
- 特征空间$\mathcal{H}$一般是高维甚至无限维
- 给定核$K(x,z)$，特征空间和映射函数不唯一

在支持向量机中，通过映射函数$\phi$把输入空间变换到新的特征空间，在新的特征空间中训练线性支持向量机。此时对偶问题的目标函数成为$W(\alpha)=\frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_{i} \alpha_{j} y_{i} y_{j} K\left(x_{i}, x_{j}\right)-\sum_{i=1}^{N} \alpha_{i}$，分类决策函数成为$f(x)=\operatorname{sign}\left(\sum_{i=1}^{N_{s}} a_{i}^{*} y_{i} \phi\left(x_{i}\right) \cdot \phi(x)+b^{*}\right)=\operatorname{sign}\left(\sum_{i=1}^{N_{s}} a_{i}^{*} y_{i} K\left(x_{i}, x\right)+b^{*}\right)$

### 3.2 正定核

> 定理 7.5 (正定核的充要条件) $\quad$ 设 $K: \mathcal{X} \times \mathcal{X} \rightarrow \mathbf{R}$ 是对称函数，则 $K(x, z)$ 为正 定核函数的充要条件是对任意 $x_{i} \in \mathcal{X}, i=1,2, \cdots, m, K(x, z)$ 对应的 Gram 矩阵:
> $$
> K=\left[K\left(x_{i}, x_{j}\right)\right]_{m \times m}
> $$
> 是半正定矩阵。

> 定义 7.7 (正定核的等价定义) $\quad$ 设 $\mathcal{X} \subset \mathbf{R}^{n}, K(x, z)$ 是定义在 $\mathcal{X} \times \mathcal{X}$ 上的对称
> 函数, 如果对任意 $x_{i} \in \mathcal{X}, i=1,2, \cdots, m, K(x, z)$ 对应的 Gram 矩阵
> $$
> K=\left[K\left(x_{i}, x_{j}\right)\right]_{m \times m}
> $$

### 3.3 常用核函数

#### 3.3.1 多项式核函数 Polynomial Kernel Function

$$
K(x, z)=(x \cdot z+1)^{p}
$$

对应的支持向量机是一个 $p$ 次多项式分类器。在此情形下，分类决策函数成为
$$
f(x)=\operatorname{sign}\left(\sum_{i=1}^{N_{s}} a_{i}^{*} y_{i}\left(x_{i} \cdot x+1\right)^{p}+b^{*}\right)
$$

#### 3.3.2 高斯核函数 Gaussian Kernel Function

$$
K(x, z)=\exp \left(-\frac{\|x-z\|^{2}}{2 \sigma^{2}}\right)
$$

对应的支持向量机是高斯径向基函数（radial basis function）分类器。在此情形下，分类决策函数成为
$$
f(x)=\operatorname{sign}\left(\sum_{i=1}^{N_{s}} a_{i}^{*} y_{i} \exp \left(-\frac{\left\|x-x_{i}\right\|^{2}}{2 \sigma^{2}}\right)+b^{*}\right)
$$

#### 3.3.3 字符串核函数 String Kernel Function

定义在离散数据集合上。

### 3.4 非线性支持向量机

> 定义 7.8 (非线性支持向量机 $) \quad$ 从非线性分类训练集，通过核函数与软间隔最大化，或凸二次规划(46)学习得到的分类决策函数
> $$
> f(x)=\operatorname{sign}\left(\sum_{i=1}^{N} \alpha_{i}^{*} y_{i} K\left(x, x_{i}\right)+b^{*}\right)
> $$
> 称为非线性支持向量机, $K(x, z)$ 是正定核涵数。



> 算法 7.4 (非线性支持向量机学习算法) 输入: 训练数据集 $T=\left\{\left(x_{1}, y_{1}\right),\left(x_{2}, y_{2}\right), \cdots,\left(x_{N}, y_{N}\right)\right\},$ 其中 $x_{i} \in \mathcal{X}=\mathbf{R}^{n}, y_{i} \in$$\mathcal{Y}=\{-1,+1\}, i=1,2, \cdots, N$
> 输出：分类决策函数。
> （1）选取适当的核函数 $K(x, z)$ 和适当的参数 $C,$ 构造并求解最优化问题
> $$
> \begin{array}{1}
> \min _{\alpha} & \frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_{i} \alpha_{j} y_{i} y_{j} K\left(x_{i}, x_{j}\right)-\sum_{i=1}^{N} \alpha_{i} \\
> \text { s.t. } & \sum_{i=1}^{N} \alpha_{i} y_{i}=0 \\
> & 0 \leqslant \alpha_{i} \leqslant C, \quad i=1,2, \cdots, N
> \end{array}
> $$
> 求得最优解 $\alpha^{*}=\left(\alpha_{1}^{*}, \alpha_{2}^{*}, \cdots, \alpha_{N}^{*}\right)^{\mathrm{T}}$ 。
> （2）选择 $\alpha^{*}$ 的一个正分量 $0<\alpha_{j}^{*}<C,$ 计算
> $$
> b^{*}=y_{j}-\sum_{i=1}^{N} \alpha_{i}^{*} y_{i} K\left(x_{i}, x_{j}\right)
> $$
> （3）构造决策函数:
> $$
> f(x)=\operatorname{sign}\left(\sum_{i=1}^{N} \alpha_{i}^{*} y_{i} K\left(x, x_{i}\right)+b^{*}\right)
> $$

- $K(x,z)$为正定核函数是，问题(46)是二次规划问题，解存在。



## 4. SMO Algorithm (sequential minimal optimization)

序列最小算法用来求解凸二次规划的对偶问题
$$
\begin{array}{ll}
\min _{\alpha} & \frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_{i} \alpha_{j} y_{i} y_{j} K\left(x_{i}, x_{j}\right)-\sum_{i=1}^{N} \alpha_{i} \\
\text { s.t. } & \sum_{i=1}^{N} \alpha_{i} y_{i}=0 \\
& 0 \leqslant \alpha_{i} \leqslant C, \quad i=1,2, \cdots, N
\end{array}
$$


## 5. Reference

[统计学习方法（第2版）](https://book.douban.com/subject/33437381/)

