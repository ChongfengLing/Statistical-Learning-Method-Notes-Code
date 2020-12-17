# 朴素贝叶斯 Naive Bayes

基于Bayes Theorem和特征条件独立假设的分类方法。

## 0. Bayes Theorem

$$
\underbrace{P(X|Y)}_{posterior}=\frac{\overbrace{P(Y|X)}^{likelihood}\overbrace{P(X)}^{prior}}{\underbrace{P(Y)}_{evidence}}=\frac{\overbrace{P(Y|X)}^{likelihood}\overbrace{P(X)}^{prior}}{\underbrace{\sum\limits_x P(Y|X)P(X)}_{evidence}}
$$

- $P(X)$：先验概率prior
- $P(X|Y)$：后验概率posterior
- $P(Y|X)$：似然likelihood。



## 1. Model

- 输入空间$\mathcal{X} \subseteq \mathbf{R}^{n}$ 为 $n$ 维向量的集合，输出空间为类集合$\mathcal{Y}=\left\{c_{1}, c_{2}, \cdots, c_{K}\right\}$
- 输入特征向量$x$，输出类标记$y$
- $X$是输入空间上的随机向量，$Y$是输出空间上的随机变量

- 训练数据集
  $$
  \begin{aligned}
  T=\left\{\left(x_{1}, y_{1}\right),\left(x_{2}, y_{2}\right), \cdots,\left(x_{N}, y_{N}\right)\right\}
  \end{aligned}
  $$
  由 $P(X, Y)$ 独立同分布产生。

  - 这边的独立指的是向量$x_1$与$x_2$等之间的独立，不是特征条件独立。

### 1.1 学习联合概率分布$P(X,Y)$

1. $P(X, Y)=P(Y) P(X \mid Y)=P(X) P(Y \mid X)$，用第一个等式

2. **先验概率分布**
   $$
   P\left(Y=c_{k}\right), \quad k=1,2, \cdots, K
   $$

   - 由测试集可知，有多少概率/比例的$c_1$标签

3. **条件概率分布**
   $$
   P\left(X=x \mid Y=c_{k}\right)=P\left(X^{(1)}=x^{(1)}, \cdots, X^{(n)}=x^{(n)} \mid Y=c_{k}\right),\\ \quad k=1,2, \cdots, K
   $$

   - 特征向量相等$\Longrightarrow$每一维度上都相等

   - 朴素贝叶斯法对条件概率做条件独立性假设，即
     $$
     \begin{align}
     P\left(X=x \mid Y=c_{k}\right) &=P\left(X^{(1)}=x^{(1)}, \cdots, X^{(n)}=x^{(n)} \mid Y=c_{k}\right)\nonumber \\
     &=\prod_{j=1}^{n} P\left(X^{(j)}=x^{(j)} \mid Y=c_{k}\right)
     \end{align}
     $$

   - 如果不加独立分布假设，那么(1,1,1,1)和(1,1,1,2)独立，没有可以利用的信息。需要更大的测试集。
   
4. **重复“词语”的处理**

   我们得到一个垃圾邮件向量$x=['发票'，‘发票’，‘发票‘，‘充值’，’购物‘]$，如何处理出现3次的“发票”

   1. 多项式模型
      - 每一个“发票”都是独立的，统计多次，即$P(X=('发票'，‘发票’，‘发票‘)|Y=c_k)=P^3('发票|Y=c_k')$
   2. 伯努利模型
      - 只分成出现，不出现的二元情况，即$P(X=('发票'，‘发票’，‘发票‘)|Y=c_k)=P('发票|Y=c_k')$
   3. 混合模型
      - 计算单一句子的概率时用伯努利模型
      - 计算全文词语的概率时用多项式模型
   4. 高斯模型
      - 适用于连续变量
      - 假设：在给定一个类别$c_k$后，各个特征符合正态分布
      - $P\left(x^{(i)} \mid y=c_k\right)=\frac{1}{\sqrt{2 \pi \sigma_{y}^{2}}} \exp \left(-\frac{\left(x^{(i)}-\mu_{y}\right)^{2}}{2 \sigma_{y}^{2}}\right)$
        - $\mu_y:$ 在类别为 $y$ 的样本中，特征 $x^{(i)}$ 的均值。
          $\sigma_{y}:$ 在类别为 $y$ 的样本中，特征 $x^{(i)}$ 的标准差。

### 1.2 计算后验分布

通过贝叶斯定理，得到在给定了一个特征向量的情况下，标签为$c_k$的概率。
$$
\begin{align}
P(Y \mid X)&=\frac{P(X, Y)}{P(X)}=\frac{P(Y) P(X \mid Y)}{\sum_{Y} P(Y) P(X \mid Y)}\\
P\left(Y=c_{k} \mid X=x\right)&=\frac{P\left(X=x \mid Y=c_{k}\right) P\left(Y=c_{k}\right)}{\sum_{k} P\left(X=x \mid Y=c_{k}\right) P\left(Y=c_{k}\right)}\\
&=\frac{P\left(Y=c_{k}\right) \prod P\left(X^{(j)}=x^{(j)} \mid Y=c_{k}\right)}{\sum_{k} (P\left(Y=c_{k}\right) \prod_{j} P\left(X^{(j)}=x^{(j)} \mid Y=c_{k}\right))}, \quad k=1,2, \cdots, K
\end{align}
$$

- 把(2)带入到(5)，得到等式(6)

### 1.3 后验概率最大化

得到了概率表达式之后，我们便去寻找使得概率$P\left(Y=c_{k} \mid X=x\right)$最大的标签$c_k$

朴素贝叶斯分类器：
$$
\begin{array}{l}
y=f(x)=\arg \max _{c_{k}} \frac{P\left(Y=c_{k}\right) \prod_{j} P\left(X^{(j)}=x^{(j)} \mid Y=c_{k}\right) \\ }{\sum_{k} P\left(Y=c_{k}\right) \prod_{j} P\left(X^{(j)}=x^{(j)} \mid Y=c_{k}\right)}
\end{array}
$$
(7)中的分母本质上为$P(X=x)$，与$c_k$的取值无关，故(7)可表示成
$$
y=\arg \max _{c_{k}} P\left(Y=c_{k}\right) \prod_{j} P\left(X^{(j)}=x^{(j)} \mid Y=c_{k}\right)
$$

#### 1.3.1 含义

- 后验概率最大化等价于期望风险最小化

### 1.4 朴素贝叶斯法中的参数估计

为了学习朴素贝叶斯法中的$P\left(Y=c_{k}\right)$与$P\left(X=x \mid Y=c_{k}\right)$

#### 1.4.1 极大似然估计

核心思想为根据以有的测试数据集，得到每一类的频率，即为概率

> 算法 4.1 (朴素贝叶斯算法 (Naive Bayes algorithm) ) 
>
> 输入: 训练数据 $T=\left\{\left(x_{1}, y_{1}\right),\left(x_{2}, y_{2}\right), \cdots,\left(x_{N}, y_{N}\right)\right\},$ 其中 $x_{i}=\left(x_{i}^{(1)}, x_{i}^{(2)}, \cdots,\right.$
> $\left.x_{i}^{(n)}\right)^{\mathrm{T}}, x_{i}^{(j)}$ 是第 $i$ 个样本的第 $j$ 个特征, $x_{i}^{(j)} \in\left\{a_{j 1}, a_{j 2}, \cdots, a_{j S_{j}}\right\}, a_{j l}$ 是第 $j$ 个特
> 征可能取的第 $l$ 个值, $j=1,2, \cdots, n, l=1,2, \cdots, S_{j}, y_{i} \in\left\{c_{1}, c_{2}, \cdots, c_{K}\right\} ;$ 实例 $x$;
> 输出：实例 $x$ 的分类。
> （1）计算先验概率及条件概率
> $$
> P\left(Y=c_{k}\right)=\frac{\sum_{i=1}^{N} I\left(y_{i}=c_{k}\right)}{N}, \quad k=1,2, \cdots, K
> $$
>
> $$
> \begin{array}{c}
> P\left(X^{(j)}=a_{j l} \mid Y=c_{k}\right)=\frac{\sum_{i=1}^{N} I\left(x_{i}^{(j)}=a_{j l}, y_{i}=c_{k}\right) }{\sum_{i=1}^{N} I\left(y_{i}=c_{k}\right)} \\
> j=1,2, \cdots, n ; \quad l=1,2, \cdots, S_{j} ; \quad k=1,2, \cdots, K
> \end{array}
> $$
> （2）对于给定的实例 $x=\left(x^{(1)}, x^{(2)}, \cdots, x^{(n)}\right)^{\mathrm{T}},$ 计算
> $$
> P\left(Y=c_{k}\right) \prod_{j=1}^{n} P\left(X^{(j)}=x^{(j)} \mid Y=c_{k}\right), \quad k=1,2, \cdots, K
> $$
> （3）确定实例 $x$ 的类
> $$
> y=\arg \max _{c_{k}} P\left(Y=c_{k}\right) \prod_{j=1}^{n} P\left(X^{(j)}=x^{(j)} \mid Y=c_{k}\right)
> $$



#### 1.4.2 贝叶斯估计

**问题**：通过极大似然估计，可能会出现(8)中的几个$P\left(X^{(j)}=x^{(j)} \mid Y=c_{k}\right)$为0（我们把“发票”当成了一个特征，但句子中没有出现“发票”这一词），这个情况很常见

**原因**：训练集太少，没有满足大数定律。

**解决**：

给定先验概率，赋值为$\lambda \ge0$

- 在一个训练集中，对一个本应该为0 的概率赋予一个较小值，那么就会降低其他词语的概率
- $\lambda=0$：极大似然估计
- $\lambda=1$：Laplace Smoothing
- ”加上“基础的概率$1/K,\;1/S_j$

**算法**：

- 计算先验概率的等式(10)变成
  $$
  P_{\lambda}\left(Y=c_{k}\right)=\frac{\sum_{i=1}^{N} I\left(y_{i}=c_{k}\right)+\lambda}{N+K \lambda}
  $$

- 计算条件概率的等式(11)变成
  $$
  P_{\lambda}(X^{(j)}=a_{jl}|Y=c_k)=\frac{\sum\limits_{i=1}^NI(x_i^{j}=a_{jl},y_j=c_k)+\lambda}{\sum\limits_{i=1}^NI(y_j=c_k)+S_j\lambda}
  $$



## 3. Trick

### 3.1 Log function

在等式(12)中，我们可以取对数，把乘法运算转化为加法运算，提高计算速度。



## 4. Coding

```python

class NaiveBayes:
    def __init__(self):
        self.model='Naive Bayes'

    def predict(self, Py, Pxi_y, x):
        """
        :param Py: 先验概率
        :param Pxi_y: 条件概率
        :param x: test array
        :return: 通过极大似然法得出最大概率的标签
        """
        (labelNumber,featureNumber,features)=Pxi_y.shape
        # p存放所有的似然
        p=Py
        for i in range(labelNumber):
            sum = 0
            for j in range(featureNumber):
                sum += Pxi_y[i][j][x[j]] # 因为转化成了log，所以累乘变累加
            p[i] += sum
        return np.argmax(p)
        

    def getAllProbability(self, labelArray, labelNumber, trainArray, featureNumber, features, para=1):
        """
        :param labelArray: [int,int,...,int] 1*n
        :param labelNumber: 一共有N个标签
        :param trainArray: train data, ~*m
        :param featureNumber: 一共有m个特征标签
        :param features: int k, 对第m个特征标签，有k_m个特征。
        :param para: lambda
        :return: 先验概率Py, 条件概率Pxi_y
        """
        '''
        目前（没有完善load function）对于输入的形式非常的苛刻，也出现了重复的参数，日后完善。
        example:
        trainArray=[[0,0],[0,1],[0,1],[0,0],[0,0],[1,0],[1,1],[1,1],[1,2],[1,2],[2,2],[2,1],[2,1],[2,2],[2,2]]
        trainArray=np.asarray(trainArray)
        featureNumber=2
        features=3
        labelArray=[0,0,1,1,0,0,0,1,1,1,1,1,1,1,0]
        labelArray=np.asarray(labelArray)
        labelNumber=2
        '''
        # 一共有N个标签
        Py = np.zeros((labelNumber, 1))
        # 对标签为y，特征为m，第？个特征取值
        Pxi_y = np.zeros((labelNumber, featureNumber, features))

        # 计算先验概率Py
        for i in range(labelNumber):
            numerator = np.sum(labelArray==i) + para
            denominator = len(labelArray) + para * labelNumber
            Py[i] = numerator/denominator
        Py = np.log(Py) # 取对数加快运算

        # 计算条件概率Pxi_y
        (a,b) = trainArray.shape # a:几条数据  b:几个特征

        for i in range(a): # 第几条数据，选取他们的label和train data array
            label = labelArray[i]
            x=trainArray[i]

            for j in range(featureNumber): # 对x的每一个标签上的值，出现一次加一次
                Pxi_y[label][j][x[j]] +=1

        for label in range(labelNumber): # 得到次数后计算概念
            for i in range(featureNumber):
                denominator = np.sum(Pxi_y[label][i]) + features * para
                for j in range(features):
                    numerator = Pxi_y[label][i][j] + para
                    Pxi_y[label][i][j]=np.log(numerator/denominator)

        return Py, Pxi_y

    def loadData(self):
        pass

    def modelTest(self):
        pass


```

- [More details and examples](https://github.com/ChongfengLing/Statistical-Learning-Method-Notes-Code)



## 5. Proof

### 5.1 用极大似然法推出等式(10)

![253e0884011c7a303fda71c8962cfa6](https://raw.githubusercontent.com/ChongfengLing/typora-picBed/main/img/SLM-4%20Proof%201.png)



## 6. Reference

[统计学习方法（第2版）](https://book.douban.com/subject/33437381/)[Dod-o/Statistical-Learning-Method_Code](https://github.com/Dod-o/Statistical-Learning-Method_Code)

[Dod-o/Statistical-Learning-Method_Code](https://github.com/Dod-o/Statistical-Learning-Method_Code)

[NLP系列(4)_朴素贝叶斯实战与进阶](https://blog.csdn.net/han_xiaoyang/article/details/50629608)

[fengdu78/lihang-code](https://github.com/fengdu78/lihang-code)

[SmirkCao, Lihang, (2018), GitHub repository](https://github.com/SmirkCao/Lihang)

