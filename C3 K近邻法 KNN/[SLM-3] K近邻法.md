# k 近邻法

k-nearest neighbor (KNN)，一种基本的分类与回归方法。这里只介绍分类问题。**k值的选择、距离度量、分类决策规则**是KNN的3个基本要素。



## 1. Algorithm

> **算法 $3.1(k$ 近邻法 $)$**
> 输入: 训练数据集
> $$
> \begin{aligned}
> T=\left\{\left(x_{1}, y_{1}\right),\left(x_{2}, y_{2}\right), \cdots,\left(x_{N}, y_{N}\right)\right\}
> \end{aligned}
> $$
> 其中， $x_{i} \in \mathcal{X} \subseteq \mathbf{R}^{n}$ 为实例的特征向量, $y_{i} \in \mathcal{Y}=\left\{c_{1}, c_{2}, \cdots, c_{K}\right\}$ 为实例的类别， $i=1,2, \cdots, N ;$ 实例特征向量 $x$;
> 输出：实例 $x$ 所属的类 $y$。
> (1) 根据给定的距离度量，在训练集 $T$ 中找出与 $x$ 最邻近的 $k$ 个点，涵盖这 $k$ 个点的 $x$ 的邻域记作 $N_{k}(x)$;
> (2) 在 $N_{k}(x)$ 中根据分类决策规则 (如多数表决) 决定 $x$ 的类别 $y$ :
> $$
> y=\arg \max _{c_{j}} \sum_{x_{i} \in N_{k}(x)} I\left(y_{i}=c_{j}\right), \quad i=1,2, \cdots, N ; j=1,2, \cdots, K
> $$
> 式 (1) 中， $I$ 为指示函数，即当 $y_{i}=c_{j}$ 时 $I$ 为 $1,$ 否则 $I$ 为 0 。

- 算法没有显式的学习过程。
- 例子：
  1. S市划分成住宅区、商业区、工业区（K=3），每个区域都有各自地标建筑，共N座。你的坐标为x，选取离你最近的k个地标建筑，k中包括哪类地标建筑最多，你就位于哪个区。



## 2. Model

- KNN的模型对应于对特征空间的划分。

- 在训练集、距离度量、k值、分类决策鬼册确定后，对任意新输入实例，所属类别唯一。
- 每一个训练实例点$x_i$，存在一个单元cell，在单元内到此$x_i$的距离最小。所有实例点的单元划分了特征空间。

### 2.1 距离度量

计算特征空间中两点的距离。

$L_{p}$ (Minkowski) distance
$$
L_{p}\left(x_{i}, x_{j}\right)=\left(\sum_{l=1}^{n}\left|x_{i}^{(l)}-x_{j}^{(l)}\right|^{p}\right)^{\frac{1}{p}}
$$

- - 设特征空间 $\mathcal{X}$ 是 $n$ 维实数向量空间 $x_{i}, x_{j} \in \mathbf{R}^{n}, x_{i}=\left(x_{i}^{(1)}, x_{i}^{(2)}, \cdots, x_{i}^{(n)}\right)^{\mathrm{T}},$
    $x_{j}=\left(x_{j}^{(1)}, x_{j}^{(2)}, \cdots, x_{j}^{(n)}\right)^{\mathrm{T}}$
  - $p\geq1$
- $p=1$，Manhattan distance
- $p=2$， Euclidean distance
- $p=\infin$，'Max' distance

### 2.2 k值选择

选取一些较小k值，通过**交叉验证法**选取最优k值。

- 小k：
  - 近似误差减小，估计误差增大。对近邻实例点非常敏感
  - 模型更复杂，容易过拟合。
- 大k：
  - 近似误差增大，估计误差减小。较远错误标签实例点也能造成影响。
  - 模型更简单。
  - $k=N$时，则为测试集最多类。



### 2.3 分类决策规则

一般为**多数表决(majority voting rule)**

对给定的实例$x$，最近邻的$k$个训练实例点存在$c$个类别。为了使得分类错误的概率最小，所以选取类别中最多的那一类。



## 3. Implement: kd tree

线性扫描 (linear scan) 要计算实例与每一个训练实例的距离，耗时。

### 3.1 kd树的构造

选取一个训练实例点，构造一个垂直于某一轴的超平面，使得特征空间被分成左右两个子空间（左小右大）。在2个特征子空间中重复，直到子空间中没有训练实例点。

#### 3.1.1平衡kd树

- 训练实例点的选取为选定坐标轴上的中位数。
- 效率不一定最优。

> 算法 3.2 (构造平衡 $k d$ 树） 
>
> 输入: $k$ 维空间数据集 $T=\left\{x_{1}, x_{2}, \cdots, x_{N}\right\},$ 其中 $x_{i}=\left(x_{i}^{(1)}, x_{i}^{(2)}, \cdots, x_{i}^{(k)}\right)^{\mathrm{T}},$
> $i=1,2, \cdots, N$
> 输出: $k d$ 树。
> （1）开始：构造根结点，根结点对应于包含 $T$ 的 $k$ 维空间的超矩形区域。
> 选择 $x^{(1)}$ 为坐标轴，以 $T$ 中所有实例的 $x^{(1)}$ 坐标的中位数为切分点，将根结点对应的超矩形区域切分为两个子区域。切分由通过切分点并与坐标轴 $x^{(1)}$ 垂直的超平面实现。
> 由根结点生成深度为 1 的左、右子结点: **左子结点对应坐标 $x^{(1)}$ 小于切分点的子 区域，右子结点对应于坐标 $x^{(1)}$ 大于切分点的子区域**。
> 将落在切分超平面上的实例点保存在根结点。
> （2）重复: 对深度为 $j$ 的结点，选择 $x^{(l)}$ 为切分的坐标轴， $l=j(\bmod k)+1,$ 以该结点的区域中所有实例的 $x^{(l)}$ 坐标的中位数为切分点，将该结点对应的超矩形区域切分为两个子区域。切分由通过切分点并与坐标轴 $x^{(l)}$ 垂直的超平面实现。
> 由该结点生成深度为 $j+1$ 的左、右子结点: 左子结点对应坐标 $x^{(l)}$ 小于切分点 的子区域，右子结点对应坐标 $x^{(l)}$ 大于切分点的子区域。
> 将落在切分超平面上的实例点保存在该结点。
>
> （3）直到两个子区域没有实例存在时停止。从而形成 $k d$ 树的区域划分.

**构造过程图示**：

![SLM-3 平衡kd树图示](https://raw.githubusercontent.com/ChongfengLing/typora-picBed/main/img/SLM-3%20%E5%B9%B3%E8%A1%A1kd%E6%A0%91%E5%9B%BE%E7%A4%BA.png)

- 按照黄、绿、蓝、红，垂直x，y，x，y的顺序切分。
- 左右节点确定：根据在第$l$维度上2个子节点对应坐标大小确定，小左大右。
- $l=j(\bmod k)+1$：在k维方向都按顺序切上之后，回过头在循环切。
  - 公式细节因初始根节点的深度是0或1而有不同
- 停止条件：所有实例点都位于一个超平面上。
- 偶数个数的中位数：自己确定，代码上别忘了。

### 3.2 kd树的搜索

#### 3.2.1 $k=1$

> 算法 3.3 (用 $k d$ 树的最近邻搜索)
> 输入: 已构造的 $k d$ 树，目标点 $x$;
> 输出: $x$ 的最近邻，$k=1$。
> (1) 在 $k d$ 树中找出包含目标点 $x$ 的叶结点: 从根结点出发，递归地向下访问 $k d$ 树。若目标点 $x$ 当前维的坐标小于切分点的坐标，则移动到左子结点，否则移动到右子结点。直到子结点为叶结点为止。
> (2) 以此叶结点为“当前最近点”。
> (3) 递归地向上回退，在每个结点进行以下操作:
> 		(a）如果该结点保存的实例点比当前最近点距离目标点更近，则以该实例点 为“当前最近点”。
> 		(b）当前最近点一定存在于该结点一个子结点对应的区域。检查该子结点的父结点的另一子结点对应的区域是否有更近的点。具体地，检查另一子结点对应的区域是否与以目标点为球心、以目标点与“当前最近点”问的距离为半径的超球体相交。
>
> ​		如果相交，可能在另一个子结点对应的区域内存在距目标点更近的点，移动到另一个子结点。接着，递归地进行最近邻搜索;
> ​		如果不相交，向上回退。
> (4）当回退到根结点时，搜索结束。最后的“当前最近点"即为 $x$ 的最近邻点。

#### 3.2.2 $k>1$

> 算法 3.04 (用 $k d$ 树的最近邻搜索)
> 输入: 已构造的 $k d$ 树，目标点 $x$;
> 输出: $x$ 的k最近邻。
>
> 1. 根据p的坐标和kd树的结点向下进行搜索 (如果树的结点是以 $x^{(l)}=c$ 来切分的, 那么如果p的 $x^{(l)}$ 坐标小于c, 则走左子结点, 否则走右子结点)
>
> 2. 到达叶子结点时，将其标记为已访问。如果S中不足k个点, 则将该结点加入到S中; 如果S不空且当前结点与p点的距离小于S中最长的距离，则用当前结点替换S中离p最远的点
>
> 3. 如果当前结点不是根节点, 执行（a）; 否则，结束算法
>
>   (a). 回退到当前结点的父结点, 此时的结点为当前结点 (回退之后的结点) ，将当前结点标 记为已访问, 执行 (b) 和（c) ; 如果当前结点已经被访过, 再次执行（a）。
>
>   (b). 如果此时S中不足k个点, 则将当前结点加入到S中; 如果S中已有k个点, 且当前结点与p 点的距离小于S中最长距离，则用当前结点替换S中距离最远的点。
>
>   (c). 计算p点和当前结点切分线的距离。如果该距离大于等于S中距离p最远的距离并且S中已 有k个点, 执行3; 如果该距离小于S中最远的距离或S中没有k个点, 从当前结点的另一子节点开始执行1; 如果当前结点没有另一子结点, 执行3。

**算法3.04 (用 $k d$ 树的最近邻搜索)搜索过程**：目标点P（-1，-5），k=3，S：存储k个近邻点。初始所有点=[0, 0] ，即 [未访问, 不在S中]

![SLM-3 平衡kd树图示](https://raw.githubusercontent.com/ChongfengLing/typora-picBed/main/img/SLM-3%20%E5%B9%B3%E8%A1%A1kd%E6%A0%91%E5%9B%BE%E7%A4%BA.png)

> 1. 执行算法1：
>    - (-1,-5)的-1与A点(6,5)的6小，往左到B
>    - -5<-3，往左到D
>    - D子节点唯一，到H
> 2. 执行算法2：
>    - H=[1, 0]，H=[1, 1]
> 3. 执行算法3：
>    - 执行(a)，到D。D=[1, 0]
>    - 执行(b)，D=[1, 1]
>    - 执行(c)，P到D的切分线$x^1=-6$距离为5。S未满。D没有另一子节点。
> 4. 执行算法3：
>    - 执行(a)，到B。B=[1, 0]
>    - 执行(b)，B=[1,1]，S={H, D, B}
>    - 执行(c)，P到B的切分线$x^2=-3$距离为2，小于S的最大距离。从B另一子节点E执行算法1.
> 5. 执行算法1：
>    - -1>-2，往右到J
> 6. 执行算法2：
>    - J=[1,0]，d(J, P)大于S的最大距离。不加入
> 7. 执行算法3：
>    - 执行(a)，到E，E=[1, 0]
>    - 执行(b)，d(E,P)小于S的最大距离。E=[1, 1]，H=[1, 0]，S={E, D, B}
>    - 执行(c)，P到E的切分线$x^1=-2$距离为1，小于S的最大距离。从E另一子节点I执行算法1.
> 8. 执行算法1：
>    - I是子节点，到I
> 9. 执行算法2：
>    - I=[1, 0]，d(I, P)大于S的最大距离。不加入。
> 10. 执行算法3：
>     - 执行若干(a)，到A，A=[1, 0]
>     - 执行(b)，d(A, P)大于S的最大距离。
>     - 执行(c)，P到E的切分线$x^1=6$距离为7，大于最长距离。不加入。
> 11. 执行算法3：
>     - A是根节点。算法结束。已按顺序访问{H, D, B, J, E, I, A}，最终结果S={E, D, B}

### 3.3 Summary

1. kd Tree的平均复杂度$O(logN)$, $N$为训练实例数。适合训练实例数远大于空间维数的KNN。

2. 把k维大空间多次对半分成k维小空间。

3. 目标点与实例点的距离$\geq$目标点与该实例点对应切分超平面的距离。因此：

   1. 当此实例点当前应该替换S中某点从而加入到S集合中时，另一半空间存在子空间，子空间的点到目标点的距离小于实例点到目标点的距离。
      1. 此时如果存在实例点位于子空间，那么应该继续循环（例子中没有体现）。
      2. 没有实例点在此子空间，那么另一半的空间的实例点都不会加入到集合S中。
   2. 这个实例点不应该加入到S中，情况如2.1.2

4. 动态规划DP的思想？

5. 别人笔记的原话：

   1、找到叶子结点，看能不能加入到S中

   2、回退到父结点，看父结点能不能加入到S中

   3、看目标点和回退到的父结点切分线的距离，判断另一子结点能不能加入到S中



## 4. Question

1. kd方差的代码！
   - 有点点难搞-_-
2. 非平衡/方差kd树
   - 同上，难搞



## 5. Code

```python
class KNN:
    def __init__(self, train_set, label_set):
        """
        :param train_set: m*n ndarray, m:dims, n:train points
        :param label_set: 1*n int ndarray and from 0 to n-1
        """
        self.train_set = train_set
        self.label_set = label_set
        self.m, self.n = self.train_set.shape

    def knn_naive(self, test_set, k=1,p=2):
        """
        :param test_set: m*n ndarray, m:dims, n:test points
        :param k: k nearest neighbor
        :param p: order of norm
        :return: 1*n list for n points
        """
        (m,n)=test_set.shape
        # 有几个点，输出对于shape的list
        final=[-1]*n

        for i in range(n):
            # 转置后取点再转置
            test_point=test_set.T[i].T
            # 有些可能是(m,)的ndarray
            test_point=test_point.reshape([m,1])
            # 计算距离
            distance=np.linalg.norm(self.train_set-test_point,ord=p,axis=0)
            # 最近k个点的index
            nearestK=np.argsort(distance)[:k]
            # label是0，1，2，...的顺序，存储对应label出现的总次数
            labelList=[0]*(max(self.label_set) + 1)
            
            for index in nearestK:
                labelList[int(self.label_set[index])] +=1
                point_label=labelList.index(max(labelList))
            final[i]=point_label
        return final

    def knn_kdtree(self):
        pass

    def score(self):
        pass

    def plot(self, points=None):
        '''
        画不同k从而分割成不同子%tb域的图
        画结果图
        '''
        if self.m != 2:
            return "Unable to draw a picture"
```

- [More details and examples](https://github.com/ChongfengLing/Statistical-Learning-Method-Notes-Code)
- Github求start！！！^_^



## 6. Reference

[统计学习方法（第2版）](https://book.douban.com/subject/33437381/)

[KNN算法和kd树详解（例子+图示）](https://blog.csdn.net/zzpzm/article/details/88565645)

[sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier.predict)

[fengdu78/lihang-code](https://github.com/fengdu78/lihang-code)

[Dod-o/Statistical-Learning-Method_Code](https://github.com/Dod-o/Statistical-Learning-Method_Code)

