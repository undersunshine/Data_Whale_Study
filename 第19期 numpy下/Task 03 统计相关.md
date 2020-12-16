# 1 学习

## 1.1 次序统计

### 1.1.1 计算最小值
- `numpy.amin(a[, axis=None, out=None, keepdims=np._NoValue, initial=np._NoValue, where=np._NoValue])`Return the minimum of an array or minimum along an axis.

  ​       

 【例】计算最小值
```python
import numpy as np

x = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])
y = np.amin(x)
print(y)  # 11

y = np.amin(x, axis=0)
print(y)  # [11 12 13 14 15]

y = np.amin(x, axis=1)
print(y)  # [11 16 21 26 31]
```
### 1.1.2 计算最大值
- `numpy.amax(a[, axis=None, out=None, keepdims=np._NoValue, initial=np._NoValue, where=np._NoValue])`Return the maximum of an array or maximum along an axis.


【例】计算最大值
```python
import numpy as np

x = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])
y = np.amax(x)
print(y)  # 35

y = np.amax(x, axis=0)
print(y)  # [31 32 33 34 35]

y = np.amax(x, axis=1)
print(y)  # [15 20 25 30 35]
```
### 1.1.3 计算极差
- `numpy.ptp(a, axis=None, out=None, keepdims=np._NoValue)` Range of values (maximum - minimum) along an axis. The name of the function comes from the acronym for 'peak to peak'.

【例】计算极差
```python
import numpy as np

np.random.seed(20200623)
x = np.random.randint(0, 20, size=[4, 5])
print(x)
# [[10  2  1  1 16]
#  [18 11 10 14 10]
#  [11  1  9 18  8]
#  [16  2  0 15 16]]

print(np.ptp(x))  # 18
print(np.ptp(x, axis=0))  # [ 8 10 10 17  8]
print(np.ptp(x, axis=1))  # [15  8 17 16]
```

### 1.1.4 计算分位数

- `numpy.percentile(a, q, axis=None, out=None, overwrite_input=False, interpolation='linear', keepdims=False)` Compute the q-th percentile of the data along the specified axis. Returns the q-th percentile(s) of the array elements.
    - a：array，用来算分位数的对象，可以是多维的数组。
    - q：介于0-100的float，用来计算是几分位的参数，如四分之一位就是25，如要算两个位置的数就[25,75]。
    - axis：坐标轴的方向，一维的就不用考虑了，多维的就用这个调整计算的维度方向，取值范围0/1。


【例】计算分位数
```python
import numpy as np

np.random.seed(20200623)
x = np.random.randint(0, 20, size=[4, 5])
print(x)
# [[10  2  1  1 16]
#  [18 11 10 14 10]
#  [11  1  9 18  8]
#  [16  2  0 15 16]]

print(np.percentile(x, [25, 50]))  
# [ 2. 10.]

print(np.percentile(x, [25, 50], axis=0))
# [[10.75  1.75  0.75 10.75  9.5 ]
#  [13.5   2.    5.   14.5  13.  ]]

print(np.percentile(x, [25, 50], axis=1))
# [[ 1. 10.  8.  2.]
#  [ 2. 11.  9. 15.]]
```


## 1.2 均值与方差

### 1.2.1 计算中位数
- `numpy.median(a, axis=None, out=None, overwrite_input=False, keepdims=False)` Compute the median along the specified axis. Returns the median of the array elements.

【例】计算中位数
```python
import numpy as np

np.random.seed(20200623)
x = np.random.randint(0, 20, size=[4, 5])
print(x)
# [[10  2  1  1 16]
#  [18 11 10 14 10]
#  [11  1  9 18  8]
#  [16  2  0 15 16]]
print(np.percentile(x, 50))
print(np.median(x))
# 10.0

print(np.percentile(x, 50, axis=0))
print(np.median(x, axis=0))
# [13.5  2.   5.  14.5 13. ]

print(np.percentile(x, 50, axis=1))
print(np.median(x, axis=1))
# [ 2. 11.  9. 15.]
```

### 1.2.2 计算平均值

- `numpy.mean(a[, axis=None, dtype=None, out=None, keepdims=np._NoValue)])`Compute the arithmetic mean along the specified axis.

【例】计算平均值（沿轴的元素的总和除以元素的数量）。
```python
import numpy as np

x = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])
y = np.mean(x)
print(y)  # 23.0

y = np.mean(x, axis=0)
print(y)  # [21. 22. 23. 24. 25.]

y = np.mean(x, axis=1)
print(y)  # [13. 18. 23. 28. 33.]
```

### 1.2.3 计算加权平均值
- `numpy.average(a[, axis=None, weights=None, returned=False])`Compute the weighted average along the specified axis.


`mean`和`average`都是计算均值的函数，在不指定权重的时候`average`和`mean`是一样的。指定权重后，`average`可以计算加权平均值。

【例】计算加权平均值（将各数值乘以相应的权数，然后加总求和得到总体值，再除以总的单位数。）
```python
import numpy as np

x = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])
y = np.average(x)
print(y)  # 23.0

y = np.average(x, axis=0)
print(y)  # [21. 22. 23. 24. 25.]

y = np.average(x, axis=1)
print(y)  # [13. 18. 23. 28. 33.]


y = np.arange(1, 26).reshape([5, 5])
print(y)
# [[ 1  2  3  4  5]
#  [ 6  7  8  9 10]
#  [11 12 13 14 15]
#  [16 17 18 19 20]
#  [21 22 23 24 25]]

z = np.average(x, weights=y)
print(z)  # 27.0

z = np.average(x, axis=0, weights=y)
print(z)
# [25.54545455 26.16666667 26.84615385 27.57142857 28.33333333]

z = np.average(x, axis=1, weights=y)
print(z)
# [13.66666667 18.25       23.15384615 28.11111111 33.08695652]
```

### 1.2.4 计算方差
- `numpy.var(a[, axis=None, dtype=None, out=None, ddof=0, keepdims=np._NoValue])`Compute the variance along the specified axis.
    - ddof=0：是“Delta Degrees of Freedom”，表示自由度的个数。

要注意方差和样本方差的无偏估计，方差公式中分母上是`n`；样本方差无偏估计公式中分母上是`n-1`（`n`为样本个数）。

【例】计算方差
```python
import numpy as np

x = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])
y = np.var(x)
print(y)  # 52.0
y = np.mean((x - np.mean(x)) ** 2)
print(y)  # 52.0

y = np.var(x, ddof=1)
print(y)  # 54.166666666666664
y = np.sum((x - np.mean(x)) ** 2) / (x.size - 1)
print(y)  # 54.166666666666664

y = np.var(x, axis=0)
print(y)  # [50. 50. 50. 50. 50.]

y = np.var(x, axis=1)
print(y)  # [2. 2. 2. 2. 2.]
```

### 1.2.5 计算标准差

- `numpy.std(a[, axis=None, dtype=None, out=None, ddof=0, keepdims=np._NoValue])`Compute the standard deviation along the specified axis.

标准差是一组数据平均值分散程度的一种度量，是方差的算术平方根。


【例】计算标准差
```python
import numpy as np

x = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35]])
y = np.std(x)
print(y)  # 7.211102550927978
y = np.sqrt(np.var(x))
print(y)  # 7.211102550927978

y = np.std(x, axis=0)
print(y)
# [7.07106781 7.07106781 7.07106781 7.07106781 7.07106781]

y = np.std(x, axis=1)
print(y)
# [1.41421356 1.41421356 1.41421356 1.41421356 1.41421356]
```

## 1.3 相关
### 1.3.1 计算协方差矩阵

$$
Cov(X,Y) = E(XY) - E(X)E(Y)
$$

- `numpy.cov(m, y=None, rowvar=True, bias=False, ddof=None, fweights=None,aweights=None)` Estimate a covariance matrix, given data and weights.

【例】计算协方差矩阵
```python
import numpy as np

x = [1, 2, 3, 4, 6]
y = [0, 2, 5, 6, 7]
print(np.cov(x))  # 3.7   #样本方差
print(np.cov(y))  # 8.5   #样本方差
print(np.cov(x, y))
# [[3.7  5.25]
#  [5.25 8.5 ]]

print(np.var(x))  # 2.96    #方差
print(np.var(x, ddof=1))  # 3.7    #样本方差
print(np.var(y))  # 6.8    #方差
print(np.var(y, ddof=1))  # 8.5    #样本方差

z = np.mean((x - np.mean(x)) * (y - np.mean(y)))    #协方差
print(z)  # 4.2

z = np.sum((x - np.mean(x)) * (y - np.mean(y))) / (len(x) - 1)   #样本协方差
print(z)  # 5.25

z = np.dot(x - np.mean(x), y - np.mean(y)) / (len(x) - 1)     #样本协方差     
print(z)  # 5.25
```

### 计算相关系数
- `numpy.corrcoef(x, y=None, rowvar=True, bias=np._NoValue, ddof=np._NoValue)` Return Pearson product-moment correlation coefficients.

理解了`np.cov()`函数之后，很容易理解`np.correlate()`，二者参数几乎一模一样。

`np.cov()`描述的是两个向量协同变化的程度，它的取值可能非常大，也可能非常小，这就导致没法直观地衡量二者协同变化的程度。相关系数实际上是正则化的协方差，`n`个变量的相关系数形成一个`n`维方阵。

【例】计算相关系数
```python
import numpy as np

np.random.seed(20200623)
x, y = np.random.randint(0, 20, size=(2, 4))

print(x)  # [10  2  1  1]
print(y)  # [16 18 11 10]

z = np.corrcoef(x, y)
print(z)
# [[1.         0.48510096]
#  [0.48510096 1.        ]]

a = np.dot(x - np.mean(x), y - np.mean(y))
b = np.sqrt(np.dot(x - np.mean(x), x - np.mean(x)))
c = np.sqrt(np.dot(y - np.mean(y), y - np.mean(y)))
print(a / (b * c))  # 0.4851009629263671
```


## 1.4 直方图

- `numpy.digitize(x, bins, right=False)`Return the indices of the bins to which each value in input array belongs.
    - x：numpy数组
    - bins：一维单调数组，必须是升序或者降序
    - right：间隔是否包含最右
    - 返回值：x在bins中的位置。

【例】

```python
import numpy as np

x = np.array([0.2, 6.4, 3.0, 1.6])
bins = np.array([0.0, 1.0, 2.5, 4.0, 10.0])
inds = np.digitize(x, bins)
print(inds)  # [1 4 3 2]
for n in range(x.size):
    print(bins[inds[n] - 1], "<=", x[n], "<", bins[inds[n]])

# 0.0 <= 0.2 < 1.0
# 4.0 <= 6.4 < 10.0
# 2.5 <= 3.0 < 4.0
# 1.0 <= 1.6 < 2.5
```

【例】
```python
import numpy as np

x = np.array([1.2, 10.0, 12.4, 15.5, 20.])
bins = np.array([0, 5, 10, 15, 20])
inds = np.digitize(x, bins, right=True)
print(inds)  # [1 2 3 4 4]

inds = np.digitize(x, bins, right=False)
print(inds)  # [1 3 3 4 5]
```


# 2 练习

## 2.1 $y = X \beta $

对于简单线性回归，向量记法等同于


\begin{bmatrix}
Y_1 \\
Y_2 \\
\vdots\\
Y_n \\
\end{bmatrix}

=

\begin{bmatrix}
\alpha + \beta * X_1 \\
\alpha + \beta * X_2 \\
\vdots\\
\alpha + \beta * X_n \\
\end{bmatrix}






```python
from numpy . linalg import inv 
from numpy import dot, transpose 
X = [[1, 6, 2] , [1, 8, 1] , [1, 10, 0] , [1 , 14, 2] , [1, 18, 0]] 
y = [[7] , [9] , [13] , [17.5] , [18]] 
```


```python
print(dot(inv(dot(transpose(X) , X)) , dot(transpose(X) , y))) 
```

    [[1.1875    ]
     [1.01041667]
     [0.39583333]]


NumPy 库也提供了一个最小二乘函数， 它能被用来更简洁地解出参数值


```python
from numpy.linalg import lstsq
print(lstsq(X,y)[0])
```

    [[1.1875    ]
     [1.01041667]
     [0.39583333]]


    <ipython-input-4-5c34e17b9dbc>:2: FutureWarning: `rcond` parameter will change to the default of machine precision times ``max(M, N)`` where M and N are the input matrix dimensions.
    To use the future default and silence this warning we advise to pass `rcond=None`, to keep using the old, explicitly pass `rcond=-1`.
      print(lstsq(X,y)[0])


## 2.2 计算给定数组中每行的最大值。


- `a = np.random.randint(1, 10, [5, 3])`

【知识点：统计相关】
- 如何在二维numpy数组的每一行中找到最大值？


```python
import numpy as np

np.random.seed(100)
a = np.random.randint(1, 10, [5, 3])
print(a)

b = np.amax(a, axis=1)
print(b)
```

    [[9 9 4]
     [8 8 1]
     [5 3 6]
     [3 3 3]
     [2 1 9]]
    [9 8 6 3 9]


## 2.3 计算数组的元素最大值与最小值之差（极值）

【知识点：统计相关】

数组为：
A=[[3 7 5]
[8 4 3]
[2 4 9]]


```python
import numpy as np

x = np.array([[3, 7, 5], 
              [8, 4, 3],
              [2, 4, 9]])

print(np.ptp(x))
```

    7


## 2.4 计算s的均值，方差，标准差，协方差

【知识点：统计相关】

s=[9.7, 10, 10.3, 9.7,10,10.3,9.7,10,10.3]


```python
import numpy as np

s = [9.7, 10, 10.3, 9.7, 10, 10.3, 9.7, 10, 10.3]

average = np.mean(s)
print(average)

var = np.var(s)
print(var)

std = np.std(s)
print(std)

cov = np.cov(s)
print(cov)
```

    10.0
    0.06000000000000029
    0.2449489742783184
    0.06750000000000032

