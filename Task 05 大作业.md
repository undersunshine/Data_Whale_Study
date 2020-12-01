```python

import numpy as np
outfile = r'.\iris.data'
iris_data = np.loadtxt(outfile, dtype=object, delimiter=',', skiprows=1)
print(iris_data[0:10])
# [['5.1' '3.5' '1.4' '0.2' 'Iris‐setosa']
# ['4.9' '3.0' '1.4' '0.2' 'Iris‐setosa']
# ['4.7' '3.2' '1.3' '0.2' 'Iris‐setosa']
# ['4.6' '3.1' '1.5' '0.2' 'Iris‐setosa']
# ['5.0' '3.6' '1.4' '0.2' 'Iris‐setosa']
# ['5.4' '3.9' '1.7' '0.4' 'Iris‐setosa']
# ['4.6' '3.4' '1.4' '0.3' 'Iris‐setosa']
# ['5.0' '3.4' '1.5' '0.2' 'Iris‐setosa']
# ['4.4' '2.9' '1.4' '0.2' 'Iris‐setosa']
# ['4.9' '3.1' '1.5' '0.1' 'Iris‐setosa']]


import numpy as np
outfile = r'.\iris.data'
sepalLength = np.loadtxt(outfile, dtype=float, delimiter=',', skiprows=1, usecols=[0])
print(sepalLength[0:10])
# [5.1 4.9 4.7 4.6 5. 5.4 4.6 5. 4.4 4.9]
print(np.mean(sepalLength))
# 5.843333333333334
print(np.median(sepalLength))
# 5.8
print(np.std(sepalLength))
# 0.8253012917851409


import numpy as np
outfile = r'.\iris.data'
sepalLength = np.loadtxt(outfile, dtype=float, delimiter=',', skiprows=1, usecols=[0])
# Method 1
aMax = np.amax(sepalLength)
aMin = np.amin(sepalLength)
x = (sepalLength ‐ aMin) / (aMax ‐ aMin)
print(x[0:10])
# [0.22222222 0.16666667 0.11111111 0.08333333 0.19444444 0.30555556
# 0.08333333 0.19444444 0.02777778 0.16666667]
# Method 2
x = (sepalLength ‐ aMin) / np.ptp(sepalLength)
print(x[0:10])
# [0.22222222 0.16666667 0.11111111 0.08333333 0.19444444 0.30555556
# 0.08333333 0.19444444 0.02777778 0.16666667]

import numpy as np
outfile = r'.\iris.data'
sepalLength = np.loadtxt(outfile, dtype=float, delimiter=',', skiprows=1, usecols=[0])
x = np.percentile(sepalLength, [5, 95])
print(x) # [4.6 7.255]


import numpy as np
outfile = r'.\iris.data'
# Method 1
iris_data = np.loadtxt(outfile, dtype=object, delimiter=',', skiprows=1)
i, j = iris_data.shape
np.random.seed(20200621)
iris_data[np.random.randint(i, size=20), np.random.randint(j, size=20)] = np.nan
print(iris_data[0:10])
# [['5.1' '3.5' '1.4' '0.2' 'Iris‐setosa']
# ['4.9' '3.0' '1.4' '0.2' 'Iris‐setosa']
# ['4.7' '3.2' '1.3' '0.2' 'Iris‐setosa']
# ['4.6' '3.1' '1.5' '0.2' 'Iris‐setosa']
# ['5.0' '3.6' '1.4' '0.2' 'Iris‐setosa']
# ['5.4' nan '1.7' '0.4' 'Iris‐setosa']
# ['4.6' '3.4' '1.4' '0.3' 'Iris‐setosa']
# ['5.0' '3.4' '1.5' '0.2' 'Iris‐setosa']
# ['4.4' '2.9' '1.4' '0.2' nan]
# ['4.9' '3.1' '1.5' '0.1' 'Iris‐setosa']]
# Method 2
iris_data = np.loadtxt(outfile, dtype=object, delimiter=',', skiprows=1)
i, j = iris_data.shape
np.random.seed(20200620)
iris_data[np.random.choice(i, size=20), np.random.choice(j, size=20)] = np.nan
print(iris_data[0:10])
# [['5.1' '3.5' '1.4' '0.2' 'Iris‐setosa']
# ['4.9' '3.0' '1.4' '0.2' 'Iris‐setosa']
# ['4.7' '3.2' '1.3' '0.2' 'Iris‐setosa']
# ['4.6' '3.1' '1.5' '0.2' 'Iris‐setosa']
# [nan '3.6' '1.4' '0.2' 'Iris‐setosa']
# ['5.4' '3.9' '1.7' '0.4' 'Iris‐setosa']
# ['4.6' '3.4' '1.4' '0.3' 'Iris‐setosa']
# ['5.0' '3.4' '1.5' '0.2' 'Iris‐setosa']
# ['4.4' '2.9' '1.4' '0.2' 'Iris‐setosa']
# ['4.9' '3.1' '1.5' nan 'Iris‐setosa']]


import numpy as np
outfile = r'.\iris.data'
iris_data = np.loadtxt(outfile, dtype=float, delimiter=',', skiprows=1, usecols=[0, 1, 2,
3])
i, j = iris_data.shape
np.random.seed(20200621)
iris_data[np.random.randint(i, size=20), np.random.randint(j, size=20)] = np.nan
sepallength = iris_data[:, 0]
x = np.isnan(sepallength)
print(sum(x)) # 6
print(np.where(x))
# (array([ 26, 44, 55, 63, 90, 115], dtype=int64),)


import numpy as np
outfile = r'.\iris.data'
iris_data = np.loadtxt(outfile, dtype=float, delimiter=',', skiprows=1, usecols=[0, 1, 2,
3])
sepallength = iris_data[:, 0]
petallength = iris_data[:, 2]
index = np.where(np.logical_and(petallength > 1.5, sepallength < 5.0))
print(iris_data[index])
# [[4.8 3.4 1.6 0.2]
# [4.8 3.4 1.9 0.2]
# [4.7 3.2 1.6 0.2]
# [4.8 3.1 1.6 0.2]
# [4.9 2.4 3.3 1. ]
# [4.9 2.5 4.5 1.7]]


import numpy as np
outfile = r'.\iris.data'
iris_data = np.loadtxt(outfile, dtype=float, delimiter=',', skiprows=1, usecols=[0, 1, 2,
3])
i, j = iris_data.shape
np.random.seed(20200621)
iris_data[np.random.randint(i, size=20), np.random.randint(j, size=20)] = np.nan
x = iris_data[np.sum(np.isnan(iris_data), axis=1) == 0]
print(x[0:10])


import numpy as np
outfile = r'.\iris.data'
iris_data = np.loadtxt(outfile, dtype=float, delimiter=',', skiprows=1, usecols=[0, 1, 2,
3])
sepalLength = iris_data[:, 0]
petalLength = iris_data[:, 2]
# method 1
m1 = np.mean(sepalLength)
m2 = np.mean(petalLength)
cov = np.dot(sepalLength ‐ m1, petalLength ‐ m2)
std1 = np.sqrt(np.dot(sepalLength ‐ m1, sepalLength ‐ m1))
std2 = np.sqrt(np.dot(petalLength ‐ m2, petalLength ‐ m2))
print(cov / (std1 * std2)) # 0.8717541573048712
# method 2
x = np.mean((sepalLength ‐ m1) * (petalLength ‐ m2))
y = np.std(sepalLength) * np.std(petalLength)
print(x / y) # 0.8717541573048712
# method 3
x = np.cov(sepalLength, petalLength, ddof=False)
y = np.std(sepalLength) * np.std(petalLength)
print(x[0, 1] / y) # 0.8717541573048716
# method 4
x = np.corrcoef(sepalLength, petalLength)
print(x)
# [[1. 0.87175416]
# [0.87175416 1. ]]


import numpy as np
outfile = r'.\iris.data'
iris_data = np.loadtxt(outfile, dtype=float, delimiter=',', skiprows=1, usecols=[0, 1, 2,
3])
x = np.isnan(iris_data)
print(np.any(x)) # False


import numpy as np
outfile = r'.\iris.data'
iris_data = np.loadtxt(outfile, dtype=float, delimiter=',', skiprows=1, usecols=[0, 1, 2,
3])
i, j = iris_data.shape
np.random.seed(20200621)
iris_data[np.random.randint(i, size=20), np.random.randint(j, size=20)] = np.nan
iris_data[np.isnan(iris_data)] = 0
print(iris_data[0:10])
# [[5.1 3.5 1.4 0.2]
# [4.9 3. 1.4 0.2]
# [4.7 3.2 1.3 0.2]
# [4.6 3.1 1.5 0.2]
# [5. 3.6 1.4 0.2]
# [5.4 0. 1.7 0.4]
# [4.6 3.4 1.4 0.3]
# [5. 3.4 1.5 0.2]
# [4.4 2.9 0. 0.2]
# [4.9 3.1 1.5 0.1]]


import numpy as np
outfile = r'.\iris.data'
iris_data = np.loadtxt(outfile, dtype=object, delimiter=',', skiprows=1, usecols=[4])
x = np.unique(iris_data, return_counts=True)
print(x)
# (array(['Iris‐setosa', 'Iris‐versicolor', 'Iris‐virginica'], dtype=object), array([50, 50, 50], dtype=int64))


import numpy as np
outfile = r'.\iris.data'
iris_data = np.loadtxt(outfile, dtype=float, delimiter=',', skiprows=1, usecols=[0, 1, 2,
3])
petal_length_bin = np.digitize(iris_data[:, 2], [0, 3, 5, 10])
label_map = {1: 'small', 2: 'medium', 3: 'large', 4: np.nan}
petal_length_cat = [label_map[x] for x in petal_length_bin]
print(petal_length_cat[0:10])
# ['small', 'small', 'small', 'small', 'small', 'small', 'small', 'small', 'small', 'small']


import numpy as np
outfile = r'.\iris.data'
iris_data = np.loadtxt(outfile, dtype=object, delimiter=',', skiprows=1)
sepalLength = iris_data[:, 0]
index = np.argsort(sepalLength)
print(iris_data[index][0:10])
# [['4.3' '3.0' '1.1' '0.1' 'Iris‐setosa']
# ['4.4' '3.2' '1.3' '0.2' 'Iris‐setosa']
# ['4.4' '3.0' '1.3' '0.2' 'Iris‐setosa']
# ['4.4' '2.9' '1.4' '0.2' 'Iris‐setosa']
# ['4.5' '2.3' '1.3' '0.3' 'Iris‐setosa']
# ['4.6' '3.6' '1.0' '0.2' 'Iris‐setosa']
# ['4.6' '3.1' '1.5' '0.2' 'Iris‐setosa']
# ['4.6' '3.4' '1.4' '0.3' 'Iris‐setosa']
# ['4.6' '3.2' '1.4' '0.2' 'Iris‐setosa']
# ['4.7' '3.2' '1.3' '0.2' 'Iris‐setosa']]


import numpy as np
outfile = r'.\iris.data'
iris_data = np.loadtxt(outfile, dtype=object, delimiter=',', skiprows=1)
petalLength = iris_data[:, 2]
vals, counts = np.unique(petalLength, return_counts=True)
print(vals[np.argmax(counts)]) # 1.5
print(np.amax(counts)) # 14


import numpy as np
outfile = r'.\iris.data'
iris_data = np.loadtxt(outfile, dtype=float, delimiter=',', skiprows=1, usecols=[0, 1, 2,
3])
petalWidth = iris_data[:, 3]
index = np.where(petalWidth > 1.0)
print(index)
print(index[0][0]) # 50

```
