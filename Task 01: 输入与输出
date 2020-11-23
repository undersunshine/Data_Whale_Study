# 1 学习内容

## 1.1 numpy 二进制文件

save()、savez()和load()函数以 numpy 专用的二进制类型（npy、npz）保存和读取数据，这三个函数会自动处理ndim、dtype、shape等信息，使用它们读写数组非常方便，但是save()输出的文件很难与其它语言编写的程序兼容。

npy格式：以二进制的方式存储文件，在二进制文件第一行以文本形式保存了数据的元信息（ndim，dtype，shape等），可以用二进制工具查看内容。

npz格式：以压缩打包的方式存储文件，可以用压缩软件解压。

- numpy.save(file, arr, allow_pickle=True, fix_imports=True) Save an array to a binary file in NumPy .npy format.
- numpy.load(file, mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII') Load arrays or pickled objects from .npy, .npz or pickled files.


```python
import numpy as np

outfile = r'.\test.npy'
np.random.seed(20200619)
x = np.random.uniform(0, 1, [3, 5])
np.save(outfile, x)
y = np.load(outfile)
print(y)
```

    [[0.01123594 0.66790705 0.50212171 0.7230908  0.61668256]
     [0.00668332 0.1234096  0.96092409 0.67925305 0.38596837]
     [0.72342998 0.26258324 0.24318845 0.98795012 0.77370715]]


- numpy.savez(file, *args, **kwds) Save several arrays into a single file in uncompressed .npz format.

savez()第一个参数是文件名，其后的参数都是需要保存的数组，也可以使用关键字参数为数组起一个名字，非关键字参数传递的数组会自动起名为arr_0, arr_1, …。

savez()输出的是一个压缩文件（扩展名为npz），其中每个文件都是一个save()保存的npy文件，文件名对应于数组名。load()自动识别npz文件，并且返回一个类似于字典的对象，可以通过数组名作为关键字获取数组的内容。

【例】将多个数组保存到一个文件，可以使用numpy.savez()函数。


```python
import numpy as np

outfile = r'.\test.npz'
x = np.linspace(0, np.pi, 5)
y = np.sin(x)
z = np.cos(x)
np.savez(outfile, x, y, z_d=z)
data = np.load(outfile)
np.set_printoptions(suppress=True)
print(data.files)  

print(data['arr_0'])

print(data['arr_1'])

print(data['z_d'])
```

    ['z_d', 'arr_0', 'arr_1']
    [0.         0.78539816 1.57079633 2.35619449 3.14159265]
    [0.         0.70710678 1.         0.70710678 0.        ]
    [ 1.          0.70710678  0.         -0.70710678 -1.        ]


## 1.2 文本文件

savetxt()，loadtxt()和genfromtxt()函数用来存储和读取文本文件（如TXT，CSV等）。genfromtxt()比loadtxt()更加强大，可对缺失数据进行处理。

- numpy.savetxt(fname, X, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None) Save an array to a text file.
  - fname：文件路径
  - X：存入文件的数组。
  - fmt：写入文件中每个元素的字符串格式，默认'%.18e'（保留18位小数的浮点数形式）。
  - delimiter：分割字符串，默认以空格分隔。


- numpy.loadtxt(fname, dtype=float, comments='#', delimiter=None, converters=None, skiprows=0, usecols=None, unpack=False, ndmin=0, encoding='bytes', max_rows=None) Load data from a text file.
  - fname：文件路径。
  - dtype：数据类型，默认为float。
  - comments: 字符串或字符串组成的列表，默认为# ， 表示注释字符集开始的标志。
  - skiprows：跳过多少行，一般跳过第一行表头。
  - usecols：元组（元组内数据为列的数值索引）， 用来指定要读取数据的列（第一列为0）。
  - unpack：当加载多列数据时是否需要将数据列进行解耦赋值给不同的变量。


【例】写入和读出TXT文件。


```python
import numpy as np

outfile = r'.\test.txt'
x = np.arange(0, 10).reshape(2, -1)
np.savetxt(outfile, x)
y = np.loadtxt(outfile)
print(y)
```

    [[0. 1. 2. 3. 4.]
     [5. 6. 7. 8. 9.]]


【例】写入和读出CSV文件。


```python
import numpy as np

outfile = r'.\test.csv'
x = np.arange(0, 10, 0.5).reshape(4, -1)
np.savetxt(outfile, x, fmt='%.3f', delimiter=',')
y = np.loadtxt(outfile, delimiter=',')
print(y)
```

    [[0.  0.5 1.  1.5 2. ]
     [2.5 3.  3.5 4.  4.5]
     [5.  5.5 6.  6.5 7. ]
     [7.5 8.  8.5 9.  9.5]]


genfromtxt()是面向结构数组和缺失数据处理的。

- numpy.genfromtxt(fname, dtype=float, comments='#', delimiter=None, skip_header=0, skip_footer=0, converters=None, missing_values=None, filling_values=None, usecols=None, names=None, excludelist=None, deletechars=''.join(sorted(NameValidator.defaultdeletechars)), replace_space='_', autostrip=False, case_sensitive=True, defaultfmt="f%i", unpack=None, usemask=False, loose=True, invalid_raise=True, max_rows=None, encoding='bytes') Load data from a text file, with missing values handled as specified.
  - names：设置为True时，程序将把第一行作为列名称。

【例】


```python
import numpy as np

outfile = r'.\data.csv'
x = np.loadtxt(outfile, delimiter=',', skiprows=1)
print(x)
```

    [[  1.  123.    1.4  23. ]
     [  2.  110.    0.5  18. ]
     [  3.  164.    2.1  19. ]]



```python
x = np.loadtxt(outfile, delimiter=',', skiprows=1, usecols=(1, 2))
print(x)
```

    [[123.    1.4]
     [110.    0.5]
     [164.    2.1]]



```python
val1, val2 = np.loadtxt(outfile, delimiter=',', skiprows=1, usecols=(1, 2), unpack=True)
print(val1)  # [123. 110. 164.]
print(val2)  # [1.4 0.5 2.1]
```

    [123. 110. 164.]
    [1.4 0.5 2.1]


【例】


```python
import numpy as np

outfile = r'.\data.csv'
x = np.genfromtxt(outfile, delimiter=',', names=True)
print(x)

print(type(x))  

print(x.dtype)

print(x['id'])  # [1. 2. 3.]
print(x['value1'])  # [123. 110. 164.]
print(x['value2'])  # [1.4 0.5 2.1]
print(x['value3'])  # [23. 18. 19.]
```

    [(1., 123., 1.4, 23.) (2., 110., 0.5, 18.) (3., 164., 2.1, 19.)]
    <class 'numpy.ndarray'>
    [('id', '<f8'), ('value1', '<f8'), ('value2', '<f8'), ('value3', '<f8')]
    [1. 2. 3.]
    [123. 110. 164.]
    [1.4 0.5 2.1]
    [23. 18. 19.]


## 1.3 文本格式选项

- numpy.set_printoptions(precision=None,threshold=None, edgeitems=None,linewidth=None, suppress=None, nanstr=None, infstr=None,formatter=None, sign=None, floatmode=None, **kwarg) Set printing options.
  - precision：设置浮点精度，控制输出的小数点个数，默认是8。
  - threshold：概略显示，超过该值则以“…”的形式来表示，默认是1000。
  - linewidth：用于确定每行多少字符数后插入换行符，默认为75。
  - suppress：当suppress=True，表示小数不需要以科学计数法的形式输出，默认是False。
  - nanstr：浮点非数字的字符串表示形式，默认nan。
  - infstr：浮点无穷大的字符串表示形式，默认inf。
  
  

These options determine the way floating point numbers, arrays and other NumPy objects are displayed.


```python
import numpy as np

np.set_printoptions(precision=4)
x = np.array([1.123456789])
print(x)  # [1.1235]

np.set_printoptions(threshold=20)
x = np.arange(50)
print(x)  # [ 0  1  2 ... 47 48 49]

np.set_printoptions(threshold=np.iinfo(np.int).max)
print(x)
# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
#  24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47
#  48 49]

eps = np.finfo(float).eps
x = np.arange(4.)
x = x ** 2 - (x + eps) ** 2
print(x)  
# [-4.9304e-32 -4.4409e-16  0.0000e+00  0.0000e+00]
np.set_printoptions(suppress=True)
print(x)  # [-0. -0.  0.  0.]

x = np.linspace(0, 10, 10)
print(x)
# [ 0.      1.1111  2.2222  3.3333  4.4444  5.5556  6.6667  7.7778  8.8889
#  10.    ]
np.set_printoptions(precision=2, suppress=True, threshold=5)
print(x)  # [ 0.    1.11  2.22 ...  7.78  8.89 10.  ]
```

    [1.1235]
    [ 0  1  2 ... 47 48 49]
    [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
     24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47
     48 49]
    [-0. -0.  0.  0.]
    [-0. -0.  0.  0.]
    [ 0.      1.1111  2.2222  3.3333  4.4444  5.5556  6.6667  7.7778  8.8889
     10.    ]
    [ 0.    1.11  2.22 ...  7.78  8.89 10.  ]


- numpy.get_printoptions() Return the current print options.


```python
import numpy as np

x = np.get_printoptions()
print(x)
```

    {'edgeitems': 3, 'threshold': 5, 'floatmode': 'maxprec', 'precision': 2, 'suppress': True, 'linewidth': 75, 'nanstr': 'nan', 'infstr': 'inf', 'sign': '-', 'formatter': None, 'legacy': False}


# 2 练习

## 2.1 只打印或显示numpy数组rand_arr的小数点后3位。

- rand_arr = np.random.random([5, 3])

【知识点：输入和输出】

- 如何在numpy数组中只打印小数点后三位？


```python
import numpy as np

rand_arr = np.random.random([5, 3])
print(rand_arr)
np.set_printoptions(precision=3)
print(rand_arr)
```

    [[0.83 0.88 0.47]
     [0.74 0.89 0.19]
     [0.95 0.86 0.72]
     [0.75 0.18 0.17]
     [0.17 0.01 0.4 ]]
    [[0.833 0.883 0.466]
     [0.741 0.892 0.188]
     [0.952 0.859 0.716]
     [0.753 0.185 0.168]
     [0.175 0.014 0.401]]


## 2.2 将numpy数组a中打印的项数限制为最多6个元素。

【知识点：输入和输出】

- 如何限制numpy数组输出中打印的项目数？


```python
import numpy as np

a = np.arange(10)
print(a)
np.set_printoptions(threshold=6)
print(a)
```

    [0 1 2 ... 7 8 9]
    [0 1 2 ... 7 8 9]


## 2.3 打印完整的numpy数组a而不中断

【知识点：输入和输出】

- 如何打印完整的numpy数组而不中断？


```python
import numpy as np

a = np.arange(10)
np.set_printoptions(threshold=6)
print(a)

np.set_printoptions(threshold=np.iinfo(np.int).max)
print(a)
```

    [0 1 2 ... 7 8 9]
    [0 1 2 3 4 5 6 7 8 9]

