# Data_Whale_Study
## 1 利用动态数组解决数据存放问题
编写一段代码，要求输入一个整数N，用动态数组A来存放2~N之间所有5或7的倍数，输出该数组.

```
def matrix_57(N):
    a = []
    for i in range(2,N+1):
        if i%5==0 or i%7==0:
            a.append(i)
    return a
```


## 2 托普利茨矩阵问题
如果一个矩阵的每一方向由左上到右下的对角线上具有相同元素，那么这个矩阵是托普利茨矩阵。
给定一个M x N的矩阵，当且仅当它是托普利茨矩阵时返回True。

```
def isToeplitzMatrix(a):
    row = len(a)
    col = len(a[0])
    
    for i in range(row-1):
        for j in range(col-1):
            if(matrix[i][j]!=matrix[i+1][j+1]):
                return False
    return True
```

## 3 三数之和
给定一个包含 n 个整数的数组nums，判断nums中是否存在三个元素a，b，c，使得a + b + c = 0？找出所有满足条件且不重复的三元组。

注意：答案中不可以包含重复的三元组。

```
def threeSum(nums) :
    nums.sort()
    res, k = [], 0
    for k in range(len(nums) - 2):
        if nums[k] > 0: break # 1. because of j > i > k.
        if k > 0 and nums[k] == nums[k - 1]: continue # 2. skip the same `nums[k]`.
        i, j = k + 1, len(nums) - 1
        while i < j: # 3. double pointer
            s = nums[k] + nums[i] + nums[j]
            if s < 0:
                i += 1
                while i < j and nums[i] == nums[i - 1]: i += 1
            elif s > 0:
                j -= 1
                while i < j and nums[j] == nums[j + 1]: j -= 1
            else:
                res.append([nums[k], nums[i], nums[j]])
                i += 1
                j -= 1
                while i < j and nums[i] == nums[i - 1]: i += 1
                while i < j and nums[j] == nums[j + 1]: j -= 1
    return res
```
