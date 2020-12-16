# 栈与递归
根据要求完成车辆重排的程序代码

## 1 用数组实现一个顺序栈
```
class ArrayStack():
    def __init__(self):
        self.items = []
    def IsEmpty(self):
        return self.items == []
    def size(self):
        return len(self.items)
    def top(self):
        if not self.IsEmpty():
            return self.items[-1]
        else:
            print("Stack is empty")
            return None
    
    def pop(self):
        if not self.IsEmpty():
            return self.items.pop()
        else:
            print("栈为空")
            return None
    def push(self,item):
        self.items.append(item)
```

## 2 用链表实现一个链栈。
```
class ListNode():
    def __init__(self,x):
        self.val=x
        self.next=None
class LinkedLiskStack():
    def __init__(self):
        self.val = None
        self.next = None
    
    def isempty(self):
        return self.next == None
    def size(self):
        i=0
        p=self.next
        while p != None:
            p = p.next
            i += 1
        return i
    def top(self):
        if self.next != None:
            return self.next.val
        else:
            print("栈为空，无法获取栈顶数据")
            return None
        
    def pop(self):
        if self.next != None:
            p = self.next
            self.next = self.next.next
            return p
        else:
            print("栈为空，无法出栈")
            return None
    def push(self,e):
        p = ListNode(e)
        if self.next != None:
            p.next = self.next
        self.next = p
```


## 3 完成车辆重排

```
def output(stacks, n):
    global minVal, minStack
    stacks[minStack].pop()
    print('移动车厢 %d 从缓冲铁轨 %d 到出轨。' % (minVal, minStack))
    minVal = n + 2
    minStack = -1
    for index, stack in enumerate(stacks):
        if((not stack.isempty()) and (stack.top() < minVal)):
            minVal = stack.top()
            minStack = index

def inputStack(i, stacks, n):
    global minVal, minStack
    beskStack = -1  # 最小车厢索引值所在的缓冲铁轨编号
    bestTop = n + 1  # 缓冲铁轨中的最小车厢编号
    for index, stack in enumerate(stacks):
        if not stack.isempty():  # 若缓冲铁轨不为空
            # 若缓冲铁轨的栈顶元素大于要放入缓冲铁轨的元素，并且其栈顶元素小于当前缓冲铁轨中的最小编号
            a = stack.top()
            # print('stack.top()的类型是', a)
            if (a > i and bestTop > a):
                bestTop = stack.top()
                beskStack = index
        else:  # 若缓冲铁轨为空
            if beskStack == -1:
                beskStack = index
                break
    if beskStack == -1:
        return False
    stacks[beskStack].push(i)
    print('移动车厢 %d 从入轨到缓冲铁轨 %d。' % (i, beskStack))
    if i < minVal:
        minVal = i
        minStack = beskStack
    return True

def rail_road(list, k):
    global minVal, minStack
    stacks = []
    for i in range(k):
        stack = stack1.ArrayStack()
        stacks.append(stack)
    nowNeed = 1
    n = len(list)
    minVal = n + 1
    minStack = -1
    for i in list:
        if i == nowNeed:
            print('移动车厢 %d 从入轨到出轨。' % i)
            nowNeed += 1
            # print("minVal", minVal)
            while (minVal == nowNeed):
                output(stacks, n)  # 在缓冲栈中查找是否有需求值
                nowNeed += 1
        else:
            if(inputStack(i, stacks, n) == False):
                return False
    return True
    
if __name__ == "__main__":
    list = [3, 6, 9, 2, 4, 7, 1, 8, 5]
    k = 3
    minVal = len(list) + 1
    minStack = -1
    result = rail_road(list, k)
    while(result == False):
        print('需要更多的缓冲轨道，请输入需要添加的缓冲轨道数量。')
        k = k + int(input())
        result = rail_road(list, k)
```
