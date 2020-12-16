# Task04：队列

```
import time, threading

class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0,item)

    def dequeue(self):
        if self.items != []:
            return self.items.pop()
        else:
            return False

    def size(self):
        return len(self.items)

    def top(self):
        if self.items != []:
            return self.items[len(self.items)-1]
        else:
            return False

class Counter(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.waitQueue = Queue() ## 初始化等待的队伍
        self.lock = threading.Lock()

    def callIng(self):
        while True:
            ### 柜台一直叫号，要一直循环
            time.sleep(5)
            if not self.waitQueue.isEmpty():
                self.lock.acquire()
                print("请客户{},到{}窗口办理业务".format(self.waitQueue.top(), threading.current_thread().name))
                self.waitQueue.dequeue()
                self.lock.release()


class bankSystem:
    def __init__(self):
        self.serviceQueue = Queue()
        self.nowNum = 0
        # self.windows = k  # 银行柜台数目
        self.maxSize = 100

    def getNumber(self):
        if self.nowNum < self.maxSize:
            self.nowNum += 1
            return self.nowNum
        else:
            print("现在业务繁忙，请稍后再来")


if __name__ == "__main__":
    res = bankSystem()
    windowcount = 3
    serviceWindow = [None] * windowcount
    threadList = [None] * windowcount

    for i in range(windowcount):
        serviceWindow[i] = Counter()
        serviceWindow[i].waitQueue = res.serviceQueue
        threadList[i] = threading.Thread(name=(i + 1), target=serviceWindow[i].callIng, args=())
        threadList[i].start()
        # threadList[i].join()

    while True:
        input("请点击触摸屏获取号码：")
        # print()
        callNumber = res.getNumber()
        if res.serviceQueue != None:
            print("当前您的的号码为" + str(callNumber) + "，您前面还有" + str(res.serviceQueue.size()) + "个人")
            res.serviceQueue.enqueue(res.nowNum)
        else:
            print('您的号码是：%d，您前面有 0 位' % (callNumber))
```






























