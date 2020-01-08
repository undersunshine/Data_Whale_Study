# Data_Whale_Study
## 1 合并两个有序链表
将两个有序链表合并为一个新的有序链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。 

```
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        ans = tmp = ListNode(0)
        while l1 and l2:
            if l1.val < l2.val:
                l1, tmp.next = l1.next, l1
            else:
                l2, tmp.next = l2.next, l2
            tmp = tmp.next
        tmp.next = l1 or l2
        return ans.next
```


## 2 删除链表的倒数第N个节点
给定一个链表，删除链表的倒数第 n 个节点，并且返回链表的头结点。

```
class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        slow = fast = head
        for i in range(n):          
            fast = fast.next
        if fast == None:            
            return head.next
        
        while fast.next != None:    
            slow = slow.next        
            fast = fast.next
        slow.next = slow.next.next  
        return head
```

## 3 旋转链表
给定一个链表，旋转链表，将链表每个节点向右移动 k 个位置，其中 k 是非负数。

```
class Solution:
    def rotateRight(self, head: ListNode, k: int) -> ListNode:
        if head is None or head.next is None: return head
        start, end, len = head, None, 0
        while head:
            end = head
            head = head.next
            len += 1
        end.next = start
        pos = len - k % len
        while pos > 1:
            start = start.next
            pos -= 1
        ret = start.next
        start.next = None
        return ret
```




