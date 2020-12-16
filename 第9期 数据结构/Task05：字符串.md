# 字符串
## 1. 无重复字符的最长子串
```
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        """
        :type s: str
        :rtype: int
        """
        from collections import defaultdict
        lookup = defaultdict(int)
        start = 0
        end = 0
        max_len = 0
        counter = 0
        while end < len(s):
            if lookup[s[end]] > 0:
                counter += 1
            lookup[s[end]] += 1
            end += 1
            while counter > 0:
                if lookup[s[start]] > 1:
                    counter -= 1
                lookup[s[start]] -= 1
                start += 1
            max_len = max(max_len, end - start)
        return max_len
```

## 2. 串联所有单词的子串
```
class Solution:
    def findSubstring(self, s: str, words: List[str]) -> List[int]:
        from collections import Counter
        if not s or not words:return []
        all_len = sum(map(len, words))
        n = len(s)
        words = Counter(words)
        res = []
        for i in range(0, n - all_len + 1):
            tmp = s[i:i+all_len]
            flag = True
            for key in words:
                if words[key] != tmp.count(key):
                    flag = False
                    break
            if flag:res.append(i)
        return res
```

## 3. 替换子串得到平衡字符串
```
class Solution:
    def balancedString(self, s: str) -> int:
        n = len(s)
        b = n // 4
        from collections import Counter
        counter = Counter(s)
        counter = {key:value for key,value in counter.items() if value > b}
        
        if not counter or n < 4:
            return 0
        rmove = True
        
        l,r = 0,0
        minlen = n
        
        while l <= r and r < n:
            
            if s[r] in counter and rmove:
                counter[s[r]] -= 1
            elif l > 0 and s[l - 1] in counter and not rmove:
                counter[s[l - 1]] += 1

            if {key:value for key,value in counter.items() if value > b}:
                r += 1
                rmove = True
            else:
                minlen = min(minlen, r - l + 1)
                l += 1
                rmove = False
                         
        return minlen
```











