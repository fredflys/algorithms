Memoization Search (From DFS to DP)
#### [509. Fibonacci Number <span style="color:green">Easy</span>](https://leetcode-cn.com/problems/fibonacci-number/)

```python
# recursion and memorization: from the top down
class Solution:
    def fib(self, n: int) -> int:
        memo = [0 for _ in range(n + 1)]
        
        return self.calculate(n, memo)

    def calculate(self, n, memo):
        if n < 2:
            return n
        if memo[n] != 0:
            return memo[n]
        memo[n] = self.calculate(n - 1, memo) + self.calculate(n - 2, memo)
        return memo[n]
```

```java
// DP: from the bottom up 
class Solution {
    public int fib(int n) {
        if (n < 2)
            return n;
        int[] dp = new int[n + 1];
        dp[1] = dp[2] = 1;
        for (int i = 3; i <= n; i++)
            dp[i] = dp[i - 1] + dp[i - 2];
        return dp[n]; 
    }
}
```

```java
// to save more space: for every iteration, only three numbers are involved
class Solution {
    public int fib(int n) {
        if (n < 2)
            return n;
        int prev = 1, cur = 1;
        for (int i = 3; i <= n; i++) {
            int sum = prev + cur;
            prev = cur;
            cur = sum;
        }
        return cur;
    }
}
```

[120. Triangle](https://leetcode.com/problems/triangle/) Medium

##### DFS
For a n-level triangle, there will be n(n-1)*2 points.
For every point there will be two routes. A triangle is thus 
turned into a binary tree for DFS traversing.
e.g. routes
1 
2 3
45 56
Time Complexity: $2^n$
```python
class Solution(object):
    def minimumTotal(self, triangle):
        self.minimum = sys.maxsize
        self.traverse(0, 0, triangle, 0)
        return self.minimum
        
    """
      y
    x 1     (0,0)
      2 3   (1,0) (1,2)
      4 5 6 ...
    either move down (x +1, y) or mover down right (x + 1, y + 1)
    """
    def traverse(self, x, y, triangle, result):
        if x == len(triangle):
            self.minimum = min(result, self.minimum)
            return
        
        # result is defined the sum from the start to the point before x, y
        result += triangle[x][y]    
        self.traverse(x + 1, y, triangle, result)
        self.traverse(x + 1, y + 1, triangle, result)
```
##### Divide and Conquer
Time Complexity: still $2^n$
```python
class Solution(object):
    def minimumTotal(self, triangle):
        return self.divide_conquer(0, 0, triangle)

    # the least sum from start to the end 
    def divide_conquer(self, x, y, triangle):
        if x == len(triangle):
            return 0
        
        left_reult = self.divide_conquer(x + 1, y, triangle)
        right_result = self.divide_conquer(x + 1, y + 1, triangle)
        return min(left_reult, right_result) + triangle[x][y]
```

##### Divide and conquer with memory search
The routes are reduced to the total of points.
Time Complexity: $n^2$
```python
class Solution(object):
    def minimumTotal(self, triangle):
        return self.divide_conquer(0, 0, triangle, {})

    def divide_conquer(self, x, y, triangle, memo):
        if x == len(triangle):
            return 0

        # before getting into recursion, let's see if the result is already computed
        if (x, y) in memo:
            return memo[(x, y)]
        
        left_reult = self.divide_conquer(x + 1, y, triangle, memo)
        right_result = self.divide_conquer(x + 1, y + 1, triangle, memo)
        # save the result into memo for later reuse
        memo[(x, y)] = min(left_reult, right_result) + triangle[x][y]
        return memo[(x, y)]
```
