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