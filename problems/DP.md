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

Memoization Search (From DFS to DP)
- Like cahce in system design
- Applies to pure function

Dynamic Programming
- A problems's solution rely on its sub-problems' solutions
- reduce subproblems that overlap (no overlap for divide-and-conquer)
- unlike the greedy algorithm, which always maxmizes its interests for every step, dp will trade a short-term loss for long-term interests 

#### [120. Triangle](https://leetcode.com/problems/triangle/) Medium

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

##### DP from bottom up
- state: coordinates
- function: possible moves
- initializtion: ending points
- answer: start points
1
2 3
4 5 6 <- start from here
```python
class Solution(object):
    def minimumTotal(self, triangle):
        n = len(triangle)
        """
        [0]
        [0,0]
        [0,0,0]
        ...
        """
        # dp[i][j] represents the shortest distance frm (i, j) to the bottom level
        dp = [[0] * (i + 1) for i in range(n)]

        """
        [0]
        [0,0]
        [4,5,6] <-
        """ 
        for i in range(n):
            dp[n - 1][i] = triangle[n - 1][i]
        
        """
        [0]
        [*, 0] <- iteration starts from *, i = n - 2, j = 0
         |\ |\
        [4, 5,6]  for dp[1][0] : 4 -> dp[i + 1][j] 5 -> dp[i + 1][j + 1] 
        """
        for i in range(n - 2, -1, -1):
            # count the shortest distance for every point on this level
            for j in range(i + 1):
                dp[i][j] = min(dp[i + 1][j], dp[i + 1][j + 1]) + triangle[i][j]
        return dp[0][0]  
```

##### DP from top down
1    <- start from here
2 3
4 5 6
```python
class Solution(object):
    def minimumTotal(self, triangle):
        n = len(triangle)
        """
        [0]
        [0,0]
        [0,0,0]
        ...
        """
        # dp[i][j] represents the shortest distance frm (0, 0) to (i, j)
        dp = [[0] * (i + 1) for i in range(n)]

        """
        [1]    <-
        [3,4]  3 = triangle[1][0] + dp[1 - 1][0] = 1 + 2
        [7,0,10] 10 = triangle[2][2] + dp[2-1][2-1]
        """ 
        # intialization: points on the left-most and right-most side because down right move (i + 1, j + 1) is not possible 
        dp[0][0] = triangle[0][0]    
        for i in range(1, n):
            # 
            dp[i][0] = triangle[i][0] + dp[i - 1][0]
            dp[i][i] = triangle[i][i] + dp[i - 1][i - 1]
        
        """
        [1]
        [3, 4]
        [7, *, 10] <- iteration starts from *, i = 2, j = 1, min(3, 4)
        """
        for i in range(2, n):
            for j in range(1, i):
                dp[i][j] = triangle[i][j] + min(dp[i - 1][j - 1], dp[i - 1][j])
        # the shortest distance on the bottom level, min([7, 8, 10])
        return min(dp[n - 1])  
```



#### [292. Nim Game](https://leetcode.com/problems/nim-game/) Easy
Memoization will stack over flow when dealing with DP problems with $O(n)$ time complexity.
If time complexity and recursion depth both reaches O(n), a stack overflow error is likely to ensue. n might be a very large number, say $10^8$, and this is not allowed for recursion depth.($10^5$ is the maximum for python) But when time complexity is $O(n^2)$, say $(10^4)^2$, recursion depth is only $O(n)$, i.e. $10^4$, and this is acceptable.
```pyhton
class Solution(object):
    def canWinNim(self, n):
        return self.memo_search(n, {})
    
    def memo_search(self, n, memo):
        # less than 3 stones
        if n <= 3:
            return True
        
        if n in memo:
            return memo[n]
        
        # remove 1 to 3 stones
        for i in range(1, 4):
            # the opponent remove n - i stones and lose
            if not self.memo_search(n - i, memo):
                memo[n] = True
                return True
            
        # all possible moves lead to not even a single win
        memo[n] = False
        return False
```
Any number of stones that is 4 times will lead the one who go first to lose.
```python
class Solution(object):
    def canWinNim(self, n):
        return n % 4 > 0
```
