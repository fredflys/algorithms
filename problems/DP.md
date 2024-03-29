### Background

##### Recursion vs Dynamic Programming

<img src="https://s2.loli.net/2022/02/27/IqKkAerxpcDJsH5.png" style="zoom:50%;" />

##### Scenarios

![image-20220308181206142.png](https://s2.loli.net/2022/03/08/PciAqnNtrZERa25.png)

- maxmium or minimum
- whether a solution exists
- total number of solutions

##### Limits

![image-20220308181344404.png](https://s2.loli.net/2022/03/08/VLnXrkaxDcKusjP.png)

##### Problem Type

![image-20220308180358329.png](https://s2.loli.net/2022/03/08/Uzl8Net4DaI9qsQ.png)

- coordinates \*
  - dp[i][j]: solutions/max/min/is possilbe from start to end
- prefix \*
  - dp[i] ... for the first i chars
  - dp[i][j] ... for j parts of the first i chars
  - dp[i][j] ... for the first 1 chars of the first string and the first j chars of the second string
- backpack
  - ... for the first i items that weigh no more than j in total
- interval
  - dp[i][j] ... between i and j
- game
- tree
- state compression

##### Memoization Search (From DFS to DP)

- Like cahce in system design
- Applies to pure function

##### Dynamic Programming

- A problems's solution rely on its sub-problems' solutions
- reduce subproblems that overlap (no overlap for divide-and-conquer)
- unlike the greedy algorithm, which always maxmizes its interests for every step, dp will trade a short-term loss for long-term interests

### Memoization

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

#### [139. Word Break](https://leetcode.com/problems/word-break/) Medium

<img src="https://s2.loli.net/2022/03/08/V1vpmzxlMD48tid.png" alt="image-20220308151400417.png" style="zoom:50%;" />

"leetcode" breakable for ["leet", "code"] ?

- **l** is in word dictionary and the remaining sub-string breakable
- or **le** is in word dictionary and the remaining sub-string breakable
- .......

<img src="https://s2.loli.net/2022/03/08/1xnkIKOjmG84sM5.png" alt="image-20220308111425087.png" style="zoom:50%;" />

##### DFS + Memoization

Recursion

```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        if not s:
            return True

        if not wordDict:
            return False

        max_length = len(max(wordDict, key = len))
        return self.dfs(s, 0, max_length, wordDict, {})

    def dfs(self, s, index, max_length, wordDict, memo):
        if index in memo:
            return memo[index]


        length = len(s)
        if index == len(s):
            return True


        for end in range(index + 1, length + 1):
            # end will only get bigger in the next iteration
            # so it is safe to stop the iteration
            if (end - index) > max_length:
                break

            partial_word = s[index : end]

            # give it another shot when the word is not found
            if partial_word not in wordDict:
                continue

            if self.dfs(s, end, max_length, wordDict, memo):
                return True

        memo[index] = False
        return False
```

Iteration

```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        length = len(s)
        return self.canBreak(0, length, s, wordDict, {})

    def canBreak(self, index, length, s, wordDict, memo):
        if index == length:
            return True

        if index in memo:
            return memo[index]

        for end in range(index + 1, length + 1):
            partial = s[index : end]

            if partial not in wordDict:
                continue

            if self.canBreak(end, length, s, wordDict, memo):
                memo[index] = True
                return True

        memo[index] = False
        return False
```

##### BFS

<img src="https://s2.loli.net/2022/03/08/mnfqRBsltovkZPI.png" alt="image-20220308113200235.png" style="zoom:50%;" />

```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        length = len(s)
        visited = []
        queue = collections.deque([])
        queue.append(0)
        while len(queue) > 0:
            start = queue.popleft()
            if start in visited:
                continue
            visited.append(start)

            for end in range(start + 1, length + 1):
                partial_word = s[start : end]

                if partial_word not in wordDict:
                    continue
                if end == length:
                    return True

                queue.append(end)
       	return False
```

##### DP

Division, DP
Divide into sub-problems

- the the sub-string starting from 0 and ending at i is breakable
- the remaining part (from i to end + 1) is in word dict

<img src="https://s2.loli.net/2022/03/08/TZ52rJVtLYumWMD.png" alt="image-20220308124812728.png" style="zoom:50%;" />

where will that final cut be?

```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        length = len(s)
        # dp[i] represents if the sub-string starting from 0 and ending at i (not included) is breakable
        dp = [False for _ in range(length + 1)]
        dp[0] = True

        """
              e
        l e e t c o d e
            s
          s
        s
        """
        # from bottom up
        for end in range(1, length + 1):
            for start in range(end - 1, -1, -1):
                partial = s[start : end]
                # the latter part is in word dictionary and the former part is breakable
                if partial in wordDict and dp[start]:
                    dp[end] = True
                    break

        return dp[length]
```

```java
class Solution {
    public boolean wordBreak(String s, List<String> wordDict) {
        int n = s.length();
        boolean[] dp = new boolean[n + 1];

        dp[0] = true;

        for (int end = 1; end <= n; end++) {
            for (int start = 0; start < end; start++) {
                if (!dp[start]) {
                    continue;
                }

                String word = s.substring(start, end);
                if (wordDict.contains(word)) {
                    dp[end] = true;
                    // one way to break the word  is found, that's enough
                    break;
                }
            }
        }

        return dp[n];
    }
}
```

Improvement: a typical word, unlike string, will not be too long. A time complexity of n\*_2 is too much.
Will be better if it is n _ word length.

```java
class Solution {
    public boolean wordBreak(String s, List<String> wordDict) {
        int maxLength = 0;
        for (String word: wordDict) {
            maxLength = Math.max(maxLength, word.length());
        }

        int n = s.length();
        boolean[] dp = new boolean[n + 1];
        dp[0] = true;

        for (int end = 1; end <= n; end++) {
            // cut from higher end
            // so start will be end minus wordLen
            for (int wordLen = 1; wordLen <= maxLength; wordLen++) {
                if (end < wordLen>) {
                    break;
                }
                if (!dp[end - wordLen]) {
                    continue;
                }

                String word = s.substring(end - wordLen, end);
                if (wordDict.contains(word)) {
                    dp[end] = true;
                    // one way to break the word  is found, that's enough
                    break;
                }
            }
        }
        return dp[n];
    }
}
```

#### [140. Word Break II](https://leetcode-cn.com/problems/word-break-ii/) Hard

Use DP to quicken the process

```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> List[str]:
        length = len(s)
        dp = [False for _ in range(length + 1)]
        dp[0] = True

        for end in range(1, length + 1):
            for start in range(end - 1, -1, -1):
                partial = s[start : end]
                if partial in wordDict and dp[start]:
                    dp[end] = True
                    break

        results = []
        if dp[length]:
            path = collections.deque([])
            self.dfs(wordDict, dp, path, results, s, length)
        return results

    def dfs(self, wordDict, dp, path, results, s, length):
        # break is finished when breaking from the start
        if length == 0:
            results.append(" ".join(path))
        """
                i
        l e e t c o d e
        <dp[4]> {  p  }

        i
        l e e t
  dp[0] {  p  }

        """
        # break from right to left
        for	i in range(length - 1, -1 , -1):
            partial = s[i : length]
            if dp[i] and partial in wordDict:
                path.appendleft(partial)
                self.dfs(wordDict, dp, path, results, s, i)
                path.popleft()
```

#### [131. Palindrome Partitioning](https://leetcode-cn.com/problems/palindrome-partitioning/) Medium

#### [LintCode 829 · Word Pattern II](https://www.lintcode.com/problem/829/) Medium

DFS

Memoization cannot be applied because when parameters remain the same (pattern, target) the output is not stable since mapping is constantly changing.

![image-20220308161320667.png](https://s2.loli.net/2022/03/08/6jO7oHAsau9Y4Xh.png)

```python
class Solution:
    """
    @param pattern: a string,denote pattern string
    @param target: a string, denote matching string
    @return: a boolean
    """
    def word_pattern_match(self, pattern: str, target: str) -> bool:
        return self.is_match(pattern, target, {})

    def is_match(self, pattern, target, mapping):
        # nothing in pattern for match
        if not pattern:
            # nothing in target for match
            return not target

        char = pattern[0]

        if char in mapping:
            word = mapping[char]
            if not target.startswith(word):
                return False
            return self.is_match(pattern[1:], target[len(word):], mapping)

        # if char has not a mapping in target string
        # every prefix is a possible mapping
        for i in range(len(target)):
            word = target[:i + 1]

            # char -> word must be a unique pair
            # e.g. if x is mapped to ab, y cannot be mapped to ab
            if word in mapping.values():
                continue

            mapping[char] = word
            if self.is_match(pattern[1:], target[i + 1:], mapping):
                return True
            del mapping[char]

        return False
```

#### [44. Wildcard Matching](https://leetcode.com/problems/wildcard-matching/) Hard

DFS + Memoization

![image-20220308172953248.png](https://s2.loli.net/2022/03/08/LgxyNMuz5fvj3km.png)

Pruning

![image-20220308172801752.png](https://s2.loli.net/2022/03/08/wUvbmiknyWLq4rp.png)

```java
class Solution {
    public boolean isMatch(String s, String p) {
        boolean[][] memo = new boolean[s.length()][p.length()];
        boolean[][] visited = new boolean[s.length()][p.length()];
        return dfs(s, 0, p, 0, memo, visited);
    }

    public boolean dfs(String source, int sIndex, String pattern, int pIndex, boolean[][] memo, boolean[][] visited) {
        // nothing in pattern for match
        if (pIndex == pattern.length()) {
            // nothing in source for match
            return sIndex == source.length();
        }

        // nothing in source for match
        if (sIndex == source.length()) {
            // pattern is not exhuasted and all left is stars, which can match empty string
            // this is considered matched
            return allLeftStars(pattern, pIndex);
        }

        if (visited[sIndex][pIndex]) {
            return memo[sIndex][pIndex];
        }

        char sChar = source.charAt(sIndex);
        char pChar = pattern.charAt(pIndex);
        boolean matched;

        if (pChar != '*') {
            matched = charMatch(sChar, pChar) && dfs(source, sIndex + 1, pattern, pIndex + 1, memo, visited);
        } else {
            // instead of matching (matching will move sIndex and pIndex at the same time), * eats a char from source string
            matched = dfs(source, sIndex + 1, pattern, pIndex, memo, visited)
			// * matches empty string
            || dfs(source, sIndex, pattern, pIndex + 1, memo, visited);
        }

        visited[sIndex][pIndex] = true;
        memo[sIndex][pIndex] = matched;
        return matched;


        // * can match zero or many chars
        // for (int i = sIndex; i <= source.length(); i++) {
        //     // match zero to the whole source string
        //     if (dfs(source, i, pattern, pIndex + 1)) {
        //         return true;
        //     }
        // }
        // return false;

        /*
        pruning by removing the iteration
        */
        // return dfs(source, sIndex + 1, pattern, pIndex) || dfs(source, sIndex, pattern, pIndex + 1)

    }

    public boolean allLeftStars(String pattern, int pIndex) {
        for (int i = pIndex; i < pattern.length(); i++){
            if (pattern.charAt(i) != '*') {
                return false;
            }
        }

        return true;
    }

    public boolean charMatch(char sChar, char pChar) {
        // ? can match any char
        return sChar == pChar || pChar == '?';
    }
}
```

#### [10. Regular Expression Matching ](https://leetcode-cn.com/problems/regular-expression-matching/)Hard

### Dynamic Programming

#### [120. Triangle](https://leetcode.com/problems/triangle/) Medium

##### DFS

For a n-level triangle, there will be n(n-1)\*2 points.
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

        # initialization
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

1 <- start from here
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

##### Use Rolling array to save space

<img src="https://s2.loli.net/2022/03/14/AF5r8OT974aBdSJ.png" alt="image-20220314171240868.png" style="zoom:50%;" />

<img src="https://s2.loli.net/2022/03/10/uYbITGasvP5jSyK.png" alt="image-20220310162627894.png" style="zoom:50%;" />

Modulo operation (% 2) will roll the array. Space complexity is reduced from $O(n^2)$ to $O(n)$.

```python
class Solution(object):
    def minimumTotal(self, triangle):
        n = len(triangle)

        # [[0] * n ] * 2 is not the right way to build the matrix
        dp = [[0] * n, [0] * n]
        dp[0][0] = triangle[0][0]

        for i in range(1, n):
            dp[i % 2][0] = triangle[i][0] + dp[(i - 1) % 2][0]
            dp[i % 2][i] = triangle[i][i] + dp[(i - 1) % 2][i - 1]
            for j in range(1, i):
                dp[i % 2][j] = triangle[i][j] + min(dp[(i - 1) % 2][j - 1], dp[(i - 1) % 2][j])
        return min(dp[(n - 1) % 2])
```

#### [292. Nim Game](https://leetcode.com/problems/nim-game/) Easy

Memoization will stack over flow when dealing with DP problems with $O(n)$ time complexity.
If time complexity and recursion depth both reaches O(n), a stack overflow error is likely to ensue. n might be a very large number, say $10^8$, and this is not allowed for recursion depth.($10^5$ is the maximum for python) But when time complexity is $O(n^2)$, say $(10^4)^2$, recursion depth is only $O(n)$, i.e. $10^4$, and this is acceptable.

```python
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

#### [62. Unique Paths](https://leetcode.com/problems/unique-paths/) Medium

coordinates, DP

##### DP from top down

```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        # possible unique paths from starting point to grid[i][j]
        dp = [[0] * n for _ in range(m)]
        # initialization
        """
        to get to the first column, there is only one possible path
        o ...
        o ...
        """
        for i in range(m):
            dp[i][0] = 1
        """
        to get to the first row, there is only one possible path
        o o o ..
        """
        for j in range(n):
            dp[0][j] = 1

        """
                           o o <- from [i -1][j]
        from [i][j - 1] -> o * -- to get here
        """
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]

        return dp[m - 1][n - 1]
```

##### DP from bottom up

```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        # possible unique paths from grid[i][j] to grid[m - 1][n - 1] (end point)
        dp = [[0] * n for _ in range(m)]
        # initialization
        """
        to reach the end, there is only one possible path for the last column
        ... o
        ... o
        ... *
        """
        for i in range(m):
            dp[i][n - 1] = 1
        """
        to reach the end, there is only one possible path for the last row
        .....
        .....
        o o *
        """
        for j in range(n):
            dp[m - 1][j] = 1

        """
            to get here -- * o <- from [i][j + 1]
        from [i + 1][j] -> o o
        """
        for i in range(m - 2, -1, -1):
            for j in range(n - 2, -1, -1):
                dp[i][j] = dp[i][j + 1] + dp[i + 1][j]

        return dp[0][0]
```

#### [63. Unique Paths II](https://leetcode.com/problems/unique-paths-ii/) Medium

coordinates, DP

```python
class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        def blocked(i,j):
            return obstacleGrid[i][j] > 0

        # if the starting point blocked, it is not possible to move
        if blocked(0, 0):
            return 0

        # build
        m = len(obstacleGrid) # rows
        n = len(obstacleGrid[0]) # columns
        # dp[i][j] represents the number of unique paths to the ending point
        dp = [[0] * n for _ in range(m)]

        # initialize
        dp[0][0] = 1
        for i in range(1, m):
            if blocked(i, 0):
                break
            dp[i][0] = 1
        for j in range(1, n):
            if blocked(0, j):
                break
            dp[0][j] = 1

        # calculate
        for i in range(1, m):
            for j in range(1, n):
                if not blocked(i, j):
                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1]

        return dp[m - 1][n - 1]
```

#### [55. Jump Game](https://leetcode.com/problems/jump-game/) Medium

coordinates
DP: Wheter a solution exists.

```java
class Solution {
    public boolean canJump(int[] nums) {
        // dp[i] whether nums[i] can be reached
        boolean[] dp = new boolean[nums.length];
        // initialzation: starting point is certainly reachable
        dp[0] = true;

        // get states for every position other than the starting point
        for (int i = 1; i < nums.length; i++) {
            // for a position, as long as it can be reached from any position that comes before it, mark it as reachable
            // try every position before
            // example: . . . i => j .. i or . j . i or . . j i
            for (int j = 0; j < i; j++) {
                // j if reachable and i can be reached from j
                if (dp[j] && nums[j] + j >= i) {
                    dp[i] = true;
                    break;
                }
            }
        }

        return dp[nums.length - 1];
    }
}
```

#### [45. Jump Game II](https://leetcode-cn.com/problems/jump-game-ii/) Medium

coordinates, DP

```python
class Solution:
    def jump(self, nums: List[int]) -> int:
        # dp[i] represents the leats number of jumps to arrive at nums[i]
        n = len(nums)
        # initialize: since the end is reachable, the maximum jumps are n - 1
        dp = [n for _ in nums]
        dp[0] = 0

		"""
		   i
		[o,2,o,o,o]
		     _ _
		"""
		# calculdate for every position other than the starting point
        for position in range(n):
            # jump from one position to nums[i] positions (included)
            # e.g. 2 .. for index 0, jump 1 to index 1 and jump 2 to index 2
            for jump in range(1, nums[position] + 1):
                # out of bound
                if position + jump >= n:
                    return dp[n - 1]
                dp[position + jump] = min(dp[position + jump], dp[position] + 1)

        return dp[n - 1]
```

#### [LintCode 630 · Knight Shortest Path II](https://www.lintcode.com/problem/630/) Medium

coordinates
Since knight is allowed to move only right, there is no circular dependency and DP can be used.

```python
class Solution:
    """
    @param grid: a chessboard included 0 and 1
    @return: the shortest path
    """
    def shortest_path2(self, grid: List[List[bool]]) -> int:
        # write your code here
        # since
        """
        Possible Moves
        o x o o -2 1
        o o x o -1 2
        * o o o
        o o x o 1  2
        o x o o 2  1

        Possilbe Previous Positions (all on the left)
        o x o o -2 -1
        x o o o -1 -2
        o o * o
        x o o o 1  -2
        o x o o 2  -1
        """
        # since we are build dp from top down, previous positions instead of possible moves are needed
        previous_positions = [(-2, -1), (-1, -2), (1, -2), (2, -1)]

        m = len(grid)
        n = len(grid[0])
        # dp[i][j] represents the least moves to arrive at grid[i][j]
        dp = [[m * n] * n for _ in range(m)]
        dp[0][0] = 0

        """
        any previous position is on the left, so all the points in the first column will not be reached
        iterating from the frist row will mark the first row as non-reachable, which is wrong

        dp when Iterating from the first row (wrong):
        o o o o
        o o 1 o
        o 1 o o
        dp when Iterating from the first column:
        o o 2 o
        o o 1 o
        o 1 o 3
        """
        for j in range(n):
            for i in range(m):
                if grid[i][j] > 0:
                    continue
                for _x, _y in previous_positions:
                    x = i + _x
                    y = j + _y
                    if 0 <= x < m and 0 <= y < n:
                        dp[i][j] = min(dp[i][j], dp[x][y] + 1)

        if dp[m - 1][n - 1] == m * n:
            return -1

        return dp[m - 1][n - 1]
```

#### [300. Longest Increasing Subsequence ]([Longest Increasing Subsequence - LeetCode](https://leetcode.com/problems/longest-increasing-subsequence/))**Medium**

coordinates
DP: from left to right, best solution

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        length = len(nums)
        # dp[i] represents the longest subsequence ending at i and starting from 0
        dp = [1 for _ in nums]
        for i in range(1, length):
            for j in range(i):
                if nums[j] < nums[i]:
                    dp[i] = max(dp[i], dp[j] + 1)

        # the longest subsequence may not end at length - 1
        # so instead of returning dp[length - 1], return max(dp)
        return max(dp)
```

Soliatre DP, combine greedy algorithm to improve efficiency

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        if not nums:
            return 0

        # incrementaly update the LIST (longest increasing subsequence)
        # 4 5 6 1 2 3
        #       *
        #  0 1 2 3 4 5
        # -x x x x x x
        # -x 4 5 6 . .
        # -x 1 5 6
        lis = [float('inf')] * (len(nums) + 1)
        lis[0] = -float('inf')

        longest = 0
        for num in nums:
            index = self.find_first_gte(lis, num)
            lis[index] = num
            longest = max(longest, index)

        return longest

    # the first num in the sequence that is the first to be greater than or equal to the target number
    def find_first_gte(self, nums, target):
        start, end = 0, len(nums) - 1
        while start + 1 < end:
            mid = (start + end) // 2
            if nums[mid] >= target:
                end = mid
            else:
                start = mid

        if nums[start] == target:
            return start
        return end
```

[354. Russian Doll Envelopes](https://leetcode.com/problems/russian-doll-envelopes/) Hard
Solitare DP, LIS
Envelopes have width and height and small envelopes can be put in bigger ones. If envelopes are arranged in ascending order by width, does it mean envelopes[2] can fit into envelopes[3】? No, height also has to be taken into consideration. If sorted envelopes are now arranged in descending order by height, the problem is turned into one of finding the longest increasing sequence.\
[1, 8]
[2, 3] _
[5, 4] _
[5, 2]
[6, 7] \*
[6, 4]

```python
class Solution:
    def maxEnvelopes(self, envelopes: List[List[int]]) -> int:
        # sort by asceding width and descending height
        envelopes.sort(key = lambda x: (x[0], -x[1]))
        # print('envelopes: ', envelopes)

        n = len(envelopes)
        lis = [float('inf')] * (n + 1)
        lis[0] = -float('inf')
        # print('lis: ', lis)

        longest = 0
        for (_, h) in envelopes:
            index = self.find_first_gte(lis, h)
            # print("height width, index: ", lis, h, index)
            lis[index] = h
            longest = max(longest, index)
            print(longest, index)
            # print('now longest: ', longest)
            # print(lis)

        return longest

        # the first num in the sequence that is the first to be greater than or equal to the target number
    def find_first_gte(self, nums, target):
        start, end = 0, len(nums) - 1
        while start + 1 < end:
            mid = (start + end) // 2
            if nums[mid] >= target:
                end = mid
            else:
                start = mid

        if nums[start] >= target:
            return start
        return end
```

#### [329. Longest Increasing Path in a Matrix](https://leetcode.com/problems/longest-increasing-path-in-a-matrix/) **Hard**

coordinates
DP: from smaller numbers to big numbers (instead of matrix direction), best solution

```python
class Solution:
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        def is_valid(x,  y):
            return 0 <= x < m and 0 <= y < n

        def get_max(dp):
            result = 0
            for i in range(m):
                row_max = max(dp[i])
                if result < row_max:
                    result = row_max
            return result

        m, n = len(matrix), len(matrix[0])
        # dp[i][j] represents the longest increasing sequence ending at matrix[i][j]
        dp = [[1 for _ in range(n)] for _ in range(m)]

        # reorder matrix in increasing order for iteration
        points = []
        for i in range(m):
            for j in range(n):
                # value, position (row, column)
                points.append((matrix[i][j], i, j))

        """
        Why bothering to sort? The problem is about increasing sequence, i.e. from
        small numbers to big numbers. This indicates a direction. Starting from the
        smallest number will ensure that samller numbers are already updated for dp
        before iterating to larger numbers. If iteration starts from left to right
        and top to bottom, the first row in dp will be updated to 2 at most, which
        is not right.
        """
		# sorting based on the first element of the tuple
        points.sort()

        # move in four possible directions
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        for i in range(len(points)):
            x = points[i][1]
            y = points[i][2]

            for _x, _y in directions:
                new_x, new_y = x + _x, y + _y
                # skip if the possible point is not in matrix
                if not is_valid(new_x, new_y):
                    continue

                # update
                if matrix[new_x][new_y] < matrix[x][y]:
                    dp[x][y] = max(dp[x][y], dp[new_x][new_y] + 1)

        return get_max(dp)
```

#### [368. Largest Divisible Subset](https://leetcode.com/problems/largest-divisible-subset/) Medium

coordinates, DP

```python
class Solution:
    def largestDivisibleSubset(self, nums: List[int]) -> List[int]:
        # check if a can be divided by b or vice versa
        def divisible(a, b):
            return a % b == 0 or b % a == 0

        # build the answer from prev
        def build_result(dp, prev, nums):
            result = []
            # the largetst divisible set may not end at the tail
            # so get the ending index
            last = dp.index(max(dp))
            cursor = last
            while cursor > -1:
                result.insert(0, nums[cursor])
                cursor = prev[cursor]
            return result

        nums.sort()
        n = len(nums)
        # dp[i] represents the size of the largest divisible set ending at nums[i]
        dp = [1 for _ in nums]
        # since a specific set is what is nedded, instead of a max length, a prev set is used to store the previous indices
        # prev[i] represens the index of the previous divisible number to nums[i]
        prev = [-1 for _ in nums]

        for i in range(1, n):
            for j in range(i):
                if not divisible(nums[i], nums[j]):
                    continue

                """
                if the largest number can be divided by all the other numbers and the other numbers cannot
                be divided by each other, prev will be wrongly updated.
                e.g.
                nums is [2,4,5,20] and prev is updated whenever num[i] and nums[j] are divisible
                prev would be updated to [-1, 0, 1, 2]， which should be [-1, 0, -1, 1]
                """
                if dp[j] + 1 > dp[i]:
                    dp[i] = max(dp[i], dp[j] + 1)
                    prev[i] = j
        result = build_result(dp, prev, nums)
        return result
```

Solitare DP, divisible， DP as a list of lists, easy to understand

```python
class Solution:
    def largestDivisibleSubset(self, nums: List[int]) -> List[int]:
        nums.sort()
        # dp is a list of lists, like [[1], [2], ...]
        dp = [[x] for x in nums]
        # 0-th element remain the same
        for j in range(1, len(nums)):
            # backward from end to start
            # the further away it is from nums[i], the less likely it will be processed
            for i in range(j - 1, -1, -1):
                if nums[j] % nums[i] == 0 and len(dp[i]) + 1 > len(dp[j]):
                    dp[j] = dp[i] + [nums[j]]

        return max(dp, key=len)
```

#### [LintCode 92 · Backpack](https://www.lintcode.com/problem/92/) Medium

0-1 Knapsack: An item is either put in or left outside.

```python
class Solution:
    def back_pack(self, backpack_size: int, weights) -> int:
        # dp[i][j] represents the max weight, which is less than or equal to j (a possible backpack size), of the first i items
        count = len(weights)
        dp = [[0] * (backpack_size + 1) for _ in range(count + 1)]
        # if backpack is strong enough to hold the number i item

        for i in range(1, count + 1):
            for j in range(0, backpack_size + 1):
                # if backpack is large enough to hold the i (index is i - 1) item
                if j >= weights[i - 1]:
                    # either not putting in the i itme in the same backpack: dp[i - 1][j]
                    # or putting in the i item by reserving the space for the i item and putting in the i item
                    dp[i][j] = max(dp[i - 1][j], dp[i -1][j - weights[i - 1]] + weights[i - 1])
                    continue

                # the i item is too heavy to fit into the backpack
                dp[i][j] = dp[i - 1][j]

        return dp[count][backpack_size]
```

#### [LintCode 125 · Backpack II](https://www.lintcode.com/problem/125/) Medium

0-1 Knapsack with values: Now every item has a different value.

The solution is almost identical to that to LintCode 92. Just modify one line by adding up values instead of weights

```python
class Solution:
    def back_pack_i_i(self, backpack_size: int, weights: List[int], values: List[int]) -> int:
        # dp[i][j] represents the max value of the some of the first i items put in a j-size bag
        count = len(weights)
        dp = [[0] * (backpack_size + 1) for _ in range(count + 1)]

        for i in range(1, count + 1):
            for j in range(0, backpack_size + 1):
                if j >= weights[i - 1]:
                    # this line is the only difference
                    # adding up values instead of weights
                    dp[i][j] = max(dp[i - 1][j], dp[i -1][j - weights[i - 1]] + values[i - 1])
                    continue

                dp[i][j] = dp[i - 1][j]

        return dp[count][backpack_size]
```

#### [LintCode 440 · Backpack III](https://www.lintcode.com/problem/440/) Medium

multiple Knapsack: to get the max value of items that can be put in a backpack. Items can be put in multiple times.

$ dp[i][j] = max(dp[i - 1][j], dp[i -1]j - repeat _ weights[i - 1]] + repeat _ values[i - 1])$ $0 <= repeat <= j /A[i - 1]$

<img src="https://s2.loli.net/2022/03/07/8BKZI15bkQOWUlj.png" alt="image-20220304171040811.png" style="zoom:50%;" />

```python
class Solution:
    def back_pack_i_i_i(self, weights, values, backpack_size: int) -> int:
        # dp[i][j] represents the max value of the some of the first i items put in a j-size bag
        count = len(weights)
        dp = [[0] * (backpack_size + 1) for _ in range(count + 1)]

        for i in range(1, count + 1):
            for j in range(0, backpack_size + 1):
                if j >= weights[i - 1]:
                    # 1. not putting in the i-th item
                    # 2. putting in the i-th item and still try to reserve the space for another i-th item
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - weights[i - 1]] + values[i - 1])
                    continue

                dp[i][j] = dp[i - 1][j]
        return dp[count][backpack_size]
```

#### [LintCode 562 · Backpack IV](https://www.lintcode.com/problem/562/) Medium

Complete Knapsack: number of ways to put in several items to reach a certain sum

```
[2,3,6,7]
7
dp:
               0  1  2  3  4  5  6  7
first 0 item  [1, 0, 0, 0, 0, 0, 0, 0],
first 1 items [1, 0, 1, 0, 1, 0, 1, 0],
first 2 items [1, 0, 1, 1, 1, 1, 2, 1],
first 3 items [1, 0, 1, 1, 1, 1, 3, 1],
first 4 items [1, 0, 1, 1, 1, 1, 3, 2]
```

```python
class Solution:
    def backPackIV(self, weights, size):
        count = len(weights)
        # dp[i][j] represents the number of ways to reach sum j choosing from the first i-th items
        dp = [[0] * (size + 1) for _ in range(count + 1)]
        # there is one way to get sum of 0
        dp[0][0] = 1

        for i in range(1, count + 1):
            dp[i][0] = 1
            for j in range(1, size + 1):
                # # if the i-th item can be put in
                if weights[i - 1] <= j:
                    # either without the i-th item pr with
                    dp[i][j] = dp[i - 1][j] + dp[i][j - weights[i - 1]]
                    continue

                dp[i][j] = dp[i - 1][j]

        return dp[count][size]
```

#### [LintCode 563 · Backpack V](https://www.lintcode.com/problem/563/) Medium

#### [322. Coin Change](https://leetcode.com/problems/coin-change/) Medium

min value

```java
class Solution {
    public int coinChange(int[] coins, int amount) {
        // dp[i] the minimum number of coins to make up amount i
        int[] dp = new int[amount + 1];
        for (int i = 0; i < amount + 1; i++) {
            dp[i] = amount + 1;
        }
        dp[0] = 0;

        for (int i = 1; i < amount + 1; i++) {
            for (int coin: coins) {
                if (coin <= i) {
                    dp[i] = Math.min(dp[i], 1 + dp[i - coin]);
                }
            }
        }

        return dp[amount] == amount + 1 ? -1 : dp[amount];
    }
}
```

#### [518. Coin Change 2](https://leetcode.com/problems/coin-change-2/) Medium

combinations

```java
class Solution {
    public int change(int amount, int[] coins) {
        int[] dp = new int[amount + 1];
        dp[0] = 1;
        for (int coin : coins) {
            for (int i = 1; i < amount + 1; i++) {
                if (coin <= i) dp[i] = dp[i] + dp[i - coin];
            }
        }
        return dp[amount];
    }
}
```

#### [2035. Partition Array Into Two Arrays to Minimize Sum Difference ](https://leetcode.com/problems/partition-array-into-two-arrays-to-minimize-sum-difference/)Hard

[LintCode 476 · Merging Stones by Two](https://www.lintcode.com/problem/476/)
Interval, longer intervals depend on shorter intervals
Greedy algorithm deals with how the first step is carried out
DP deals with the last step.

```python
class Solution:
    """
    @param a: An integer array
    @return: An integer
    """
    def stone_game(self, a: List[int]) -> int:
        n = len(a)

        # nothing to merge
        if n < 2:
            return 0

        range_sum = self.get_range_sum(a)

        dp = [[float('inf')] * n for _ in range(n)]

        for i in range(n):
            dp[i][i] = 0

        # length from 2 to n
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                for mid in range(i, j):
                    dp[i][j] = min(dp[i][j], dp[i][mid] + dp[mid + 1][j] + range_sum[i][j])
        return dp[0][n - 1]

    def get_range_sum(self, a):
        n = len(a)

        # also uses dp to create the range sum
        range_sum = [[0] * n for _ in range(n)]

        # initialization
        for i in range(n):
            range_sum[i][i] = a[i]

        for i in range(n):
            for j in range(i + 1, n):
                range_sum[i][j] = range_sum[i][j - 1] + a[j]

        return range_sum
```

#### [312. Burst Balloons](https://leetcode.com/problems/burst-balloons/) Hard

Subarray related, Interval, DP
last to burst
L ... k ... R
k: last baloon to be burst
L: L is on B's left and to be burst. All ballons between L and B are burst
R: R is on B's right and to be burst. All baloons between B and R are burst
dp[i][j] means the maximum point when bursting all baloons between nums[i] and nums[j], both end points not included (nums[i] and nums[j]). To make this work, two points are inserted at the start and the end.
i |start .... baloon last to be burst .... end| k

dp[i][j] = max(dp[i][k] + dp[k][j] + nums[i] _ nums[k] _ nums[j]) (i < k < j)

```python
class Solution:
    def maxCoins(self, nums: List[int]) -> int:
        # insert virtual points on both ends
        # tuple unpacking
        nums = [1, *nums, 1]
        n = len(nums)

        dp = [[0] * n for _ in range(n)]

        # dp[0][2] means burst the middle balloon nums[1]
        # less than 3 balloons makes no sense because end points are not included
        for length in range(3, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                for k in range(i + 1, j):
                    dp[i][j] = max(
                        dp[i][j],
                        dp[i][k] + dp[k][j] + nums[i] * nums[k] * nums[j]
                    )

        return dp[0][n - 1]
```

```java
class Solution {
    public int maxCoins(int[] nums) {
        if (nums == null) {
            return 0;
        }

        // initialization
        // insert two virtual points on both ends
        int n = nums.length + 2;
        int[] extended = new int[n];
        // push original array one position right
        for (int i = 0; i < nums.length; i ++) {
            extended[i + 1] = nums[i];
        }
        // assign 1 to virtual points
        extended[0] = extended[n - 1] = 1;

        int[][] dp = new int[n][n];
        for (int length = 3; length <= n; length ++) {
            for (int i = 0; i < n - length + 1; i ++) {
                int j = i + length - 1;
                for (int k = i + 1; k < j; k++) {
                    dp[i][j] = Math.max(
                        dp[i][j],
                        dp[i][k] + dp[k][j] + extended[i] * extended[k] * extended[j]
                    );
                }
            }
        }

        return dp[0][n - 1];

    }
}
```

#### [5. Longest Palindromic Substring](https://leetcode-cn.com/problems/longest-palindromic-substring/) Medium

Interval, DP

- bigger intervals depend on smaller intervals

#### [44. Wildcard Matching](https://leetcode.com/problems/wildcard-matching/) Hard

prefix, matching, DP
dp[i][j] : a boolean value that shows if the first i characters of the source string matches the first j characters of the pattern
last step: how the last character is matched?
state
pattern[j - 1] == '_' // if the j-th char of the pattern is _
dp[i][j] = dp[i][j - 1] // \* matches no characters. the first i chars of the source already matches the first j - 1 chars of the pattern

    or

    dp[i][j] = dp[i - 1][j] // * matches one or more characters
        dp[i][j] = dp[i][j - 1] or | dp[i - 1][j - 1] or dp[i - 2][j - 1] or ... or dp[0][j - 1] |
                 = ...          or   dp[i - 1][j]

pattern[j - 1] != '\*'
dp[i][j] = dp[i - 1][j - 1] and source[i - 1] matches pattern[j - 1]

```python
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        if s is None or p is None:
            return False


        source_length = len(s)
        pattern_length = len(p)

        dp = [[False] * (pattern_length + 1) for _ in range(source_length + 1)]

        # initialization
        dp[0][0] = True
        # the first 0 chars matches the pattern only when the first j chars are *, which can match zero chars
        for j in range(1, pattern_length + 1):
            dp[0][j] = dp[0][j - 1] and p[j - 1] == '*'

        for i in range(1, source_length + 1):
            for j in range(1, pattern_length + 1):
                if p[j - 1] == '*':
                    dp[i][j] = dp[i - 1][j] or dp[i][j - 1]
                else:
                    dp[i][j] = dp[i - 1][j - 1] and (
                        s[i - 1] == p[j - 1] or p[j - 1] == '?'
                    )

        return dp[source_length][pattern_length]
```

#### [1143. Longest Common Subsequence](https://leetcode.com/problems/longest-common-subsequence/) Medium

matching, DP
dp[i][j]: LCS of the first i chars of the first string and the first j chars of the second string

state transition
s1[i - 1] != s2[j - 1]
dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]) // s1[i - 1] and s2[j - 1] are not in LCS
s1[i - 1] == s2[j - 1]
// although the chars might be same, one of them could be useless
// or both could be useful
// a b c  
 // b c c _ useless
// a b c _ useful
// b a c
dp[i][j] = max(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1] + 1)

inialization
dp[i][0] = dp[0][j] = 0

```python
class Solution:
    def longestCommonSubsequence(self, A: str, B: str) -> int:
        if not A or not B:
            return 0

        a = len(A)
        b = len(B)
        dp = [[0] * (b + 1) for i in range(a + 1)]

        for i in range(1, a + 1):
            for j in range(1, b + 1):
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
                if A[i - 1] == B[j - 1]:
                    dp[i][j] = max(dp[i][j], dp[i - 1][j - 1] + 1)

        return dp[a][b]
```

#### [72. Edit Distance](https://leetcode.com/problems/edit-distance/) Hard

matching, DP
dp[i][j]: the minimum steps it will take to change the first i chars of the first string to the first j chars of the second string

state transition
s1[i - 1] == s2[j - 1]
dp[i][j] = min(
// add s2[j - 1] to the first string to match
dp[i][j - 1] + 1,
// remove the last char of the first string to match
dp[i - 1][j] + 1,
// simply match the two chars, no substitution
dp[i - 1][j - 1]
)
s1[i - 1] != s2[j - 1]
dp[i]j = min(
dp[i][j - 1] + 1,
dp[i - 1][j] + 1,
// substitution, change the s1[i - 1] to s2[j - 1]
dp[i - 1][j - 1] + 1
)

initialization
dp[i][0] = i // remove i chars to be empty
dp[0][j] = j // add j chars

```python
class Solution:
    def minDistance(self, A: str, B: str) -> int:
        if not A:
            return len(B)

        if not B:
            return len(A)

        a = len(A)
        b = len(B)
        dp= [[0] * (b + 1) for _ in range(a + 1)]

        # initialization
        for i in range(a + 1):
            dp[i][0] = i
        for j in range(b + 1):
            dp[0][j] = j

        for i in range(1, a + 1):
            for j in range(1, b + 1):
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + 1
                if A[i - 1] == B[j - 1]:
                    dp[i][j] = min(dp[i][j], dp[i - 1][j - 1])
                else:
                    dp[i][j] = min(dp[i][j], dp[i - 1][j - 1] + 1)

        return dp[a][b]
```

#### [10. Regular Expression Matching](https://leetcode.com/problems/regular-expression-matching/) Hard

dp[i][j]: a boolean value that indicates whether the first i chars of the source string can be matched with the first j chars of the pattern string

state transition
p[j - 1] == '_'
dp[i][j] = dp[i][j - 1] or dp[i - 1][j]
p[j - 1] != '_'
dp[i][j] = dp[i - 1][j - 1] and s[i - 1] == p[j - 1]

```python
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        if not s or not p:
            return False

        a = len(s)
        b = len(p)

        dp = [[False] * (b + 1) for _ in range(a + 1)]

        # initialization
        dp[0][0] = True
        # remember constraint 4: there will be a previous valid character to match for each every '*'
        # whenever dealing with '*', its previous character should always be considered
        # so for a non-empty pattern to match an empty string, it mush be like 'a*b*'
        for j in range(2, b + 1):
            dp[0][j] = dp[0][j - 2] and p[j - 1] == '*'

        for i in range(1, a + 1):
            print("i is ", i, "s[i - 1] = ", s[i - 1])
            source_char = s[i - 1]
            for j in range(1, b + 1):
                pattern_char = p[j - 1]
                previous_pattern_char = p[j - 2]
                if pattern_char == '*' and j >= 2:
                    # match none, one or more
                    dp[i][j] = dp[i][j - 2] or (
                        # this is a requisite if one or more chars need to be matched
                        (previous_pattern_char == source_char or previous_pattern_char == '.')
                    and
                        dp[i - 1][j]
                    )
                else:
                    dp[i][j] = dp[i - 1][j - 1] and (source_char == pattern_char or pattern_char == '.')
        print(dp)
        return dp[a][b]
```

#### [91. Decode Ways](https://leetcode.com/problems/decode-ways/) Medium

Division possibilities, DP
final division: ways to cut at the end _ 1 or 0 (e.g. 0 is not valid) + ways to cut at the second to the last _ 1 or 0 (e.g. 01 is not valid)

```python
class Solution:
    def numDecodings(self, s: str) -> int:
        n = len(s)
        # rolling array to save space
        dp = [0, 0, 0]

        dp[0] = 1
        dp[1] = self.decoded(s[0])

        # starting from 1 will result in invalid index when evaluatng i - 2
        for i in range(2, n + 1):
                        # cut before the last char
            dp[i % 3] = dp[(i - 1) % 3] * self.decoded(s[i - 1 : i]) +\
                        # cut before the second to the last
                        dp[(i - 2) % 3] * self.decoded(s[i - 2 : i])

        return dp[n % 3]

    def decoded(self, s):
        num = int(s)
        # 01 cannot be decoded
        if len(s) == 1 and num >=1 and num <= 9:
            return 1
        if len(s) == 2 and num >= 10 and num <= 26:
            return 1
        return 0
```

#### [198. House Robber](https://leetcode.com/problems/house-robber/) Medium

dp, from bottom to top

```java
class Solution {
    public int rob(int[] nums) {
        int n = nums.length;
        // dp[i] represents the maximum gain for the first i houses
        int[] dp = new int[n + 1];

        // initialization
        dp[0] = 0;
        dp[1] = nums[0];
        // state transition
        for (int i = 2; i <= n; i++) {
            /*
            look at the last step, either the last house (nums[i - 1]) is robbed or not, whichever produces the greater gain wins
            i-th house (nums[i - 1]) robbed: the second to the previous house (nums[i - 2]) then has to be skipped, so we have the maximum gain before (dp[i - 2]) plus the i-th house value
            i-th house not robbed: simply look at the maxium gain before it, i.e. dp[i - 1]
            */
            dp[i] = Math.max(dp[i - 1], dp[i - 2] + nums[i - 1]);
        }

        return dp[n];
    }
}
```

recursion and dp, from top to bottom
Recursion solution is easier to understand than an iteration one, though at an efficieny cost. Can be improved by memorization.

```java
class Solution {
    // private int[] memo;
    public int rob(int[] nums) {
        // memo = new int[nums.length];
        // Arrays.fill(memo, -1);

        return dp(nums, 0);
    }

    private int dp(int[] nums, int start) {
        if (start >= nums.length) {
            return 0;
        }

        // if (memo[start] != -1) return memo[start];

        int res = Math.max(
            // skip current house and start from the next house
            dp(nums, start + 1),
            // rob current house and skip next house (start + 1)
            nums[start] + dp(nums, start + 2)
        );

        // memo[start] = res;
        return res;
    }
}
```

#### [213. House Robber II](https://leetcode.com/problems/house-robber-ii/) Medium

Now the first house and the last house are related.

```java
class Solution {
    public int rob(int[] nums) {
        int n = nums.length;
        if (n == 1) return nums[0];
        //
        return Math.max(
            // skip the last house and stops at the second to the last house
            // skip the first house and stops at the last house
            // whichever has the greater gain wins
            robRange(nums, 0, n - 2),
            robRange(nums, 1, n - 1)
        );
    }

    int robRange(int[] nums, int start, int end) {
        int n = nums.length;
        /*
            pre : max gain until previous house
            prePre: max gain until previous of previous house
            prePre pre cur
    prePre  pre    cur    <--
        */

        int pre = 0, prePre = 0, cur = 0;
        for (int i = end; i >= start; i--) {
            // skip current house or rob current hosue
            cur = Math.max(pre, nums[i] + prePre);
            prePre = pre;
            pre = cur;
        }

        return cur;
    }
}
```

#### [121. Best Time to Buy and Sell Stock](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/) Easy

Only one trasaction is allwed.
two pointers
If a price on a certain day is higher than a price on a later day, this day cannot be the day for buying in the stock.

```java
class Solution {
    public int maxProfit(int[] prices) {
        int max = 0;
        int buyDay = 0;
        for (int sellDay = 1; sellDay < prices.length; sellDay++) {
            if (prices[sellDay] > prices[buyDay]) {
                max = Math.max(max, prices[sellDay] - prices[buyDay]);
            } else {
                // sellDay price is already less or euqal to buyDay price
                // then why not buy on the sellDay and sell on another day?
                buyDay = sellDay;
            }
        }

        return max;
    }
}
```

DP

```java
class Solution {
    public int maxProfit(int[] prices) {
        int n = prices.length;
        int[][] dp = new int[n][2];
        for (int i = 0; i < n; i++) {
            if (i == 0) {
                dp[i][0] = 0;
                dp[i][1] = -prices[i];
                continue;
            }

            dp[i][0] = Math.max(dp[i - 1][0], dp[i - 1][1] + prices[i]);
            dp[i][1] = Math.max(dp[i - 1][1], -prices[i]);
        }

        return dp[n - 1][0];
    }
}

```

#### [53. Maximum Subarray](https://leetcode.com/problems/maximum-subarray/) Medium

Kadane’s Algorithm

```java
class Solution {
    public int maxSubArray(int[] nums) {
        int res = nums[0];
        int localMax = nums[0];
        for (int i = 1; i < nums.length; i++) {
            // two possibilities: starting afresh at current element or extending from previous subarray
            localMax = Math.max(nums[i], localMax + nums[i]);
            res = Math.max(localMax, res);
        }

        return res;
    }
}
```

#### [122. Best Time to Buy and Sell Stock II](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/) Medium

Umlimited transactions.
DP

```java
class Solution {
    public int maxProfit(int[] prices) {
        int n = prices.length;
        int[][] dp = new int[n][2];
        for (int i = 0; i < n; i++) {
            if (i == 0) {
                dp[i][0] = 0;
                dp[i][1] = -prices[i];
                continue;
            }

            dp[i][0] = Math.max(dp[i - 1][0], dp[i - 1][1] + prices[i]);
            dp[i][1] = Math.max(dp[i - 1][1], dp[i - 1][0] - prices[i]);
        }

        return dp[n - 1][0];
    }
}
```

#### [309. Best Time to Buy and Sell Stock with Cooldown](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/) Medium

Unlimited transactions with cooldown. (cool down before new buy)

```java
class Solution {
    public int maxProfit(int[] prices) {
        int n = prices.length;
        int[][] dp = new int[n][2];
        for (int i = 0; i < n; i++) {
            if (i == 0) {
                // rest
                dp[i][0] = 0;
                // buy
                dp[i][1] = -prices[i];
                continue;
            }

            if (i == 1) {
                // still rest / buy on day 0 and sell on day 1
                dp[i][0] = Math.max(dp[i - 1][0], dp[i - 1][1] + prices[i]);
                // buy on day 0 and rest today / buy today
                dp[i][1] = Math.max(dp[i - 1][1], dp[i - 1][0] - prices[i]);
                continue;
            }

            dp[i][0] = Math.max(dp[i - 1][0], dp[i - 1][1] + prices[i]);
            dp[i][1] = Math.max(dp[i - 1][1], dp[i - 2][0] - prices[i]);
        }

        return dp[n - 1][0];
    }
}
```

#### [714. Best Time to Buy and Sell Stock with Transaction Fee](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/) Medium

unlimited trasactions with a fee.

```java
class Solution {
    public int maxProfit(int[] prices, int fee) {
        int n = prices.length;
        int[][] dp = new int[n][2];
        for (int i = 0; i < n; i++) {
            if (i == 0) {
                dp[i][0] = 0;
                dp[i][1] = -prices[i] - fee;
                continue;
            }

            dp[i][0] = Math.max(dp[i - 1][0], dp[i - 1][1] + prices[i]);
            dp[i][1] = Math.max(dp[i - 1][1], dp[i - 1][0] - prices[i] - fee);
        }

        return dp[n - 1][0];
    }
}
```

#### [122. Best Time to Buy and Sell Stock III](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iii/) Hard

```java
class Solution {
    public int maxProfit(int[] prices) {
        int n = prices.length;
        int maxK = 2;
        int[][][] dp = new int[n][maxK + 1][2];

        for (int i = 0; i < n; i++) {
            for (int k = maxK; k >= 1; k--) {
                if (i == 0) {
                    dp[i][k][0] = 0;
                    dp[i][k][1] = -prices[i];
                    continue;
                }

                dp[i][k][0] = Math.max(dp[i - 1][k][0], dp[i - 1][k][1] + prices[i]);
                dp[i][k][1] = Math.max(dp[i - 1][k][1], dp[i - 1][k - 1][0] - prices[i]);
            }
        }

        return dp[n - 1][maxK][0];
    }
}
```

#### [188. Best Time to Buy and Sell Stock IV](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iv/) Hard

```java
class Solution {
    public int maxProfit(int k, int[] prices) {
        int n = prices.length;
        int maxK = k;
        int[][][] dp = new int[n][maxK + 1][2];

        for (int i = 0; i < n; i++) {
            for (k = maxK; k >= 1; k--) {
                if (i == 0) {
                    dp[i][k][0] = 0;
                    dp[i][k][1] = -prices[i];
                    continue;
                }

                dp[i][k][0] = Math.max(dp[i - 1][k][0], dp[i - 1][k][1] + prices[i]);
                dp[i][k][1] = Math.max(dp[i - 1][k][1], dp[i - 1][k - 1][0] - prices[i]);
            }
        }

        return dp[n - 1][maxK][0];
    }
}
```

#### [70. Climbing Stairs](https://leetcode.com/problems/climbing-stairs/) Easy

DP

```java
class Solution {
    public int climbStairs(int n) {
        /*
        state:  ways to reach k-th step
        choice: 1 step or 2 steps
        */

        int[] dp = new int[n + 1];
        for (int i = 1; i < n + 1; i++) {
            if (i == 1) {
                dp[0] = 1;
                dp[1] = 1;
                continue;
            }


            dp[i] = dp[i -2] + dp[i - 1];
        }

        return dp[n];
    }
}
```

#### [1706. Where Will the Ball Fall](https://leetcode.com/problems/where-will-the-ball-fall/) Medium

reference: https://leetcode.cn/problems/where-will-the-ball-fall/solution/java-shuang-bai-di-gui-by-ethan-jx-yvx6/
DP

```java
class Solution {
    int row;
    int col;
    public int[] findBall(int[][] grid) {
        row = grid.length;
        col = grid[0].length;
        int[] res = new int[col];
        for (int i = 0; i < col; i++) {
            res[i] = move(grid, 0, i);
        }
        return res;
    }

    int move(int[][] grid, int x, int y) {
        if (x == row) {
            return y;
        }

        // stuck on left
        if (y == 0 && grid[x][y] == -1) {
            return -1;
        }
        // stuck on right
        if (y == col - 1 && grid[x][y] == 1) {
            return -1;
        }

        // stuck in the middle
        if (grid[x][y] == 1 && grid[x][y + 1] == -1) {
            return -1;
        }
        if (grid[x][y] == -1 && grid[x][y - 1] == 1) {
            return -1;
        }

        // move to next row
        return move(grid, x + 1, y + grid[x][y]);
    }
}
```

#### [221. Maximal Square](https://leetcode.com/problems/maximal-square/) Medium

Intuition:
![](https://s2.loli.net/2022/11/10/xd1rnEGeZjfpoY3.png)

dp[i][j] : the longest side length of the rectangle that has matrix[i][j] as its bottom right corner and matrix[0][o] as its left top corner

```java
class Solution {
    public int maximalSquare(char[][] matrix) {
        int maxSideLength = 0;
        int rows = matrix.length;
        int cols = matrix[0].length;
        int[][] dp = new int[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (matrix[i][j] == '1') {
                    if (i == 0 || j == 0) {
                        dp[i][j] = 1;
                    } else {
                        dp[i][j] = getLeast(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]) + 1;
                    }
                    maxSideLength = Math.max(dp[i][j], maxSideLength);
                }
            }
        }

        return maxSideLength * maxSideLength;
    }

    int getLeast(int x, int y, int z) {
        return Math.min(Math.min(x, y), z);
    }
}
```

#### [1277. Count Square Submatrices with All Ones](https://leetcode.com/problems/count-square-submatrices-with-all-ones/) Medium

Similar to 221.
Once the max square edge length is known at every position, the amount of all sub sqaures can be counted by summing up these

```java
class Solution {
    public int countSquares(int[][] matrix) {
        int res = 0;
        int rows = matrix.length;
        int cols = matrix[0].length;
        int[][] dp = new int[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (matrix[i][j] == 1) {
                    if (i == 0 || j == 0) {
                        dp[i][j] = 1;
                    } else {
                        dp[i][j] = getLeast(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]) + 1;

                    }

                    res += dp[i][j];
                }
            }
        }

        return res;
    }

    int getLeast(int x, int y, int z) {
        return Math.min(Math.min(x, y), z);
    }
}
```

#### [1547. Minimum Cost to Cut a Stick](https://leetcode.com/problems/minimum-cost-to-cut-a-stick/) Medium

#### [64. Minimum Path Sum](https://leetcode.com/problems/minimum-path-sum/) Medium

backtrack, will cause TLE

```java
class Solution {
    int m, n;
    int pathSum;
    int[][] moves = {{1 ,0}, {0, 1}};

    private boolean isValid(int[] pos) {
        return pos[0] < m && pos[1] < n;
    }

    public int minPathSum(int[][] grid) {
        m = grid.length;
        n = grid[0].length;
        pathSum = Integer.MAX_VALUE;
        travel(grid, new int[]{0, 0}, 0);
        return pathSum;
    }

    private void travel(int[][] grid, int[] pos, int currentSum) {
        if (!isValid(pos)) {
            return;
        }

        currentSum += grid[pos[0]][pos[1]];

        if (pos[0] == m - 1 && pos[1] == n - 1) {
            if (currentSum < pathSum) {
                pathSum = currentSum;
            }
            return;
        }

        for (int i = 0; i < moves.length; i++) {
            pos[0] += moves[i][0];
            pos[1] += moves[i][1];
            travel(grid, pos, currentSum);
            pos[0] -= moves[i][0];
            pos[1] -= moves[i][1];
        }
    }
}
```

dp with memorization

```java
class Solution {
    int[][] memo;

    public int minPathSum(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;

        memo = new int[m][n];
        for (int[] row: memo) {
            Arrays.fill(row, -1);
        }

        return dp(grid, m - 1, n - 1);
    }

    private int dp(int[][] grid, int i, int j) {
        if (i == 0 && j == 0) {
            return grid[0][0];
        }

        if (i < 0 || j < 0) {
            return Integer.MAX_VALUE;
        }

        if (memo[i][j] != -1) {
            return memo[i][j];
        }

        memo[i][j] = Math.min(
            dp(grid, i - 1, j),
            dp(grid, i, j - 1)
        ) + grid[i][j];

        return memo[i][j];
    }
}
```

#### [Kadane's Algorithm](https://www.geeksforgeeks.org/problems/kadanes-algorithm-1587115620/1?page=1&sprint=a663236c31453b969852f9ea22507634&sprint=a663236c31453b969852f9ea22507634&sortBy=submissions)

```java
class Solution{
    // arr: input array
    // n: size of array
    //Function to find the sum of contiguous subarray with maximum sum.
    long maxSubarraySum(int arr[], int n){
        long local_max = arr[0], global_max = arr[0];
        for (int i = 1; i < n; i++) {
            local_max = Math.max(arr[i], local_max + arr[i]);
            if (local_max > global_max) global_max = local_max;
        }

        return global_max;
    }

}
```
