## Array

- two pointers
- sliding window
- binary search
- DFS
#### 26 Remove Duplicates <span style="color:green">Easy</span>

two-pointers: 不要想着从原数组内移除重复的元素，而是从空数组开始，一点点将不重复的元素添加进去。快指针用来遍历原数组，慢指针则负责追踪结果数组的最后一个位置。

```java
class Solution {
    public int removeDuplicates(int[] nums) {
        int slow = 0;
        for(int fast = 1; fast < nums.length; fast++){
            // duplicate elements will be skipped
            // when a new element is found, move it into the result array
            if(nums[fast] != nums[slow]){
                slow++;
                nums[slow] = nums[fast];
            }
        }
        return ++slow;
    }
}
```

#### 293 Move Zeros <span style="color:green">Easy</span>

two-pointers: 快指针找到一个非 0 数就往前放，慢指针标记非零数组的末尾。快指针遍历完成后，则所有非 0 数都已挪到原数组前部，但是原数组后部还保持不变，因此需要再依次更新后部为 0。

```java
class Solution {
    public void moveZeroes(int[] nums) {
        int slow = 0;
        for(int fast = 0; fast < nums.length; fast++){
            if(nums[fast] != 0){
                nums[slow] = nums[fast];
                slow++;
            }
        }
        while(slow < nums.length){
            nums[slow] = 0;
            slow++;
        }

    }
}
```

#### 844 Backspace String Compare <span style="color:green">Easy</span>

two-pointers: 碰到退格键，满指针不再是跳过，而是要向前退一格。因此涉及到减法，要额外注意确保慢指针不要小于 0。

```java
class Solution {
    public boolean backspaceCompare(String s, String t) {
        char backspace = '#';
        return getRealString(s, backspace).equals(getRealString(t, backspace));
    }

    public String getRealString(String s, char backspace){
        char[] chars = s.toCharArray();
        int slow = 0;
        for(int fast = 0; fast < chars.length; fast++){
            if(chars[fast] != backspace){
                chars[slow] = chars[fast];
                slow++;
            } else {
                // be careful when doing deduction
                if(slow > 0)
                    slow--;
            }
        }
        return new String(chars).substring(0, slow);
    }
}
```

#### [944. Delete Columns to Make Sorted ](https://leetcode-cn.com/problems/delete-columns-to-make-sorted/) <span style="color:green">Easy</span>
two-pointers
```java
class Solution {
    public int minDeletionSize(String[] A) {
        int result = 0;
        int stringLen = A[0].length();
        int arrayLen = A.length;
        for (int stringPointer = 0; stringPointer < stringLen; ++stringPointer)
            for (int arrayPointer = 0; arrayPointer < A.length - 1; ++arrayPointer)
                if (A[arrayPointer].charAt(stringPointer) > A[arrayPointer + 1].charAt(stringPointer)) {
                    result++;
                    break;
                }

        return result;
    }
}
```

#### [977 Squares of a Sorted Array](https://leetcode.com/problems/squares-of-a-sorted-array/) <span style="color:green">Easy</span>

two-pointers: 谈到到平方，就会涉及负数，就要想到比较绝对值。从“原数列是有序数列”这点出发，又可知平方后最小的数无法判断，但是最大值一定在两端之中。除此之外，不能再有其他假设。我第一次的错误在于假设每次插入都是成对进行，即最右侧是最大，最左侧一定是次大，这种假设让情况变得复杂，也不符合实际，因为我们在任意时刻能确定的只有最大值。我们是一个一个往结果数组插入，因此该去控制循环的是指向结果数组元素的指针，左右指针是指向原数组的两端，因此我们要从两者中总是取较大值。

```java
class Solution {
    public int[] sortedSquares(int[] nums) {
        int[] result = new int[nums.length];
        // left points to the left end of the original array
        int left = 0;
        // right points to the right end of the original array
        int right = nums.length - 1;
        // cursor points to the right end of the result array
        // since we are putting elements here one by one starting from the largest one
        for(int cursor = result.length - 1; cursor >= 0; cursor--){
            if(Math.abs(nums[left]) > Math.abs(nums[right])){
                result[cursor] = nums[left] * nums[left];
                // current leftmost element is inserted, go to the next one
                left++;
            } else {
                result[cursor] = nums[right] * nums[right];
                // current rightmost element is inserted, go to the next one
                right--;
            }
        }
        return result;
    }
}
```
#### [1513. Number of Substrings With Only 1s](https://leetcode.com/problems/number-of-substrings-with-only-1s/) Medium
two-pointers, same direction
1. get answer incrementally
```python
# this solution will cause TLE
class Solution:
    def numSub(self, s: str) -> int:
        n = len(s)
        left = 0
        result = 0

        for left in range(n):
            # "1" instead of 1
            if s[left] != '1':
                continue
            right = left + 1
            while right < n and s[right] == '1':
                right += 1
            """
              l l   l l l
            0 1 1 0 1 1 1
                  r       r
              2 1   3 2 1
            """
            result += right - left

        return result
```
2. get substring lengths of 1s and then calculate the combinations of every substring and add them up
```python
class Solution:
    def numSub(self, s: str) -> int:
        n = len(s)
        left = 0
        # store lenghts of every substring of 1s
        subs = []

        while left < n:
            if s[left] == "0":
                # remember to move left pointer
                # a simple continue will cause infinite loop here
                left += 1
                continue

            # the start of a substring is detected
            right = left + 1
            # find the right boundary of current substring
            while right < n and s[right] == "1":
                right += 1

            subs.append(right - left)
            left = right

        result = 0
        for s in subs:
            result += s * (s + 1) / 2 % 1000000007

        return int(result)
```
#### [424. Longest Repeating Character Replacement](https://leetcode.com/problems/longest-repeating-character-replacement/) Medium
two pointers
longest repeating -> least replacements  -> replace infrequent letters with the most frequent letter
```python
class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        def convert(char):
            return ord(char) - ord('A')

        # replacing a character with most frequent characters leads to least replacements
        max_frequency = 0
        # note down frequencies
        freqs = [0 for _ in range(26)]
        n = len(s)
        right = 0
        result = 0

        # move left pointer
        for left in range(n):
            # least replacements <= k
            while right < n and right - left - max_frequency <= k:
                freqs[convert(s[right])] += 1
                max_frequency = max(max_frequency, freqs[convert(s[right])])

                # right pointer is moved after the update
                # [left, right)
                # move right pointer and never go back
                right += 1

            # requires more than k replacements
            if right - left - max_frequency > k:
                # sub from the previous position
                # [left, right - 1)
                result = max(result, right - left - 1)
            else:
                # requires k replacements
                # [left, right)
                result = max(result, right - left)

            # update because left pointer is moved
            freqs[convert(s[left])] -= 1
            max_frequency = max(freqs)

        return result
```

### Sliding window

#### [209 Minimum Size Subarray Sum](https://leetcode.com/problems/minimum-size-subarray-sum/) <span style="color:orange">Medium</span>

brute force

```java
class Solution {
    public int minSubArrayLen(int target, int[] nums) {
        // make it an impossible value
        int result = -1;
        int sum = 0;
        int subLength = 0;
        for(int start = 0; start < nums.length; start++){
            sum = 0;
            for(int end = start; end < nums.length; end++){
                sum += nums[end];
                if(sum >= target){
                    subLength = end - start + 1;
                    // get a real result for the first time
                    if(result == -1)
                        result = subLength;
                    else
                        result = result < subLength ? result : subLength;
                }
            }
        }
        return result == -1 ? 0 : result;
    }
}
```

sliding window

上述暴力解法无非循环嵌套，不断确定 start 和 end 的位置，再不断对区间内的值累加求和。注意外层循环中，每一次都会重置 sum，这也就带来了重复运算。而滑动窗口则可以利用之前的结果。

```java
class Solution {
    public int minSubArrayLen(int target, int[] nums) {
        int start = 0;
        // make it a value that is greater than any possibilities, oblivating the need to use Integer.MAX_VALUE (which is ugly)
        int result = nums.length + 1;
        for(int end = 0; end < nums.length; end++){
            // think reversely: deduct from target, instead of calculating a sum
            target -= nums[end];
            while(target <= 0){
                int candidate = end - start + 1;
                // make sure result is the smallest
                result = result < candidate ? result : candidate;
                // exclude start from the result and then start from the next position
                target += nums[start];
                // move forward the starting point
                start++;
            }
        }
        return result % (nums.length + 1);
    }
}
```

```python
class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        left = right = sum = 0
        # set result to the maximum value plus one
        result = len(nums) + 1
        while(right < len(nums)):
            # enlarge the window to the right
            sum += nums[right]
            # now the subarray between left and right has a sum that meets the condition
            while(sum >= target):
                # compare with the old result and get a new result
                result = min(result, right - left + 1)
                # shrink the window from the left
                sum -= nums[left]
                # increment the left pointer so that the iteration continues
                left += 1
            # drive the outer iteration
            right += 1

        return result if result != len(nums) + 1 else 0
```

#### [904 Fruits into Baskets](https://leetcode.com/problems/fruit-into-baskets/) <span style="color:orange">Medium</span>

自己想出来的循环法，空间占用小，但耗时较多。start 用来标记水果 A 的起点，middle 用来标记水果 B 的起点，end 用来标记挑选停止的位置。

```java
class Solution {
    public int totalFruit(int[] fruits) {
        int middle = 0;
        int end = 0;
        int result = 0;
        int candidate = 0;
        for(int start = 0; start < fruits.length; start++){
            // stops when fruit B appears
            while(middle < fruits.length && fruits[middle] == fruits[start]){
                middle++;
            }
            // initializes end
            end = middle;
            // stops when curent fruit is neither fruit A nor fruit B
            while(end < fruits.length && (fruits[end] == fruits[middle] || fruits[end] == fruits[start])){
                end++;
            }

            candidate = end - start;
            result = result > candidate ? result : candidate;
            // jumps to the middle since the start-to-middle part is already visisted
            // in the next iteration, start will be incremented so 1 is deducted in advance
            start = middle - 1;
        }
        return result;
    }
}
```

sliding window: 用键值对的形式来表示某一时刻篮子中各类果子的数量，并保证任一时刻篮子中最多都只有三种水果，并且只要有了第三种水果，就要一个一个往外扔了。

```java
class Solution {
    public int totalFruit(int[] fruits) {
        // two baskets holding a type each can be seen as one basket holding two types
        // fruit -> count of this kind of fruit
        Map<Integer, Integer> basket = new HashMap<Integer, Integer>();
        int slow = 0;
        int fast;
        for(fast = 0; fast < fruits.length; fast++){
            // put current fruit in the basket (from left to right)
            basket.put(fruits[fast], basket.getOrDefault(fruits[fast], 0) + 1);
            // more than 2 types of fruits are detected
        	if(basket.size() > 2){
            	// throw away one of the type you put earlier
            	basket.put(fruits[slow], basket.get(fruits[slow]) - 1);
                // try to clear the basket of the first kind of fruits if the kind of fruits you put earilier is all gone
                basket.remove(fruits[slow], 0);
                slow++;
        	}
        }
        return fast - slow;
    }
}
```

#### 76 Minimum Window Substring <span style="color:red">Hard</span>

sliding window
第一次用自己的思路写，只能通过一半的测试用例，错误在于我没有考虑到 t 中可以有重复字符，错将 t 中字符与该字符在 s 中出现的位置当成了 HashMap 的 key 与 value，应该记录出现的次数。在看了力扣的视频题解后，才对滑动窗口有了更深的认识，并改写了自己的版本。

```java
class Solution {
    public String minWindow(String s, String t) {
        if(s == null || s == "" || t == null || t == "" || t.length() > s.length()) return "";
        // character frequency in sliding window. This will be updated in the process as expected characters are found.
		// ASCII table has 128 characters so the array is initialized to 128-element long.
        int[] windowFreq = new int[128];
		// character frequency in target string. This will stay the same after initialization.
        int[] targetFreq = new int[128];

		// iterate target string to initialize targetFreq
        for(int i = 0; i < t.length(); i++){
            targetFreq[t.charAt(i)]++;
        }
        // attach s length and t length to variables for re-use
        int sLen = s.length();
        int tLen = t.length();

        int left = 0;
        char leftChar;
        int right;
        char rightChar;
        int begin = 0;
		// minimum length of the string in the window that contains all the characters needed
        int minLen = sLen + 1;
		// this variable is used to count how many expected characters are covered in the window
        int covered = 0;
		// move the right end of the window first
        for(right = 0; right < sLen; right++){
            rightChar = s.charAt(right);
			// current character at right pointer is included in t
            if(targetFreq[rightChar] > 0){
				// covered is incremented only when the desired amount of the expected character has not been reached
				// it helps to decide when to shrink the window
				// for example, for s = "aac" and t = "ac", without this if-statement, covered would be equal to 2 when right is equal to 1
				// continue on and the window would be shrinked but "aa" is not a valid window, which means window-shrinking happens too early
                if(windowFreq[rightChar] < targetFreq[rightChar]){
                    covered++;
                }
				// expected characters in the window can be more than enough so the frequency of the current character is incremented anyway
                windowFreq[rightChar]++;
            }

			// when enough or more expected characters are found in the window
            while(covered == tLen){
                if(right - left < minLen){
                    minLen = right - left;
                    begin = left;
                }

                leftChar = s.charAt(left);
				// current character at left pointer is included in t
                if(targetFreq[leftChar] > 0){
					// since left will always be decremented in the while loop, if current character is just what we need, moving on means we have one less character covered in the window. So wee need to decrement covered.
                    if(windowFreq[leftChar] == targetFreq[leftChar]){
                        covered--;
                    }
					// window will be shrinked anyway so the current character in the window is lost
                    windowFreq[leftChar]--;
                }
				// now shrink the window
                left++;
            }

        }

        if(minLen == sLen + 1){
            return "";
        }
        // note that right pointer stops just at the right end of the window, which means we have to increment when we calculate the expected length using end points
        return s.substring(begin, begin + minLen + 1);

    }
}
```

#### [567. Permutation in String](https://leetcode-cn.com/problems/permutation-in-string/) <span style="color:orange">Medium</span>
fixed-size sliding window / two pointers
When it comes to permutaion, only chars' count matters and how they are arranged doesn't matter. Also, a sub-string is required, thus comes the sliding window.

```python
class Solution:
    def checkInclusion(self, pattern: str, text: str) -> bool:
        left = 0
        right = len(pattern) - 1
        pattern_counter = collections.Counter(pattern)
        # window size is pattern length minus 1 now
        window_counter = collections.Counter(text[0: right])
        while right < len(text):
            # for every iteration, the char on the right end is first added to the window
            window_counter[text[right]] += 1
            if pattern_counter == window_counter:
                return True
            window_counter[text[left]] -= 1
            # if count is zero, remove it from the counter
            if window_counter[text[left]] == 0:
                del window_counter[text[left]]
            left += 1
            right += 1

        return False
```

#### 1248 Count Number of Nice Subarrays <span style="color:orange">Medium</span>

自己想的解法，自以为完备，真正执行时，先会遇到语法错误。语法问题解决后，再去执行，常常是连第一个测试用例也无法通过。再以第一个测试用例，调整思路，使其通过后，再执行多个测试用例，发现又有错误。多半是自己只盯着特殊的测试用例，问题没理解充分，忽略了限制条件。好不容易让多个测试用例也通过了，真正提交时又会遇到要么是一半的测试用例没通过，要么是超时了，瓶颈在哪里自己也搞不清楚。这道题还是看了讨论中的题解才依样画葫芦给做出来的。

```java
class Solution {
    public int numberOfSubarrays(int[] nums, int k) {
		int len = nums.length;
		int left = 0;
		int right;
		// track how many odd numbers are found in the window
		int windowOdds = 0;
		int result = 0;
		// track how many subarrays are found in the window
		int tempResult = 0;
		for(right = 0; right < len; right++){
			if(nums[right] % 2 == 1){
				windowOdds++;
				if(windowOdds == k) tempResult = 0;
				while(windowOdds == k){
					// as long as the window remains
					tempResult++;

					// left stops at an odd number, which means we'll lose an odd number after moving left one step further, so the window should shrink
					if(nums[left] % 2 == 1){
						windowOdds--;
					}
					left++;
				}
				// the window no longer remains, so we add the temp result from this round to the final result
				result += tempResult;
			} else {
				// current character is even after moving forward, which means it does not affect the result from last round, so we simply reuse it and add it to the result
				// e.g. for nums = [2 2 1 2 1 2] and k = 2, when right pointer stops at the latter 1, there are 3 sub arrays for now. Then we move the right pointer forward and it stops at the last 2, since it is not odd, we have another 3 sub arrays, every one of which is simply the sub array from last round with the even number appended. (22121 -> 221212, 2121->21212, 121->1212)
				result += tempResult;
			}
		}
		return result;
    }
}
```

#### [930 Binary Subarrays With Sum](https://leetcode-cn.com/problems/binary-subarrays-with-sum/) <span style="color:orange">Medium</span>

So we are trying to count continuous sub-arrays where the number of ones within is equal to a desired goal. So only ones count in this case. Consider the array [0,0,1,0,1,0,0,0] and the goal is 2 and [1,0,1] in the middle is the base case from which we can go left or right. Since two zeros are on the left and three on the right, we have 2 x 3 possibilities. Also, don't forget [1,0,1] in the middle, therefore we have (2+1) x (3+1) possibilities in total. Starting from this example, it's safe to say that the things we care about here are the number of ones and zeros in between. But we have to be careful when the goal is 0 because in this cases we can not get the result simply by summing the number of the zeros in between.

```python
class Solution:
    def numSubarraysWithSum(self, nums: List[int], goal: int) -> int:
        result = 0
        # count zeros between ones in each iteration
        count_zeros = 0
        # use an array to save the number of zeros between ones
        # example: [0,0,1,0,1,0,0,0] zeros_between_ones: [3, 2, 4]
        zeros_between_ones = []

        for num in nums:
            if(num == 0):
                count_zeros += 1
            else:
                zeros_between_ones.append(count_zeros + 1)
                count_zeros = 0
        zeros_between_ones.append(count_zeros + 1)

        # when goal is 0, count the possibilities of zero combinations, ones excluded
        # so only adding is involved here and no multiplication
        # example: [0,0,0,1,0,0] zeros_between_ones: [4, 3]
        if(goal == 0):
            for count in zeros_between_ones:
                for i in range(count):
                    result += i
            return result

        # when goal is not zero, multiplication is needed
        for i in range(len(zeros_between_ones)):
            # ones are exhausted
            if(i + goal == len(zeros_between_ones)):
                break
            result += zeros_between_ones[i] * zeros_between_ones[i + goal]
        return result
```

前缀和加哈希表的解法，我始终无法理解，看了许多题解，还是不懂为什么要sum-goal，统计出来的map又有怎么样的实际意义。不想再在这道题上花时间了，挫败感太强了，之后再碰到前缀和的题目然后再看吧。

#### [1. Two Sum](https://leetcode-cn.com/problems/two-sum/) <span style="color:green">Easy</span>

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        num_index = {}
        for index, num in enumerate(nums):
            if target - num in num_index:
                return [num_index[target - num], i]
            num_index[nums[index]] = index
        return []
```

#### [992. Subarrays with K Different Integers](https://leetcode-cn.com/problems/subarrays-with-k-different-integers/) <span style="color:red">Hard</span>

**恰好**由k个不同整数组成的子序列个数 = **最多**由k个不同整数组成的子序列个数 - **最多**由k - 1个不同整数组成的子序列个数

```python
class Solution:
    def subarraysWithKDistinct(self, nums: List[int], k: int) -> int:
        return self.distinctAtMost(nums, k) - self.distinctAtMost(nums, k - 1)
    def distinctAtMost(self, nums, k):
        length = len(nums)
        # to track how many different integers are in the window
        distinct = 0
        # to count frequency of every different integer in the window
        counter = collections.Counter()
        # let and right pointer
        left, right = 0, 0
        result = 0
        while right < length:
            # right points to a number that has not been seen in the window
            if counter[nums[right]] == 0:
                distinct += 1
            # right moves and
            counter[nums[right]] += 1
            while distinct > k:
                counter[nums[left]] -= 1
                # left points to a number that disappears from the window now
                if counter[nums[left]] == 0:
                    distinct -= 1;
                # left moves forwar
                left += 1
            result += right - left + 1

            # right moves forward
            right += 1
        return result
```

#### [862. Shortest Subarray with Sum at Least K](https://leetcode-cn.com/problems/shortest-subarray-with-sum-at-least-k/) <span style="color:red">Hard</span>

求连续子数组之和至少为k的最短长度，既然有和，就可以想到使用前缀和数组，问题可转化为连续子数组之和为prefix_sums[right] - prefix_sums[left] >= K时，right - left的最小值。如果以暴力求解，可以想到每一个点都要当作right，也要被当作left，即内外双层循环。如此方法，自然费时。有没有办法可以减少循环的次数呢？此问题的难点在于数组中可以有负数，前缀和数组自然就不一定是单调增加。从这一点出发，前缀和数组中就可能出现这样的情况：有i < j使得prefix_sums[i] >= prefix_sums[j]，也就是说i到j这一段的和小于0，往后再延长，长度变长自不必说，结果反而比起不带i到j这一段变小了。那么，i作为left自然不可能是最后答案。而假如我们已经找到了prefix_sums[j] - prefix_sums[i] >= K，j再往后走也没有意义，因为无论之后是正数还是负数，长度都变长了，即便符合条件，也不是我们要找的最后答案。因此，一旦遇到一个符合条件的left（即此刻的i），就意味着可以求出一个可能的结果，不必再考虑该点作为左侧起始点的情况。

```python
class Solution:
    def shortestSubarray(self, nums: List[int], k: int) -> int:
        # prefix sum array
        prefix_sums = [0]
        # assume the result is the largest posssible number plus one
        result = len(nums) + 1;
        #
        left_candidates = collections.deque()
        for index, num in enumerate(nums):
            prefix_sums.append(prefix_sums[index] + num)

        for right in range(len(prefix_sums)):
			# if sum[j] <= sum[i], sum from i to j is negative. i as the starting point can be removed
            while(left_candidates and prefix_sums[right] <= prefix_sums[left_candidates[-1]]):
                left_candidates.pop()
			# if sum[j] - sum[i] >= k, we can compute a result and i as the starting point can be removed
            while(left_candidates and prefix_sums[right] - prefix_sums[left_candidates[0]] >= k):
                result = min(result, right - left_candidates.popleft())
            # every prefix sum is considered as a starting point candidate at first and then we try to shrink the candidates list in the loop to skip unnecessary calculations
            left_candidates.append(right)

        return result if result < len(nums) + 1 else -1
```

#### [54. Spiral Matrix](https://leetcode-cn.com/problems/spiral-matrix/) <span style="color:orange">Medium</span>



#### [59. Spiral Matrix II](https://leetcode-cn.com/problems/spiral-matrix-ii/) <span style="color:orange">Medium</span>

```java
class Solution {
    public int[][] generateMatrix(int n) {
        // left and top border, set to 0
		int left = 0, top = 0;
        // right and bottom border, set to n - 1
		int right = n - 1, bottom = n - 1;
		int[][] matrix = new int[n][n];
		int current = 1;
		int end = n * n;
		while(current <= end){
			// from top left to top right
			for(int i = left; i <= right; i++) matrix[top][i] = current++;
			top++;
			// from right top to right bottom
			for(int i = top; i <= bottom; i++) matrix[i][right] = current++;
			right--;
			// from bottom right to bottom left
			for(int i = right; i >= left; i--) matrix[bottom][i] = current++;
			bottom--;
			// from left bottom to left top
			for(int i = bottom; i >= top; i--) matrix[i][left] = current++;
			left++;
		}
		return matrix;
    }
}
```

#### [912. Sort an Array](https://leetcode-cn.com/problems/sort-an-array/) <span style="color:orange">Medium</span>

自己写出来的，错误真多，有用错方法，忘记类型，忘记返回值，拼写错误。

先整体有序，再局部有序，分而治之。

```java
// quick sort
class Solution {
    public int[] sortArray(int[] nums) {
        if(nums.length <= 1)
            return nums;

		quicksort(nums, 0, nums.length - 1);
        return nums;
    }

    private void quicksort(int[] nums, int start, int end){
        if(start >= end)
            return;

        int left = start, right = end;
        int pivot = nums[(start + end) / 2];
        // make sure that left and right do not stop at the same position to avoid stack over flow error
        while(left <= right){
            // try best to create balanced subarrays
            // nums[left] == pivot will cause inbalance when most elements are the same integer
            while(left <= right && nums[left] < pivot)
                left++;
            while(left <= right && nums[right] > pivot)
                right--;
            if(left <= right){
                int temp = nums[left];
                nums[left] = nums[right];
                nums[right] = temp;
                left++;
                right--;
            }
        }

        quicksort(nums, start, right);
        quicksort(nums, left, end);
    }
}
```

```java
// merge sort
public class Solution {
    public int[] sortArray(int[] nums) {
        int[] sorted = new int[nums.length];
        mergeSort(nums, 0, nums.length - 1, sorted);
        return sorted;
    }

    public void mergeSort(int[] nums, int start, int end, int[] temp) {
        if(start >= end)
            return;

        int mid = (start + end) / 2;

        mergeSort(nums, start, mid, temp);
        mergeSort(nums, mid + 1, end, temp);
        merge(nums, start, mid, end, temp);
    }

    public void merge(int[] nums, int start, int mid, int end, int[] temp) {
        int left = start, right = mid + 1, tempIndex = left;

        while (left <= mid && right <= end) {
        	if(nums[left] < nums[right]) {
                temp[tempIndex++] = nums[left++];
            } else {
                temp[tempIndex++] = nums[right++];
            }
        }

        while (left <= mid)
            temp[tempIndex++] = nums[left++];
        while (right <= end)
        	temp[tempIndex++] = nums[right++];

        for (int i = start; i <= end; i++)
            nums[i] = temp[i];
    }
}
```

#### [88. Merge Sorted Array](https://leetcode-cn.com/problems/merge-sorted-array/) <span style="color:green">Easy</span>

```java
// basically the merging process of the merge sort
class Solution {
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        int[] temp = new int[m];
        for(int i = 0; i < m; i++)
            temp[i] = nums1[i];
        // System.out.println("Temp Array: " + Arrays.toString(temp));

        int left = 0, right = 0, index = 0;
        while(left < m && right < n) {
            if(temp[left] < nums2[right]) {
                nums1[index++] = temp[left++];
            } else {
                nums1[index++] = nums2[right++];
            }
        }

        while(left < m)
            nums1[index++] = temp[left++];
        while(right < n)
            nums1[index++] = nums2[right++];

    }
}
```

```java
// another perspective: moving nums2 into nums1 from t
class Solution {
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        int index = nums1.length;

        while(n > 0) {
            if (m > 0 && nums1[m - 1] > nums2[n - 1]) {
                nums1[--index] = nums1[--m];
            }
            else {
                nums1[--index] = nums2[--n];
            }
        }
    }
}
```
### Binary search

Keywords for binary search:
array, target, sorted, equal or close to target, $O(nlogn)$

#### [34. Find First and Last Position of Element in Sorted Array](https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/) <span style="color:orange">Medium</span>

Binary search: Decrease and Conquer

 ```python
class Solution:
    def searchRange(self, nums, target):
        if not nums:
            return [-1, -1]

        def binary_search_left(nums, target):
            left, right = 0, len(nums) - 1
            while left < right:
                mid = (left + right) // 2
                if nums[mid] < target:
                    left = mid + 1
                else:
                    right = mid
            return left if nums[left] == target else -1

        def binary_search_right(nums, target):
            left, right = 0, len(nums) - 1
            while left < right:
                mid = (left + right) // 2 + 1
                if nums[mid] > target:
                    right = mid - 1
                else:
                    left = mid
            return left if nums[left] == target else -1

        return [binary_search_left(nums, target), binary_search_right(nums, target)]
 ```



#### 702.Search in a Sorted Array of Unknown Size <span style="color:orange">Medium</span>

Binary search with incremental range.

```python
class Solution:
    """
    @param reader: An instance of ArrayReader.
    @param target: An integer
    @return: An integer which is the first index of target.
    """
    def searchBigSortedArray(self, reader, target):
        # write your code here
        range_total = 1
        while reader.get(range_total - 1) < target:
            range_total = range_total * 2

        # binary search template code
        def binary_search(length, reader):
            start, end = 0, length - 1
            while start + 1 < end:
                mid = start + (end- start) // 2
                if reader.get(mid) < target:
                    start = mid
                else:
                    end = mid

            if reader.get(start) == target:
                return start
            if reader.get(end) == target:
                return end
            return -1

        return binary_search(range_total, reader)
```

#### [658. Find K Closest Elements ](https://leetcode-cn.com/problems/find-k-closest-elements/)<span style="color:orange">Medium</span>

Binary search is used to find a middle line instead of an exact target. Two pointers are used to merge from both sides.

```python
class Solution:
    def findClosestElements(self, nums: List[int], k: int, target: int) -> List[int]:
        right = self.findUpperClosest(nums, target)
        left = right - 1

        results = []
        for _ in range(k):
            if self.isLeftCloser(nums, target, left, right):
                results.append(nums[left])
                left -= 1
            else:
                results.append(nums[right])
                right += 1

        return sorted(results)

    # binary search template code
    def findUpperClosest(self, nums, target):
        start, end = 0, len(nums) - 1
        while start + 1< end:
            mid = start + (end - start) // 2
            if nums[mid] >= target:
                end = mid
            else:
                start = mid

        if nums[start] >= target:
            return start
        if nums[end] >= target:
            return end

        return len(nums)

    def isLeftCloser(self, nums, target, left, right):
        if left < 0:
            return False

        if right >= len(nums):
            return True

        return abs(target - nums[left]) <= abs(target - nums[right])
```

#### [852. Peak Index in a Mountain Array](https://leetcode-cn.com/problems/peak-index-in-a-mountain-array/) <span style="color:green">Easy</span>

<img src="https://s2.loli.net/2021/12/30/nJgIpRw2Ld3kN9V.png" alt="image-20211230224558606" style="zoom:33%;" />

XXXYYYY  Binary search is used to find the turning point.

```java
class Solution {
    public int peakIndexInMountainArray(int[] nums) {
        if(nums == null || nums.length == 0)
            return -1;

        int start = 0, end = nums.length - 1;
        while(start + 1 < end){
            int mid = start + (end - start) / 2;
            // decremental after mid, downward slope in the right, cut the right part
            if(nums[mid] > nums[mid + 1]){
                end = mid;
            } else {
                start = mid;
            }
        }

        if(nums[start] > nums[end])
            return start;
        return end;
    }
}
```

#### [153. Find Minimum in Rotated Sorted Array](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array/) <span style="color:orange">Medium</span>

4 5 6 7 0 1 2 3 Incremental, incremental, opposite of mountain sequence.

Binary search to to get the middle line

```python
class Solution:
    def findMin(self, nums: List[int]) -> int:
        if not nums:
            return -1

        start, end = 0, len(nums) - 1
        while start + 1 < end:
            mid = (start + end) // 2
            if nums[mid] > nums[end]:
                # cut left
                start = mid
            else:
                # cut right
                end = mid
        return min(nums[start], nums[end])
```



#### [33. Search in Rotated Sorted Array](https://leetcode-cn.com/problems/search-in-rotated-sorted-array/) <span style="color:orange">Medium</span>

Two binary seraches: the first to get the minium point, the second to find the target in one side of the array

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        if not nums:
            return -1

        lowest = self.get_lowest(nums)
        if target > nums[len(nums) - 1]:
            # search on the left side
            return self.binary_search(nums, 0, lowest, target)
        # search on the right side
        return self.binary_search(nums, lowest, len(nums) - 1, target)


    def get_lowest(self, nums):
        start, end = 0, len(nums) - 1
        while start + 1 < end:
            mid = (start + end) // 2
            if nums[mid] > nums[end]:
                # cut left
                start = mid
            else:
                # cut right
                end = mid
        if nums[start] < nums[end]:
            return start
        return end

    def binary_search(self, nums, start, end, target):
        while start + 1 < end:
            mid = (start + end) // 2
            if nums[mid] > target:
                end = mid
            else:
                start = mid

        if nums[start] == target:
            return start
        if nums[end] == target:
            return end

        return -1
```

One binary search

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        if not nums:
            return -1

        start, end = 0 , len(nums) - 1
        while start + 1 < end:
            mid = (start + end) // 2
            if nums[mid] > nums[end]:
                if nums[start] <= target <= nums[mid]:
                    end = mid
                else:
                    start = mid
            else:
                if nums[mid] <= target <= nums[end]:
                    start = mid
                else:
                    end = mid

        if nums[start] == target:
            return start
        if nums[end] == target:
            return end

        return -1
```

#### [162. Find Peak Element](https://leetcode-cn.com/problems/find-peak-element/) <span style="color:orange">Medium</span>

Binary search to find element in a unsorted array

```java
class Solution {
    public int findPeakElement(int[] nums) {
        if(nums == null || nums.length == 0) {
            return -1;
        }

        if(nums.length == 1){
            return 0;
        }

        if(nums.length == 2) {
            return nums[0] > nums[1] ? 0 : 1;
        }

        int start = 0, end = nums.length - 1;
        while (start + 1 < end) {
            int mid = start + (end - start) / 2;
            // upward on the right, then a peak must be on the right part
            if (nums[mid] < nums[mid + 1]) {
                start = mid;
			// upward on the left, then a peak must be on the left
            } else if (nums[mid] < nums[mid - 1]) {
                end = mid;
            } else {
                return mid;
            }
        }

        return nums[start] > nums[end] ? start : end;
    }
}
```

#### [183 · Wood Cut - LintCode  ](https://www.lintcode.com/problem/183/)<span style="color:orange">Medium</span>

Binary search on an answer set

```python
class Solution:
    """
    @param L: Given n pieces of wood with length L[i]
    @param k: An integer
    @return: The maximum length of the small pieces
    """
    def woodCut(self, L, k):
        # write your code here
        if not L:
            return 0

        start, end = 1, sum(L) // k

        if end < 1:
            return 0

        while start + 1 < end:
            mid = (start + end) // 2
            if self.cut_by(L, mid) >= k:
                start = mid
            else:
                end = mid

        return end if self.cut_by(L, end) >= k else start

    def cut_by(self, L, length):
        return sum(l // length for l in L)
```

#### [74. Search a 2D Matrix](https://leetcode.com/problems/search-a-2d-matrix/) Medium

binary search on a sorted input

- find a row: binary search on first column to get the largest point smaller than the target
- find target in the row: binary search  on the row

This approach involves two binary searches, which is not so efficient. The matrix can also be treated as a one-dimensional sorted array so that only one binary search is needed.

```python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        # get a matrix value from an index
        def get_value(index):
            """
            0 [0][0] 1 [0][1] 2 [0][2]
            3 [1 * 3][+0] 4 [1 * 3][+1] 5 [1 * 3][+2]
            Think of the matrix (m * n) as an m*n-long array. Row number is index divided by columns
            and column number is its modulo.
            """
            x = index // n
            y = index % n
            return matrix[x][y]

        m = len(matrix)
        n = len(matrix[0])

        start = 0
        end = m * n - 1
        while start + 1 < end:
            mid = (start + end) // 2
            mid_num = get_value(mid)
            if target < mid_num:
                end = mid
            else:
                start = mid

        if target == get_value(start):
            return True
        if target == get_value(end):
            return True
```

#### [240. Search a 2D Matrix II](https://leetcode.com/problems/search-a-2d-matrix-ii/) Medium

Binary search cannot be used here because the matrix is not strictly ascending. But it is known that each row and each column is ascending. The solution may start from here. Take a look at corner points.

$matrix[0][0]$ is the smallest point and $matrix[m - 1][n - 1]$ the largest. Both points is of no help to exclude certain points when searching for a target. For $matrix[m - 1][0]$, all the points that come before it in the column are smaller and all the points that come after it in the row are bigger. And vice versa for $matrix[0][n - 1]$. This knowledge will come to handy for excluding certain points. Say $target < matrix[m - 1][0]$ or $target > matrix[m - 1][0]$ , it's safe to conclude that target is not in $matrix[m - 1]$ or not in $matrix[0..m-1][0]$. This way a row or column is excluded and the search range is narrowed down.

```python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        m = len(matrix)
        n = len(matrix[0])

        # start from left bottom corner
        x = m - 1
        y = 0
        # x will only decrease and y will only increase
        while x >= 0 and y < n:
            if target > matrix[x][y]:
                # the leftmost column is excluded
                y += 1
            elif target < matrix[x][y]:
                # the bottom row is excluded
                x -= 1
            else:
                return True
        return False
```

#### [600 · Smallest Rectangle Enclosing Black Pixels - LintCode ](https://www.lintcode.com/problem/600/)Hard

leetcode: [302. Smallest Rectangle Enclosing Black Pixels'](https://leetcode.com/problems/smallest-rectangle-enclosing-black-pixels/)

binary search on unsorted input

A rectangle that can enclose all black pixels must cover leftmost/rightmost/top/bottom black pixels. Binary search is used to find these dotted pixels.

```python
class Solution:
    def min_area(self, image: List[List[str]], x: int, y: int) -> int:
        def is_column_dotted(column_num):
            return '1' in [row[column_num] for row in image]

        def is_row_dotted(row_num):
            return '1' in image[row_num]

        # searching for the first one means the left part is prefered
        def search_first(start, end, is_dotted):
            while start + 1 < end:
                mid = (start + end) // 2
                print(start, mid, end)
                if is_dotted(mid):
                    # try to find the first, search in the left part
                    end = mid
                else:
                    # no better option, have to search in the right part
                    start = mid

            # still try left part as the first option
            if is_dotted(start):
                return start
            return end

        def search_last(start, end, is_dotted):
            while start + 1 < end:
                mid = (start + end) // 2
                if is_dotted(mid):
                    # try to find the first, search in the left part
                    start = mid
                else:
                    # no better option, have to search in the right part
                    end = mid

            # still try left part as the first option
            if is_dotted(end):
                return end
            return start

        if not image or not image[0]:
            return 0

        m = len(image)
        n = len(image[0])

        # the leftmost point must not be on the right to the given point
        # search from 0-th column to y-th column
        left = search_first(0, y, is_column_dotted)
        # the top point must
        top = search_first(0, x, is_row_dotted)
        right = search_last(y, n - 1, is_column_dotted)
        bottom = search_last(x, m - 1, is_row_dotted)
        print(top, bottom, left, right)

        return (right - left + 1) * (bottom - top + 1)
```

#### [437 · Copy Books - LintCode](https://www.lintcode.com/problem/437/) Medium

binary search on an answer set

There is a turning point in the answer set: less than certain time copying cannot be finished for k people and more than a certain point coping can be completed for k people in whatever manners. The fact that such a turning point exists makes binary search a possible choice.

```python
class Solution:
    def copy_books(self, pages: List[int], k: int) -> int:
        def get_copiers_copying_within_limit(pages, time_limit):
            page_per_minute = 1
            copiers = 0
            # for forst book initialization
            copy_time_spent = time_limit

            for page in pages:
                time_needed = page * page_per_minute
                # not sufficient time
                if time_needed > time_limit:
                    return len(pages) + 1

                # copy time already spent plus the time for copying current book is more than time limit
                # this is when a new copier is needed
                if copy_time_spent + time_needed > time_limit:
                    # add a copier
                    copiers += 1
                    # the copier hasn't started copying yet
                    copy_time_spent = 0

                # current copier is able to handle
                copy_time_spent += time_needed

            return copiers

        # early exit for abnormal input
        if not pages:
            return 0
        if k <= 0:
            return 0

        # define the range of the answer set
        # bottom: there is a copier for every book
        start = max(pages)
        # top: suppose a copier copies all the books
        end = sum(pages)

        while start + 1 < end:
            mid = (start + end) // 2
            if get_copiers_copying_within_limit(pages, mid) <= k:
                end = mid
            else:
                start = mid

        if get_copiers_copying_within_limit(pages, start) <= k:
            return start
        return end

```

### DFS

#### [78. Subsets](https://leetcode.com/problems/subsets/) <span style="color:orange">Medium</span>

Combinatorial Search Problem -> DFS.

Every element either exist in a subset or does not exist in a subset and in total there are $2^n$ combinations.

 									[ ]

1?					[1] 						[ ]

2?		[1, 2] 				 	[1] 				[2] [ ]

3? 	[1, 2, 3] [1, 2] 	[1, 3]  [1]	 [2] [2, 3] 	[3] [ ]

```java
class Solution {
    public List<List<Integer>> subsets(int[] nums) {
		List<List<Integer>> results = new ArrayList<>();
        if (nums == null)
            return results;
        Arrays.sort(nums);
        dfs(nums, 0, new ArrayList<Integer>(), results);
        return results;
    }

    // current index, current subset, final results
    private void dfs(int[] nums, int index, List<Integer> subset, List<List<Integer>> results) {
        // stops when index if out of scope and a subset is found
        if (index == nums.length) {
            results.add(new ArrayList<Integer>(subset));
            // stops
            return;
        }

        // any element is either added or not added
        // element at index is added
        subset.add(nums[index]);
        dfs(nums, index + 1, subset, results);

        // element at index is not added
        // backtrack
        subset.remove(subset.size() - 1);
        dfs(nums, index + 1, subset, results);
    }
}
```

BFS 1
```java
class Solution {
    public List<List<Integer>> subsets(int[] nums) {
        if (nums == null) {
            return new ArrayList<>();
        }
        
        List<List<Integer>> queue = new ArrayList<>();
        int index = 0;
        
        // to avoid add the same number repetitively
        Arrays.sort(nums);
        // initialize: start from empty subset
        queue.add(new ArrayList<Integer>());
        while (index < queue.size()) {
            List<Integer> subset = queue.get(index++);
            for (int i = 0; i < nums.length; i++) {
                if (subset.size() != 0 && nums[i] <= subset.get(subset.size() - 1)) {
                    continue;
                }

                // add every element as individual subset
                // then make every subset longer
                // []. [1], [2], [3], [1,2] ...
                List<Integer> newSubset = new ArrayList<>(subset);
                newSubset.add(nums[i]);
                queue.add(newSubset);
            }
        }
        return queue;
    }
}
```
BFS 2
[] | [1] | [2] [1,2] | [3] [1,3] [2,3] [1,2,3]
```java
class Solution {
    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> candidates = new ArrayList<>();
        // initialize: empty subset
        candidates.add(new LinkedList<>());
        Arrays.sort(nums);

        for (int num: nums) {
            int size = candidates.size();
            for (int i = 0; i < size; i++) {
                List<Integer> subset = new ArrayList<>(candidates.get(i));
                // there is no replicates in candidates
                subset.add(num);
                candidates.add(subset);
            }
        }
        return candidates;
    }
}
```


A more general solution: Any subset starts with nums[0] .. nums[n - 1]. Since every subset is ascending, there is no need to go back, i.e. we are not gettting all the permutations here.

​									[ ]

​			[1] 					[2] 				[3]

​		[1, 2]  [1, 3] 	[2, 3]

​	[1, 2, 3]

```java
class Solution {
    public List<List<Integer>> subsets(int[] nums) {
		List<List<Integer>> results = new ArrayList<>();
        if (nums == null)
            return results;
        Arrays.sort(nums);
        dfs(nums, 0, new ArrayList<Integer>(), results);
        return results;
    }

    private void dfs(int[] nums, int startIndex, List<Integer> subset, List<List<Integer>> results) {
		// use iteration to contorl the flow and end
        // so there is no need to place a stop case here
        // for every subset, simply add it to the results
        results.add(new ArrayList<Integer>(subset));
        // never look back, start from startIndex instead of 0
        for (int i = startIndex; i < nums.length; i++) {
			subset.add(nums[i]);
            dfs(nums, i + 1, subset, results);
            // backtrack
            subset.remove(subset.size() - 1);
        }
    }
}
```

#### [90. Subsets II  ](https://leetcode.com/problems/subsets-ii/)<span style="color:orange">Medium</span>

Combinatorial Search Probelm --> DFS while removing duplicates along the way

Keep in mind that getting all possibilites and then remoing duplicates is too high in time complexity. (Consider the test case: [1, 1, 1, 1, 1])

```java
class Solution {
    public List<List<Integer>> subsetsWithDup(int[] nums) {
		List<List<Integer>> results = new ArrayList<>();
        if (nums == null)
            return results;
        Arrays.sort(nums);
        dfs(nums, 0, new ArrayList<Integer>(), results);
        return results;
    }

    private void dfs(int[] nums, int startIndex, List<Integer> subset, List<List<Integer>> results) {
        results.add(new ArrayList<Integer>(subset));
        for (int i = startIndex; i < nums.length; i++) {
            // skip if a duplicate is encountered
            if(i > 0 && nums[i] == nums[i - 1] && i != startIndex) {
               continue;
            }
			subset.add(nums[i]);
            dfs(nums, i + 1, subset, results);
            // backtrack
            subset.remove(subset.size() - 1);
        }
    }
}
```

#### [46. Permutations](https://leetcode.com/problems/permutations/) <span style="color:orange">Medium</span>

[<img src="https://s4.ax1x.com/2022/01/13/7lSavT.png" alt="7lSavT.png" style="zoom:50%;" />](https://imgtu.com/i/7lSavT)

Permutation Search --> DFS

$n!$ permutations for a n-element list

```java
class Solution {
    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> results = new ArrayList<>();
        if (nums == null)
            return results;
        boolean[] visited = new boolean[nums.length];
        dfs(nums, visited, new ArrayList<>(), results);
        return results;
    }

    private void dfs(int[] nums, boolean[] visited, List<Integer> subset, List<List<Integer>> results) {
    	// deeper and deeper until all elements are in the subset
        // add to the results only when all elements are contained
        if (subset.size() == nums.length) {
            // deep copy to avoid copying a reference
            results.add(new ArrayList<Integer>(subset));
            return;
        }

        // for every level all elements are iterated so start from 0
        for (int i = 0; i < nums.length; i++) {
            if (visited[i]) {
            	continue;
            }

            subset.add(nums[i]);
            visited[i] = true;
            dfs(nums, visited, subset, results);
            // backtrack
            visited[i] = false;
            subset.remove(subset.size() - 1);
        }
    }
}
```
#### [LintCode 90 · k Sum II](https://www.lintcode.com/problem/90/)
Combination, DFS
```java
public class Solution {
    public List<List<Integer>> kSumII(int[] A, int k, int target) {
        List<List<Integer>> combinations = new ArrayList();
        List<Integer> combination = new ArrayList();
        dfs(A, 0, k, target, combination, combinations);
        return combinations;
    }

    public void dfs(int[] A, int index, int k, int target, List<Integer> combination, List<List<Integer>> combinations) {
        // stop conditions: target is reached and k and only k numbers are used
        if (k == 0 && target == 0) {
            System.out.println(Arrays.toString(combination.toArray()));
            // add a copy instead of a reference because every element addedd into the combination will be removed afterwards
            combinations.add(new ArrayList(combination));
            return;
        }

        // stop condition: k numbers are used but target is not met or target is met but not k numbers are used
        if (k == 0 || target == 0) {
            return;
        }

        for (int i = index; i < A.length; i++) {
            combination.add(A[i]);
            dfs(A, i + 1, k - 1, target - A[i], combination, combinations);
            combination.remove(combination.size() - 1);
        }
    }
}
```
#### [39. Combination Sum](https://leetcode.com/problems/combination-sum/) <span style="color:orange">Medium</span>
Combination DFS
```python
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        results = []
        if not candidates:
            return results
        self.dfs(0, target, sorted(candidates), [], results)
        return results

    def dfs(self, index, remaining_target, candidates, combination, combinations):
        if remaining_target == 0:
            return combinations.append(list(combination))

        for i in range(index, len(candidates)):
            if remaining_target < candidates[i]:
                break
            combination.append(candidates[i])
            # every candidate can be used more than once, so start from i again instead of i + 1
            self.dfs(i, remaining_target - candidates[i], candidates, combination, combinations)
            combination.pop()
```

#### [51. N-Queens](https://leetcode.com/problems/n-queens/) Hard
DFS, search tree
```python
class Solution:
    def solveNQueens(self, n):
        # entry
        results = []
        # try column places in every row, starting from an empty board
        self.search(n, [], results)
        return results
    
    # hom many queens (n) are placed at which columns one by one (queen_columns)
    # answers are collected into results (results)
    def search(self, n, queen_columns, results):
        queen_row = len(queen_columns)
        # exit: there is a queen in all rows now
        if queen_row == n:
            results.append(self.draw_board(queen_columns))
            return
        
        # try every column in the current row
        for queen_column in range(n):
            # contradiction exists if a queen is placed in the current column
            if not self.is_valid(queen_columns, queen_row, queen_column):
                continue
            queen_columns.append(queen_column)
            self.search(n, queen_columns, results)
            # backtrack
            queen_columns.pop() 
        
        
    def is_valid(self, queen_columns, new_queen_row, new_queen_col):
        # check column
        if new_queen_col in queen_columns:
            return False
        
        # check diagonal
        for queen_row, queen_col in enumerate(queen_columns):
            # from top left to bottom right: increasing in row number and column number
            if queen_row - queen_col == new_queen_row - new_queen_col:
                return False
            # from bottom left to top right: decreasing in row number and increasing in column number
            if queen_row + queen_col == new_queen_row + new_queen_col:
                return False
        return True
        
    def draw_board(self, queen_columns):
        n = len(queen_columns)
        board = []
        for row in range(n):
            row_desc = ['Q' if col == queen_columns[row] else '.' for col in range(n)]
            board.append(''.join(row_desc))
        return board
```

#### [37. Sudoku Solver](https://leetcode.com/problems/sudoku-solver/) Hard
DFS
```python
class Solution:
    def solveSudoku(self, board: List[List[str]]) -> None:
        used = self.prepare_occupied_info(board)
        # starting from 0 and ending at 81
        self.search(board, 0, used)
        return board
        
    def prepare_occupied_info(self, board):
        used = {
            'rows': [set() for _ in range(9)],
            'cols': [set() for _ in range(9)],
            'boxes': [set() for _ in range(9)]
        }
        for row in range(9):
            for col in range(9):
                if board[row][col] == '.':
                    continue
                used['rows'][row].add(board[row][col])
                used['cols'][col].add(board[row][col])
                # convert (x, y) coordinate to the x-th box 
                used['boxes'][row // 3 * 3 + col // 3].add(board[row][col])
                
        return used
    
    def search(self, board, index, used):
        # exit: the last slot is already taken 
        if index == 81:
            return True
        
        x, y = index // 9, index % 9
        # current slot is alreay taken
        if board[x][y] != '.':
            return self.search(board, index + 1, used)
        
        for val in range(1, 10):
            val = str(val)
            if not self.is_valid(used, x, y, val):
                continue
                
            board[x][y] = val
            used['rows'][x].add(val)
            used['cols'][y].add(val)
            used['boxes'][x // 3 * 3 + y // 3].add(val)
            if self.search(board, index + 1, used):
                return True
            
            board[x][y] = '.'
            used['rows'][x].remove(val)
            used['cols'][y].remove(val)
            used['boxes'][x // 3 * 3 + y // 3].remove(val)
        
            
    def is_valid(self, used, row, col, val):
        if val in used['rows'][row]:
            return False
        if val in used['cols'][col]:
            return False
        if val in used['boxes'][row // 3 * 3 + col // 3]:
            return False
        return True
```
```java
class Solution {
    public void solveSudoku(char[][] board) {
        dfs(board, 0);
    }
    
    private boolean dfs(char[][] board, int index) {
        if (index == 81) {
            return true;
        }
        
        int x = index / 9;
        int y = index % 9;
        
        // alreay filled
        if (board[x][y] != '.') {
            return dfs(board, index + 1);
        }
        
        for (char val = '1'; val <= '9'; val++) {
            if (!isValid(board, x, y, val)) {
                continue;
            }
            
            board[x][y] = val;
            if (dfs(board, index + 1)) {
                return true;
            }
            board[x][y] = '.';
        }
        
        return false;
    }
    
    private boolean isValid(char[][] board, int x, int y, char val) {
        for (int i = 0; i < 9; i++) {
            // row
            if (board[x][i] == val) {
                return false;
            }

            // column
            if (board[i][y] == val) {
                return false;
            }
            
            /** 
              0 1 2 3 4 5 6 7 8
            / 0 0 0 1 1 1 2 2 2 (row)
            % 0 1 2 0 1 2 0 1 2 (column)
            / 3 * 3 : set starting point to the top left corner in a box, then move from left to right and top to bottom
            0 . . 3 . . 6 . .
            . . . . . . . . .
            . . . . . . . . .
            3 . . 3 . . 6 . .
            . . . . . . . . .
            . . . . . . . . .
            6 . . 3 . . 6 . .
            . . . . . . . . .
            . . . . . . . . .
            **/
            if (board[x / 3 * 3 + i / 3][y / 3 * 3 + i % 3] == val) {
                return false;
            }
        }
        return true;
    } 
}
```
Reduce search range for every slot so that the process can speed up
```python
class Solution:
    def solveSudoku(self, board: List[List[str]]) -> None:
        self.dfs(board)

    def dfs(self, board):
        x, y, possibilities = self.get_possible(board)
        
        if x is None:
            return True

        for p in possibilities:
            board[x][y] = p
            if self.dfs(board):
                return True
            board[x][y] = '.'
        
        return False

    def get_possible(self, board):
        x, y, possibilities = None, None, ['.'] * 10

        for i in range(9):
            for j in range(9):
                if  board[i][j] != '.':
                    continue
                vals = []
                for num in range(1, 10):
                    val = str(num)
                    if self.is_valid(board, i, j, val):
                        vals.append(val)

                x, y, possibilities = i, j, vals
                return x, y, possibilities

        return x, y, possibilities            


    def is_valid(self, board, x, y, val):
        for i in range(9):
            if board[x][i] == val:
                return False
            if board[i][y] == val:
                return False
            if board[x // 3 * 3 + i // 3][y // 3 * 3 + i % 3] == val:
                return False
        return True
```


### Other

Algorithms that has time complexity better than linear, i.e. less than $O(n)$

#### [4. Median of Two Sorted Arrays](https://leetcode.com/problems/median-of-two-sorted-arrays/) Hard
recursive, something like a binary search in terms of perspective

Think from the perspective of time complexity. A better solution than a linear one would be a logrithmatic one. Since there are two sorted arrays, the searching range can be narrowed by half if search can be down in two arrays at the same time by seaching k/2-th number in both arrays.
```java
public class Solution {
    public double findMedianSortedArrays(int A[], int B[]) {
        int n = A.length + B.length;

        if (n % 2 == 0) {
            // get average of two middle numbers
            return (
                // middle left
                findKth(A, 0, B, 0, n / 2) +
                // middle right
                findKth(A, 0, B, 0, n / 2 + 1)
            ) / 2.0;
        }

        return findKth(A, 0, B, 0, n / 2 + 1);
    }

    // find kth number of two sorted array
    // a is the search starting point in A and so is b in B
    public static int findKth(int[] A, int a,
                              int[] B, int b,
                              int k){
        // starting point is not in A, then the first k numbers must be in B
        if (a >= A.length) {
            return B[b + k - 1];
        }
        // starting point is not in B, then the first k numbers must be in A
        if (b >= B.length) {
            return A[a + k - 1];
        }
        // trying to find the first number? either the first element of A or that of B
        if (k == 1) {
            return Math.min(A[a], B[b]);
        }

        // get the k/2-th number in A and B
        int halfKthOfA = a + k / 2 - 1 < A.length
            ? A[a + k / 2 - 1]
            : Integer.MAX_VALUE;
        int halfKthOfB = b + k / 2 - 1 < B.length
            ? B[b + k / 2 - 1]
            : Integer.MAX_VALUE;

        if (halfKthOfA < halfKthOfB) {
            // k-th number is not in the first k/2 part of A starting from a
            return findKth(A, a + k / 2, B, b, k - k / 2);
        } else {
            // k-th number is not in the first k/2 part of B starting from b
            return findKth(A, a, B, b + k / 2, k - k / 2);
        }
    }
}
```
```python
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        length = len(nums1) + len(nums2)
        if length % 2 == 1:
            return self.findKth(nums1, 0, nums2, 0, length // 2 + 1)

        middle_left = self.findKth(nums1, 0, nums2, 0, length // 2)
        middle_right = self.findKth(nums1, 0, nums2, 0, length // 2 + 1)
        return (middle_left + middle_right) / 2

    def findKth(self, A, a, B, b, k):
        # A start is out, all k numbers are in B
        if a >= len(A):
            return B[b + k - 1]
        # B start is out, all k numbers are in A
        if b >= len(B):
            return A[a + k - 1]
        # first number is either the first of A or the first of B
        if k == 1:
            return min(A[a], B[b])

        """
        What if k/2 > len(A) ?
        k/2-th number might exist in A, because if k/2-th exists in first k/2 elements of B, A does not have enough elements to contain the other k/2-th half
                 k/2
        A . . .
        B . . . . . . . . .
        """
        A_key = math.inf
        B_key = math.inf
        if a + (k // 2) <= len(A):
            A_key = A[a + k // 2 - 1]
        if b + (k // 2) <= len(B):
            B_key = B[b + k // 2 - 1]


        if A_key < B_key:
            # k/2 numbers are removed by searching
            return self.findKth(A, a + k // 2, B, b, k - k // 2)
        return self.findKth(A, a, B, b + k // 2,  k - k // 2)
```
#### [315. Count of Smaller Numbers After Self](https://leetcode.com/problems/count-of-smaller-numbers-after-self/) Hard
block search, my solution not fast for this problem, will cause TLE sometimes, check out Java solution for reference

```python
class Block:
    def __init__(self):
        self.total = 0
        self.counter = {}
        
class BlockArray:
    def __init__(self, max_value):
        self.blocks = [
            Block()
            for _ in range(10000)
        ]
        
    def get_position(self, val):
        return val // 100
        
    def insert(self, val):
        position = self.get_position(val)
        self.blocks[position].counter[val] = self.blocks[position].counter.get(val, 0) + 1
        self.blocks[position].total += 1
    
    def count_smaller(self, val):
        count = 0
        position = self.get_position(val)
        # count of total from all blocks that come before the block where the val lies
        for i in range(position):
            count += self.blocks[i].total
        
        # count of all smaller numbers in this block
        block = self.blocks[position]
        for number in block.counter:
            if number < val:
                count += block.counter[number]
                
        return count
        
class Solution:
    def countSmaller(self, nums: List[int]) -> List[int]:
        results = []
        block_array = BlockArray(10000)
        # offset is used to deal with negative numbers
        offset = 10000
        for i in range(len(nums) - 1, -1, -1):
            count = block_array.count_smaller(nums[i] + offset)
            results.insert(0, count)
            block_array.insert(nums[i] + offset)
        return results
```
```java
class Solution {
    // not sure why 20000 is chosen
    // maybe the largest number from input is less than 20000?
    private final int SIZE = 20000;
    // offset deals with negative numbers
    // though I'm not sure why 10000 is chosen
    // maybe the least negative number is larger than -10000
    private static final int OFFSET = 10000;
    public List<Integer> countSmaller(int[] nums) {
        LinkedList<Integer> res = new LinkedList<>();
        if (nums == null || nums.length == 0) {
            return res;
        }
        
        BlockArray blockArray = new BlockArray(SIZE);
        // iterate nums backward
        for (int i = nums.length - 1; i >= 0; i--) {
            res.addFirst(blockArray.countSmaller(nums[i] + OFFSET));
            blockArray.insert(nums[i] + OFFSET);
        }
        
        return res;
    }
}


class Block {
    public int total;
    public int[] counter;
    public Block(int size) {
        this.total = 0;
        this.counter = new int[size];
    }
}

class BlockArray {
    public Block[] blocks;
    public int blockSize;
    public BlockArray(int capacity) {
        this.blockSize = (int) Math.sqrt(capacity);
        int numberOfBlocks = capacity / this.blockSize + 1;
        this.blocks = new Block[numberOfBlocks];
        for (int i = 0; i < numberOfBlocks; i++) {
            this.blocks[i] = new Block(this.blockSize);
        }
    }
    
    public void insert(int value) {
        int indexOfBlock = value / blockSize;
        this.blocks[indexOfBlock].total++;
        this.blocks[indexOfBlock].counter[value - this.blockSize * indexOfBlock]++;
    }
    
    public int countSmaller(int value) {
        int indexOfBlock = value / blockSize;
        int count = 0;
        for (int i = 0; i < indexOfBlock; i++) {
            count += this.blocks[i].total;
        }
        
        for (int i = 0; i < value - this.blockSize * indexOfBlock; i++) {
            count += this.blocks[indexOfBlock].counter[i];
        }
        
        return count;
    }
}
```