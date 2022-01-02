

## Array

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



#### 977 Squares of a Sorted Array <span style="color:green">Easy</span>

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

#### 209 Minimum Size Subarray Sum <span style="color:orange">Medium</span>

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



#### 904 Fruits into Baskets <span style="color:orange">Medium</span>

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

#### [88. Merge Sorted Array ](https://leetcode-cn.com/problems/merge-sorted-array/) <span style="color:green">Easy</span>

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

Keywords for binary search: 

array, target, sorted, equal or close to target, O(NlogN)

#### [34. Find First and Last Position of Element in Sorted Array](https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/) <span style="color:orange">Medium</span>

Binary search: Decrease and Conqu

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

#### LintCode 183 · Wood Cut

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



## Linked List

#### [203. Remove Linked List Elements](https://leetcode-cn.com/problems/remove-linked-list-elements/) <span style="color:green">Easy</span>

```java
class Solution {
    public ListNode removeElements(ListNode head, int val) {
        if (head == null) return null;
        head.next = removeElements(head.next, val);
        return head.val == val ? head.next : head;
    }
}
```

```python
class Solution(object):
    def removeElements(self, head, val):
        """
        :type head: ListNode
        :type val: int
        :rtype: ListNode
        """
        dummy_head = ListNode(-1)
        dummy_head.next = head
        
        current_node = dummy_head
        while current_node.next:
            if current_node.next.val == val:
                current_node.next = current_node.next.next
            else:
                current_node = current_node.next
        return dummy_head.next
```

#### [707. Design Linked List](https://leetcode-cn.com/problems/design-linked-list/) <span style="color:orange">Medium</span>

My Implementation is bat at run time and memory usage. There are simply too many special cases involved that cause additional complexity.

```java
class MyLinkedList {

    class Node {
        int val;
        Node next;
        Node(int val){
            this.val = val;
        }
    }
    
    Node head;
    int length;
    
    
    public MyLinkedList() {
        this.head = null;
        this.length = 0;
    }
    
    public int get(int index) {
        if(index < 0 || index >= this.length){
            return -1;
        } else {
            System.out.println(index);
            System.out.println(this.length);
            int count = 0;
            Node current = this.head;
            while(count != index){
                if(current != null){
                    current = current.next;   
                }
                count++;
            }
            return current.val;
        }
    }
    
    public Node getNode(int index){
        int count = 0;
        Node current = head;
        while(count < index){
            current = current.next;
            count++;
        }
        return current;
    }
    
    public void addAtHead(int val) {
        Node newHeadNode = new Node(val);
        Node currHeadNode = this.head;
        newHeadNode.next = currHeadNode;
        this.head = newHeadNode;
        this.length++;
    }
    
    public void addAtTail(int val) {
        if(this.length == 0){
            this.addAtHead(val);
            return;
        }
        Node newTailNode = new Node(val);
        Node currentTailNode = this.getNode(this.length - 1);
        currentTailNode.next = newTailNode;
        this.length++;
    }
    
    public void addAtIndex(int index, int val) {
        if(index == 0){
            this.addAtHead(val);
            return;
        }
        if(index == this.length){
            this.addAtTail(val);
            return;
        }
        if(index < 0 || index > this.length){
            return;
        }
        Node node = new Node(val);
        Node prevNode = this.getNode(index - 1);
        Node currNode = prevNode.next;
        prevNode.next = node;
        node.next = currNode;
        this.length++;
    }
    
    public void deleteAtIndex(int index) {
        if(index < 0 || index >= this.length){
            return;
        }
        if(index == 0){
            Node node = this.head.next;
            this.head = node;
            this.length--;
            return;
        }
        Node prevNode = this.getNode(index - 1);
        if(prevNode.next.next == null){
            prevNode.next = null;
        } else {
            prevNode.next = prevNode.next.next;
        }
        this.length--;
    }
}
```

Clean and elegant implementation that saves time and storage.  addAtHead and addAtTail are treated as special cases of addAtIndex.

```python
class Node:
    def __init__(self, val):
        self.val = val
        self.next = None

class MyLinkedList(object):

    def __init__(self):
        self.head = None
        self.size = 0

    def get(self, index):
        if index < 0 or index >= self.size:
            return -1
        current = self.head
        for _ in range(0, index):
            current = current.next
        return current.val
    
	# defined as a special case of addAtIndex 
    def addAtHead(self, val):
        self.addAtIndex(0, val)
        
	# defined as a special case of addAtIndex 
    def addAtTail(self, val):
        self.addAtIndex(self.size, val)
        

    def addAtIndex(self, index, val):
        if index > self.size:
            return
        current = self.head
        node = Node(val)
        
        if index <= 0:
            node.next = current
            self.head = node
        else:
            for _ in range(index  - 1):
                current = current.next
            node.next = current.next
            current.next = node
            
        self.size += 1

    def deleteAtIndex(self, index):
        if index < 0 or index >= self.size:
            return
        
        if index == 0:
            self.head = self.head.next
        else:
            current = self.head
            for _ in range(index - 1):
                current = current.next
            current.next = current.next.next
            
        self.size -= 1

```

#### [206. Reverse Linked List](https://leetcode-cn.com/problems/reverse-linked-list/) <span style="color:green">Easy</span>

My implementation. I treat the reversing process as a whole by converting the linked list to a list, reversing it and building the reversed linked list from scratch. This is inefficient and memory consuming.

```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if not head:
            return None
        current = head
        nodes = []
        while current.next:
            nodes.append(current)
            current = current.next
        nodes.append(current)
        nodes.reverse()
        for index, node in enumerate(nodes):
            if index == len(nodes) - 1:
                node.next = None
            else:
                node.next = nodes[index + 1]
            if index == 0:
                head = node
        return head
```

Iteration is a different perspective. The whole process is divided into sub-processes chained together.

```java
// iterative 
public ListNode reverseList(ListNode head) {
    /*-- 
    e.g. null ->   1    ->  2
              prev ->  head  -> next
    --*/
    ListNode prev = null;
    while(head != null){
        ListNode next = head.next;
       	// link the current node to its previous node, thus the reversing is done
        head.next = prev;
        // to keep the iteration moving
        prev = head;
        head = next;
    }
    return prev;
}

// recursive
public ListNode reverseList(ListNode head) {
    return reverse(head, null);
}
private ListNode reverse(ListNode head, ListNode prev){
    if(head == null)
        return prev;
    ListNode next = head.next;
    head.next = prev;
    return reverse(next, head);
}

```

#### [24. Swap Nodes in Pairs ](https://leetcode-cn.com/problems/swap-nodes-in-pairs/) <span style="color:orange">Medium</span>

```java
class Solution {
    /*-- 
    e.g. null ->  1  ->  2  ->  3
         f        s      t
    --*/
    public ListNode swapPairs(ListNode head) {
		ListNode resultHead = new ListNode();
        resultHead.next = head;
        
        ListNode current = resultHead;
        while(current != null && current.next != null && current.next.next != null){
            // swap second and third
            ListNode first = current;
            ListNode second = current.next;
            ListNode third = second.next;
            
            first.next = third;
            second.next = third.next;
            third.next = second;
            
            current = current.next.next;
        }
        return resultHead.next;
    }
}
```

```java
// recursive
class Solution {
    public ListNode swapPairs(ListNode head) {
        if(head == null || head.next == null)
            return head;
        
        ListNode next = head.next;
        head.next = swapPairs(next.next);
        next.next = head;
        return next;
    }
}
```

#### [19. Remove Nth Node From End of List](https://leetcode-cn.com/problems/remove-nth-node-from-end-of-list/) <span style="color:orange">Medium</span>

```java
// recursive 
class Solution {
    // index is used to determine if n is reached
    private int index;
    public ListNode removeNthFromEnd(ListNode head, int n) {
        // exit condition
        if(head == null){
            this.index = 0;
            return head;
        }
        
        // what to do at current step: link head to its next or next's next
        head.next = removeNthFromEnd(head.next, n);
        this.index += 1;
        
        // return value: when index does not reach n, return current node
        // when index is equal to n, current node is skipped and return its next
        return this.index == n ? head.next : head;
    }
}
```

```java
// Calculate the length of the linked list
class Solution {
    public int getLength(ListNode head) {
        int length = 0;
        while (head != null) {
            ++length;
            head = head.next;
        }
        return length;
    }
    
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode dummy = new ListNode(0, head);
        int length = getLength(head);
        ListNode current = dummy;
        for(int i = 0; i < length - n; ++i){
            // current will be moved to the next when the index stops so 
            current = current.next;
        }
        current.next = current.next.next;
        return dummy.next;
    }
}
```

```java
// Stack's first-in-last-out mechanism is the right fit for couting backwards
class Solution {
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode dummy = new ListNode(0, head);
        Deque<ListNode> stack = new LinkedList<ListNode>();
        ListNode current = dummy;
        while(current != null){
            stack.push(current);
            current = current.next;
        }
        // when n nodes are poped current will be just the node before the node to be deleted
        for(int i = 0; i < n; ++i){
            stack.pop();
        }
        ListNode prev = stack.peek();
        prev.next = prev.next.next;
        ListNode ans = dummy.next;
        return ans;
    }
}
```

```java
// Tow Pointers
class Solution {
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode dummy = new ListNode(0, head);
        ListNode fast = head;
        ListNode slow = dummy;
        for(int i = 0; i < n; i++){
            fast = fast.next;
        }
        while(fast != null){
            fast = fast.next;
            slow = slow.next;
        }
        slow.next = slow.next.next;
        return dummy.next;
    }
}
```

Why not simply return head instead of the next of dummy? If head is returned, that means the linked list will never be empty. If the linked list is {1] and n is 1, the result will be empty after removal. Returning dummy.next will get null, but returning head, in this case, will return a wrong answer. When dealing with linked lists, it's better to make sure that you always stay in the linked list. A variable that points to a node is not the node itself. A node is always located right after its previous node, which starts from the head node. And keep in mind that a dummy node may be quite useful in linked list problems.

#### [面试题 02.07. Intersection of Two Linked Lists LCCI ](https://leetcode-cn.com/problems/intersection-of-two-linked-lists-lcci/)<span style="color:green">Easy</span>

```java
// Two pointers
class Solution{
    public ListNode getIntersectionNode(ListNode headA, ListNode headB){
        ListNode a = headA;
        ListNode b = headB;
        while(a != b){
			a = a == null ? headB : a.next;
            b = b == null ? headA : b.next;
        }
        return a;
    }
}
```

#### [142. Linked List Cycle II](https://leetcode-cn.com/problems/linked-list-cycle-ii/) <span style="color:orange">Medium</span>

The node where the cycle begins, if there do exists a cycle, must be the node that re-appears first.

```java
public class Solution {
    public ListNode detectCycle(ListNode head) {
        ListNode current = head;
        ArrayList<ListNode> seen = new ArrayList<ListNode>();
        
        while(current != null){
            if(seen.contains(current))
                return current;
            else{
                seen.add(current);
                current = current.next;
            }
        }
        return null;
    }
}
```

**从头结点出发一个指针，从相遇节点也出发一个指针，这两个指针每次只走一个节点， 那么当这两个指针相遇的时候就是环形入口的节点。**

```java
// two pointers
public class Solution{
    public ListNode detectCycle(ListNode head){
        ListNode slow = head;
        ListNode fast = head;
        while(fast != null && fast.next != null){
            slow = slow.next;
            fast = fast.next.next;
            if(slow == fast){
                ListNode f = fast;
                LIstNode h = head;
                while(f != h){
                    f = f.next;
                    h = h.next;
                }
                return f;
            }
        }
        return null;
    }
}
```

#### [141. Linked List Cycle](https://leetcode-cn.com/problems/linked-list-cycle/) <span style="color:green">Easy</span>

```java
// tow pointers: if two pointers meet, then there must be a cycle
public class Solution {
    public boolean hasCycle(ListNode head) {
        ListNode fast = head;
        ListNode slow = head;
        while(fast != null && fast.next != null){
            fast = fast.next.next;
            slow = slow.next;
            if(fast == slow){
                return true;
            }
        }
        return false;
    }
}
```

## Tree

#### [102. Binary Tree Level Order Traversal ](https://leetcode-cn.com/problems/binary-tree-level-order-traversal/)<span style="color:orange">Medium</span>

BFS: Single Queue(FIFO)

```python
class Solution:
    def levelOrder(self, root):
        results = []
        
        if not root:
            return results
        # initialization: put first level nodes in queue(FIFO)
        queue = collections.deque([root])
        while queue:
            results.append([node.val for node in queue])
            for _ in range(len(queue)):
                # already traversed, so  pop it out
                node = queue.popleft()
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        return results

```

Dummy Node

```python
class Solution:
    def levelOrder(self, root):
        results = []
        if not root:
            return results
        
        queue = collections.deque([root, None])
        level = []
        while queue:
            node = queue.popleft()
            if node is None:
                results.append(level)
                level = []
                if queue:
                    queue.append(None)
                continue
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        return results
```

#### [257. Binary Tree Paths](https://leetcode-cn.com/problems/binary-tree-paths/) <span style="color:green">Easy</span>

Backtracking

```python
class Solution:
    def binaryTreePaths(self, root):
        path, paths = [root], []
        self.findPaths(root, path, paths)
        return paths
	
    def findPaths(self, node, path, paths):
        if node is None:
            return
        # leat node
        if node.left is None and node.right is None:
            paths.append("->".join([str(n.val) for n in path]))
            return
        
        path.append(node.left)
        self.findPaths(node.left, path, paths)
        path.pop()
        
        path.append(node.right)
        self.findPaths(node.right, path, paths)
        path.pop()
```

Divideand conquer 

Whole tree paths = left tree paths + right tree paths

后序遍历：先左右后根节点

```python
class Solution:
    def binaryTreePaths(self, root):
        paths = []
        if root is None:
            return paths
        # leat node
        if root.left is None and root.right is None:
            return [str(root.val)]
        
        for path in self.binaryTreePaths(root.left):
            paths.append(str(root.val) + "->" + path)
        for path in self.binaryTreePaths(root.right):
            paths.append(str(root.val) + "->" + path)
            
        return paths
```

#### [110. Balanced Binary Tree](https://leetcode-cn.com/problems/balanced-binary-tree/) <span style="color:green">Easy</span>

Divide and conquer

```python
class Solution:
    def isBalanced(self, root):
        isBalanced, _ = self.detectBalanced(root)
        return isBalanced
    
    def detectBalanced(self, root):
        if not root:
            return True, 0
        
        is_left_balanced, left_height = self.detectBalanced(root.left)
        is_right_balanced, right_height = self.detectBalanced(root.right)
        root_height = max(left_height, right_height) + 1
        
        if not is_left_balanced or not is_right_balanced:
            return False, root_height
        if abs(left_height - right_height) > 1:
            return False, root_height
        return True, root_height
```

#### [173. Binary Search Tree Iterator](https://leetcode-cn.com/problems/binary-search-tree-iterator/) <span style="color:orange">Medium</span>

Without recursion, user has to use a stack to keep running paths and control backtracking. 

```java
class BSTIterator {
    private Stack<TreeNode> stack = new Stack<>();

    public BSTIterator(TreeNode root) {
        // find the minimum point and make it the starting point by putting it on the stack top
        // the next node will always be on the stack top
        while(root != null) {
            stack.push(root);
            root = root.left;
        }
    }
    
    // inorder -> left, root, right
    public int next() {
        // peek to get the next instad of poppping because the stack has to be re-arranged for next iteration
        TreeNode current = stack.peek();
        TreeNode node = current;

        // if current node has a right node, then get the least possible node on its right tree
        if(node.right != null) {
            // as it has a right sub-tree, the next to be returned will exist there
            // current node will be used for later backtracking, don't pop it
            node = node.right;
            // looks like another initialization
            while(node != null) {
                stack.push(node);
                node = node.left;
            }
        // if current node has no right node, return 
        } else {
            // current node is the end of the sub-tree we are travasing here 
            // it will not be used again, so pop it
            stack.pop();
            // backtrack by removing current sub-tree and moving up
            while (this.hasNext() && stack.peek().right == node) {
                node = stack.pop();
            }
        }
        return current.val;
    }
    
    public boolean hasNext() {
        return !stack.isEmpty();
    }
}
```

A simpler implementaion: only unused nodes are put in the stack

```python
class BSTIterator:
    def __init__(self, root: Optional[TreeNode]):
        self.stack = []
        self.stack_from_most_left(root)
    
    # as this code snippet will be reused, make it a function
    def stack_from_most_left(self, node):
        while node:
            self.stack.append(node)
            node = node.left

    def next(self) -> int:
        node = self.stack.pop()
        if node.right:
            self.stack_from_most_left(node.right)
        return node.val

    def hasNext(self) -> bool:
        return bool(self.stack)
```

## Hash Table

#### [242. Valid Anagram](https://leetcode-cn.com/problems/valid-anagram/) <span style="color:green">Easy</span>

Easy! FIgured it out right away! Now I see I do make progress.

```java
class Solution {
    public boolean isAnagram(String s, String t) {
        // if lengths are equal, they cannot be anagrams
        // change not-equal to less-than and you have the solution to problem 383
        if(s.length() != t.length())
            return false;
        
        int[] count = new int[26];
        for(int i = 0; i < s.length(); i++){
            count[s.charAt(i) - 'a']++;
        }
        for(int j = 0; j < t.length(); j++){
            // improvement: there is no need to walk through count another time. Comparing while deducting.
            if(--count[t.charAt(j) - 'a'] < 0)
                return false;
        }
       
        return true;
    }
}
```

```python
	return sorted(s) == sorted(t)
```

#### [383. Ransom Note](https://leetcode-cn.com/problems/ransom-note/) <span style="color:green">Easy</span> 

Solution almost identical as that to problem 242. 

```python
# another solution
a = Counter(ransomNote) 
b = Counter(magazine)
# get the intersection set
return (a & b) == a
```

#### [49. Group Anagrams](https://leetcode-cn.com/problems/group-anagrams/) <span style="color:orange">Medium</span>

```python
# Passs, but running is slow
class Solution(object):
    def groupAnagrams(self, strs):
        """
        :type strs: List[str]
        :rtype: List[List[str]]
        """
        sorts = []
        result = []
        for s in strs:
            _ = sorted(s)
            if _ not in sorts:
                sorts.append(_)
                result.append([])
            result[sorts.index(_)].append(s)
        return result
        
```

```java
// java 8 new API stream， used to organize collections
class Solution {
    public List<List<String>> groupAnagrams(String[] strs) {
        return new ArrayList<>(Arrays.stream(strs)
            .collect(
                Collectors.groupingBy(str -> {
                    char[] array = str.toCharArray();
                    Arrays.sort(array);
                    return new String(array);
                    })
            )
            .values()
        );
    }
}

```

```java
// hashmap
class Solution{
    public List<List<String>> groupAnagrams(String[] strs) {
        int length = strs.length;
        // use hash map to group results
        Map<String, List<String>> map = new HashMap<>();
        for(String str: strs){
            char[] chars = str.toCharArray();
            Arrays.sort(chars);
            // used sorted string as the key
            String key = String.valueOf(chars);
            // if not contained in the map, create a empty array to as a holder
            if(!map.containsKey(key)){
                map.put(key, new ArrayList<>());
            }
            map.get(key).add(str);
        }
        return new ArrayList<>(map.values());
    }
}
```

#### [438. Find All Anagrams in a String](https://leetcode-cn.com/problems/find-all-anagrams-in-a-string/) <span style="color:orange">Medium</span>

```java
// A, but running is slow
class Solution {
    public List<Integer> findAnagrams(String s, String p) {
        int sLen = s.length();
        int pLen = p.length();
        char[] chars = p.toCharArray();
        Arrays.sort(chars);
        ArrayList result = new ArrayList();
        for(int i = 0;i <= sLen - pLen; i++){
            char[] subs = s.substring(i, i + pLen).toCharArray();
            Arrays.sort(subs);
            if(Arrays.equals(chars, subs)){
                result.add(i);
            }
        }
        return result;
    }
}
```

#### [349. Intersection of Two Arrays](https://leetcode-cn.com/problems/intersection-of-two-arrays/) <span style="color:green">Easy</span>

```python
class Solution(object):
    def intersection(self, nums1, nums2):
        map = {}
        result = []
        for num in nums1:
            if map.get(num):
                map[num] += 1
            else:
                map[num] = 1
        
        for num in nums2:
            if map.get(num) and num not in result:
                result.append(num)
        
        return result
```

#### [350. Intersection of Two Arrays II]( https://leetcode-cn.com/problems/intersection-of-two-arrays-ii/) <span style="color:green">Easy</span>

```python
class Solution(object):
    def intersect(self, nums1, nums2):
        map1 = {}
        map2 = {}
        result = []
        for num in nums1:
            if map1.get(num):
                map1[num] += 1
            else:
                map1[num] = 1

        for num in nums2:
            if map2.get(num):
                map2[num] += 1
            else:
                map2[num] = 1
        
        for key in map1.keys():
            if key in map2.keys():
                result = result + [key] * (map1[key] if map1[key] < map2[key] else map2[key])

        return result
```

#### [202. Happy Number](https://leetcode-cn.com/problems/happy-number/) <span style="color:green">Easy</span>

```python
class Solution(object):
    def isHappy(self, n):
        sums = []
        while n != 1:
            n = self.convert(n)
            if n in sums:
                return False
            sums.append(n)
        return True

    def convert(self, n):
        sum = 0
        for digit in str(n):
            digit = int(digit)
            sum += digit * digit
        return sum
```

#### [1. Two Sum](https://leetcode-cn.com/problems/two-sum/) <span style="color:green">Easy</span> **

```python
class Solution(object):
    def twoSum(self, nums, target):
        for i in range(len(nums)):
            other = target - nums[i]
            if other in nums and i != nums.index(other):
                return [i, nums.index(other)]
        return []
```

```python
class Solution(object):
    def twoSum(self, nums, target):
        hashset = set()
        for num in nums:
            if target - num in hashset:
                return num, target - number
           	hashset.add(num)
        return []
```

```python
# two pointers
class Solution(object):
    def twoSum(self, nums, target):
        if not nums:
            return []
        
        # save index and number at the same time
        # because original index will be lost after sort
        # e.g. [3,3] 6 ans: [0,1] instead of [0,0] 
        num_tuples = [
            (num, index)
            for index, num in enumerate(nums)
        ]
        num_tuples.sort()
        left, right = 0, len(nums) - 1
        while left < right:
            if num_tuples[left][0] + num_tuples[right][0] > target:
                right -= 1
            elif num_tuples[left][0] + num_tuples[right][0] < target:
                left += 1
            else:
                return num_tuples[left][1], num_tuples[right][1]
        return []
```



#### [454. 4Sum II](https://leetcode-cn.com/problems/4sum-ii/) <span style="color:orange">Medium</span>

```python
# official solution
class Solution(object):
    def fourSumCount(self, nums1, nums2, nums3, nums4):
        counter = collections.Counter(i + j for i in nums1 for j in nums2)
        ans = 0
        for i in nums3:
            for j in nums4:
                if -(i+j) in counter:
                    ans += counter[-(i+j)]
        return ans
```

```java
class Solution {
    public int fourSumCount(int[] A, int[] B, int[] C, int[] D) {
        int ans = 0;
		Map<Integer, Integer> counter = new HashMap<Integer, Integer>();
        for(int a: A){
            for(int b: B){
                counter.put(a + b, counter.getOrDefault(a + b, 0) + 1)
            }
        }
        for(int c: C){
            for(int d: D){
                if(counter.containsKey(-c-d)){
                    ans += counter.get(-c-d);
                }
            }
        }
        return ans;
    }
}
```

#### [15. 3Sum](https://leetcode-cn.com/problems/3sum/) <span style="color:orange">Medium</span>

```python
# my solution not accepted because of low efficiency
class Solution(object):
    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        counter = collections.Counter()
        result = []
        for i in range(len(nums) - 1):
            for j in range(i + 1,len(nums)):
                twosum = nums[i] + nums[j] 
                if -twosum in nums and nums.index(-twosum) != i and nums.index(-twosum) != j:
                    candidate = sorted([nums[i], nums[j], - nums[i] - nums[j]])
                    if candidate not in result:
                        result.append(candidate)
        return result
```

```javascript
// two-pointer after sort
var threeSum = function(nums) {
  if (nums.length < 3) {
    return [];
  }
  // 从小到大排序
  const arr = nums.sort((a,b) => a-b);
  // 最小值大于 0 或者 最大值小于 0，说明没有无效答案
  if (arr[0] > 0 || arr[arr.length - 1] < 0) {
    return [];
  }
  const n = arr.length;
  const res = [];
  for (let i = 0; i < n; i ++) {
    // 如果当前值大于 0，和右侧的值再怎么加也不会等于 0，所以直接退出
    if (nums[i] > 0)
      return res;
	  
    // 当前循环的值和上次循环的一样，就跳过，避免重复值
    if (i > 0 && nums[i] === nums[i - 1])
      continue;

    // 双指针
    let l = i + 1;
    let r = n - 1;
    while(l < r) {
      const _sum = nums[i] + nums[l] + nums[r];
      if (_sum > 0)
        r --;
      if (_sum < 0) 
        l ++;
      if (_sum === 0) {
        res.push([nums[i], nums[l], nums[r]]);
        // 跳过重复值
        while(l < r && nums[l] === nums[l + 1])
          l ++;
        // 同上
        while(l < r && nums[r] === nums[r - 1])
          r --;
        l ++;
        r --;
      }
    }
  }
  return res;
};
```

#### [18. 4Sum](https://leetcode-cn.com/problems/4sum/) <span style="color:orange">Medium</span>

```python
# two pointer
class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        nums.sort()
        n = len(nums)
        results = []

        if not nums or n < 4:
            return results
		# stops when c is 1 from the last
        for a in range(n - 3):
            if a > 0 and nums[a] == nums[a - 1]:
                continue
            for b in range(a + 1, n - 2):
                if b > a + 1 and nums[b] == nums[b - 1]:
                    continue
                c = b + 1
                d = n - 1
                while c < d:
                    sum = nums[a]+nums[b]+nums[c]+nums[d]
                    if sum < target:
                        c += 1
                    elif sum > target:
                        d -= 1
                    else:
                        results.append([nums[a], nums[b], nums[c], nums[d]])
                        while c < d and nums[c + 1] == nums[c]:
                            c += 1
                        while c < d and nums[d - 1] == nums[d]:
                            d -= 1
                        c += 1
                        d -= 1
        return results
```

```java
// backtrack and dfs
class Solution {
    List<List<Integer>> ans = new ArrayList<>();
    List<Integer> list = new ArrayList<>();
    int cur = 0;
    public List<List<Integer>> fourSum(int[] nums, int target) {
        Arrays.sort(nums);
        dfs(nums, target,0);
        return ans;
    }
    void dfs(int[] nums, int target, int begin){
		// condition where recursion stops
        if(list.size() == 4){
            if(cur == target){
                ans.add(new ArrayList<>(list));
            }
            return;
        }

        for(int i = begin; i < nums.length; i++ ){
            // cutting branches
			// remaining candidates are not enough
			if(nums.length - i  < 4 - list.size())
				return;
			// current item is equal to the previous one
            if(begin != i && nums[i - 1] == nums[i])
				continue;
			// current sum plus the least possible combination of remaining values is still larger than target   
            if(i < nums.length - 1 && cur + nums[i] > target - (3 - list.size()) * nums[i + 1])		
                return;
			// current sum plus the largest possible combination of remaining values is still less then target
            if(i < nums.length - 1 && cur + nums[i] < target - (3 - list.size()) * nums[nums.length - 1])
				continue;
			
            cur += nums[i];
            list.add(nums[i]);
            dfs(nums, target, i + 1);
            list.remove(list.size() - 1);
            cur -= nums[i];
        }
    }
}
```

## String

#### [344. Reverse String](https://leetcode-cn.com/problems/reverse-string/) <span style="color:green">Easy</span>

#### [541. Reverse String II](https://leetcode-cn.com/problems/reverse-string-ii/) <span style="color:green">Easy</span>

#### [151. Reverse Words in a String](https://leetcode-cn.com/problems/reverse-words-in-a-string/) <span style="color:orange">Medium</span>

#### [28. Implement strStr()](https://leetcode-cn.com/problems/implement-strstr/) <span style="color:green">Easy</span>

```java
// Brute force n**2
public int strStr(String source, String target){
    if (target == null || target.equals(""))
        return -1;
    
    // limit possible start points
    for(int i = 0; i < source.length() - target.length() + 1; i++){
        // default flag is true, only set it to false when unequal chars are found
        boolean equal = true;
        for(int j = 0; j < target.length(); j++){
            if(source.charAt(i + j) != target.charAt(j)){
                equal = false;
                break;
            }
        }
        
        if(equal)
            return i;
    }
    
    return -1;
}
```

Rabin-Karp: use hash code to speed comparisons

<img src="https://s2.loli.net/2021/12/19/YLfja3DWJAXv4nQ.png" alt="image-20211219225340858" style="zoom: 33%;" /> 

```java
public int BASE = 1000000;
public int strStr(String source, String target){
	if (source == null || target == null || target.equals(""))
        return -1;
    
    int targetLen = target.length();
    int sourceLen = source.length();
    if(sourceLen < targetLen)
        return -1;
    
    int power = 1;
    for(int i = 0; i < targetLen; i++)
        power = (power * 31) % BASE;
    
    // map target string to hash code 
    int targetHashed = 0;
    for(int i = 0; i < targetLen; i++)
        targetHashed = (targetHashed * 31 + target.charAt(i)) % BASE;
    
    int hashed = 0;
    for(int i = 0; i < sourceLen; i++){
        hashed = (hashed * 31 + source.charAt(i)) % BASE;
        // current length is less than target length, hashing continues
        if(i < targetLen - 1)
            continue;
        
        // current length longer than target length, remove the starting point from current window
        if(i >= targetLen){
            hashed = hashed - (source.charAt(i - targetLen) * power) % BASE;
            if(hashed < 0)
                hashed += BASE;
        }
        
        // double check is required in case of hash collision
        if(hashed == targetHashed){
			if(source.substring(i - targetLen + 1, i + 1).equals(target))
                return i - targetLen + 1;
        }
    }
    
    return -1;
}
```

#### [459. Repeated Substring Pattern](https://leetcode-cn.com/problems/repeated-substring-pattern/) <span style="color:green">Easy</span>

#### [5. Longest Palindromic Substring](https://leetcode-cn.com/problems/longest-palindromic-substring/) <span style="color:orange">Medium</span>

```python
# brute force with nested loop
class Solution:
    def longestPalindrome(self, s: str) -> str:
        if not s:
            return None

        length = len(s)
        # possible substring length from full length to 1 
        for possibleLength in range(length, 0, -1):
            for start in range(length - possibleLength + 1):
                if self.is_palindrome(s, start, start + possibleLength - 1):
                    return s[start : start + possibleLength]
        return ""
    
    def is_palindrome(self, s, left, right):
        while left < right and s[left] == s[right]:
            left += 1
            right -= 1
        return left >= right

```

Expanding from a middle line

<img src="https://s2.loli.net/2021/12/20/ZB91efyuRP5TiU8.png" alt="image-20211220220032016" style="zoom:33%;" /> 

```java
// build a palindrome from the bottom up instead of checking every possibility 
public String longestPalindrome(String s){
    if(s == null)
        return null;
    
    String longest = "";
    for(int i = 0; i < s.length(); i++){
        String oddPalindrome = getPalindromeFrom(s, i, i);
        if(longest.length() < oddPalindrome.length())
            longest = oddPalindrome;
        
        String evenPalindrome = getPalindromeFrom(s, i, i + 1);
        if(longest.length() < evenPalindrome.length())
            longest = evenPalindrome;
    }
    return longest;
}

private String getPalindromeFrom(String s, int left, int right){
    while(left >= 0 && right < s.length()){
        if(s.charAt(left) != s.charAt(right))
            break;
        left--;
        right++;
    }
    return s.substring(left + 1, right);
}
```

Dynamic Programming

s[i : j] is a palindrome only if 

- s[i + 1 : j - 1] is a palindome

- s[i] = s[j]

```java
public String longestPalindrome(String s){
    if(s == null && s.equals(""))
        return "";
    
    int n = s.length();
    boolean[][] isPalindrome = new boolean[n][n];
   	
    // matrix initialization
    int longest = 1, start = 0;
    for(int i = 0; i < n; i++){
    	// base case: a single char is always a palindrome
        isPalindrome[i][i] = true;
        if(i < n - 1){
            // base case: two chars as a palindrome if s[i] == s[i + 1]
            isPalindrome[i][i + 1] = s.charAt(i) == s.charAt(i + 1);
            if(isPalindrome[i][i + 1]){
                start = i;
                longest =2;
            }
        }
    }
    // since whether s[i : j] is a palindrome depends on s[i + 1: j - 1], start the iteration from i = n - 1
    for(int i = n - 1;i >= 0; i--){
        // two chars are treated as a base case so j starts from i + 2
        for(int j = i + 2; j < n; j++){
            isPalindrome[i][j] = isPalindrome[i + 1][j - 1] && s.charAt(i) == s.charAt(j);
            if(isPalindrome[i][j] && j - i + 1 > longest){
                start = i;
                longest = j - i + 1;
            }
        }
    }
    
    return s.substring(start, start + longest);
}
```

#### [125. Valid Palindrome](https://leetcode-cn.com/problems/valid-palindrome/) <span style="color:green">Easy</span>

```python
class Solution:
    def isPalindrome(self, s):
        left, right = 0, len(s) - 1
        while left < right:
            while left < right and not self.is_valid(s[left]):
                left += 1
            while left < right and not self.is_valid(s[right]):
                right -= 1
            if left < right and s[left].lower() != s[right].lower():
                return False
            left += 1
            right -= 1
        return True
    def is_valid(self, char):
        return char.isdigit() or char.isalpha()
```

#### [680. Valid Palindrome II](https://leetcode-cn.com/problems/valid-palindrome-ii/) <span style="color:green">Easy</span>

```java
class Solution {
    class Pair {
        int left, right;
        public Pair(int left, int right){
            this.left = left;
            this.right = right;
        }
    }

    public boolean validPalindrome(String s) {
        if(s == null)
            return false;

        Pair pair = findDIfference(s, 0, s.length() - 1);
        if(pair.left >= pair.right)
            return true;

        return isPalindrome(s, pair.left + 1, pair.right) || isPalindrome(s, pair.left, pair.right - 1);   
    }

	private Pair findDIfference(String s, int left, int right){
        while(left < right && s.charAt(left) == s.charAt(right)){
            left++;
            right--;
        }
        return new Pair(left, right);
    }
    
    private boolean isPalindrome(String s, int left, int right) {
        Pair pair = findDIfference(s, left, right);
        return pair.left >= pair.right;
    }
}
```

## Recursion and Iteration

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

