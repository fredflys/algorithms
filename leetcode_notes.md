## Array

26 Remove Duplicates <span style="color:green">Easy</span>

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

293 Move Zeros <span style="color:green">Easy</span>
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

844 Backspace String Compare <span style="color:green">Easy</span>
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

977 Squares of a Sorted Array <span style="color:green">Easy</span>

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

209 Minimum Size Subarray Sum <span style="color:orange">Medium</span>

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

904 Fruits into Baskets <span style="color:orange">Medium</span>

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

76 Minimum Window Substring <span style="color:red">Hard</span>
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

1248 Count Number of Nice Subarrays <span style="color:orange">Medium</span>
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

[930 Binary Subarrays With Sum](https://leetcode-cn.com/problems/binary-subarrays-with-sum/) <span style="color:orange">Medium</span>

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

[1. Two Sum](https://leetcode-cn.com/problems/two-sum/) <span style="color:green">Easy</span>

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

[992. Subarrays with K Different Integers](https://leetcode-cn.com/problems/subarrays-with-k-different-integers/) <span style="color:red">Hard</span>

恰好由k个不同整数组成的子序列个数 = **最多**由k个不同整数组成的子序列个数 - **最多**由k - 1个不同整数组成的子序列个数   

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
            
            # right  moves forward
            right += 1
        return result
```

