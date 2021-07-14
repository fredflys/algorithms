## Array

26 Remove Duplicates

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

293 Move Zeros
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

844 Backspace String Compare
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

977 Squares of a Sorted Array

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

209 Minimum Size Subarray Sum

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

904 Fruits into Baskets

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

76 Minimum Window Substring Hard
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
