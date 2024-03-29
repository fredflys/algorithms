## Array

- two pointers: no extra space needed
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
longest repeating -> least replacements -> replace infrequent letters with the most frequent letter

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

#### [11. Container With Most Water](https://leetcode.com/problems/container-with-most-water/description/) Medium

naive brute force, tle

```java
class Solution {
    public int maxArea(int[] height) {
        int maxArea = 0;
        int n = height.length;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                int h = Math.min(height[i], height[j]);
                int w = j - i;
                maxArea = Math.max(maxArea, h * w);
            }
        }

        return maxArea;
    }
}
```

two pointers

```java
class Solution {
    public int maxArea(int[] height) {
        int maxArea = 0;
        int n = height.length;
        int l = 0, r = n - 1;
        int h, w;
        while (l < r) {
            w = r - l;
            h = Math.min(height[l], height[r]);
            maxArea = Math.max(maxArea, w * h);
            // only move pointer when there a chance of having a bigger area
            if (height[l] < height[r]) {
                l++;
            } else {
                r--;
            }
        }

        return maxArea;
    }
}
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

sliding window, two pointers
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

#### [438. Find All Anagrams in a String](https://leetcode.com/problems/find-all-anagrams-in-a-string/) Medium

sliding window

```java
class Solution {
    public List<Integer> findAnagrams(String s, String p) {
        List<Integer> res = new LinkedList<>();

        int[] target = new int[26];
        int[] window = new int[26];
        for (char c : p.toCharArray()) {
            target[c - 'a']++;
        }

        int l = 0, r = 0;
        while (r < s.length()) {
            window[s.charAt(r) - 'a']++;
            r++;

            if (r - l > p.length()) {
                window[s.charAt(l) - 'a']--;
                l++;
            }

            if (r - l == p.length() && check(window, target)) res.add(l);
        }


        return res;
    }

    // compare two arrays
    boolean check(int[] window, int[] target) {
        for (int i = 0; i < 26; i++) {
            if (window[i] != target[i]) return false;
        }
        return true;
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

#### [3. Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/) Medium

sliding window
Consider:

1. when to move right poninter and what needs to be updated
2. when to move left pointer and what needs to be updated
3. when to update the result

```java
class Solution {
    public int lengthOfLongestSubstring(String s) {
        if (s.length() == 0) {
            return 0;
        }

        int res = 0;
        int[] window = new int[128];
        int l = 0, r = 0;
        while (r < s.length()) {
            char rc = s.charAt(r);
            window[rc]++;
            r++;

            while (window[rc] > 1) {
                char lc = s.charAt(l);
                l++;
                window[lc]--;
            }

            res = Math.max(res, r - l);
        }

        return res;
    }
}
```

#### [187. Repeated DNA Sequences](https://leetcode.com/problems/repeated-dna-sequences/) Medium

brute force

```java
class Solution {
    public List<String> findRepeatedDnaSequences(String s) {
        int n = s.length();
        int L = 10;
        HashSet<String> seen = new HashSet<>();
        List<String> res  = new ArrayList<>();
        for (int i = 0; i + L <= n; i++) {
            // [i, i + 10)
            String sub = s.substring(i, i + L);
            if (seen.contains(sub)) {
                if (!res.contains(sub)) res.add(sub);
            } else {
                seen.add(sub);
            }
        }

        return res;
    }
}
```

sliding window

```java
class Solution {
    public List<String> findRepeatedDnaSequences(String s) {
        int n = s.length();
        int L = 10;
        HashSet<String> seen = new HashSet<>();
        List<String> res  = new ArrayList<>();
        int l = 0, r = 0;
        StringBuilder window = new StringBuilder();
        String windowStr;
        while (r < s.length()) {
            window.append(s.charAt(r));
            r++;

            if (r - l == L) {
                windowStr = window.toString();
                if (seen.contains(windowStr)) {
                    if (!res.contains(windowStr)) res.add(windowStr);
                } else {
                    seen.add(windowStr);
                }
                window.deleteCharAt(0);
                l++;
            }
        }

        return res;
    }
}
```

sliding window hash - variation of Rabin-Karp text pattern algorithm
Convert text information to numbers, which only can perform digit removal and adding in O(1), instead of O(L).

```java
class Solution {
    public List<String> findRepeatedDnaSequences(String s) {
        int[] nums = convertToQuaternay(s);
        int L = 10;
        int R = 4;
        int RL = (int) Math.pow(R, L - 1);

        HashSet<Integer> seen = new HashSet<>();
        List<String> res = new LinkedList<>();

        int windowHash = 0;
        int l = 0, r = 0;
        while (r < nums.length) {
            // append a digit to the right
            /*
            8324
            res = 0
            8   -> res = res * 10 + nums[0] 0 + 8
            83  -> res = res * 10 + nums[1] 80 + 3
            ...
             */
            windowHash = R * windowHash + nums[r];
            r++;

            if (r - l == L) {
                if (seen.contains(windowHash)) {
                    String sub = s.substring(l, r);
                    if (!res.contains(sub)) res.add(sub);
                } else {
                    seen.add(windowHash);
                }

                // remove the leftmost digit
                /*
                8324
                res = 8324
                324 -> res - nums[i] * 10^(4 - 1)
                 */
                windowHash = windowHash - nums[l] * RL;
                l++;
            }
        }

        return res;
    }

    // conver s to a number in base 4
    int[] convertToQuaternay(String s) {
        int[] nums = new int[s.length()];
        for (int i = 0; i < nums.length; i++) {
            switch (s.charAt(i)) {
                case 'A' : nums[i] = 0; break;
                case 'C' : nums[i] = 1; break;
                case 'G' : nums[i] = 2; break;
                case 'T' : nums[i] = 3; break;
            }
        }

        return nums;
    }
}
```

#### [992. Subarrays with K Different Integers](https://leetcode.com/problems/subarrays-with-k-different-integers/) Hard

sliding window
if there are exactly k distinct integers between [i, j], there are at least k distinct integers between [i, j + 1]
if there are exactly k distinct integers between [i, j], there are at most k distinct integers between [i + 1, j]
For every possible position for i, a total amount of subarrays with at least k distinct integers can be counted and and summed up, which includes k subarrays, k + 1 subarrays and so forth. By the same logic, a total amount of subarrays with at least k + 1 distinct integers can also be calculated, which includes k + 1 subarrays, k + 2 subarrays and so forth. Subtract latter from former will yield the desired result.
e.g.

a b a c b d k = 3
i j
for i, there are 6 - 4 + 1 subarrays where at least k distinct integers exist, i.e. [i, j - 1], [i, j], [i, j + 1]

```java
class Solution {
    public int subarraysWithKDistinct(int[] nums, int k) {
        return countSubWithLeastKInt(nums, k) - countSubWithLeastKInt(nums, k + 1);
    }

    int countSubWithLeastKInt(int[] nums, int k) {
        int[] window = new int[nums.length + 1];
        int res = 0;
        for (int i = 0, j = 0, distinctInts = 0; i < nums.length; i++) {
            // stretch the window by incrementing j
            // a new interger is found
            for (; distinctInts < k && j < nums.length; j++) {
                if (++window[nums[j]] == 1) {
                    ++distinctInts;
                }
            }

            if (distinctInts < k) break;

            // now outside the inner loop, j already is incremented to a position where [i, j] has **at least** k distinct integers
            res += nums.length - j + 1;

            // shrink the window by decrementing i
            // prepare for the next iteration as i will be incremented
            if (--window[nums[i]] == 0) {
                --distinctInts;
            }
        }

        return res;
    }
}
```

### [1456. Maximum Number of Vowels in a Substring of Given Length](https://leetcode.com/problems/maximum-number-of-vowels-in-a-substring-of-given-length/description/) Medium

naive brute force, will cause TLE

```java
class Solution {
    public int maxVowels(String s, int k) {
        int maxVowls = 0;
        int n = s.length();
        for (int i = 0; i + k <= n; i++) {
            int cnt = 0;
            for (int j = 0; j < k; j++) {
                if (isVowl(s, i + j)) cnt++;
            }
            if (cnt > maxVowls) maxVowls = cnt;
        }

        return maxVowls;
    }

    private boolean isVowl(String s, int index) {
        if (s.charAt(index) == 'a' || s.charAt(index) == 'e' || s.charAt(index) == 'i' || s.charAt(index) == 'o' || s.charAt(index) == 'u') {
            return true;
        }
        return false;
    }
}
```

sliding window

```java
class Solution {
    public int maxVowels(String s, int k) {
        int res = 0;
        int n = s.length();
        int l = 0, r = 0;

        int vowls = 0;
        // [l, r] closed interval on both ends
        while (r < n) {
            if (isVowl(s, r)) vowls++;
            if (r - l + 1 > k) {
                if (isVowl(s, l)) vowls--;
                l++;
            }

            if (vowls > res) res = vowls;
            r++;
        }

        return res;
    }

    private boolean isVowl(String s, int index) {
        if (s.charAt(index) == 'a' || s.charAt(index) == 'e' || s.charAt(index) == 'i' || s.charAt(index) == 'o' || s.charAt(index) == 'u') {
            return true;
        }
        return false;
    }
}
```

#### [2379. Minimum Recolors to Get K Consecutive Black Blocks](https://leetcode.com/problems/minimum-recolors-to-get-k-consecutive-black-blocks/description/) Easy

sliding window

```java
class Solution {
    public int minimumRecolors(String blocks, int k) {
        int n = blocks.length();
        int r = 0, l = 0, cnt = 0;
        int res = n;
        while (r < n) {
            if (recolorNeeded(blocks, r)) cnt++;
            if (r - l + 1 == k) {
                res = Math.min(cnt, res);
                if (recolorNeeded(blocks, l)) cnt--;
                l++;
            }
            r++;
        }
        return res;
    }

    private boolean recolorNeeded(String blocks, int index) {
        return blocks.charAt(index) == 'W';
    }
}
```

Two Pointers

#### [870. Advantage Shuffle](https://leetcode.com/problems/advantage-shuffle/) Medium

two pointers

```java
class Solution {
    public int[] advantageCount(int[] nums1, int[] nums2) {
        int n = nums1.length;
        // max heap: used to produce currently largest element in nums2
        PriorityQueue<int[]> candidates = new PriorityQueue<>(
            (int[] a, int[] b) -> {
                return b[1] - a[1];
            }
        );
        for (int i = 0; i < n; i++) {
            candidates.offer(new int[]{i, nums2[i]});
        }

        Arrays.sort(nums1);

        int[] res = new int[n];
        int l = 0, r = n - 1;
        int[] candidate;
        int index, currentMax;
        while (!candidates.isEmpty()) {
            candidate = candidates.poll();
            index = candidate[0];
            currentMax = candidate[1];
            if (nums1[r] > currentMax) {
                res[index] = nums1[r];
                r--;
            } else {
                res[index] = nums1[l];
                l++;
            }
        }

        return res;
    }
}
```

#### [15. 3Sum](https://leetcode.com/problems/3sum/) Medium

Two pointers, dimension reduction
This problem is turned into a two sum one with an outer loop.

```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums = sorted(nums)
        n = len(nums)
        results = []

        # dimension reduction
        # change 3 sum problem to a two sum one with a loop
        for i in range(n):
            # avoid duplicates
            #
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            self.find_two_sums(nums, i + 1, -nums[i], results)
        return results

    def find_two_sums(self, nums, left, target, results):
        right = len(nums) - 1
        last_pair = None
        while left < right:
            two_sum = nums[left] + nums[right]
            if two_sum == target:
                # avoid duplicates: if the pair is seen, never add it to the results
                if (nums[left], nums[right]) != last_pair:
                    results.append([-target, nums[left], nums[right]])
                last_pair = (nums[left], nums[right])
                right -= 1
                left += 1
            elif two_sum > target:
                right -= 1
            else:
                left += 1
```

#### [16. 3Sum Closest](https://leetcode.com/problems/3sum-closest/description/) Medium

two pointers

```java
class Solution {
    public int threeSumClosest(int[] nums, int target) {
        Arrays.sort(nums);
        int n = nums.length;
        int delta = Integer.MAX_VALUE;
        for (int i = 0; i < n - 2; i++) {
            int sum = nums[i] + twoSumCloest(nums, i + 1, target - nums[i]);
            if (Math.abs(delta) > Math.abs(target - sum)) {
                delta = target - sum;
            }
        }
        return target - delta;
    }

    private int twoSumCloest(int[] nums, int start, int target) {
        int l = start, r = nums.length - 1;
        int delta = Integer.MAX_VALUE;
        while (l < r) {
            int sum = nums[l] + nums[r];
            if (Math.abs(delta) > Math.abs(target - sum)) {
                delta = target - sum;
            }

            if (sum < target) {
                l++;
            } else {
                r--;
            }
        }
        return target - delta;
    }
}
```

#### [611. Valid Triangle Number](https://leetcode.com/problems/valid-triangle-number/) Medium

a triangle definition: a <= b <= c and a + b < c
two pointers

```python
class Solution:
    def triangleNumber(self, nums: List[int]) -> int:
        nums = sorted(nums)
        n = len(nums)
        answer = 0

        for i in range(n):
            left = 0
            right = i - 1
            while left < right:
                if nums[left] + nums[right] > nums[i]:
                    answer += right - left
                    right -= 1
                else:
                    left += 1

        return answer
```

#### [75. Sort Colors](https://leetcode.com/problems/sort-colors/submissions/) Medium

partition, two pointers

```python
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        n = len(nums)
        index = 0
        left = 0
        right = n - 1
        #
        # loop invariant
        # - all in [0, left) == 0
        # - all in [left, index) == 1
        # - all in [index, right] unknown
        # starting from empty partitions, gradually enlarge each partition by moving pointers left or right
        # - all in (right, n - 1] == 2
        while index <= right:
            # find 1
            if nums[index] == 1:
                index += 1
            # find 0
            elif nums[index] < 1:
                nums[index], nums[left] = nums[left], nums[index]
                left += 1
                index += 1
            # find 2
            else:
                nums[index], nums[right] = nums[right], nums[index]
                # after the swap, nums[index] is not known, so index will not be incremented
                # so that in the next loop nums[index] will be dealt with
                right -= 1
```

for loop

```java
class Solution {
    public void sortColors(int[] nums) {
        /*
        0s : [0, i)
        1: [i, j)
        unknown: [j, k]
        2: (k, nums.length - 1]
        */

        for (int i = 0, j = 0, k = nums.length - 1; j <= k; j++) {
            if (nums[j] == 0) {
                // swap nums[j] and nums[i]
                // 0 zone is larger now
                nums[j] = nums[i];
                nums[i++] = 0;
            } else if (nums[j] == 2) {
                // nums[k] is moved to position j
                // nums[k] is still unknown so j needs to be moved backward
                // so that it will be visited in the next iteration
                nums[j--] = nums[k];
                nums[k--] = 2;
            }
        }
    }
}
```

#### [LintCode 143 · Sort Colors II](https://www.lintcode.com/problem/143/) Medium

partition, merge sort, quick sort, two pointers
nlogk

```python
from typing import (
    List,
)

class Solution:
    """
    @param colors: A list of integer
    @param k: An integer
    @return: nothing
    """
    def sort_colors2(self, colors: List[int], k: int):
        self.sort(colors, 1, k, 0, len(colors) - 1)

    def sort(self, colors, color_from, color_to, index_from, index_to):
        # no colors to sort or space to sort from
        if color_from == color_to or index_from == index_to:
            return

        middle_color = (color_from + color_to) // 2
        left, right = index_from, index_to
        # partition
        while left <= right:
            while left <= right and colors[left] <= middle_color:
                left += 1
            while left <= right and colors[right] > middle_color:
                right -= 1
            if left <= right:
                colors[left], colors[right] = colors[right], colors[left]
                left += 1
                right -= 1
        # merge
        self.sort(colors, color_from, middle_color, index_from, index_to)
        self.sort(colors, middle_color + 1, color_to, index_from, index_to)
```

#### [992. Subarrays with K Different Integers](https://leetcode-cn.com/problems/subarrays-with-k-different-integers/) <span style="color:red">Hard</span>

two pointers
**恰好**由 k 个不同整数组成的子序列个数 = **最多**由 k 个不同整数组成的子序列个数 - **最多**由 k - 1 个不同整数组成的子序列个数

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

求连续子数组之和至少为 k 的最短长度，既然有和，就可以想到使用前缀和数组，问题可转化为连续子数组之和为 prefix_sums[right] - prefix_sums[left] >= K 时，right - left 的最小值。如果以暴力求解，可以想到每一个点都要当作 right，也要被当作 left，即内外双层循环。如此方法，自然费时。有没有办法可以减少循环的次数呢？此问题的难点在于数组中可以有负数，前缀和数组自然就不一定是单调增加。从这一点出发，前缀和数组中就可能出现这样的情况：有 i < j 使得 prefix_sums[i] >= prefix_sums[j]，也就是说 i 到 j 这一段的和小于 0，往后再延长，长度变长自不必说，结果反而比起不带 i 到 j 这一段变小了。那么，i 作为 left 自然不可能是最后答案。而假如我们已经找到了 prefix_sums[j] - prefix_sums[i] >= K，j 再往后走也没有意义，因为无论之后是正数还是负数，长度都变长了，即便符合条件，也不是我们要找的最后答案。因此，一旦遇到一个符合条件的 left（即此刻的 i），就意味着可以求出一个可能的结果，不必再考虑该点作为左侧起始点的情况。

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

#### [54. Spiral Matrix](https://leetcode.com/problems/spiral-matrix/) Medium

matrix, multi-dimension array

```java
class Solution {
    public List<Integer> spiralOrder(int[][] matrix) {
        int m = .length, n = matrix[0].length;
        int upperbound = 0, rightbound = n - 1, leftbound = 0, lowerbound = m - 1;
        List<Integer> res = new LinkedList<>();

        while (res.size() < m * n) {
            // from left to right
            // a row on the top is traversed
            if (res.size() < m * n && upperbound <= lowerbound) {
                traverseRow(matrix, res, upperbound, leftbound, rightbound);
                upperbound++;
            }


            // from top to bottom
            // a column on the rightest is traversed
            if (res.size() < m * n && rightbound >= leftbound) {
                traverseCol(matrix, res, rightbound, upperbound, lowerbound);
                rightbound--;
            }

            // from right to left
            // a bottom row is traversed
            if (res.size() < m * n && lowerbound >= upperbound) {
                traverseRow(matrix, res, lowerbound, rightbound, leftbound);
                lowerbound--;
            }

            // from bottom to top
            // a column on the leftest is traversed
            if (res.size() < m * n && leftbound <= rightbound) {
                traverseCol(matrix, res, leftbound, lowerbound, upperbound);
                leftbound++;
            }
        }

        return res;
    }

    void traverseRow(int[][] matrix, List<Integer> res, int row, int start, int end) {
        if (start < end) {
            for (int i = start; i <= end; i++) {
                res.add(matrix[row][i]);
            }
            return;
        }

        for (int i = start; i >= end; i--) {
            res.add(matrix[row][i]);
        }
    }

    void traverseCol(int[][] matrix, List<Integer> res, int col, int start, int end) {
        if (start < end) {
            for (int i = start; i <= end; i++) {
                res.add(matrix[i][col]);
            }
            return;
        }


        for (int i = start; i >= end; i--) {
            res.add(matrix[i][col]);
        }
    }
}
```

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

#### [766. Toeplitz Matrix](https://leetcode.com/problems/toeplitz-matrix/) Easy

```java
class Solution {
    public boolean isToeplitzMatrix(int[][] matrix) {
        int base;
        int m = matrix.length;
        int n = matrix[0].length;
        for (int i = 1 ; i < m; i++) {
            for (int j = 1; j < n; j++) {
                if (matrix[i][j] != matrix[i - 1][j - 1]) return false;
            }
        }
        return true;
    }
}
```

#### [1504. Count Submatrices With All Ones](https://leetcode.com/problems/count-submatrices-with-all-ones/) Medium

Kind of know how it works but still not fully understand.

```java
class Solution {
    public int numSubmat(int[][] mat) {
        int m = mat.length;
        int n = mat[0].length;
        int rowConsectiveOne;
        int[][] leftOnes = new int[m][n];
        for (int i = 0; i < m; i++) {
            rowConsectiveOne = 0;
            for (int j = 0; j < n; j++) {
                if (mat[i][j] == 1) rowConsectiveOne++;
                else rowConsectiveOne = 0;
                leftOnes[i][j] = rowConsectiveOne;
            }
        }
        int res = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                int rowMin = leftOnes[i][j];
                for (int k = i; k >= 0; k--) {
                    rowMin = Math.min(rowMin, leftOnes[k][j]);
                    res += rowMin;
                }
            }
        }
        return res;
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

XXXYYYY Binary search is used to find the turning point.

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

#### [183 · Wood Cut - LintCode ](https://www.lintcode.com/problem/183/)<span style="color:orange">Medium</span>

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

#### []() Medium

max of all possible minimum values -> binary search on an answer set
the less tastiness is, the more choices there are and vice versa

```java
class Solution {
    // max of all minimums
    // binary search on an answer set
    public int maximumTastiness(int[] price, int k) {
        Arrays.sort(price);

        // max tastiness is the max difference of the array
        if (k == 2) return price[price.length - 1] - price[0];

        // all candies have the same price so the max tastiness will be 0
        int l = 0, r = price[price.length - 1] - price[0];
        while (l + 1 < r) {
            int mid = l + (r - l) / 2;
            if (guess(price, k, mid)) {
                l = mid;
            } else {
                r = mid;
            }
        }
        return l;
    }

    private boolean guess(int[] price, int k, int tastiness) {
        int candyCnt = 1;
        int candy = price[0];
        for (int i = 1; i < price.length; i++) {
            if (price[i] >= candy + tastiness) {
                candyCnt++;
                candy = price[i];
            }
            if (candyCnt == k) return true;
        }

        return false;
    }
}
```

### [2560. House Robber IV](https://leetcode.com/problems/house-robber-iv/description/) Medium

binary search

```java
import java.util.Arrays;

class Solution {
    public int minCapability(int[] nums, int k) {
        int[] copiedNums = Arrays.copyOf(nums, nums.length);
        Arrays.sort(copiedNums);
        int l = copiedNums[0], r = copiedNums[nums.length - 1];

        if (k == 1) return l;

        while (l < r) {
            int mid = l + (r - l) / 2;
            if (guess(nums, k, mid)) {
                r = mid;
            } else {
                l = mid + 1;
            }
        }

        return l;
    }

    private boolean guess(int[] nums, int k, int cap) {
        int robbed = 0;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] <= cap) {
                robbed++;
                i++;
            }

            if (robbed >= k) return true;
        }

        return false;
    }
}
```

dp and binary search

```java
class Solution {
    public int minCapability(int[] nums, int k) {
        int n = nums.length;
        if (n == 1) {
            return nums[0];
        }
        int l = 0, r = (int) 1e9;
        while (l < r) {
            int mid = l + (r - l) / 2;
            // dp[i] means based on the guessed minimum capability (mid)
            // to steal from the first i-th houses
            // how many houses at least can be robbed
            int[] dp = new int[n];
            // the guessed capability is indeed the maximum
            if (nums[0] <= mid) dp[0] = 1;
            if (Math.min(nums[0], nums[1]) <= mid) {
                dp[1] = 1;
            }

            for (int i = 2; i < n; i++) {
                if (nums[i] > mid) {
                    // i-th house cannot be robbed
                    dp[i] = dp[i - 1];
                } else {
                    // i-th house will be robbed or not
                    dp[i] = Math.max(dp[i - 1], dp[i - 2] + 1);
                }
            }

            if (dp[n - 1] >= k) {
                r = mid;
            } else {
                l = mid + 1;
            }
        }
        return r;
    }
}
```

#### [74. Search a 2D Matrix](https://leetcode.com/problems/search-a-2d-matrix/) Medium

binary search on a sorted input

- find a row: binary search on first column to get the largest point smaller than the target
- find target in the row: binary search on the row

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

Division, minimum, prefix DP
dp[i][j]: the minimum cost for the first i books to be copied by the first j persons
j - 1 j (last copier)  
 i
. . . . . . . | . . .
previous pages remaining pages for last copier

                   cost: sum(pages[prev...i - 1])

cost: max(dp[prev][j - 1])
Since the last copier and other copiers start their work at the same time, whichever group spend the most time will be final result: max(sum(pages[prev...i - 1]), max(dp[prev][j - 1])). Then find the minimum result from all the possibilities.

```python
from typing import (
    List,
)

class Solution:
    """
    @param pages: an array of integers
    @param k: An integer
    @return: an integer
    """
    def copy_books(self, pages: List[int], k: int) -> int:
        # no books or no copiers
        if not pages or not k:
            return 0

        # initialization
        n = len(pages)
        dp = [[float('inf')] * (k + 1) for _ in range(n + 1)]
        # copying a book of 0 pages for whatever copiers costs no time
        for j in range(k + 1):
            dp[0][j] = 0
        # no need to initialize
        # hiring 0 person to copy any amount of books is a task never to be finished

        prev_books = [0] * (n + 1)
        for p in range(1, n + 1):
            # pages of previous i - 1 books plus the pages of current book
            prev_books[p] = prev_books[p - 1] + pages[p - 1]

        # the first i books
        for i in range(1, n + 1):
            # the first j copiers
            for j in range(1, k + 1):
                # all possibile ways to distribute current books to available copiers
                for prev in range(i):
                    final_copier_cost = prev_books[i] - prev_books[prev]
                    dp[i][j] = min(dp[i][j], max(dp[prev][j - 1], final_copier_cost))

        return dp[n][k]
```

#### [69. Sqrt(x)](https://leetcode.com/problems/sqrtx/description/) Easy

binary search on an answer set

```java
class Solution {
    public int mySqrt(int x) {
        long left = 0, right = x;
        while (left + 1 < right) {
            long mid = left + (right - left) / 2;
            if (mid * mid <= x) {
                left = mid;
            } else {
                right = mid;
            }
        }
        return (int) (left - 1);
    }
}
```

#### [1482. Minimum Number of Days to Make m Bouquets](https://leetcode.com/problems/minimum-number-of-days-to-make-m-bouquets/description/) Medium

binary search on an answer set, guess until a right answer is reached

```java
class Solution {
    public int minDays(int[] bloomDay, int m, int k) {
        int flowers = bloomDay.length;
        // not enough flowers to make m bouquets
        if (flowers / k < m) {
            return -1;
        }

        int l = -1, r = 1000000000 + 1;
        while (l + 1 < r) {
            int mid = l + (r - l) / 2, remainingBouquets = m;
            for (int i = 0, flowersInBouquet = 0; remainingBouquets > 0 && i < flowers; i++) {
                // flowers that take less than mid hours to bloom
                // can be in a bouquet
                if (bloomDay[i] <= mid) {
                    if (++flowersInBouquet == k) {
                        flowersInBouquet = 0;
                        --remainingBouquets;
                    }
                } else {
                    // only adjacent flowers can be in a bouquet
                    flowersInBouquet = 0;
                }
            }

            if (remainingBouquets == 0) {
                r = mid;
            } else {
                l = mid;
            }
        }
        return r;
    }
}
```

#### [1552. Magnetic Force Between Two Balls](https://leetcode.com/problems/magnetic-force-between-two-balls/description/) Medium

binary search on an answer set
l r
l r
1 2 3 4..k | k + 1 k + 2 ... n

```java
class Solution {
    public int maxDistance(int[] position, int m) {
        Arrays.sort(position);
        // the minimum answer is 1, which means two balls are placed right next to each other
        // the maximum answer is end - 1, which means one is placed at the start and one is placed at the end
        int l = 0, r = position[position.length - 1];
        while (l + 1 < r) {
            // the first ball is placed at the 1st position already
            int mid = l + (r - l) / 2, remainingBalls = m - 1;
            for (int i = 1, lastBall = position[0]; remainingBalls > 0 && i < position.length; i++) {
                if (position[i] - lastBall >= mid) {
                    --remainingBalls;
                    lastBall = position[i];
                }
            }

            if (remainingBalls == 0) {
                l = mid;
            } else {
                r = mid;
            }
        }
        return l;
    }
}
```

#### [878. Nth Magical Number](https://leetcode.com/problems/nth-magical-number/description/) Hard

binary search on an answer set, numeric theory

```java
class Solution {
    public int nthMagicalNumber(int n, int a, int b) {
        int m = lcm(a, b);

        long l = Math.min(a, b) - 1;
        long r = (long) Math.min(a, b) * n + 1;
        while (l + 1 < r) {
            long mid = l + (r - l) / 2;
            if (mid / a + mid / b - mid / m >= n) {
                r = mid;
            } else {
                l = mid;
            }
        }
        return (int) (r % 1000000007);
    }

    private int lcm(int a, int b) {
        return a / gcd(a, b) * b;
    }

    private int gcd(int a, int b) {
        return b == 0 ? a : gcd(b, a % b);
    }
}
```

#### [719. Find K-th Smallest Pair Distance](https://leetcode.com/problems/find-k-th-smallest-pair-distance/description/) Hard

binary search on an answer set, sliding window

```java
class Solution {
    public int smallestDistancePair(int[] nums, int k) {
        Arrays.sort(nums);
        // distance between any two pairs
        // could be 0 at minimum, longest - shortest at maximum
        int l = 0 - 1, r = nums[nums.length - 1] - nums[0] + 1;
        // given a distance x, count how many pairs that are no more than x from each other
        while (l + 1 < r) {
            int mid = l + (r - l) / 2;
            int pairsCount = 0;
            for (int i = 0, j = 0; pairsCount < k && j < nums.length; j++) {
                for (; nums[j] - nums[i] > mid; i++);
                pairsCount += j - i;
            }

            // too many pairs
            if (pairsCount >= k) {
                r = mid;
            } else {
            // not enough pairs
                l = mid;
            }
        }

        return r;
    }
}
```

#### [1011. Capacity To Ship Packages Within D Days](https://leetcode.com/problems/capacity-to-ship-packages-within-d-days/description/) Medium

binary search on an answer set

```java
class Solution {
    public int shipWithinDays(int[] weights, int days) {
        // min capacity is 1
        // max capacity is package max weight multiplied by packages
        int l = 1 - 1, r = 500 * 5 * 10000 + 1;
        while (l + 1  < r) {
            int mid = l + (r - l) / 2;

            // finished within desired days
            // try to reduce the capacity
            if (isShipped(weights, mid, days)) {
                r = mid;
            } else {
                l = mid;
            }
        }
        return r;
    }

    private boolean isShipped(int[] weights, int capacity, int limitDays) {
        int spent = 0;
        int loaded = 0;

        for (int i = 0; i < weights.length; i++) {
            if (weights[i] > capacity) return false;

            loaded += weights[i];

            if (loaded > capacity) {
                ++spent;
                loaded = weights[i];
            }
        }

        // still needs one more day to ship the last packages
        // after exiting the loop, there is still some loaded weight
        return spent + 1 <= limitDays;
    }
}
```

#### [793. Preimage Size of Factorial Zeroes Function](https://leetcode.com/problems/preimage-size-of-factorial-zeroes-function/description/) Hard

```java
class Solution {
    public int preimageSizeFZF(int k) {
        return (int) (binarySearch(k + 1) - binarySearch(k));
    }

    private long binarySearch(int k) {
        long l = 0 - 1, r = 5L * k + 1;
        while (l + 1 < r) {
            long mid = l + (r - l) / 2;
            if (trailingZeroes(mid) < k) {
                l = mid;
            } else {
                r = mid;
            }
        }
        return l;
    }

    private long trailingZeroes(long n) {
        long res = 0;
        for (long d = n; d / 5 > 0; d /= 5) {
            res += d / 5;
        }
        return res;
    }
}
```

#### [1095. Find in Mountain Array](https://leetcode.com/problems/find-in-mountain-array/description/) Hard

```java
class Solution {
    public int findInMountainArray(int target, MountainArray mountainArr) {
        int peakIndex = findPeak(mountainArr);

        int beforePeakFound = binarySeach(mountainArr, peakIndex, target);
        if (beforePeakFound != -1) return beforePeakFound;

        int afterPeakFound = binarySeachDescending(mountainArr, peakIndex, target);
        if (afterPeakFound != -1) return afterPeakFound;

        return -1;
    }

    int findPeak(MountainArray ma) {
        int l = 0, r = ma.length() - 1;
        while (l + 1 < r) {
            int mid = l + (r - l) / 2;
            // next element > current element
            // still before peak
            if (ma.get(mid) > ma.get(mid - 1)) {
                l = mid;
            } else {
                r = mid;
            }
        }
        return l;
    }

    int binarySeach(MountainArray ma, int peakIndex, int target) {
        int l = 0 - 1, r = peakIndex + 1;
        while (l + 1 < r) {
            int mid = l + (r - l) / 2;
            if (target > ma.get(mid)) {
                l = mid;
            } else if (target < ma.get(mid)) {
                r = mid;
            } else {
                return mid;
            }
        }
        return -1;
    }

    int binarySeachDescending(MountainArray ma, int peakIndex, int target) {
        int l = peakIndex - 1, r = ma.length();
        while (l + 1 < r) {
            int mid = l + (r - l) / 2;
            if (target < ma.get(mid)) {
                l = mid;
            } else if (target > ma.get(mid)) {
                r = mid;
            } else {
                return mid;
            }
        }
        return -1;
    }
}
```

#### [1283. Find the Smallest Divisor Given a Threshold](https://leetcode.com/problems/find-the-smallest-divisor-given-a-threshold/description/) Medium

binary serach on an answer set

```java
class Solution {
    public int smallestDivisor(int[] nums, int threshold) {
        int l = 1 - 1, r = 1000000 + 1;
        while (l + 1 < r) {
            int mid = l + (r - l) / 2;
            int remaining = threshold;
            for (int i = 0; i < nums.length && remaining >= 0; remaining -= (nums[i++] + mid - 1) / mid);
            if (remaining < 0) {
                l = mid;
            } else {
                r = mid;
            }
        }
        return r;
    }
}
```

recursive

```java
class Solution {
    private int search(int[] nums, int l, int r, int threshold) {
        if (l + 1 >= r) {
            return -1;
        }

        int mid = l + (r - l) / 2;
        int remaining = threshold;
        for (int i = 0; i < nums.length && remaining >= 0; remaining -= (nums[i++] + mid - 1) / mid);
        if (remaining < 0) {
            return search(nums, mid, r, threshold);
        }

        int possibleAnswer = search(nums, l, mid, threshold);
        if (possibleAnswer == -1) {
            return mid;
        }
        return possibleAnswer;
    }

    public int smallestDivisor(int[] nums, int threshold) {
        return search(nums, 1 - 1, 1000000 + 1, threshold);
    }
}
```

#### [Kth smallest element](https://www.geeksforgeeks.org/problems/kth-smallest-element5635/1?page=1&sprint=a663236c31453b969852f9ea22507634&sprint=a663236c31453b969852f9ea22507634&sortBy=submissions)

partition, quick select

```java
class Solution{
    public static int kthSmallest(int[] arr, int l, int r, int k)
    {
        return quick_select(arr, l, r, k);
    }

    static int quick_select(int[] arr, int l, int r, int k) {
        int pivot = partition(arr, l, r);
        if (pivot == k - 1) return arr[pivot];
        else if (pivot > k - 1) return quick_select(arr, l, pivot - 1, k);
        else return quick_select(arr, pivot + 1, r, k);
    }

    static int partition(int[] arr, int l, int r) {
        int pivot = r;
        // i represents the index that marks the partition point where all elements before it are smaller than the pivot value
        // j is used for traversal
        int i = 0;
        for (int j = 0; j < r; j++) {
            if (arr[j] < arr[pivot]) {
                swap(arr, j, i);
                i++;
            }
        }
        swap(arr, r, i);
        return i;
    }

    static void swap(int[] arr, int i, int j) {
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }
}

```

### DFS

#### [78. Subsets](https://leetcode.com/problems/subsets/) <span style="color:orange">Medium</span>

Combinatorial Search Problem -> DFS.

Every element either exist in a subset or does not exist in a subset and in total there are $2^n$ combinations.

[ ]

1? [1] [ ]

2? [1, 2] [1] [2] [ ]

3? [1, 2, 3] [1, 2] [1, 3] [1] [2] [2, 3] [3] [ ]

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

​ [ ]

​ [1] [2] [3]

​ [1, 2] [1, 3] [2, 3]

​ [1, 2, 3]

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

#### [90. Subsets II ](https://leetcode.com/problems/subsets-ii/)<span style="color:orange">Medium</span>

Combinatorial Search Probelm --> DFS while removing duplicates along the way

Keep in mind that getting all possibilites and then removing duplicates is too high in time complexity. (Consider the test case: [1, 1, 1, 1, 1]). Prune while traversing.

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

#### [40. Combination Sum II](https://leetcode.cn/problems/combination-sum-ii/) Medium

```java
class Solution {
    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        List<List<Integer>> results = new ArrayList<>();
        Arrays.sort(candidates);
        dfs(target, 0, candidates, new ArrayList<Integer>(), results);
        return results;
    }

    void dfs(int target, int startIndex, int[] candidates, List<Integer> subset, List<List<Integer>> results) {
        // preorder position
        if (target == 0) {
            results.add(new ArrayList<Integer>(subset));
        }

        for (int i = startIndex; i < candidates.length; i++) {
            if (candidates[i] > target) {
                continue;
            }

            if (i > startIndex && candidates[i] == candidates[i - 1]) {
                continue;
            }

            subset.add(candi  dates[i]);
            dfs(target - candidates[i], i + 1, candidates, subset, results);
            subset.remove(subset.size() - 1);
        }
    }
}
```

#### [40. Combination Sum II](https://leetcode.com/problems/combination-sum-ii/description/) Medium

```java
class Solution {
    boolean[] used;

    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
    List<List<Integer>> res = new LinkedList<>();
    used = new boolean[candidates.length];
    Arrays.sort(candidates);
    List<Integer> combination = new LinkedList<>();
    backtrack(candidates, 0, target, combination, res);
    return res;
    }

    private void backtrack(int[] candidates, int start, int target, List<Integer> combination, List<List<Integer>> res) {
        for (int i = start; i < candidates.length; i++) {
            if (i > start && candidates[i] == candidates[i - 1]) continue;

            if (target - candidates[i] < 0) {
                return;
            }

            combination.add(candidates[i]);
            if (target - candidates[i] == 0) {
                res.add(new LinkedList<Integer>(combination));
            } else {
                backtrack(candidates, i + 1, target - candidates[i], combination, res);
            }
            combination.remove(combination.size() - 1);
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

#### [1239. Maximum Length of a Concatenated String with Unique Characters](https://leetcode.com/problems/maximum-length-of-a-concatenated-string-with-unique-characters/) Medium

Ugly solution. Still it works at an acceptable speed.
The problem is that I tend to make too many false and premature assumptions from example test cases, which means a lot of unnecessary trials. For this problem, it is obvious that some strings contain overlapping characters, which mean that they should not be used. But from here I wrongly assume that strings themselves do not contain same characters, which is totally wrong. When I ran into a test case like this, e.g. ["aa", "bb"], I made another false assumption that the characters in a string are ordered, which is still wrong. After two attempts, I finally came to the coclution that I had to filter the input array first before other things.

```java
class Solution {
    int maxLength = 0;

    public int maxLength(List<String> arr) {
        List<Integer> removed = filterCandidates(arr);
        dfs(arr, 0, new StringBuilder(), removed);
        return maxLength;
    }

    void dfs(List<String> arr, int start, StringBuilder sb, List<Integer> removed) {
        if (sb.length() > maxLength) {
            maxLength = sb.length();
        }


        for (int i = start; i < arr.size(); i++) {
            if (removed.contains(i)) {
                continue;
            }

            String candidate = arr.get(i);
            boolean flag = true;
            for (int j = 0; j < candidate.length(); j++) {
                // indexOf only accepts string as an input
                if (sb.indexOf(String.valueOf(candidate.charAt(j))) > -1) {
                    flag = false;
                    break;
                }
            }


            if (!flag) {
                continue;
            }
            sb.append(candidate);
            dfs(arr, i + 1, sb, removed);
            sb.delete(sb.length() - candidate.length() ,sb.length());
        }
    }

    List<Integer> filterCandidates(List<String> arr) {
        int[] counter;
        int n = arr.size();
        List<Integer> removed = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            String candidate = arr.get(i);
            counter = new int[26];

            for (int j = 0; j < candidate.length(); j++) {
                counter[candidate.charAt(j) - 97]++;
                if (counter[candidate.charAt(j) - 97] > 1) {
                    removed.add(i);
                    continue;
                }
            }
        }
        return removed;
    }
}
```

#### [1655. Distribute Repeating Integers](https://leetcode.com/problems/distribute-repeating-integers/description/) Hard

dfs

```java
class Solution {
    public boolean canDistribute(int[] nums, int[] quantity) {
        int[] freq = getFrequency(nums);
        Arrays.sort(quantity);
        // processing orders from largest to smallest
        return dfs(quantity, freq, quantity.length - 1);
    }

    // only frequency matters, number doesn't count'
    int[] getFrequency(int[] nums) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int num: nums) {
            map.put(num, map.getOrDefault(num, 0) + 1);
        }

        int[] freq = new int[map.size()];
        int index = 0;
        for (Map.Entry<Integer, Integer> entry: map.entrySet()) {
            freq[index++] = entry.getValue();
        }
        Arrays.sort(freq);
        return freq;
    }

    boolean dfs(int[] orders, int[] stock, int processingOrder) {
        // even 1st(0-index) is processed now
        // which means all orders are finished
        if (processingOrder == -1) {
            return true;
        }

        for (int i = stock.length - 1; i >= 0; i--) {
            if (stock[i] >= orders[processingOrder]) {
                stock[i] -= orders[processingOrder];
                if (dfs(orders, stock, processingOrder - 1)) {
                    return true;
                }
                stock[i] += orders[processingOrder];
            }
        }

        return false;
    }
}
```

#### [1593. Split a String Into the Max Number of Unique Substrings](https://leetcode.com/problems/split-a-string-into-the-max-number-of-unique-substrings/description/) Medium

dfs

```java
class Solution {
    private int res;

    public int maxUniqueSplit(String s) {
        res = 1;
        dfs(s, new HashSet<String>(), 0);
        return res;
    }

    private void dfs(String s, HashSet<String> cutParts, int cutPosition) {
        // max cut from remaining string plus parts alreay cut
        if (s.length() - cutPosition + cutParts.size() <= res) {
            return;
        }

        // nothing to cut
        if (cutPosition == s.length()) {
            res = cutParts.size();
            return;
        }

        for (int i = cutPosition + 1; i <= s.length(); i++) {
            String cut = s.substring(cutPosition, i);
            if (!cutParts.contains(cut)) {
                cutParts.add(cut);
                dfs(s, cutParts, i);
                cutParts.remove(cut);
            }
        }
    }
}
```

#### [1415. The k-th Lexicographical String of All Happy Strings of Length n](https://leetcode.com/problems/the-k-th-lexicographical-string-of-all-happy-strings-of-length-n/description/) Medium

```java
class Solution {
    private List<String> res;
    public String getHappyString(int n, int k) {
        char[] candidates = {'a', 'b', 'c'};
        res = new ArrayList<>();
        dfs(n, "", candidates);
        if (k > res.size()) {
            return "";
        }

        return res.get(k - 1);
    }

    private void dfs(int desiredLength, String currentString, char[] candidates) {
        if (currentString.length() == desiredLength) {
            res.add(currentString);
            return;
        }

        for (char candidate: candidates) {
            if (currentString.length() != 0 && currentString.charAt(currentString.length() - 1) == candidate) {
                continue;
            }

            dfs(desiredLength, currentString + candidate, candidates);
        }
    }
}
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

#### 1248 Count Number of Nice Subarrays <span style="color:orange">Medium</span>

prefix sum
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

prefix sum
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

前缀和加哈希表的解法，我始终无法理解，看了许多题解，还是不懂为什么要 sum-goal，统计出来的 map 又有怎么样的实际意义。不想再在这道题上花时间了，挫败感太强了，之后再碰到前缀和的题目然后再看吧。

```java
class Solution {
    public int numSubarraysWithSum(int[] nums, int goal) {
        // [i...j]
        int count = 0;
        int presum = 0;
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, 1);
        /*
        k: count sum[i, j] == k
        previous presum: [0, i - 1]
        current presum: [0, j]

               presum
        .  .  .  .  .  .  .  .
              .  .  .  .
              i        j
        0 i-1
         previous presum + k = current presum
      => current presum - k = previous presum

        */
        for (int num: nums) {
            presum += num;
            if (map.containsKey(presum - goal)) {
                count = count + map.get(presum - goal);
            }

            // store previous presum
            map.put(presum, map.getOrDefault(presum, 0) + 1);
        }



        return count;
    }
}
```

#### [1. Two Sum](https://leetcode-cn.com/problems/two-sum/) <span style="color:green">Easy</span>

prefix sum

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

#### [724. Find Pivot Index](https://leetcode.com/problems/find-pivot-index/) Easy

prefix sum

```java
class Solution {
    public int pivotIndex(int[] nums) {
        int preSum = 0;
        for (int num: nums) {
            preSum += num;
        }

        int leftSum = 0;
        int pivotNum;
        for (int i = 0; i < nums.length; i++) {
            pivotNum = nums[i];
            int rightSum = preSum - pivotNum - leftSum;
            if (leftSum == rightSum) {
                return i;
            }

            // get ready for next loop: extend left
            leftSum += nums[i];
        }

        return -1;
    }
}
```

Another solution: gradually narrow the right and extend the left and always leave the pivot in between

```java
class Solution {
    public int pivotIndex(int[] nums) {
        int leftSum = 0;
        int rightSum = 0;
        for (int num: nums) {
            rightSum += num;
        }

        // now it is the status when i == -1
        /*
          2 1 -1
        |
     l       r
        */
        for (int i = 0; i < nums.length; i++) {
            // nums[i] is deducted from right sum and is not in left sum now
            // so it is the pivot now
            rightSum -= nums[i];
            if (leftSum == rightSum) {
                return i;
            }

            // only now one number is added to the left for the next loop
            leftSum += nums[i];
        }

        return -1;
    }
}
```

#### [560. Subarray Sum Equals K](https://leetcode.com/problems/subarray-sum-equals-k/) Medium

prefix sum
Same solution can be used to solve 930.

```java
class Solution {
    public int subarraySum(int[] nums, int k) {
        // [i...j]
        int count = 0;
        int presum = 0;
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, 1);
        /*
        k: count sum[i, j] == k
        previous presum: [0, i - 1]
        current presum: [0, j]

               presum
        .  .  .  .  .  .  .  .
              .  .  .  .
              i        j
        0 i-1
         previous presum + k = current presum
      => current presum - k = previous presum

        */
        for (int num: nums) {
            presum += num;
            if (map.containsKey(presum - k)) {
                count = count + map.get(presum - k);
            }

            // store previous presum
            map.put(presum, map.getOrDefault(presum, 0) + 1);
        }



        return count;
    }
}
```

#### [974. Subarray Sums Divisible by K](https://leetcode.com/problems/subarray-sums-divisible-by-k) Medium

prefix sum

Modular arithmetic: a % c == b % c <-> (a - b) % c = 0

```java
class Solution {
    public int subarraysDivByK(int[] nums, int k) {
        // [i...j]
        int count = 0;
        int presum = 0;
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, 1);

        for (int num: nums) {
            presum += num;
            int modulus  = (presum % k + k) % k;
            if (map.containsKey(modulus )) {
                count = count + map.get(modulus );
            }

            // store previous presum
            map.put(modulus, map.getOrDefault(modulus, 0) + 1);
        }

        return count;
    }
}
```

prefix sum, no corner cases need to be considered, though another loop is needed

```java
class Solution {
    public boolean checkSubarraySum(int[] nums, int k) {
        int n = nums.length;
        int[] presum = new int[n + 1];
        for (int i = 1; i <= n; i++) {
            presum[i] = presum[i - 1] + nums[i - 1];
        }
        Set<Integer> set = new HashSet<>();
        // subarray length must be greater than 1
        // for presum, it means starting from 2
        for (int i = 2; i <= n; i++) {
            set.add(presum[i - 2] % k);
            // get modulus for presum from index 2 to index n
            if (set.contains(presum[i] % k)) {
                return true;
            }
        }

        return false;
    }
}
```

#### [523. Continuous Subarray Sum](https://leetcode.com/problems/continuous-subarray-sum/) Medium

Modular arithmetic, prefix sum, hash table

```java
class Solution {
    public boolean checkSubarraySum(int[] nums, int k) {
        if (nums.length == 1) {
            return false;
        }

        int n = nums.length;
        Map<Integer, Integer> map = new HashMap<>();
        int presum = 0;
        // a desired subarray can be possibly found at index 1 (second element)
        // if -1 is not inserted here, this case will be missed
        // e.g. [2,4] k = 6
        // also, the first element of prefix sum is actually 0, which should be included.
        map.put(0, -1);
        for (int i = 0; i < n; i++) {
            presum += nums[i];
            int modulus = (presum % k + k) % k;
            // System.out.println(modulus);
            if (map.containsKey(modulus)) {
                if (i - map.get(modulus) > 1) {
                    return true;
                }
                /*
                do not update agiain when the same modulus is repeated. only keep the smalleest
                e.g. [5, 0, 0, 0] k = 3

                */
                continue;
            }

            map.put(modulus, i);
        }

        return false;
    }
}
```

#### [912. Sort an Array](https://leetcode.com/problems/sort-an-array/) Medium

merge sort

```java
class Solution {
    int[] temp;

    public int[] sortArray(int[] nums) {
        sort(nums, 0, nums.length - 1);
        return nums;
    }

    void sort(int[] nums, int lo, int hi) {
        if (lo == hi) {
            return;
        }
        int mid = lo + (hi - lo) / 2;
        sort(nums, lo, mid);
        sort(nums, mid + 1, hi);
        merge(nums, lo, mid, hi);
    }

    void merge(int[] nums, int lo, int mid, int hi) {
        temp = new int[hi - lo + 1];

        int i = 0, l = lo, r = mid + 1;
        while (l <= mid && r <= hi) {
            if (nums[l] < nums[r]) {
                temp[i++] = nums[l++];
            } else {
                temp[i++] = nums[r++];
            }
        }

        while (l <= mid) {
            temp[i++] = nums[l++];
        }

        while (r <= hi) {
            temp[i++] = nums[r++];
        }

        // copy
        for (int tempIndex = 0, numsIndex = lo; tempIndex < temp.length; tempIndex++, numsIndex++) {
            nums[numsIndex] = temp[tempIndex];
        }
    }
}
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

count[i] = COUNT(j) where j > i and nums[j] < nums[i]
merge sort https://labuladong.github.io/algo/2/21/41/

```java
class Solution {
    private class PosVal {
        int val, index;
        PosVal (int index, int val) {
            this.val = val;
            this.index = index;
        }
    }

    private PosVal[] temp;
    private int[] count;

    public List<Integer> countSmaller(int[] nums) {
        int n = nums.length;
        count = new int[n];
        temp = new PosVal[n];
        PosVal[] pairs = convertNums(nums);
        sort(pairs, 0, n - 1);
        return convertAnswer(count);
    }

    PosVal[] convertNums(int[] nums) {
        PosVal[] res = new PosVal[nums.length];
        for (int i = 0; i < nums.length; i++) {
            res[i] = new PosVal(i, nums[i]);
        }
        return res;
    }

    List<Integer> convertAnswer(int[] count) {
        List<Integer> res = new LinkedList<>();
        for (int c: count) {
            res.add(c);
        }
        return res;
    }

    void sort(PosVal[] pairs, int lo, int hi) {
        if (lo == hi) {
            return;
        }

        int mid = lo + (hi - lo) / 2;
        sort(pairs, lo, mid);
        sort(pairs, mid + 1, hi);
        merge(pairs, lo, mid, hi);
    }

    void merge(PosVal[] pairs, int lo, int mid, int hi) {
        for (int i = lo; i <= hi; i++) {
            temp[i] = pairs[i];
        }

        int l = lo, r = mid + 1;
        for (int j = lo; j <= hi; j++) {
            if (l == mid + 1) {
                pairs[j] = temp[r++];
            } else if (r == hi + 1) {
                pairs[j] = temp[l++];
                count[pairs[j].index] += r - mid - 1;
            } else if (temp[l].val <= temp[r].val) {
                pairs[j] = temp[l++];
                count[pairs[j].index] += r - mid - 1;
            } else {
                pairs[j] = temp[r++];
            }
        }
    }


}
```

#### [493. Reverse Pairs](https://leetcode.com/problems/reverse-pairs/) Hard

count[i] = COUNT(j) where j > i and nums[i] > 2 \* nums[j] https://labuladong.github.io/algo/2/21/41/
merge sort. Before every merging, nums[lo..mid] and nums[mid + 1...hi] are already sorted. Make use of this.

```java
class Solution {


    public int reversePairs(int[] nums) {
        sort(nums);
        return count;
    }

    private int[] temp;
    int count = 0;
    void sort(int[]  nums) {
        temp = new int[nums.length];
        sort(nums, 0, nums.length - 1);
    }

    void sort(int[] nums, int lo, int hi) {
        if (lo == hi) {
            return;
        }

        int mid = lo + (hi - lo) / 2;
        sort(nums, lo, mid);
        sort(nums, mid + 1, hi);
        merge(nums, lo, mid, hi);
    }


    void merge(int[] nums, int lo, int mid, int hi) {
        for (int i = lo; i <= hi; i++) {
            temp[i] = nums[i];
        }

        // now nums[lo..mid], nums[mid + 1, hi] are both sorted now
        // make sure [mid + 1, end)
        int end = mid + 1;
        for (int i = lo; i <= mid; i++) {
            while (end <= hi && (long) nums[i] > (long) nums[end] * 2) {
                end++;
            }

            count += end - (mid + 1);
        }

        int l = lo, r = mid + 1;
        for (int j = lo; j <= hi; j++) {
            if (l == mid + 1) {
                nums[j] = temp[r++];
            } else if (r == hi + 1) {
                nums[j] = temp[l++];
            } else if (temp[l] > temp[r]) {
                nums[j] = temp[r++];
            } else {
                nums[j] = temp[l++];
            }
        }
    }
}
```

#### [327. Count of Range Sum](https://leetcode.com/problems/count-of-range-sum/) Hard

count[i] = COUNT(j) where lower <= preSum[j] - preSum[i] <= upper
merge sort on a prefix sum set
寻找满足条件的区间和实际上只需要找到任意两个满足差值为[lower, upper]的前缀和就可以，前缀和的顺序无关紧要
reference: https://leetcode.cn/problems/count-of-range-sum/solution/327qu-jian-he-de-ge-shu-ti-jie-zong-he-by-xu-yuan-/

```java
class Solution {
    int lower;
    int upper;
    int count;
    public int countRangeSum(int[] nums, int lower, int upper) {
        this.lower = lower;
        this.upper = upper;
        this.count = 0;
        long[] presum = getPresum(nums);
        sort(presum);
        return count;
    }

    long[] getPresum(int[] nums) {
        long[] presum = new long[nums.length + 1];
        for (int i = 0; i < nums.length; i++) {
            presum[i + 1] = (long) nums[i] + presum[i];
        }
        return presum;
    }

    long[] temp;
    void sort(long[] nums) {
        temp = new long[nums.length];
        sort(nums, 0, nums.length - 1);
    }

    void sort(long[] nums, int lo, int hi) {
        if (lo == hi) return;

        int mid = lo + (hi - lo) / 2;
        sort(nums, lo, mid);
        sort(nums, mid + 1, hi);
        merge(nums, lo, mid, hi);
    }

    void merge(long[] nums, int lo, int mid, int hi) {
        for (int i = lo; i <= hi; i++) {
            temp[i] = nums[i];
        }

        // nums[lo...mid] and num[mid + 1...hi] are both sorted now
        int start = mid + 1, end = mid + 1;
        for (int k = lo; k <= mid; k++) {
            while (start <= hi && nums[start] - nums[k] < lower) {
                start++;
            }
            // start now stands at a position where range sum is bigger than lower
            while (end <= hi && nums[end] - nums[k] <= upper) {
                end++;
            }
            // end now stands at a position where range sum is just bigger than upper
            count += end - start; // end is not included
        }

        int l = lo, r = mid + 1;
        for (int p = lo; p <= hi; p++) {
            if (l == mid + 1) {
                nums[p] = temp[r++];
            } else if (r == hi + 1) {
                nums[p] = temp[l++];
            } else if (temp[l] > temp[r]) {
                nums[p] = temp[r++];
            } else {
                nums[p] = temp[l++];
            }
        }
    }
}
```

#### [215. Kth Largest Element in an Array](https://leetcode.com/problems/kth-largest-element-in-an-array/) Medium

quick sort, quick select
top k-th -> k-th in descending order -> n - k in ascending order

```java
class Solution {
    public int findKthLargest(int[] nums, int k) {
        int lo = 0, hi = nums.length - 1;
        // convert k to ascending order rank
        int _k = nums.length - k;
        while (lo <= hi) {
            int pivot = partition(nums, lo, hi);
            if (pivot > _k) {
                hi = pivot - 1;
            } else if (pivot < _k) {
                lo = pivot + 1;
            } else {
                return nums[pivot];
            }
        }

        return -1;
    }

    int partition(int[] nums, int lo, int hi) {
        int pivotVal = nums[hi];

        // nums[lo..i) lower than pivot val
        // nums[i..j) greater than pivot val
        int i = lo, j = lo;
        // compare until the second to the last item
        while (j < hi) {
            // i moves only when a smaller value is found
            if (nums[j] < pivotVal) {
                swap(nums, i, j);
                i++;
            }
            // no matter what j moves forward
            j++;
        }
        swap(nums, i, hi);
        return i;
    }

    void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
}
```

max heap - reversed min heap

```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        heap = []
        result = []
        for num in nums:
            heapq.heappush(heap, -num)
        while k > 0:
            result = -heapq.heappop(heap)
            k -= 1
        return result
```

#### [703. Kth Largest Element in a Stream](https://leetcode.com/problems/kth-largest-element-in-a-stream/) Easy

min heap
remove n - k items in a min heap and you will have the k-th larget on the top

```java
class KthLargest {
    private int k;
    private PriorityQueue<Integer> pq = new PriorityQueue<>();

    public KthLargest(int k, int[] nums) {
        for (int num: nums) {
            pq.offer(num);
            if (pq.size() > k) {
                pq.poll();
            }
        }
        this.k = k;
    }

    public int add(int val) {
        pq.offer(val);
        if (pq.size() > k) {
            pq.poll();
        }

        return pq.peek();
    }
}
```

#### [118. Pascal's Triangle](https://leetcode.com/problems/pascals-triangle/) Easy

```java
class Solution {
    public List<List<Integer>> generate(int numRows) {
        List<List<Integer>> res = new ArrayList<>();
        List<Integer> firstRow = new ArrayList<>();
        firstRow.add(1);

        res.add(firstRow);

        List<Integer> row;
        List<Integer> previousRow = firstRow;
        for (int i = 2; i < numRows + 1; i++) {
            row = new ArrayList<>();
            row.add(1);
            int j = 0;
            while (j + 1 < previousRow.size()) {
                int num = previousRow.get(j) + previousRow.get(j + 1);
                row.add(num);
                j++;
            }

            row.add(1);

            previousRow = row;
            res.add(row);
        }

        return res;
    }
}
```

#### [594. Longest Harmonious Subsequence](https://leetcode.com/problems/longest-harmonious-subsequence/) Easy

two pointers using while loop

```java
class Solution {
    public int findLHS(int[] nums) {
        int res = 0;
        Arrays.sort(nums);
        // System.out.println(Arrays.toString(nums));
        int l = 0, r = 0;
        while (r < nums.length) {
            if (nums[r] - nums[l] == 1) {
                res = Math.max(res, r - l + 1);
            }
            // r is fixed and move forawrd l until the desired condition is met
            // if only a if is used here, instead of a while loop,
            while (l < r && nums[r] - nums[l] > 1) {
                l++;
            }
            r++;
        }

        return res;
    }
}
```

two pointers using for loop

```java
class Solution {
    public int findLHS(int[] nums) {
        int res = 0;
        Arrays.sort(nums);
        // System.out.println(Arrays.toString(nums));
        for (int l = 0, r = 0; r < nums.length; r++) {
            while (l < r && nums[r] - nums[l] > 1) l++;
            if (nums[r] - nums[l] == 1) res = Math.max(res, r - l + 1);
        }

        return res;
    }
}
```

Hash table

```java
class Solution {
    public int findLHS(int[] nums) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int num: nums) {
            map.put(num,  map.getOrDefault(num, 0) + 1);
        }
        int res = 0;
        for (int num: nums) {
            if (map.containsKey(num - 1)) {
                res = Math.max(res, map.get(num) + map.get(num - 1));
            }
        }
        return res;
    }
}
```

#### [219. Contains Duplicate II](https://leetcode.com/problems/contains-duplicate-ii/) Easy

two pointers

```java
class Solution {
    public boolean containsNearbyDuplicate(int[] nums, int k) {
        int j = nums.length - 1;
        while (j >= 0) {
            int i = j - k;
            // i could be less than 0
            if (i < 0) {
                i = 0;
            }
            // exit when i == j
            while (i < j) {
                // whenever using a index, make sure it is not out of boundary
                // no need to worry about j because j is always bigger than zero
                if (nums[i] == nums[j]) {
                    return true;
                }
                i++;
            }
            j--;
        }

        return false;
    }
}
```

#### [220. Contains Duplicate III](https://leetcode.com/problems/contains-duplicate-iii/) Hard

brute force

```java
class Solution {
    public boolean containsNearbyAlmostDuplicate(int[] nums, int indexDiff, int valueDiff) {
        for (int i = 0; i < nums.length; i++) {
            for (int j = i + 1; j <= i + indexDiff && j < nums.length; j++) {
                if (Math.abs(nums[j] - nums[i]) > valueDiff) continue;
                return true;
            }
        }

        return false;
    }
}
```

reference: https://leetcode.cn/problems/contains-duplicate-iii/solution/c-li-yong-tong-fen-zu-xiang-xi-jie-shi-b-ofj6/
bucket division

```java
class Solution {
    public boolean containsNearbyAlmostDuplicate(int[] nums, int indexDiff, int valueDiff) {
        int n = nums.length;
        Map<Integer, Integer> buckets = new HashMap<>();
        for (int i = 0; i < n; i++) {
            int value = nums[i];
            int id = getBucketId(value, valueDiff);

            // already in the same bucket
            if (buckets.containsKey(id)) return true;
            // check previous and next bucket
            if (buckets.containsKey(id - 1) && Math.abs(buckets.get(id - 1) - value) <= valueDiff)
                return true;
            if (buckets.containsKey(id + 1) && Math.abs(buckets.get(id + 1) - value) <= valueDiff)
                return true;
            buckets.put(id, value);
            // make sure
            if (i >= indexDiff)
                buckets.remove(getBucketId(nums[i - indexDiff], valueDiff));
        }
        return false;
    }

    int getBucketId(int value, int valueDiff) {
        if (value >= 0) {
            return value / (valueDiff + 1);
        }

        return (value + 1) / (valueDiff + 1) - 1;
    }
}
```

#### [1637. Widest Vertical Area Between Two Points Containing No Points](https://leetcode.com/problems/widest-vertical-area-between-two-points-containing-no-points/) Medium

The intuitive solution would be sorting the input and calcuate the gap between all non-equal candidates. Sorting means nlogn time complexity. with bucket division, the time complexity can be reduced to (3)n.
bucket division, pigeonhole principle

```java
class Solution {
    public int maxWidthOfVerticalArea(int[][] points) {
		int n = points.length;

        // find the leftmost and rightmost point
        int min = points[0][0];
        int max = points[0][0];
        // only horizontal ordinate is relevant here
        for (int[] point: points) {
            min = Math.min(min, point[0]);
            max = Math.max(max, point[0]);
        }

        // all points are on the same vertical line, no rectangle is possible
        if (min == max) {
            return 0;
        }

        // prepare buckets, with the first point in the first bucket and the last point in the last point
        // a bucket will hold two values, the min value in the bucket and the max vlue in the bucket
        int[][] buckets = new int[n + 1][];
        final int partition = max - min;
        for (int[] point: points) {
            // since there is multiplication here, result could exceed the int limit
            int bucketNumber = (int) ((point[0] - min) * (long) n / partition);
            // bucket is still empty, initialize
            if (buckets[bucketNumber] == null) {
                buckets[bucketNumber] = new int[] {point[0], point[0]};
            } else {
                // update the values in the bucket
                buckets[bucketNumber][0] = Math.min(buckets[bucketNumber][0], point[0]);
                buckets[bucketNumber][1] = Math.max(buckets[bucketNumber][1], point[0]);
            }
        }

        int res = 0;
        for (int lastMax = buckets[0][1], i = 1; i <= n; i++) {
            if (buckets[i] != null) {
                res = Math.max(res, buckets[i][0] - lastMax);
                lastMax = buckets[i][1];
            }
        }

        return res;
    }
}
```

#### [643. Maximum Average Subarray I](https://leetcode.com/problems/maximum-average-subarray-i/) Easy

two pointers

```java
class Solution {
    public double findMaxAverage(int[] nums, int k) {
        int n = nums.length;
        double average = 0;
        double sum = 0;
        for (int i = 0; i < k; i++) {
            sum += nums[i];
        }
        average = sum / k;
        // System.out.println(sum);
        for (int l = 1, r = k; r < nums.length; r++,l++) {
            sum = sum + nums[r] - nums[l - 1];
            average = Math.max(average, sum / k);
        }

        return average;
    }
}
```

#### [1984. Minimum Difference Between Highest and Lowest of K Scores](https://leetcode.com/problems/minimum-difference-between-highest-and-lowest-of-k-scores/) Easy

two pointers

```java
class Solution {
    public int minimumDifference(int[] nums, int k) {
        if (k == 1) {
            return 0;
        }
        Arrays.sort(nums);
        int diff = Integer.MAX_VALUE;
        for (int l = 0, r = k -1; r < nums.length; l++, r++) {
            if (nums[r] - nums[l] < diff) {
                diff = nums[r] - nums[l];
            }
        }
        return diff;
    }
}
```

#### [Segregate 0s and 1s](https://www.geeksforgeeks.org/problems/segregate-0s-and-1s5106/1?itm_source=geeksforgeeks&itm_medium=article&itm_campaign=bottom_sticky_on_article)

two pointers

```java

class Solution {
    void segregate0and1(int[] arr, int n) {
        // 0: 0 - low - 1
        // 1: high + 1, n
        int low = 0, high = n - 1;
        while (low <= high) {
            if (arr[low] == 1) {
                swap(arr, low, high);
                high--;
            } else {
                low++;
            }
        }
     }

     void swap(int[] arr, int i, int j) {
         int temp = arr[i];
         arr[i] = arr[j];
         arr[j] = temp;
     }
}

```

#### [Sort an array of 0s, 1s and 2s](https://www.geeksforgeeks.org/problems/sort-an-array-of-0s-1s-and-2s4231/1?page=1&sprint=a663236c31453b969852f9ea22507634&sprint=a663236c31453b969852f9ea22507634&sortBy=submissions)

three pointers, Dutch National Flag problem

```java
class Solution
{
    public static void sort012(int arr[], int n)
    {
        int low = 0, mid = 0, high = n - 1;
        // 0: 0 - low - 1
        // 1: low - mid - 1
        // 2: high + 1 - end
        while (mid <= high) {
            if (arr[mid] == 1) {
                mid++;
            } else if (arr[mid] == 0) {
                swap(arr, mid, low);
                low++;
                mid++;
            } else {
                swap(arr, mid, high);
                high--;
            }
        }
    }

    static void swap(int[] arr, int i, int j) {
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }
}
```

#### [303. Range Sum Query - Immutable](https://leetcode.com/problems/range-sum-query-immutable/) Easy

prefix sum

```java
class NumArray {
    int[] prefix;

    public NumArray(int[] nums) {
        int n = nums.length;
        int[] prefix = new int[n + 1];
        for (int i = 1; i < n + 1; i++) {
            prefix[i] = nums[i - 1] + prefix[i - 1];
        }

        this.prefix = prefix;
    }

    public int sumRange(int left, int right) {
        return prefix[right + 1] - prefix[left];
    }
}
```

#### [Equilibrium Poin](https://www.geeksforgeeks.org/problems/equilibrium-point-1587115620/1?page=1&sprint=a663236c31453b969852f9ea22507634&sprint=a663236c31453b969852f9ea22507634&sortBy=submissions)

```java
class Solution {


    // a: input array
    // n: size of array
    // Function to find equilibrium point in the array.
    public static int equilibriumPoint(long arr[], int n) {
        if (n == 1) res = 1;

        long[] prefix = new long[n + 1];
        for (int i = 1; i <= n; i++) {
            prefix[i] = prefix[i - 1] + arr[i - 1];
        }

        int res = -1;
        for (int i = 2; i < n; i++) {
            if (prefix[i - 1] == prefix[n] - prefix[i]) res = i;
        }
        return res;
    }
}

```

#### [304. Range Sum Query 2D - Immutable](https://leetcode.cn/problems/range-sum-query-2d-immutable/) Medium

```java
class NumMatrix {
    // prefix[i][j] means the sum from [0,0] to [i-1, j-1]
    int[][] prefix;

    public NumMatrix(int[][] matrix) {
        int m = matrix.length;
        int n = matrix[0].length;
        prefix = new int[m + 1][n + 1];
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                /*
                . . .
                . * .
                . . .

                * + . + . . - .
                    .
                */
                prefix[i][j] = matrix[i - 1][j - 1] + prefix[i - 1][j] + prefix[i][j - 1] - prefix[i -1][j - 1];
            }
        }
    }

    public int sumRegion(int x1, int y1, int x2, int y2) {
        return prefix[x2 + 1][y2 + 1] - prefix[x1][y2 + 1] - prefix[x2 + 1][y1] + prefix[x1][y1];
    }
}
```

#### []()

#### [1109. Corporate Flight Bookings](https://leetcode.com/problems/corporate-flight-bookings/) Medium

difference array, range update in O(1)

```java
class Solution {
    public int[] corpFlightBookings(int[][] bookings, int n) {
        int[] diff = new int[n];
        int[] res = new int[n];

        for (int i = 0; i < bookings.length; i++) {
            diff[bookings[i][0] - 1] += bookings[i][2];
            if (bookings[i][1] < n) {
                diff[bookings[i][1]] -= bookings[i][2];
            }
        }

        res[0] = diff[0];
        for (int i = 1; i < n; i++) {
            res[i] = res[i - 1] + diff[i];
        }


        return res;
    }
}
```

#### [1526. Minimum Number of Increments on Subarrays to Form a Target Array](https://leetcode.com/problems/minimum-number-of-increments-on-subarrays-to-form-a-target-array/) Hard

range update, difference array

````java
class Solution {
    public int minNumberOperations(int[] target) {
        int res = 0, pre = 0;
        for (int cur: target) {
            res += Math.max(cur - pre, 0);
            pre = cur;
        }
        return res;
    }
}

#### [1094. Car Pooling](https://leetcode.com/problems/car-pooling/) Medium
```java
class Solution {
    public boolean carPooling(int[][] trips, int capacity) {
        int n = 0;
        for (int[] trip: trips) {
            if (trip[2] > n) {
                n = trip[2];
            }
        }

        int[] diff = new int[n + 1];
        int[] res = new int[n + 1];
        for (int[] trip: trips) {
            diff[trip[1]] += trip[0];

            // wrong condition and end should not be incremented in index
            // if (end < n - 1) {
            //     diff[end + 1] -= passengers;
            // }

            diff[trip[2]] -= trip[0];

        }

        res[0] = diff[0];
        if (res[0] > capacity)  {
            return false;
        }

        for (int i = 1; i < n + 1; i++) {
            res[i] = res[i - 1] + diff[i];
            if (res[i] > capacity) {
                return false;
            }
        }


        return true;
    }
}
````

#### [48. Rotate Image](https://leetcode.com/problems/rotate-image/) Medium

matrix, multi-dimension array

```java
class Solution {
    public void rotate(int[][] matrix) {
        int n = matrix.length;
        int temp;
        // diagnoal swap from left top to right bottom
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                temp = matrix[i][j];
                matrix[i][j] = matrix[j][i];
                matrix[j][i] = temp;
            }
        }

        // row reversal line by line
        for (int[] row : matrix) {
            reverseRow(row);
        }
    }

    void reverseRow(int[] nums) {
        int i = 0, j = nums.length - 1, temp;
        while (i < j) {
            temp = nums[i];
            nums[i] = nums[j];
            nums[j] = temp;
            i++;
            j--;
        }
    }
}
```

#### [1524. Number of Sub-arrays With Odd Sum](https://leetcode.com/problems/number-of-sub-arrays-with-odd-sum/) Medium

```java
/**
for presum[j], count i's where i < j and presum[i] and presum[j] have different parities -> sum of i to j = presum[j] - presum[i]
e.g.'
         1 2 3 4  5
presum 0 1 3 6 10 15
parity e o o e e  o
count of even and odd presums count[0] even count[1] odd
i   (e,o)   odd subarrays
0   (1,0)   0
1   (1,1)   1
3   (1,2)   3 is odd, find even presums before it, 1 + previous count[0] = 2
6   (2,2)   6 is even, find odd presums before it, 2 + previous count[1] = 4
    - [1] sum is odd, exlucding it from subarrays that end with arr[2] and subarray sum will be odd, [2, 3]
    - [1,2] sum is odd, excluded and the desired subarray will be [3]
10  (3,2)   ...
15  (3,3)   ...
 */
class Solution {
    public int numOfSubarrays(int[] arr) {
        final int MODULO = 1000000007;

        // count of odd and even presums
        int odd = 0, even = 1;
        int presum = 0;
        int res = 0;
        for (int i = 0; i < arr.length; i++) {
            presum += arr[i];
            if (presum % 2 == 0) {
                res = (res + odd) % MODULO;
                even++;
            } else {
                res = (res + even) % MODULO;
                odd++;
            }
        }
        return res;
    }
}
```

use bitwise operation

```java
/**
0 even
1 odd
for sum[j], count sum[i] where i < j and sum[i] have different parities

odd + even = odd
even + odd = odd
odd + odd = even
even + even = even
 */
class Solution {
    public int numOfSubarrays(int[] arr) {
        // presumParity[0] count of even presums
        // presumParity[1] count of odd presums
        int[] presumParity = new int[] {1, 0};
        long res = 0;
        int parity;
        for (int i = 0, sum = 0; i < arr.length; i++) {
            /**
            bitwise AND
            if arr[i] is even, 0
            if arr[i] is odd, 1
            */
            parity = arr[i] & 1;
            /**
            bitwise XOR
            only when
             */
            ++presumParity[sum ^= parity];
            res += presumParity[sum ^ 1];
        }
        return (int) (res % 1000000007);
    }
}
```

#### [1491. Average Salary Excluding the Minimum and Maximum Salary](https://leetcode.com/problems/average-salary-excluding-the-minimum-and-maximum-salary/) Easy

```java
class Solution {
    public double average(int[] salary) {
        int min = 1000001;
        int max = 999;
        double sum = 0;
        for (int s: salary) {
            if (s < min) min = s;
            if (s > max) max = s;
            sum += s;
        }

        return (sum - max - min) / (salary.length - 2);
    }
}
```

#### [1512. Number of Good Pairs](https://leetcode.com/problems/number-of-good-pairs/) Easy

```java
class Solution {
    public int numIdenticalPairs(int[] nums) {
        Arrays.sort(nums);
        int count = 1;
        int total = 0;
        for (int i = 0, j = 1; j < nums.length; j++) {
            if (nums[j] == nums[i]) {
                count++;
                continue;
            }

            for (int k = 1; k < count; k++) {
                total += k;
            }
            i = j;
            count = 1;
        }

        for (int k = 1; k < count; k++) {
            total += k;
        }

        return total;
    }
}
```

#### [268. Missing Number](https://leetcode.com/problems/missing-number/) Easy

array as map

```java
class Solution {
    public int missingNumber(int[] nums) {
        int[] count = new int[nums.length + 1];
        for (int num: nums) {
            count[num]++;
        }

        int res = 0;
        for (int i = 0; i < count.length; i++) {
            if (count[i] == 0) res = i;
        }
        return res;
    }
}
```

#### [448. Find All Numbers Disappeared in an Array](https://leetcode.com/problems/find-all-numbers-disappeared-in-an-array/) Easy

array as map

```java
class Solution {
    public List<Integer> findDisappearedNumbers(int[] nums) {
        List<Integer> res = new ArrayList<>();
        int[] count = new int[nums.length];
        for (int num: nums) {
            count[num - 1]++;
        }

        for (int i = 0; i < nums.length; i++) {
            if (count[i] == 0) res.add(i + 1);
        }

        return res;
    }
}
```

#### [442. Find All Duplicates in an Array](https://leetcode.com/problems/find-all-duplicates-in-an-array/) Medium

array as map
I figured out a way to store frequency in the input array provided.

```java
class Solution {
    public List<Integer> findDuplicates(int[] nums) {
        List<Integer> res = new ArrayList<>();

        for (int i = 0; i < nums.length; i++) {
            int correctIndex = Math.abs(nums[i]) - 1;
            // the position is now visted twice, indicating a duplicate
            if (nums[correctIndex] < 0) {
                res.add(correctIndex + 1);
            } else {
                nums[correctIndex] = -nums[correctIndex];
            }
        }

        return res;
    }
}
```

#### [41. First Missing Positive](https://leetcode.com/problems/first-missing-positive/) Hard

array as map

```java
class Solution {
    public int firstMissingPositive(int[] nums) {
        int n = nums.length;

        for (int i = 0; i < n; i++) {
            // keep moving nums[i] to the position where it belongs
            while (nums[i] >= 1 && nums[i] <= n && nums[i] != nums[nums[i] - 1]) {
                swap(nums, i, nums[i] - 1);
            }
        }

        // now all eleements should be in the right positions
        for (int i = 0; i < n; i++) {
            if (nums[i] != i + 1) return i + 1;
        }

        return n + 1;
    }

    void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
}
```

#### [287. Find the Duplicate Number](https://leetcode.com/problems/find-the-duplicate-number/) Medium

array as a linked list with cycle, turtoise and hare algorithms for detecting a cycle in a linked list
Pay close attention to the constraints, which make this algorithms not only possible but also desired.

```java
class Solution {
    public int findDuplicate(int[] nums) {
        // same as dummy nodes
        int slow = 0;
        int fast = 0;
        do {
            slow = nums[slow]; // slow = slow.next
            fast = nums[nums[fast]]; // fast = fast.next.next
        } while (slow != fast);

        slow = 0;
        while (slow != fast) {
            slow = nums[slow];
            fast = nums[fast];
        }
        return slow;
    }
}
```

#### [1657. Determine if Two Strings Are Close](https://leetcode.com/problems/determine-if-two-strings-are-close/) Medium

use array as map, reverse thinking

```java
class Solution {
    public boolean closeStrings(String word1, String word2) {
        int[] count1 = count(word1);
        int[] count2 = count(word2);
        return haveSameChars(count1, count2) && haveSameFrequency(count1, count2);
    }

    int[] count(String word) {
        int[] count = new int[26];
        for (int i = 0; i < word.length(); i++) {
            ++count[word.charAt(i) - 'a'];
        }
        return count;
    }

    boolean haveSameChars(int[] count1, int[] count2) {
        for (int i = 0; i < 26; i++) {
            if ((count1[i] == 0 && count2[i] != 0)
                || (count1[i] != 00 && count2[i] == 0))
                return false;
        }

        return true;
    }

    boolean haveSameFrequency(int[] count1, int[] count2) {
        Arrays.sort(count1);
        Arrays.sort(count2);
        for (int i = 0; i < 26; i++) {
            if (count1[i] != count2[i])
                return false;
        }
        return true;
    }
}
```

#### [Subarray with given sum](https://www.geeksforgeeks.org/problems/subarray-with-given-sum-1587115621/1?page=1&sprint=a663236c31453b969852f9ea22507634&sprint=a663236c31453b969852f9ea22507634&sortBy=submissions)

```java
class Solution
{
    //Function to find a continuous sub-array which adds up to a given number.
    static ArrayList<Integer> subarraySum(int[] arr, int n, int s)
    {
        int[] prefix = new int[n + 1];
        for (int i = 1; i < n + 1; i++) {
            prefix[i] = prefix[i - 1] + arr[i - 1];
        }
        ArrayList<Integer> res = new ArrayList<>();
        for (int i = 0, j = 1; i < n + 1 && j < n + 1; ) {
            if (prefix[j] - prefix[i] == s) {
                // make sure left bound is on the left
                if (i + 1 <= j) {
                    res.add(i + 1);
                    res.add(j);
                }
                break;
            } else if (prefix[j] - prefix[i] > s) {
                i++;
            } else {
                j++;
            }
        }
        if (res.size() == 0) res.add(-1);
        return res;
    }
}
```

#### [945. Minimum Increment to Make Array Unique](https://leetcode.com/problems/minimum-increment-to-make-array-unique/description/) Medium

```java
class Solution {
    public int minIncrementForUnique(int[] nums) {
        int count = 0;
        Arrays.sort(nums);
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] <= nums[i - 1]) {
                int temp = nums[i - 1] + 1;
                count += temp - nums[i];
                nums[i] = temp;
            }
        }

        return count;
    }
}
```
