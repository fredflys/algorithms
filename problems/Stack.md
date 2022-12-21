Monotonic stack is used to find next greater or smaller number.

#### [496. Next Greater Element I](https://leetcode.com/problems/next-greater-element-i/) Easy
monotonic stack
```java
class Solution {
    public int[] nextGreaterElement(int[] nums1, int[] nums2) {
        Map<Integer, Integer> numToFirstGreater = new HashMap<>();
        Stack<Integer> greaters = new Stack<>();
        for (int i = nums2.length - 1; i >= 0; i--) {
            int num = nums2[i];
            while (!greaters.isEmpty() && num >= greaters.peek()) {
                greaters.pop();
            }
            numToFirstGreater.put(num, greaters.isEmpty() ? -1 : greaters.peek());
            greaters.push(num);
        }
        
        int[] res = new int[nums1.length];
        for (int i = 0; i < nums1.length; i++) {
            res[i] = numToFirstGreater.get(nums1[i]);
        }
        
        return res;
    }
}
```

#### [503. Next Greater Element II](https://leetcode.com/problems/next-greater-element-ii/) Medium
monotonic stack. Use modular operation and doubling array size to emulate a circular array.
```java
class Solution {
    public int[] nextGreaterElements(int[] nums) {
        int n = nums.length;
        Stack<Integer> stack = new Stack<>();
        int[] res = new int[n];
        for (int i = 2  * n - 1; i >= 0; i--) {
            int index = i % n;
            int num = nums[index];
            while (!stack.isEmpty() && num >= stack.peek() ) {
                stack.pop();
            }
            
            res[index] = stack.isEmpty() ? -1 : stack.peek();
            stack.push(num);    
        }
        
        return res;
    }
}
```

#### [1019. Next Greater Node In Linked List](https://leetcode.com/problems/next-greater-node-in-linked-list/) Medium
my solution
```java
class Solution {
    public int[] nextLargerNodes(ListNode head) {
        ListNode cur = head;
        int n = 0;
        for(ListNode node = head; node != null; node = node.next) {
            n++;
        }
        
        int[] res = new int[n];
        Stack<int[]> stack = new Stack<>();
        int index = 0;
        while (head != null) {
            while (!stack.isEmpty() && head.val > stack.peek()[1]) {
                int[] temp = stack.pop();
                res[temp[0]] = head.val;
            }

            stack.push(new int[]{index, head.val});
        
            head = head.next;
            index++;
        }
        
        return res;
    }
}
```

#### [739. Daily Temperatures](https://leetcode.com/problems/daily-temperatures/) Medium
```java
class Solution {
    public int[] dailyTemperatures(int[] temperatures) {
        Stack<Integer> warmerTempIndices = new Stack<>();
        int[] res = new int[temperatures.length];
        for (int i = temperatures.length - 1; i >= 0; i--) {
            int todayTemp = temperatures[i];
            while (!warmerTempIndices.isEmpty() && todayTemp >= temperatures[warmerTempIndices.peek()]) {
                warmerTempIndices.pop();
            }
            if (!warmerTempIndices.isEmpty()) {
                res[i] = warmerTempIndices.peek() - i;
            }
            
            warmerTempIndices.push(i);
        }

        return res;
    }
}
```
#### [907. Sum of Subarray Minimums](https://leetcode.com/problems/sum-of-subarray-minimums/) Medium
brute force, will not pass
```java
class Solution {
    public int sumSubarrayMins(int[] arr) {
        int sum = 0;
        int min;
        for (int i = 0; i < arr.length; i++) {
            min = arr[i];
            for (int j = i; j < arr.length; j++) {
                if (arr[j] < min) min = arr[j];
                sum += min;
            }
        }
        return sum;
    }
}
```
reference: https://leetcode.cn/problems/sum-of-subarray-minimums/solution/xiao-bai-lang-dong-hua-xiang-jie-bao-zhe-489q/
https://leetcode.com/problems/sum-of-subarray-minimums/discuss/2118729/Very-detailed-stack-explanation-O(n)-or-Images-and-comments
monotonic stack
```java
class Solution {
    private static final long MOD = (long) 1e9 + 7;
    
    public int sumSubarrayMins(int[] arr) {
        int n = arr.length;
        Stack<Integer> stack = new Stack<>();
        int[] left = new int[n];
        int[] right = new int[n];
        
        int num;
        for (int i = 0; i < n; i++) {
            num = arr[i];
            // only smaller candidates will be retained
            while (!stack.isEmpty() && arr[stack.peek()] > num) {
                stack.pop();
            }
            
            if (stack.isEmpty()) {
                left[i] = -1;
            } else {
                left[i] = stack.peek();
            }
            
            stack.push(i);
        }
        
        stack.clear();
        for (int j = arr.length - 1; j >= 0; j--) {
            num = arr[j];
            // only smaller condidates will be retained
            while (!stack.isEmpty() && arr[stack.peek()] >= num) {
                stack.pop();
            }
            
            if (stack.isEmpty()) {
                right[j] = n;
            } else {
                right[j] = stack.peek();
            }
            
            stack.push(j);
        }

        long res = 0;
        for (int i = 0; i < n; i++) {
            res = (res + (long) (i - left[i]) * (right[i] - i) * arr[i]) % MOD;
        }
        return (int) res;
    }
}
```
monotonic stack, one loop
```java
class Solution {
    private static long M = (long)1e9 + 7;
    
    public int sumSubarrayMins(int[] arr) {
        long res = 0;
        Stack<Integer> stack = new Stack<>();
        stack.push(-1);
        
        int n = arr.length;
        int num;
        for (int j = 0; j <= n; j++) {
            if (j < n) {
                num = arr[j];
            } else {
                num = 0;
            }
            
            while (stack.peek() != -1 && num < arr[stack.peek()]){
                int index = stack.pop();
                int i = stack.peek();
                int left = index - i;
                int right = j - index;
                
                long add = (long)(left * right * (long) arr[index]) % M;
                res += add ;
                res %= M;
            }
            
            stack.push(j);
        }
        
        return (int) res;
    }
}
```
#### [402. Remove K Digits](https://leetcode.com/problems/remove-k-digits/) Medium
monotonic stack.
The solution I come up with is flawed. I have to keep adding conditions to make it pass more test cases. But seversal failed attempts, still not all cases can be passed, which is pretty frustrating. 

WRONG SOLUTION, UGLY CODE
```java
class Solution {
    public String removeKdigits(String num, int k) {
        if (k >= num.length()) {
            return "0";
        }
        
        
        char[] digits = num.toCharArray();
        int n = digits.length;
        Deque<Character> stack = new LinkedList<>();
        StringBuilder res = new StringBuilder();

        for (int i = 0; i < n; i++) {
            while (k > 0 && !stack.isEmpty() && digits[i] < stack.peek()) {
                stack.pop();
                res.delete(res.length() - 1, res.length());
                k--;
                if (k == 0) {
                    break;
                }
            }
            
            stack.push(digits[i]);
            res.append(digits[i]);
        }
        
        if (res.length() == 1) {
            return res.toString();
        }
        
        int i;
        for (i = 0; i < res.length(); i++) {
            if (res.charAt(i) != '0') break;
        }
        
        res.delete(0, i);
        return res.toString();
    }
}
```

all corner cases are beautiflly handled, much better than I do.
If you'd like to modify the string in traversal, it's better to use while loop than for loop.
```java
class Solution {
    public String removeKdigits(String num, int k) {
        int n = num.length();
        if (k == 0) return num;
        if (k == n) return "0";
        
        Stack<Character> stack = new Stack<>();
        for (int i = 0; i < n; i++) {
            while (k > 0 && !stack.isEmpty() && num.charAt(i) < stack.peek()) {
                stack.pop();
                k--;
            }
            
            stack.push(num.charAt(i));
        }
        
        // k is not fully spent, e.g. 123 k = 2
        while (k-- > 0) stack.pop();
        
        // build the result
        StringBuilder res = new StringBuilder();
        while (!stack.isEmpty()) {
            res.insert(0, stack.pop());
        }
        
        // delete leading zeros
        while (res.length() > 1 && res.charAt(0) == '0') {
            res.delete(0, 1);
        }
        
        return res.toString();
    }
}
```
#### [1475. Final Prices With a Special Discount in a Shop](https://leetcode.com/problems/final-prices-with-a-special-discount-in-a-shop/) Easy
My solution. Running pretty slow.
```java
class Solution {
    public int[] finalPrices(int[] prices) {
        int n = prices.length;
        int[] res = new int[n];
        Stack<Integer> stack = new Stack<>();
        for (int i = 0; i < n; i++) {
            int num = prices[i];
            while (!stack.isEmpty() &&  num <= prices[stack.peek()]) {
                res[stack.peek()] = prices[stack.peek()] - num;
                stack.pop();
            }
            stack.push(i);
        }
        
        while (!stack.isEmpty()) {
            int index = stack.pop();
            res[index] = prices[index];
        }
        
        return res;
    }
}
```
two pointers. It is actually easier than a monotonic solution and runs faster.
```java
class Solution {
    public int[] finalPrices(int[] prices) {
        int days = prices.length;
        int[] res = prices;
        for (int today = 0, nextLowerDay; today < days - 1; today++) {
            nextLowerDay = today + 1;
            while (nextLowerDay < days && prices[today] < prices[nextLowerDay]) {
                nextLowerDay++;
            }
            if (nextLowerDay < days && prices[today] >= prices[nextLowerDay]) {
                res[today] = prices[today] - prices[nextLowerDay];
            }
        }
        return res;
    }
}
```

#### [456. 132 Pattern](https://leetcode.com/problems/132-pattern/) Medium
```java
class Solution {
    public boolean find132pattern(int[] nums) {
        Stack<Integer> stack = new Stack<>();
        int middle = Integer.MIN_VALUE;
        for (int i = nums.length - 1; i >= 0; i--) {
            if (nums[i] < middle) {
                return true;
            }
            
            while (!stack.isEmpty() && nums[i] > stack.peek()) {
                middle = stack.pop();
            }            
            stack.push(nums[i]);
        }
        return false;
    }
}
```

#### [316. Remove Duplicate Letters](https://leetcode.com/problems/remove-duplicate-letters/) Medium
#### [1081. Smallest Subsequence of Distinct Characters](https://leetcode.com/problems/smallest-subsequence-of-distinct-characters/) Medium
```java
class Solution {
    public String removeDuplicateLetters(String s) {
        char[] chars = s.toCharArray();
        int[] count = countOccurences(chars);
            
        StringBuilder res = new StringBuilder();
        boolean[] contains = new boolean[26];
        for (char c: chars) {
            // visited
            count[c - 'a']--;
            
            if (contains[c - 'a']) continue;
            
            while (res.length() > 0 && c < res.charAt(res.length() - 1)) {
                // do not pop if there are no duplicates after 
                if (count[peek(res) - 'a'] == 0) break;
                
                pop(res, contains);
            }
            
            push(res, contains, c);
        }
        
        return res.toString();
    }
    
    int[] countOccurences(char[] chars) {
        int[] count = new int[26];
        for (char c: chars) {
            count[c - 'a']++;
        }
        return count;
    }
    
    void push (StringBuilder res, boolean[] inStack, char c) {
        res.append(c);
        inStack[c - 'a'] = true;
    }
    
    void pop (StringBuilder res, boolean[] inStack) {
        char stackTopChar = peek(res);
        res.delete(res.length() - 1, res.length());
        inStack[stackTopChar - 'a'] = false;
    }
    
    char peek (StringBuilder res) {
        return res.charAt(res.length() - 1);
    }
}
```

#### [901. Online Stock Span](https://leetcode.com/problems/online-stock-span/) Medium
Stupid solution. Will cause TLE.
```java
class StockSpanner {
    ArrayList<Integer> prices;
    Stack<Integer> stack;

    public StockSpanner() {
        this.prices = new ArrayList<>();
        this.stack = new Stack<>();
    }
    
    public int next(int price) {
        stack.clear();
        prices.add(price);

        for (int i = prices.size() - 1; i >= 0; i--) {
            if (!stack.isEmpty() && prices.get(i) > price) break;
            stack.push(prices.get(i));
        }

        return stack.size();
    }
}
```
Push a two-element array into stack to store the price and its span at the same time. When popping smaller or equal elements from the stack, the desired span can be calculated from summing up the previous spans.
```java
class StockSpanner {
    Stack<int[]> stack;
    public StockSpanner() {
        this.stack = new Stack<>();
    }
    
    public int next(int price) {
        int span = 1;
        while (!stack.isEmpty() && price >= stack.peek()[0]) {
            span += stack.pop()[1];
        }
        
        stack.push(new int[]{price, span});
        return span;
    }
}
```

#### [1944. Number of Visible People in a Queue](https://leetcode.com/problems/number-of-visible-people-in-a-queue/) Hard
Guess it is not hard enough beacause I worked it out. Haha.
```java
class Solution {
    public int[] canSeePersonsCount(int[] heights) {
        // an increasing order will be maintain in the stack from top to bottom
        Stack<Integer> stack = new Stack<>();
        int[] res = new int[heights.length];
        int visible;
        // iterated from backwards
        // so that heights[i] is the start and all items after it are already iterated
        for (int i = heights.length - 1; i >= 0; i--) {
            visible = 0;
            // shorter people, can be seen
            while (!stack.isEmpty() && heights[i] > heights[stack.peek()]) {
                stack.pop();
                visible++;
            }
            
            // the top person in the stack, if not empty, is higher than current person
            // who will block the view
            if (!stack.isEmpty()) visible++;
            
            res[i] = visible;
            stack.push(i);
        }
        
        return res;
    }
}
```

#### [1130. Minimum Cost Tree From Leaf Values](https://leetcode.com/problems/minimum-cost-tree-from-leaf-values/) Medium
monotonic stack
```java
class Solution {
    public int mctFromLeafValues(int[] arr) {
        Stack<Integer> stack = new Stack<>();
        stack.push(Integer.MAX_VALUE);
        int res = 0;
        for (int i = 0; i < arr.length; i++) {
            while (arr[i] >= stack.peek()) {
                res += stack.pop() * Math.min(stack.peek(), arr[i]);
            }
            
            stack.push(arr[i]);
        }
        
        while (stack.size() > 2) {
            res += stack.pop() * stack.peek();
        }
        return res;
    }
}
```
DP

#### [20. Valid Parentheses](https://leetcode.com/problems/valid-parentheses/description/) Easy
```java
class Solution {
    public boolean isValid(String s) {
        Stack<Character> stack = new Stack<>();
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '(' || c == '[' || c == '{' ) {
                stack.push(c);
                continue;
            }
            if (stack.isEmpty()) return false;
            char popped = stack.pop();
            if ((c == ')' && popped != '(') || (c == ']' && popped != '[') || (c == '}' && popped != '{'))
            return false;
        }

        return stack.isEmpty();
    }
}
```

#### [84. Largest Rectangle in Histogram](https://leetcode.com/problems/largest-rectangle-in-histogram/description/) Hard
```java
class Solution {
    public int largestRectangleArea(int[] heights) {
        Stack<Integer> s = new Stack<>();
        final int n = heights.length;
        int res = 0;
        for (int i = 0; i < n; i++) {
            while (!s.isEmpty() && heights[i] <= heights[s.peek()]) {
                int height = heights[s.pop()];
                int left = s.isEmpty() ? -1 : s.peek();
                res = Math.max(res, (i - left - 1) * height);
                System.out.printf("height: %d area: %d\n", height, (i - left - 1) * height);
            }
            s.push(i);
        }

        while (!s.isEmpty()) {
            int height = heights[s.pop()];
            int left = s.isEmpty() ? -1 : s.peek();
            res = Math.max(res, (n - left - 1) * height);
            System.out.printf("height: %d area: %d\n", height, (n - left - 1) * height);
        }

        return res;
    }
}
```

#### [1441. Build an Array With Stack Operations](https://leetcode.com/problems/build-an-array-with-stack-operations/description/) Medium
```java
class Solution {
    public List<String> buildArray(int[] target, int n) {
        List<String> res = new ArrayList<>();
        for (int i = 0, j = 1; i < target.length; j++) {
            res.add("Push");
            if (j < target[i]) {
                res.add("Pop");
            } else {
                i++;
            }
        }

        return res;
    }
}
```

#### [682. Baseball Game](https://leetcode.com/problems/baseball-game/description/) Easy
```java
class Solution {
    public int calPoints(String[] operations) {
        Stack<Integer> stack = new Stack<>();
        int sum = 0;
        for (String op: operations) {
            System.out.println(op);
            if (op.equals("+")) {
                int temp = stack.pop();
                int combined = stack.peek() + temp;
                sum += combined;
                stack.push(temp);
                stack.push(combined);
                continue;
            }

            if (op.equals("C")) {
 
                sum -= stack.pop();
                continue;
            }

            if (op.equals("D")) {
                sum += stack.peek() * 2;
                stack.push(stack.peek() * 2);
                continue;
            }

            sum += Integer.valueOf(op);
            stack.push(Integer.valueOf(op));
        }

        return sum;
    }
}
```

#### [225. Implement Stack using Queues](https://leetcode.com/problems/implement-stack-using-queues/description/) Easy
```java
class MyStack {
    private Queue<Integer> q;

    public MyStack() {
        q = new LinkedList<>();
    }
    
    public void push(int x) {
        Queue<Integer> temp = new LinkedList<>();
        temp.offer(x);
        for (; !q.isEmpty(); temp.offer(q.poll()));
        q = temp;        
    }
    
    public int pop() {
        return q.poll();
    }
    
    public int top() {
        return q.peek();
    }
    
    public boolean empty() {
        return q.isEmpty();
    }
}
```

#### [239. Sliding Window Maximum](https://leetcode.com/problems/sliding-window-maximum/description/) Hard
deque
```java
class Solution {
    public int[] maxSlidingWindow(int[] nums, int k) {
        Deque<Integer> dq = new LinkedList<>();
        int[] res = new int[nums.length - k + 1];
        for (int i = 0; i < nums.length; i++) {
            // remove outdated indices
            if (!dq.isEmpty() && dq.peek() < i - k + 1) {
                dq.pop();
            }
            // remove smaller values inside the window
            while (!dq.isEmpty() && nums[i] >= nums[dq.peekLast()]) {
                dq.removeLast();
            }
            dq.add(i);
            if (i + 1 >= k) {
                res[i - k + 1] = nums[dq.peek()];
            }
        }
        return res;
    }
}
```

#### [1696. Jump Game VI](https://leetcode.com/problems/jump-game-vi/description/) Medium
```java
class Solution {
    public int maxResult(int[] nums, int k) {
        Deque<Integer> dq = new LinkedList<>();
        // int[] res = new int[nums.length];
        // jump to the starting point first
        dq.add(0);
        for (int i = 1; i < nums.length; i++) {
            if (!dq.isEmpty() && dq.peek() < i - k) {
                dq.pop();
            }
            nums[i] += nums[dq.peek()];
            while (!dq.isEmpty() && nums[dq.peekLast()] <= nums[i]) {
                dq.removeLast();
            }
            dq.add(i);
        }

        return nums[nums.length - 1];
    }
}
```