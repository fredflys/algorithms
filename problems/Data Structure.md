#### [155. Min Stack](https://leetcode.com/problems/min-stack/) Easy
User another stack to save min values until the latest push and align the min stack with the main stack.
```java
class MinStack {
    Stack<Integer> stack, minStack;

    public MinStack() {
        stack = new Stack<Integer>();
        minStack = new Stack<Integer>();
    }
    
    public void push(int val) {
        stack.push(val);
        if ( minStack.empty() || val < minStack.peek() ) {
            minStack.push(val);
        } else {
            minStack.push(minStack.peek());
        }
    }
    
    public void pop() {
        minStack.pop();
        stack.pop();
    }
    
    public int top() {
        return stack.peek();
    }
    
    public int getMin() {
        return minStack.peek();
    }
}
```

Reduce memory usage: push value into the min stack only when it is smaller to avoid repetition
```java
class MinStack {
    Stack<Integer> stack, minStack;

    ...

    public void push(int val) {
        stack.push(val);
        if ( minStack.empty() || val <= minStack.peek() ) {
            minStack.push(val);
        }
    }
    
    public void pop() {
        int popped = stack.pop();
        if ( minStack.peek() == popped ) {
            minStack.pop();
        }
    }
    
    ...
}
```

[LintCode 859 · Max Stack](https://www.lintcode.com/problem/859/) Hard
```java
class MaxStack {
    Stack<Integer> stack, maxStack;

    public MaxStack() {
        stack = new Stack<Integer>();
        maxStack = new Stack<Integer>();
    }

    /*
     * @param number: An integer
     * @return: nothing
     */    
    public void push(int x) {
        int max = maxStack.isEmpty() ? x : maxStack.peek();
        maxStack.push(max > x ? max : x);
        stack.push(x);
    }

    public int pop() {
        maxStack.pop();
        return stack.pop();
    }

    /*
     * @return: An integer
     */    
    public int top() {
        return stack.peek();
    }

    /*
     * @return: An integer
     */    
    public int peekMax() {
        return maxStack.peek();
    }

    /*
     * @return: An integer
     */    
    public int popMax() {
        int max = peekMax();
        Stack<Integer> buffer = new Stack();
        // if max is not on the top
        while ( top() != max ) {
            buffer.push(pop());
        }
        // now max is on the top
        pop();
        while ( !buffer.isEmpty() ) {
            push(buffer.pop());
        }
        return max;
    }
}
```

soft delete: heap (find max) + stack + hashset (mark to delay deletion) + incremental id (deal with repetitive numbers)
popMax will involve popping bulk numbers, which requires O(n) time, so there has to be a way to better target the max and avoid popping in bulk

```python
import heapq

class MaxStack:
    
    def __init__(self):
        self.stack = []
		# min heap with negative numbers, this is used to retrieve the max number
        self.heap = []
		# a set is used to note down which item is deleted so that soft deletion is possible
        self.popped_set = set()
        self.count = 0
    
    """
    @param: number: An integer
    @return: nothing
    """
    def push(self, x):
        # min heap, make x negative number so that top number stays on the top
        item = (-x, -self.count)
        self.stack.append(item)
        heapq.heappush(self.heap, item)
        self.count += 1

    """
    @return: An integer
    """
    # pop from stack
    def pop(self):
        self._update_stack_from_popped()
        top_item = self.stack.pop()
        self.popped_set.add(top_item)
        return -top_item[0]

    """
    @return: An integer
    """
    def top(self):
        self._update_stack_from_popped()
        top_item = self.stack[-1]
        return -top_item[0]

    """
    @return: An integer
    """
    def peekMax(self):
        self._update_heap_from_popped()
        max_item = self.heap[0]
        return -max_item[0]

    """
    @return: An integer
    """
    def popMax(self):
        self._update_heap_from_popped()
        max_item = heapq.heappop(self.heap)
        self.popped_set.add(max_item)
        return -max_item[0]

    def _update_stack_from_popped(self):
        while self.stack and self.stack[-1] in self.popped_set:
            self.popped_set.remove(self.stack[-1])
            self.stack.pop()

    def _update_heap_from_popped(self):
        while self.heap and self.heap[0] in self.popped_set:                    
            self.popped_set.remove(self.heap[0])
            heapq.heappop(self.heap)
```

[232. Implement Queue using Stacks](https://leetcode.com/problems/implement-queue-using-stacks/) Easy	
```python
class MyQueue:

    def __init__(self):
        self.main = []
        self.buffer = []
    
    # newly added item
    def push(self, x: int) -> None:
        self.main.append(x)
    
    # retrieve the oldest itemb
    def pop(self) -> int:
        self.move()
        return self.buffer.pop()
    
    # peek the oldest item
    def peek(self) -> int:
        self.move()
        return self.buffer[-1]

    def empty(self) -> bool:
        return not self.main and not self.buffer
    
    # by popping and re-inserting into another stack will reverse the stack order, which is euqal to a queue
    def move(self):
        if self.buffer:
            return
        
        while self.main:
            self.buffer.append(self.main.pop())
```

[88. Merge Sorted Array](https://leetcode.com/problems/merge-sorted-array/) Easy
```java
class Solution {
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        int index = nums1.length;

        /**
            m     i
        1 2 3 0 0 0
        2 5 6
            n

            m i    
        1 2 3 0 5 6
        2 5 6
        n

          mi    
        1 2 2 3 5 6
        2 5 6
       n
        **/
        
        // put elements in nums2 in nums1 so when n is run out loop stops
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
three pointers
```java
class Solution {
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        int i = m - 1;
        int j = n - 1;
        int k = m + n - 1;
        
        // fill nums1 from end to start
        // loop stops when the first position is filled
        while (k >= 0) {
            if (i >= 0 && j >= 0) {
                if (nums1[i] > nums2[j]) {
                    nums1[k] = nums1[i--];
                } else {
                    nums1[k] = nums2[j--];
                }
            } else if (i >= 0) {
                nums1[k] = nums1[i--];
            } else {
                nums1[k] = nums2[j--];
            }
            // k i
            --k;
        }
    }
}
```

[21. Merge Two Sorted Lists](https://leetcode.com/problems/merge-two-sorted-lists/) Easy