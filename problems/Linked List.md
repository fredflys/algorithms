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
ref: https://labuladong.github.io/algo/1/4/
two pointers
```java
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
two-pointers
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



#### [21. Merge Two Sorted Lists](https://leetcode.com/problems/merge-two-sorted-lists/) Easy
without a dummy node, you have to be very careful with boundary cases
```java
class Solution {
    public ListNode mergeTwoLists(ListNode node1, ListNode node2) {
        ListNode start;
        if (node1 == null) {
            return node2;
        }
        if (node2 == null) {
            return node1;
        }
        
        if (node1.val < node2.val) {
                start = node1;
                node1 = node1.next;
        } else {
                start = node2;
                node2 = node2.next;
        }
        
        
        ListNode current = start;
        while (node1 != null && node2 != null) {
            if (node1.val < node2.val) {
                current.next = node1;
                node1 = node1.next;
            } else {
                current.next = node2;
                node2 = node2.next;
            }
            current = current.next;
        }
        
        while (node1 != null) {
            current.next = node1;
            node1 = node1.next;
            current = current.next;
        }
        
        while (node2 != null) {
            current.next = node2;
            node2 = node2.next;
            current = current.next;
        }

        return start;
    }
}
```
with a dummy node, things are much simpler. Whenever a new linked list is needed, it's better practice to creat it with a dummy node.
```java
class Solution {
    public ListNode mergeTwoLists(ListNode list1, ListNode list2) {
        // virtual start
        ListNode dummy = new ListNode(-1);
        ListNode p = dummy, p1 = list1, p2 = list2;
        
        while (p1 != null && p2 != null) {
            if (p1.val < p2.val) {
                p.next = p1;
                p1 = p1.next;
            } else {
                p.next = p2;
                p2 = p2.next;
            }
            
            // move pointer in the new linked list
            p = p.next;
        }
        
        // p1 or p2 may not be exhausted
        if (p1 != null) {
            p.next = p1;
        }
        
        if (p2 != null) {
            p.next =p2;
        }
        
        return dummy.next;
    }
}
```

#### [86. Partition List](https://leetcode.com/problems/partition-list/) Medium
dummy node 
```java
class Solution {
    public ListNode partition(ListNode head, int x) {
        ListNode dummy1 = new ListNode(-1);
        ListNode dummy2 = new ListNode(-1);
        ListNode p = head, p1 = dummy1, p2 = dummy2;
        
        while (p != null) {
            if (p.val < x) {
                p1.next = p;
                p1 = p1.next;
            } else {
                p2.next = p;
                p2 = p2.next;
            }
            
            // cut off original link
            ListNode temp = p.next;
            p.next = null;
            
            // move to next node
            p = temp;
        }
        
        p1.next = dummy2.next;
        return dummy1.next;
    }
}
```

#### [23. Merge k Sorted Lists](https://leetcode.com/problems/merge-k-sorted-lists/) Hard
Min heap, dummy node
```java
class Solution {
    public ListNode mergeKLists(ListNode[] lists) {
        int k = lists.length;
        if (k == 0) {
            return null;
        }
        
        ListNode dummy = new ListNode(-1), p = dummy;
        // min heap
        PriorityQueue<ListNode> heap = new PriorityQueue<>(
            k, (a, b) -> (a.val - b.val)
        );
        
        for (ListNode head: lists) {
            if (head != null) {
                heap.add(head);
            }
        }
        
        while (!heap.isEmpty()) {
            ListNode node = heap.poll();
            p.next = node;
            if (node.next != null) {
                heap.add(node.next);
            }
            p = p.next;
        }
        
        return dummy.next;
    }
}
```
merge sort
```java
class Solution {
    public ListNode mergeKLists(ListNode[] originalLists) {
        if (originalLists == null || originalLists.length == 0) {
            return null;
        }
        
        // it's better to leave the input list unchanged
        List<ListNode> lists = new ArrayList<>(Arrays.asList(originalLists));
        
        return mergeHelper(lists, 0, lists.size() - 1);
    }
    
    private ListNode mergeHelper(List<ListNode> lists, int start, int end) {
        if (start == end) {
            return lists.get(start);
        }
        
        int mid = start + (end - start) / 2;
        ListNode left = mergeHelper(lists, start, mid);
        ListNode right = mergeHelper(lists, mid + 1, end);
        return merge(left, right);
    }

    // merge two lists
    public ListNode merge(ListNode node1, ListNode node2) {
        ListNode dummy = new ListNode(0);
        ListNode current = dummy;
        while (node1 != null && node2 != null) {
            if (node1.val < node2.val) {
                current.next = node1;
                node1 = node1.next;
            } else {
                current.next = node2;
                node2 = node2.next;
            }

            current = current.next;
        }

        // there is no need to loop through the remaining list
        // linking the head is enough
        if (node1 != null) {
            current.next = node1;
        } else {
            current.next = node2;
        }

        return dummy.next;
    }
}
```

#### [160. Intersection of Two Linked Lists](https://leetcode.com/problems/intersection-of-two-linked-lists/) Easy
```java
public class Solution {
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        ListNode p1 = headA, p2 = headB;
        
        while (p1 != p2) {
            if (p1 == null) {
                p1 = headB;
            } else {
                p1 = p1.next;
            }
            
            if (p2 == null) {
                p2 = headA;
            } else {
                p2 = p2.next;
            }
        }
        
        return p1;
    }
}
```

#### [2. Add Two Numbers](https://leetcode.com/problems/add-two-numbers/) Medium
Simply adding every digit is not enough. Two problems to consider:
1. There could be carries doing additions. A carry should be transferred to the next digit.
2. Two numbers may have different lengths.
```java
class Solution {
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode p1 = l1, p2 = l2, dummy = new ListNode(-1), p3 = dummy;
        int toNext = 0, sum = 0;
        while (p1 != null && p2 != null) {
            sum = p1.val + p2.val;
            if (toNext != 0) {
                sum++;
                toNext--;
            }
            if (sum >= 10) {
                sum -= 10;
                toNext++;
            }
            p3.next = new ListNode(sum);
            
            p1 = p1.next;
            p2 = p2.next;
            p3 = p3.next;
        }
        
        while (p1 != null) {
            sum = p1.val;
            
            if (toNext != 0) {
                sum++;
                toNext--;
            }
            if (sum >= 10) {
                sum -= 10;
                toNext++;
            }
            
            p3.next = new ListNode(sum);
            p1 = p1.next;
            p3 = p3.next;
        }
        
        while (p2 != null) {
            sum = p2.val;
            
            if (toNext != 0) {
                sum++;
                toNext--;
            }
            if (sum >= 10) {
                sum -= 10;
                toNext++;
            }
            
            p3.next = new ListNode(sum);
            p2 = p2.next;
            p3 = p3.next;
        }
        
        if (toNext != 0) {
            p3.next = new ListNode(1);
            toNext--;
        }
        
        return dummy.next;
    }
}
```
```java
class Solution {
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode p1 = l1, p2 = l2, dummy = new ListNode(-1), p = dummy;
        int carry = 0;
        int sum = 0;
        while (p1 != null || p2 != null || carry > 0) {
            sum = carry;
            if (p1 != null) {
                sum += p1.val;
                p1 = p1.next;
            }
            if (p2 != null) {
                sum += p2.val;
                p2 = p2.next;
            }
            
            carry = saum / 10;
            sum = sum % 10;
            
            p.next = new ListNode(sum);
            p = p.next;
        }
        return dummy.next;
    }
}
```

#### [234. Palindrome Linked List](https://leetcode.com/problems/palindrome-linked-list/) Easy
```java
class Solution {
    public boolean isPalindrome(ListNode head) {
        StringBuilder sb = new StringBuilder();
        while (head != null) {
            sb.append(head.val);
            head = head.next;
        }
        
        String res = sb.toString();        
        int i = 0, j = res.length() - 1;
        while (i < j) {
            if (res.charAt(i) != res.charAt(j)) {
                return false;
            }
            
            i++;
            j--;
        }
        
        return true;
    }
}
```
recursion
```java
class Solution {
    public boolean isPalindrome(ListNode head) {
        front = head;
        return isPali(front);
    }
    
    ListNode front;
    boolean isPali(ListNode back) {
        if (back == null) {
            return true;
        }
        
        boolean equalBefore = isPali(back.next);
        boolean equalNow = front.val == back.val;
        
        front = front.next;
        
        return equalBefore && equalNow;
    }
}
```
two pointers, recursion
find the middle node, reverse the latter part and then compare the former half and latter half
```java
class Solution {
    public boolean isPalindrome(ListNode head) {
        ListNode middleNode = getMiddleNode(head);
        ListNode newHead = reverseListNoRecursion(middleNode);
        return compareTwoParts(head, newHead);
    }
    
    ListNode getMiddleNode(ListNode head) {
        ListNode slow = head, fast = head;
        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            slow = slow.next;
        }
        
        return slow;
    }
    
    ListNode reverseList(ListNode prev, ListNode current) {
        if (current == null) {
            return prev;
        }
        
        ListNode next = current.next;
        current.next = prev;
        return reverseList(current, next);
    }
        
    // without recursion, reversing will save a lot of space becuase there is no stack calaling involved
    ListNode reverseListNoRecursion(ListNode current) {
        ListNode prev = null;
         while (current != null) {
             ListNode next = current.next;
             current.next = prev;
             prev = current;
             current = next;
         }
         return prev;
    }
    
    boolean compareTwoParts(ListNode first, ListNode second) {
        if (second == null) {
            return true;
        }
        
        if (first.val == second.val) {
            return compareTwoParts(first.next, second.next);
        }
        
        return false;
    }
}
```

#### [204. Count Primes](https://leetcode.com/problems/count-primes/) Medium
```java
class Solution {
    public int countPrimes(int n) {
        boolean[] isPrime = new boolean[n];
        // pretend every number less than n is prime and then start exclusion
        Arrays.fill(isPrime, true);
        
        // 
        for (int i = 2; i * i < n; i++) {
            if (isPrime[i]) {
                for (int j = i * i; j < n; j += i) {
                    isPrime[j] = false;
                }    
            }
        }
        
        int count = 0;
        for (int i = 2; i < n; i++) {
            if (isPrime[i]) {
                count++;
            }
        }
        
        return count;
    }
}
```

### Ugly Number Series
#### [263. Ugly Number](https://leetcode.com/problems/ugly-number/) Easy
```python
class Solution:
    def isUgly(self, n: int) -> bool:
        if n == 0:
            return False
            
        while n != 1:
            result = self.divide(n)
            if result == -1:
                return False
            n = result
        return True

    def divide(self, n):
        factors = [2, 3, 5]
        for f in factors:
            if n % f == 0:
                return n / f
        return -1
```

#### [264. Ugly Number II](https://leetcode.com/problems/ugly-number-ii/) Medium
Heap, BFS
```java
class Solution {
    public int nthUglyNumber(int n) {
        // nth ugly number might be a very large number, so it's better to use long
        PriorityQueue<Long> heap = new PriorityQueue<>();
        Set<Long> seen = new HashSet<>();
        // initialization
        heap.add(1L);
        seen.add(1L);

        int[] factors = new int[] {2, 3, 5};
        long ugly = 1, newUgly;

        // to get the nth number, n - 1 times are needed
        for (int i = 0; i < n; ++i) {
            ugly = heap.poll();
            for (int factor: factors) {
                newUgly = ugly * factor;
                if (!seen.contains(newUgly)) {
                    heap.add(newUgly);
                    seen.add(newUgly);
                }
            }
        }

        return (int)ugly;
    }
}
```
DP
```python
class Solution:
    def nthUglyNumber(self, n: int) -> int:
        """
        dp[0] = 1
        dp[1[ = min(dp[0] * 2, dp[0] * 3, dp[0] * 5) = 2
        dp[2] = min(dp[1} * 2，dp[0] * 3, dp[0] * 5) = 3
        dp[3] = min(dp[1} * 2，dp[1] * 3, dp[0] * 5) = 4
        dp[4] = min(dp[2} * 2，dp[1] * 3, dp[0] * 5) = 5

        """
        dp = [0] * n
        dp[0] = 1
        p2, p3, p5 = 0, 0, 0
        for i in range(1, n):
            dp[i] = min(dp[p2] * 2, dp[p3] * 3, dp[p5] * 5)
            if dp[i] == 2 * dp[p2]:
                p2 += 1
            if dp[i] == 3 * dp[p3]:
                p3 += 1
            if dp[i] == 5 * dp[p5]:
                p5 += 1
        return dp[n - 1]
```
merge linked list
```java
class Solution {
    public int nthUglyNumber(int n) {
    // 可以理解为三个指向有序链表头结点的指针
    int p2 = 1, p3 = 1, p5 = 1;
    // 可以理解为三个有序链表的头节点的值
    int product2 = 1, product3 = 1, product5 = 1;
    // 可以理解为最终合并的有序链表（结果链表）
    int[] ugly = new int[n + 1];
    // 可以理解为结果链表上的指针
    int p = 1;

    // 开始合并三个有序链表，找到第 n 个丑数时结束
    while (p <= n) {
        // 取三个链表的最小结点
        int min = Math.min(Math.min(product2, product3), product5);
        // 将最小节点接到结果链表上
        ugly[p] = min;
        p++;
        // 前进对应有序链表上的指针
        if (min == product2) {
            product2 = 2 * ugly[p2];
            p2++;
        }
        if (min == product3) {
            product3 = 3 * ugly[p3];
            p3++;
        }
        if (min == product5) {
            product5 = 5 * ugly[p5];
            p5++;
        }
        System.out.println(Arrays.toString(ugly));
        System.out.printf("%s %s %s \n", product2, product3, product5);
    }
        
    return ugly[n];
    }
}
```

#### [313. Super Ugly Number](https://leetcode.com/problems/super-ugly-number/) Medium
```java
class Solution {
    public int nthSuperUglyNumber(int n, int[] primes) {
        PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> {
                return a[0] - b[0];
            }
        );
        
        // {product, prime, index}
        for (int i = 0; i < primes.length; i++) {
            pq.offer(new int[]{1, primes[i], 1});
        }
        
        int[] ugly = new int[n + 1];
        int p = 1;
        int[] current;
        int product, prime, index;
        while (p <= n) {
            current = pq.poll();
            product = current[0];
            prime = current[1];
            index = current[2];
            
            if (product != ugly[p - 1]) {
                ugly[p] = product;
                p++;
            }
        
            int[] next = new int[]{ugly[index] * prime, prime, index + 1};
            pq.offer(next);
        }
        
        return ugly[n];
    }
}
```

#### [1201. Ugly Number III](https://leetcode.com/problems/ugly-number-iii/) Medium
```java

```

#### [706. Design HashMap](https://leetcode.com/problems/design-hashmap/) Easy
use linked list on array to resolve collision conflict
e.g.
20, 23, 46, 71, 32, 66  hased modulo 10 
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
-1  -1 -1 -1       -1
20  71 32 23       46
                   66 
```java
class MyHashMap {
    ListNode[] nodes;

    public MyHashMap() {
        nodes = new ListNode[10009];
    }
    
    public int get(int key) {
        int index = getIndex(key);
        ListNode prev = findPrev(index, key);
        
        return prev.next == null ? -1 : prev.next.val;
    }
    
    public void put(int key, int value) {
        int index = getIndex(key);
		ListNode prev = findPrev(index, key);
        
        if (prev.next == null) {
            prev.next = new ListNode(key, value);
        } else {
            prev.next.val = value;
        }
    }
    
    public void remove(int key) {
        int index = getIndex(key);
        ListNode prev = findPrev(index, key);
        
        if (prev.next != null) {
            // deletion simply means cutting off the link, which is done by linking previous node to next's next
            prev.next = prev.next.next;
        }
    }
    
    private ListNode findPrev(int index, int key) {
        if (nodes[index] == null) {
            nodes[index] = new ListNode(-1, -1);
            return nodes[index];
        }
        
        ListNode prev = nodes[index];
        while (prev.next != null && prev.next.key != key) {
            prev = prev.next;
        }
        
        return prev;
    }
    
    private int getIndex(int key)
	{	
		return Integer.hashCode(key) % nodes.length;
	}
    
    private static class ListNode {
		int key, val;
		ListNode next;

		ListNode(int key, int val)
		{
			this.key = key;
			this.val = val;
		}
	}
}
```

#### [83. Remove Duplicates from Sorted List](https://leetcode.com/problems/remove-duplicates-from-sorted-list/) Easy
two pointers
```java
class Solution {
    public ListNode deleteDuplicates(ListNode head) {
        ListNode dummy = new ListNode(-101, head);
        ListNode slow = dummy, fast = dummy;
        while (fast.next != null) {
            fast = fast.next;
            if (fast.val == slow.val) {
                slow.next = fast.next;
            } else {
                // make sure that slow pointer is always one step slower
                slow = slow.next;   
            }
        }
        
        return dummy.next;
    }
}
```
recursive
```java
class Solution {
    public ListNode deleteDuplicates(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        
        if (head.val == head.next.val) {
            while (head.next != null && head.val == head.next.val) {
                head = head.next;
            }
            // do not skip the remaining duplicate node
            // this is the only difference from the next problem
            return deleteDuplicates(head);
        } else {
            head.next = deleteDuplicates(head.next);
            return head;
        }
    }
}
```

#### [82. Remove Duplicates from Sorted List II](https://leetcode.com/problems/remove-duplicates-from-sorted-list-ii/) Medium
two pointers
two traversals are nedded. not so fast
```java
class Solution {
    Map<Integer, Integer> map;
    
    public ListNode deleteDuplicates(ListNode head) {
        map = new HashMap<>();
        traverse(head);
        ListNode dummy = new ListNode(-101, head);
        ListNode slow = dummy, fast = head;
        while (fast != null) {
            if (map.get(fast.val) == 1) {
                slow.next = fast;
                slow = slow.next;   
            }
            fast = fast.next;
        }
        
        // distinct nodes end at slow
        slow.next = null;
        
        return dummy.next;
    }
    
    void traverse(ListNode head) {
        if (head == null) {
            return;
        }
        
        map.put(head.val, map.getOrDefault(head.val, 0) + 1);
        traverse(head.next);
    }
}
```
only one traversal is needed
```java
class Solution {
    public ListNode deleteDuplicates(ListNode head) {
        ListNode dummy = new ListNode(-101, head);
        ListNode slow = dummy, fast = head;
        while (fast != null) {
            // duplicate nodes encountered
            while (fast.next != null && fast.val == fast.next.val) {
                fast = fast.next;
            }
            
            // one step away is the normal gap
            // fast is not moving too fast
            // so move slow now 
            if (slow.next == fast) {
                slow = slow.next;
            } else {
                // fast is now at the end of duplicates
                // linke slow to fast.next
                slow.next = fast.next;
            }
            
            fast = fast.next;
        }
        
        return dummy.next;
    }
}
```
recursive, very intuitive and straightforward
```java
class Solution {
    public ListNode deleteDuplicates(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        
        if (head.val != head.next.val) {
            // distinct nodes
            // skip it and delete from next node
            head.next = deleteDuplicates(head.next);
            return head;
        } else {
            // skip duplicates until the remaining one
            while (head.next != null && head.val == head.next.val) {
                head = head.next;
            }
            
            // also skip the remaining one so that only distinct nodes remain
            return deleteDuplicates(head.next);
        }
    }
}
```

#### [876. Middle of the Linked List](https://leetcode.com/problems/middle-of-the-linked-list/) Easy
two pointers, slow and fast
```java
class Solution {
    public ListNode middleNode(ListNode head) {
        ListNode slow = head, fast = head;
        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            slow = slow.next;
        }
        return slow;
    }
}
```

#### [206. Reverse Linked List](https://leetcode.com/problems/reverse-linked-list/) Easy

My implementation. I treat the reversing process as a whole by converting the linked list to a list, reversing it and building the reversed linked list from scratch. This is inefficient and memory consuming.

```python
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
Iteration and recursion
The whole process is divided into sub-processes chained together.
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
recursive: pay close attention to the definition of the recursive algorithms
1 -> 2 -> 3 -> 4 -> 5 -> 6 -> null
1 -> reverse (2 -> 3 -> 4 -> 5 -> 6 -> null)
h                        l
1 -> 2 <- 3 <- 4 <- 5 <- 6 
     |
    null
1 <=> 2
head.next = head.next.next
1 <-  2
head.next = null
```java
class Solution {
    // reverse the linked list that starts from head and returns the last (new first) node
    public ListNode reverseList(ListNode head) {
        // if there is only one node to be reversed, simply return it
        if (head.next == null) {
            return head;
        }
        
        ListNode last = reverseList(head.next);
        /*
        
        */
        head.next.next = head;
        head.next = null;
        return last;
    }
}
```
recursive: store list nodes in the returned values
```java
// recursive 2
class Solution {
    public ListNode reverseList(ListNode head) {
        return reverse(head).getKey();
    }
    
    
    // reverse the linked list and return the first node and last node after reversal
    Pair<ListNode, ListNode> reverse(ListNode head) {
        if (head == null || head.next == null) {
            return new Pair(head, head);
        }
        
        Pair<ListNode, ListNode> nodes = reverse(head.next);
        nodes.getValue().next = head;
        head.next = null;
        return new Pair<>(nodes.getKey(), head);
    }
}
```

#### [92. Reverse Linked List II(https://leetcode.com/problems/reverse-linked-list-ii/) Medium
```java
class Solution {
    public ListNode reverseBetween(ListNode head, int left, int right) {
        if (left == 1) {
            return reverseN(head, right);
        }
        
        head.next = reverseBetween(head.next, left - 1, right - 1);
        return head;
    }
    
    // reverse the first n nodes in a linked list
    ListNode successor = null;
    ListNode reverseN(ListNode head, int n) {
        if (n == 1) {
            successor = head.next;
            return head;
        }
        
        ListNode last = reverseN(head.next, n - 1);
        head.next.next = head;
        head.next = successor;
        return last;
    }
}
```
Iteration
```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode reverseBetween(ListNode head, int left, int right) {
        ListNode dummy = new ListNode(-501, head);
        ListNode prev = dummy;
        
        // move left - 1 steps to arrive at the previous node from dummy to the node at left
        for (int i = 0; i < left - 1; i++) {
            prev = prev.next;
        }
        ListNode current = prev.next;
        // right - left + 1 nodes 
        // needs right - left swaps
        /* 
            left = 3 right = 6
              p  c  f
            1 2 |10 20| 30 40 5
            from end to start
            c.next = f.next 10 -> 30
            f.next = c.next 20 -> 10
            p.next = f -> 2 -> 30
            1 2 20 10 30 40 5
            
            ...
            1 2 30 20 10 40 5
            1 2 40 30 20 10 5
            
        */
        for (int i = 0; i < right - left; i++) {
            // in every pass, forward will be inserted before current
            ListNode forward = current.next;
            // link end node to successor
            current.next = forward.next;
            // reverse current and forward
            forward.next = prev.next;
            // link precursor to start end
            prev.next = forward;
            
        }
        
        return dummy.next;
    }
}
```

#### [143. Reorder List](https://leetcode.com/problems/reorder-list/) Medium
convert linked list to an array to enable indexing
```java
class Solution {
    public void reorderList(ListNode head) {
        List<ListNode> nodes = buildNodes(head);
        int i = 0, j = nodes.size() - 1;
        while (i < j) {
            nodes.get(i).next = nodes.get(j);
            i++;
            if (i == j) {
                break;
            }
            nodes.get(j).next = nodes.get(i);
            j--;
        }
        // break the possible cycle
        nodes.get(i).next = null;

    }
    
    List<ListNode> buildNodes(ListNode node) {
        List<ListNode> nodes = new ArrayList<>();
        while (node != null) {
            nodes.add(node);
            node = node.next;
        }
        return nodes;
    }
}
```
cut the linked list in half, reverse the latter half and link the two seperate linked lists one by one
cut by half: [876. Middle of the Linked List](https://leetcode.com/problems/middle-of-the-linked-list/)
reverse linked list:  [206. Reverse Linked List](https://leetcode.com/problems/reverse-linked-list/)
Divide the original problem into three sub-problems. Fantastic.
```java
class Solution {
    public void reorderList(ListNode head) {
        ListNode leftMiddle = findLeftMiddle(head);
        
        // divid the linked list by half
        ListNode latterHalfHead = leftMiddle.next;
        // cut by half by disconnect the end node in the first half from the first node in the latter half
        leftMiddle.next = null;
        
        // reverse the latter half
        ListNode reversedLatterHead = reverse(latterHalfHead);
        
        // link the two halves one by one
        mergeOneByOne(head, reversedLatterHead);
    }
    
    
    ListNode findLeftMiddle(ListNode head) {
        ListNode slow = head, fast = head;
        while (fast.next != null && fast.next.next != null) {
            fast = fast.next.next;
            slow = slow.next;
        }
        return slow;
    }
    
    ListNode reverse(ListNode head) {
        ListNode prev = null;
        while (head != null) {
            ListNode next = head.next;
            head.next = prev;
            prev = head;
            head = next;
        }
        return prev;
    }
    
    void mergeOneByOne(ListNode left, ListNode right) {
        while (right != null) {
            /*
            l
            1 2
            5 4 3
            r
            build connection from back
            linking
            5 -> 2
            1 -> 5
            moving
            l: 1 to 2
            r: 5 to 4
            */
            ListNode rightNext = right.next;
            right.next = left.next;
            left.next = right;
            
            left = right.next;
            right = rightNext;   
        }
    }
}
```
recursive
```java
class Solution {
    public void reorderList(ListNode head) {
        reorder(head, head);
    }

    // return the head node that corresponds to the tail node
    ListNode reorder(ListNode head, ListNode tail) {
        // base case: return starting node for last node
        if (tail == null) {
            return head;
        }

        ListNode ret = reorder(head, tail.next);
        if (ret == null) {
            return null;
        }
        if (ret == tail || ret.next == tail) {
            tail.next = null;
            return null;
        }
        
        /*
        1 2 3 4 5

        1 2 3
        5 4
        */
        // 5 -> 2
        tail.next = ret.next;
        // 1 -> 5
        ret.next = tail;
        
        return tail.next;
    }
}
```
recursive: keep reversing from second node
It is easy to understand. But since it involves a lot of reversals, it is running very slow.
1 (5 4 3  2
1 5 (2 3  4
1 5  2 (4 3
```java

class Solution {
    public void reorderList(ListNode head) {
        if (head.next == null) {
            return;
        }
        head.next = dfs(head.next);
    }
    
    // reverse head and return the new head
    ListNode dfs(ListNode head) {
        // base case: return tail node
        if (head == null || head.next == null) {
            return head;
        }
        
        ListNode tail = reverse(head);
        tail.next = dfs(tail.next);
        return tail;
    }
    
    ListNode reverse(ListNode head) {
        ListNode prev = null;
        while (head != null) {
            ListNode next = head.next;
            head.next = prev;
            prev = head;
            head = next;
        }
        return prev;
    }
}
```

#### [25. Reverse Nodes in k-Group](https://leetcode.com/problems/reverse-nodes-in-k-group/) Hard
```java
class Solution {
    public ListNode reverseKGroup(ListNode head, int k) {
        if (head == null) {
            return null;
        }
        
        ListNode start = head, end = head;
        
        for (int i = 0; i < k; i++) {
            // less than k nodes in the current list
            // no reversal is needed
            if (end == null) {
                return head;
            }
            end = end.next;
        }
        
        // now there are k nodes between start and end
        ListNode reversedHead = reverse(start, end);
        // now start is the end of the reversed linked list
        // link it to reversed remaining nodes
        start.next = reverseKGroup(end, k);
        
        return reversedHead;
    }
    
    // revese the whole linked list
    // it is equal to reversing head to null (null not included)
    ListNode reverse(ListNode head) {
        ListNode prev = null, current = head, next = head.next;
        
        while (current != null) {
            next = current.next;
            current.next = prev;
            prev = current;
            current = next;
        }
        
        return prev;
    }
    
    // reverse from head to end (end not included) [head, end)
    ListNode reverse(ListNode head, ListNode tail) {
        ListNode prev = null, current = head, next = head.next;
        
        while (current != tail) {
            next = current.next;
            current.next = prev;
            prev = current;
            current = next;
        }
        
        return prev;
    }
}
```

#### [1290. Convert Binary Number in a Linked List to Integer](https://leetcode.com/problems/convert-binary-number-in-a-linked-list-to-integer/) Easy
bit operation
```java
class Solution {
    public int getDecimalValue(ListNode head) {
        int num = 0;
        for (; head != null; head = head.next) {
            num = num << 1 | head.val;    
        }
        return num;
    }
}
```

#### [1171. Remove Zero Sum Consecutive Nodes from Linked List](https://leetcode.com/problems/remove-zero-sum-consecutive-nodes-from-linked-list/) Medium
```java
class Solution {
    public ListNode removeZeroSumSublists(ListNode head) {
        ListNode dummy = new ListNode(0, head);
        Map<Integer, ListNode> presumNode = new HashMap<>();
        presumNode.put(0, dummy);
        for(int sum = 0; head != null; head = head.next) {
            sum += head.val;
            
            if (presumNode.containsKey(sum)) {
                ListNode node = presumNode.get(sum);
                /*
            sum 0 1 2  0 4    
                0 1 2 -3 4
                       h
                n
                sum 0 is found in the map, have to remove three nodes: 1 2 -3
                node = map.get(sum) -> dummy (starting point: this should not be removed)
        remove  0 2 -3 4
                0 -3 4
                node.next == head is true, stop here
                still, -3 needs to be removed so one more removal should be done 
                */
                for (int currentSum = sum; // initialization
                     node.next != head; // condition
                     presumNode.remove(currentSum += node.next.val),
                     node.next = node.next.next) // action: remove 
                    ;
                node.next = node.next.next;
            } else {
                presumNode.put(sum, head);
            }
        }
        
        return dummy.next;
    }
}
```

#### [237. Delete Node in a Linked List](https://leetcode.com/problems/delete-node-in-a-linked-list/) Medium
lazy deletion
```java
class Solution {
    public void deleteNode(ListNode node) {
        node.val = node.next.val;
        node.next = node.next.next;       
    }
}
```

#### [1669. Merge In Between Linked Lists](https://leetcode.com/problems/merge-in-between-linked-lists/description/) Medium
```java
class Solution {
    public ListNode mergeInBetween(ListNode list1, int a, int b, ListNode list2) {
        int distance = b - a;
        
        ListNode start = list1;
        // start will stop one node before the to-be-removed list
        while (--a >= 1) {
            start = start.next;
        }

        ListNode end = start;
        // from one node before the to-be-removed list to one node after the to-be-removed list
        while (--distance >= -2) {
            end = end.next;
        }

        // connect the start node
        start.next = list2;
        
        while (list2.next != null) {
            list2 = list2.next;
        }

        // connect the end node
        list2.next = end;

        return list1;
    }
}
```

#### [950. Reveal Cards In Increasing Order](https://leetcode.com/problems/reveal-cards-in-increasing-order/description/) Medium
```java
class Solution {
    public int[] deckRevealedIncreasing(int[] deck) {
        Arrays.sort(deck);
        final int n = deck.length;
        Deque<Integer> deq = new LinkedList<>();
        deq.push(deck[n - 1]);
        // reverse the revealing order
        // revealing order: reveal the first element and move the second to the bottom
        // reversed: put the bottom card on the top and put the second to last card on the top
        for (int i = n - 2; i >= 0; i--) {
            deq.addFirst(deq.removeLast());
            deq.addFirst(deck[i]);
        }

        int[] res = new int[n];
        for (int i = 0; i < n; i++) {
            res[i] = deq.poll();
        }
        return res;
    }
}
```
Follow the instructinos to build result array from a sorted deck
res: [2, 0, 0, 0, 0, 0, 0]
Queue: [2, 3, 4, 5, 6, 1] 
res: [2, 0, 3, 0, 0, 0, 0]
Queue: [4, 5, 6, 1, 3] 
res: [2, 0, 3, 0, 5, 0, 0]
Queue: [6, 1, 3, 5] 
res: [2, 0, 3, 0, 5, 0, 7]
Queue: [3, 5, 1] 
res: [2, 0, 3, 11, 5, 0, 7]
Queue: [1, 5] 
res: [2, 13, 3, 11, 5, 0, 7]
Queue: [5] 
```java
class Solution {
    public int[] deckRevealedIncreasing(int[] deck) {
        Arrays.sort(deck);
        final int n = deck.length;
        Queue<Integer> q = new LinkedList<>();
        for (int i = 0; i < n; i++) {
            q.offer(i);
        }

        int[] res = new int[n];
        for (int i = 0, j = 0; i < n ; i++) {
            res[q.poll()] = deck[i];
            if (!q.isEmpty()) {
                q.add(q.poll());
                // System.out.printf("res: %s\n", Arrays.toString(res));
                // System.out.printf("Queue: %s \n", q);
            }    
        } 

        return res;
    }
}
```

#### [622. Design Circular Queue](https://leetcode.com/problems/design-circular-queue/description/) Medium
```java
class MyCircularQueue {
    private int[] q;
    private int rear, size;

    public MyCircularQueue(int k) {
        q = new int[k];
        size = 0;
        // when the first element is inserted, rear becomes 0, which is exactly what it should be
        rear = k - 1;
    }
    
    public boolean enQueue(int value) {
        if (isFull()) return false;

        size++;
        rear = (rear + 1) % q.length;
        q[rear] = value;
        return true;
    }
    
    public boolean deQueue() {
        if (isEmpty()) return false;
        
        size--;
        return true;
    }
    
    public int Front() {
        if (isEmpty()) return -1;

        int front = (rear + 1 - size + q.length) % q.length;
        return q[front];
    }
    
    public int Rear() {
        if (isEmpty()) return -1;

        return q[rear];
    }
    
    public boolean isEmpty() {
        return size == 0;
    }
    
    public boolean isFull() {
        return size == q.length;
    }
}
```

#### [641. Design Circular Deque](https://leetcode.com/problems/design-circular-deque/description/) Medium
```java
class MyCircularDeque {
    private int front, rear;
    private int size;
    private int[] dq;

    public MyCircularDeque(int k) {
        size = k + 1;
        dq = new int[size];
        front = 0;
        rear = 0;
    }

    public boolean insertFront(int value) {
        if (isFull()) return false;

        front = (front - 1 + size) % size;
        dq[front] = value;
        return true;
    }
    
    public boolean insertLast(int value) {
        if (isFull()) return false;

        dq[rear] = value;
        rear = (rear + 1) % size;
        return true;
    }
    
    public boolean deleteFront() {
        if (isEmpty()) return false;

        front = (front + 1) % size;
        return true;
    }
    
    public boolean deleteLast() {
        if (isEmpty()) return false;

        // soft deletion: moving rear pointer instead of actively deleting anything 
        rear = (rear - 1 + size) % size;
        return true;
    }
    
    public int getFront() {
        if (isEmpty()) return -1;

        return dq[front];
    }
    
    public int getRear() {
        if (isEmpty()) return -1;

        return dq[(rear - 1 + size) % size];
    }
    
    public boolean isEmpty() {
        return front == rear;
    }
    
    public boolean isFull() {
        return (rear + 1) % size == front;
    }
}
```