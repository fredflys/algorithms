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