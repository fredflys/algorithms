## Tree

<img src="https://s2.loli.net/2022/01/09/DJmCGL5rSpwuMKs.png" alt="image-20220109132224589" style="zoom:50%;" /> 

<img src="https://s2.loli.net/2022/01/09/cWtZSIj36khqFdC.png" alt="image-20220109130337781" style="zoom:50%;" /> 

Categories:

1. get a value or path
2. structure the tree in a different way
3. binary search tree

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

Divide and conquer 

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

Divide and Conquer, BST, Iteration

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

A simpler implementation: only unused nodes are put in the stack

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

#### [596 · Minimum Subtree - LintCode](https://www.lintcode.com/problem/596/)

Divide and Conquer, recursion

```java
public class Solution {
    private int minSum;
    private TreeNode minRoot;

    public TreeNode findSubtree(TreeNode root) {
        minSum = Integer.MAX_VALUE;
        minRoot = null;
        getTreeSum(root);
        return minRoot;
    }

    private int getTreeSum(TreeNode root) {
        if(root == null)
            return 0;
        
        int leftSum = getTreeSum(root.left);
        int rightSum = getTreeSum(root.right);
        int rootSum = leftSum + rightSum + root.val;

        if(rootSum < minSum) {
            minSum = rootSum;
            minRoot = root;
        }

        return rootSum;
    }
}
```
#### [236. Lowest Common Ancestor of a Binary Tree](https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-tree/) <span style="color:orange">Medium</span>

Divide and Conquer, Recursion

```python
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if root is None:
            return None
        if root == p or root == q:
            return root

        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)

        # p and q exists on left and right tree respectively, so root is the common ancestor 
        if left and right:
            return root
        # p and q exists only on left tree, then left is the common ancesstor
        if left:
            return left
        # p and q exists only on right tree, then right is the common ancestor
        if right:
            return right
        # p and q exists on neither side of the tree, then there is no common ancestor
        return None
```
#### [1644. Lowest Common Ancestor of a Binary Tree II](https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-tree-ii/) <span style="color:green">Easy</span>

Divide and Conquer

```python
class Solution:
    def lowestCommonAncestorII(self, root, A, B):
        parents = set()
        current = A

        while current:
            parents.add(current)
            current = current.parent

        current = B
        while current:
            if current in parents:
                return current
            current = current.parent
        
        return None
```

Iteration

```python
class Solution:
    def lowestCommonAncestorII(self, root, A, B):
        def get_ancestors(node):
            ancestors = []
            current = node

            while current:
                ancestors.append(current)
                current = current.parent
            return ancestors
        
        def get_lowest_common_ancestor(ans1, ans2):
            common = None
            for _ in range(min(len(ans1), len(ans2))):
                a = ans1.pop()
                b = ans2.pop()
                if a == b:
                    common = a
            return common

        ancestors_A = get_ancestors(A)
        ancestors_B = get_ancestors(B)
        return get_lowest_common_ancestor(ancestors_A, ancestors_B)
        
```

#### [1650. Lowest Common Ancestor of a Binary Tree III](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree-iii/) <span style="color:orange">Medium</span>

Unlike 236, A or B might not exist on the tree. So this must be taken into consideration in writing the recursive solution.

Divide and Conquer, recursion

```python
class Solution:
    def lowestCommonAncestor3(self, root: 'TreeNode', A: 'TreeNode', B: 'TreeNode') -> 'TreeNode':
        exists_a, exists_b, lca = self.find(root, A, B)
        if exists_a and exists_b:
        	return lca
        return None
    
    def find(self, root: 'TreeNode', A: 'TreeNode', B: 'TreeNode') -> 'TreeNode':
        # Neither A nor B can be found on the tree
        if root is None:
            return False, False, None
        exists_left_a, exists_left_b, left_lca = self.find(root.left, A, B)
        exists_right_a, exists_right_b, right_lca = self.find(root.right, A, B)
        
        # A on the left or right tree, or root itself is A, then A must exist
        exists_a = exists_left_a or exists_right_a or root == A
        exists_b = exists_left_b or exists_right_b or root == B
        
        if root == A or root == B:
            return exists_a, exists_b, root
        
        if left_lca and right_lca:
            return exists_a, exists_b, root
        
        if left_lca:
            return exists_a, exists_b, left_lca
        
        if right_lca:
            return exists_a, exists_b, right_lca
        
        return exists_a, exists_b, None
```

#### [114. Flatten Binary Tree to Linked List](https://leetcode-cn.com/problems/flatten-binary-tree-to-linked-list/) <span style="color:orange">Medium</span>

The problem can be better understood this way: DFS to traverse the tree in preorder and get the result one by one.

```python
class Solution:
    def flatten(self, root: TreeNode) -> None:
        self.flatten_and_return_root(root)

    def flatten_and_return_root(self, root):
        # no nodes to flatten
        if root is None:
            return None
        
        # the last node on its left tree
        left_last = self.flatten_and_return_root(root.left)
        # the last node on its right tree
        right_last = self.flatten_and_return_root(root.right)
        # flattening steps
        # if there is no left tree, then it's alreay flattened
        # otherwise 
        if left_last:
            # link left end to right start
            left_last.right = root.right
            # linke root to left start
            root.right = root.left
            # make left empty
            root.left = None
        
        if right_last:
            return right_last
        if left_last:
            return left_last
        return root
```

#### [230. Kth Smallest Element in a BST ](https://leetcode.com/problems/kth-smallest-element-in-a-bst/) <span style="color:orange">Medium</span>

A non-recursive implementation of Inorder Traversal

[<img src="https://s4.ax1x.com/2022/01/13/7MyaH1.png" alt="7MyaH1.png" style="zoom:50%;" />](https://imgtu.com/i/7MyaH1)

```python
class Solution:
    def kthSmallest(self, root, k):
        count = 0
        stack = []
        node = root
        while node or stack:
            # left 
            while node:
                stack.append(node)
                node = node.left
            # root
            node = stack.pop()
            count += 1
            if count == k:
                return node.val
            # right
            node=node.right
```

Recursion

```python
class Solution:
    def kthSmallest(self, root, k):
        self.result = None
        self.k = k
        self.findKthNode(root)
        return self.result
        
    def findKthNode(self, root):
        if not root:
            return
        self.findKthNode(root.left)
        self.k -= 1
        if self.k == 0:
            self.result = root.val
            return
        self.findKthNode(root.right)
```



#### [270. Closest Binary Search Tree Value](https://leetcode.com/problems/closest-binary-search-tree-value/)  <span style="color:orange">Medium</span>

```java
public class Solution {
    public int closestValue(TreeNode root, double target) {
        if (root == null)
            return 0;

        TreeNode lowerBound = lowerBound(root, target);
        TreeNode upperBound = upperBound(root, target);
        
        if (lowerBound == null)
            return upperBound.val;

        if (upperBound == null)
            return lowerBound.val;

        if (target - lowerBound.val > upperBound.val - target)
            return upperBound.val;

        return lowerBound.val;
    }

    public TreeNode lowerBound(TreeNode root, double target) {
        if (root == null)
            return null;
        
        // current val is still bigger, then search in the left part
        // until current val is no less than the target
        if (root.val > target)
            return lowerBound(root.left, target);
        
        // try to get the upper limit in the lower part
        TreeNode boundNode = lowerBound(root.right, target);

        // if a upper limit exists
        if (boundNode != null)
            return boundNode;
        
        // back to last level and return
        return root;
    }

    public TreeNode upperBound(TreeNode root, double target) {
        if (root == null)
            return null;

        // current val is still smaller, then search in the right part
        // until current val is no bigger than the target
        if (root.val < target)
            return upperBound(root.right, target);

        TreeNode boundNode = upperBound(root.left, target);

        // if a upper limit exists
        if (boundNode != null)
            return boundNode;
        
        // back to last level and return
        return root;
    }
}
```

Iteration

Consider the test case below:

target = 4.1

 		3

​	2 	4

1

node.val < target True node:  3  lower:  3  upper:  3
node.val < target True node:  4  lower:  3  upper:  3
node:  None  lower:  4  upper: 3     

```python
class Solution:
    def closestValue(self, root, target):
        upper = root
        lower = root
        # don't change parameter's value
        node = root

        while node:
            if node.val < target:
                # lower is the biggest in the left part
                lower = node
                node = node.right
            elif node.val > target:
                # upper is the smallest in the right part
                upper = node
                node = node.left
            else:
                return node.val
        
        # there is a chance that upper is lower is actually upper
        # so comparison must be made between absolute values 
        if abs(upper.val - target) < abs(target - lower.val):
            return upper.val
        return lower.val
```

#### [272. Closest Binary Search Tree Value II](https://leetcode.com/problems/closest-binary-search-tree-value-ii/)

Similar to [658. Find K Closest Elements](https://leetcode.com/problems/find-k-closest-elements/)

