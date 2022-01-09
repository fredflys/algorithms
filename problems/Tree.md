## Tree

<img src="https://s2.loli.net/2022/01/09/DJmCGL5rSpwuMKs.png" alt="image-20220109132224589" style="zoom:50%;" /> 

<img src="https://s2.loli.net/2022/01/09/cWtZSIj36khqFdC.png" alt="image-20220109130337781" style="zoom:50%;" /> 

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

#### [596 · Minimum Subtree - LintCode](https://www.lintcode.com/problem/596/)

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