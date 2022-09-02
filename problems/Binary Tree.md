Binary Tree - two kinds of problems:
1. Traversal: traverse the whole tree to get an answer --> backtrack
2. Decomposition: use recursion to solve sub-probllems which lead to the whole problem --> dp

Preorder, inorder and postorder traversal corresponds to three different timings when dealing with a sub-tree.


#### [94. Binary Tree Inorder Traversal](https://leetcode.com/problems/binary-tree-inorder-traversal/) Easy
Traversal
```python
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        results = []
        self.traverse(root, results)
        return results
        
    def traverse(self, node, results):
        if node is None:
            return
        
        self.traverse(node.left, results)
        results.append(node.val)
        self.traverse(node.right, results)
```
Decomposition
inorder traversal results = inorder traversl results of its left child + root node + inorder traversal results of its right child
```java
class Solution {
    public List<Integer> inorderTraversal(TreeNode root) {
        LinkedList<Integer> results = new LinkedList<>();
        if (root == null) {
            return results;
        }
        
        results.addAll(inorderTraversal(root.left));
        results.add(root.val);
        results.addAll(inorderTraversal(root.right));
        return results;
    }
}
```
#### [144. Binary Tree Preorder Traversal](https://leetcode.com/problems/binary-tree-preorder-traversal/) Easy
Traversal
```java
class Solution {
    public List<Integer> preorderTraversal(TreeNode root) {
        List<Integer> result = new LinkedList<>();
        if (root == null) {
            return result;
        }
        
        traverse(root, result);    
        return result;
    }
    
    void traverse(TreeNode root, List<Integer> result) {
        if (root == null) {
            return;
        }
        result.add(root.val);
        traverse(root.left, result);
        traverse(root.right, result);
    }
}
```
Decomposition
```java
class Solution {
    public List<Integer> preorderTraversal(TreeNode root) {
        List<Integer> result = new LinkedList<>();
        if (root == null) {
            return result;
        }
        
        result.add(root.val);
        result.addAll(preorderTraversal(root.left));
        result.addAll(preorderTraversal(root.right));
        return result;
    }
}
```

#### [145. Binary Tree Postorder Traversal](https://leetcode.com/problems/binary-tree-postorder-traversal/) Easy
Traversal
```java
class Solution {
    public List<Integer> postorderTraversal(TreeNode root) {
        List<Integer> result = new LinkedList<>();
        if (root == null) {
            return result;
        }
        
        traverse(root, result);    
        return result;
    }
    
    void traverse(TreeNode root, List<Integer> result) {
        if (root == null) {
            return;
        }

        traverse(root.left, result);
        traverse(root.right, result);
        result.add(root.val);
    }
}
```
Decomposition
```java
class Solution {
    public List<Integer> postorderTraversal(TreeNode root) {
        List<Integer> result = new LinkedList<>();
        if (root == null) {
            return result;
        }
        
        
        result.addAll(postorderTraversal(root.left));
        result.addAll(postorderTraversal(root.right));
        result.add(root.val);
        return result;
    }
}
```


#### [100. Same Tree](https://leetcode.com/problems/same-tree/) Easy
```java
class Solution {
    public boolean isSameTree(TreeNode p, TreeNode q) {
        // only when both p and q is none, it is certain that p is equal to q
        if (p == null && q == null) {
            return true;
        }
        if (p == null || q == null) {
            return false;
        }
        // is it ok to set boolean value to true when p.val == q.val
        // no, because this only means values are euqal. whether the structure is equal has to be checked
        if (p.val != q.val) {
            return false;
        }
        // values are the same
        // then check the structure by going into its left and right nodes
        return isSameTree(p.left, q.left) && isSameTree(p.right, q.right);
    }
}
```

#### [572. Subtree of Another Tree](https://leetcode.com/problems/subtree-of-another-tree/) Easy
Similar to 100. Same Tree 
Traversal: always think in recursion. Do not try to use a linked list to store all nodes in postorder and compare. This is silly.
Wheter a tree contains a sub-tree:
1. the tree itself is the desired sub-tree
2. the left sub-tree contains the desired sub-tree
3. the right sub-tree contains the desired sub-tree
```java
class Solution {
    public boolean isSubtree(TreeNode root, TreeNode subRoot) {
        // base case
        if (root == null) {
            return subRoot == null;
        }
        
        // this is where the tree comparison happens
        if (isSameTree(root, subRoot)) {
            return true;
        }
        
        return isSubtree(root.left, subRoot) || isSubtree(root.right, subRoot);
        
    }
    
    boolean isSameTree(TreeNode p, TreeNode q) {
        // both empty?
        if (p == null && q == null) {
            return true;
        }
        
        // one side empty?
        if (p == null || q == null) {
            return false;
        }
        
        // wheter values equal
        if (p.val != q.val) {
            return false;
        }
        
        // whether left and right sub-tree are the same
        return isSameTree(p.left, q.left) && isSameTree(p.right, q.right);
    }
}
```

#### [101. Symmetric Tree](https://leetcode.com/problems/symmetric-tree/) Easy
The key problem is how to check symmetry:
- equal root val
- left child's right == right child's left
```java
class Solution {
    public boolean isSymmetric(TreeNode root) {
        if (root == null) {
            return true;
        }
        
        return check(root.left, root.right);
    }
    
    public boolean check(TreeNode a, TreeNode b) {
        if (a == null || b == null) {
            return a == b;
        }
        if (a.val != b.val) {
            return false;
        }
        // key: use function entry to check whether it is symmetric
        return check(a.left, b.right) && check(a.right, b.left);
    }
    
    
}
```

#### [112. Path Sum](https://leetcode.com/problems/path-sum/) Easy

I wrote this solution. Though it passes all test cases, it uses a lot of memory and runs slowly. Not so good.
```java
class Solution {
    boolean found = false;
    
    public boolean hasPathSum(TreeNode root, int targetSum) {
        if (root == null) {
            return found;
        }
        traverse(root, root.val, targetSum);
        return found;
    }
    
    public void traverse(TreeNode root, int currentSum, int targetSum) {
        // stop only when it is a leaf
        if (root != null && root.left == null && root.right == null) {
            if (currentSum == targetSum) {
                found = true;
            }
            return;
        }

        // make sure it has a left child
        if (root.left != null)
            traverse(root.left, currentSum + root.left.val, targetSum);
        // make sure it has a right child
        if (root.right != null)
            traverse(root.right, currentSum + root.right.val, targetSum);
    }
}
```

Decomposition: a tree has a certain path sum that equals target -> either left sub-tree or right sub-tree has certain path sum that euqals target - root.val ...
Instead of summing all node values, it is better to deduct from the target sum and 
```java
class Solution {
    public boolean hasPathSum(TreeNode root, int targetSum) {
        // base case: already reaches the end (the next to the leaf node)
        if (root == null) {
            return false;
        }
        
        // desired case: reaches leaf and target sum is met
        // only when left and right leaves are both null they will be equal
        if (root.left == root.right && root.val == targetSum) {
            return true;
        }
        
        return hasPathSum(root.left, targetSum - root.val) || hasPathSum(root.right, targetSum - root.val); 
    }
}
```

Traversal
```java
class Solution {
    // used in every sub-tree
    int target;
    boolean found = false;
    int current = 0;
    
    public boolean hasPathSum(TreeNode root, int targetSum) {
        if (root == null) {
            return false;
        }
        
        this.target = targetSum;
        traverse(root);
        return found;
    }
    
    void traverse(TreeNode root) {
        if (root == null) {
            return;
        }
        
        // preorder position: entering a sub-tree
        current += root.val;
        if (root.left == root.right) {
            System.out.println(current);
            if (current == target) {
                found = true;
            }
        }
        
        
        traverse(root.left);
        traverse(root.right);
        
        // postorder position: leaving a sub-tree
        current -= root.val;
        
    }
}
```

#### [404. Sum of Left Leaves](https://leetcode.com/problems/sum-of-left-leaves/) Easy
What is a left leaf ? No children and it is a left child to its parent node. Thus, when entering a sub-tree, a mark should be sent from the parent node to know which child it is.
I wrote this solution.
Traversal
```java
class Solution {
    int sum = 0;
    
    public int sumOfLeftLeaves(TreeNode root) {
        if (root == null) {
            return 0;
        }
        
        traverse(root, 0);
        return sum;
    }
    
    // 0 means root, 1 means left, 2 means right
    void traverse(TreeNode root, int mark) {
        if (root == null) {
            return;
        }
        
        // this is a left leaf
        if (root.left == root.right && mark == 1) {
            sum += root.val;
        }
        
        traverse(root.left, 1);
        traverse(root.right, 2);
        
    }
}
```
Decomposition: sum of all left leaves = sum of left leaves of the left sub-tree + sum of left leaves of the right sub-tree
```java
class Solution {
    public int sumOfLeftLeaves(TreeNode root) {
        return sum(root, 0);   
    }
    
    public int sum(TreeNode root, int mark) {
        if (root == null) {
            return 0;
        }
        
        if (root.left == root.right && mark == 1) {
            return root.val;
        }
        
        return sum(root.left, 1) + sum(root.right, 2);
    }
}
```

#### [543. Diameter of Binary Tree](https://leetcode.com/problems/diameter-of-binary-tree/) Easy
max diameter of a root node = max depth of left sub-tree + max depth of right sub-tree
For the whole tree, the longest path between two nodes may not run through its root node. So when traversing the whole tree, it is necessary to update the max diameter along the way. 
Decomposition
```java
class Solution {
    int maxDiameter = 0;
    
    public int diameterOfBinaryTree(TreeNode root) {
        maxDepth(root);
        return maxDiameter;
    }
    
    // to calculate the max diameter that runs through the current node
    // in the process, all nodes will be processed
    // so the max diameter can be updated along the way
    // remember the longest path between two nodes does not necessarily run through the root node 
    int maxDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        
        int leftMax = maxDepth(root.left);
        int rightMax = maxDepth(root.right);
        // postorder position: when leaving a sub-tree
        // at this point, info of sub-trees can be collected 
        maxDiameter = Math.max(maxDiameter, leftMax + rightMax);
            
        System.out.println(1 + Math.max(leftMax, rightMax));
        return 1 + Math.max(leftMax, rightMax);
    }
}
```

#### [606. Construct String from Binary Tree](https://leetcode.com/problems/construct-string-from-binary-tree/) Easy
a tree string = root.val + ( + left sub-tree string + ) + ( + right sub-tree string +)
rules:
1. leaf has no parenthesis attached
2. left child and no right child, right parenthesis is removed
I wrote this solution by trying different test cases...
Traversal
```java
class Solution {
    String treeStr = "";
    
    public String tree2str(TreeNode root) {
        traverse(root);
        return treeStr;
    }
    
    void traverse(TreeNode root) {
        if (root == null) {
            return;
        }
        
        treeStr += "" + root.val;
        // leaf
        if (root.left == root.right) {
            return;
        }

        
        treeStr += '(';
        traverse(root.left);
        treeStr += ')';
        

        // do not attach parenthesis if only left child exists
        if (!(root.left != null && root.right == null)) {
            treeStr += '(';
            traverse(root.right);
            treeStr += ')';
        }
    }
}
```
Decomposition
```java
class Solution {
    public String tree2str(TreeNode root) {
        // base
        // empty
        if (root == null) {
            return "";
        }
        
        // leaf
        if (root.left == root.right) {
            return root.val + "";
        }
        
        String leftStr = tree2str(root.left);
        String rightStr = tree2str(root.right);
        // postorder position, now children info is known
        
        // left child and no right child
        // no parenthesis for right child
        if (root.left != null && root.right == null) {
            return root.val + "(" + leftStr + ")";
        }
        
        // right cihld and no left child
        // parenthesis for left child is attached
        if (root.left == null && root.right != null) {
            return root.val + "()" + "(" + rightStr + ")";
        }
        
        return root.val + "(" + leftStr + ")" + "(" + rightStr + ")";
    }
}
```

#### [637. Average of Levels in Binary Tree](https://leetcode.com/problems/average-of-levels-in-binary-tree/) Easy
level traversal
```java
class Solution {
    public List<Double> averageOfLevels(TreeNode root) {
        List<Double> results = new ArrayList<>();
        levelTraverse(root, results);
        return results;
    }
    
    void levelTraverse(TreeNode root, List<Double> results) {
        if (root == null) {
            return;
        }
        
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            int size = queue.size();
            double levelSum = 0;
            for (int i = 0; i < size; i++) {
                TreeNode current = queue.poll();
                levelSum += current.val;
                if (current.left != null) {
                    queue.offer(current.left);
                }
                if (current.right != null) {
                    queue.offer(current.right);
                }
            }
            results.add(1.0 * levelSum / size);
        }
    }
}
```

#### [671. Second Minimum Node In a Binary Tree](https://leetcode.com/problems/second-minimum-node-in-a-binary-tree/) Easy
I wrote this solution after trying many test cases... corner cases are tricky
Traversal
```java
class Solution {
    int secondMin = Integer.MAX_VALUE;
    boolean updated = false;
    
    public int findSecondMinimumValue(TreeNode root) {
        if (root == null) {
            return -1;
        }
        
        if (root.left == root.right) {
            return -1;
        }
        
        
        int min = root.val;
        traverse(root, min);
        return updated ? secondMin : -1;
    }
    
    void traverse(TreeNode root, int min) {
        if (root == null) {
            return;
        }
        
        // root value can also be equal to Integer.MAX_VALUE
        // also update ssecondMin when it is the case
        if (root.val > min && root.val <= secondMin) {
            secondMin = root.val;
            updated = true;
        }
        
        traverse(root.left, min);
        traverse(root.right, min);
    }
}
```

Decompositoin
a node has either 0 or 2 child nodes and root value is either less than or equal to its children's value
Rules
1. If a child node has the same value as its root, then find the second minimum value in the sub-tree.
2. If a child node has a different value from its root, simply return the smaller between two child nodes.
```java
class Solution {
    public int findSecondMinimumValue(TreeNode root) {
        // zero child
        if (root.left == root.right) {
            return -1;
        }
        
        // two children
        int left = root.left.val;
        int right = root.right.val;
        // root value is equal to one of its child's value

        // find the second to minimum in the left sub-tree
        if (root.val == root.left.val) {
            left = findSecondMinimumValue(root.left);
        }
        
        // find the second to minimum in the right sub-tree
        if (root.val == root.right.val) {
            right = findSecondMinimumValue(root.right);
        }
        
        if (left == -1) {
            return right;
        }
        
        if (right == -1) {
            return left;
        }
        
        // return the smaller one between its two children
        return Math.min(left, right);
    }
}
```

#### [872. Leaf-Similar Trees](https://leetcode.com/problems/leaf-similar-trees/) Easy
```java
class Solution {
    public boolean leafSimilar(TreeNode root1, TreeNode root2) {
        List<Integer> list1 = new ArrayList<>();
        List<Integer> list2 = new ArrayList<>();
        traverse(root1, list1);
        traverse(root2, list2);
        return compareSeq(list1, list2);
    }
    
    void traverse(TreeNode root, List<Integer> list) {
        if (root == null) return;
        
        traverse(root.left, list);
        traverse(root.right, list);
        if (root.left == root.right) {
            list.add(root.val);
        }
    }
    
    boolean compareSeq(List<Integer> list1, List<Integer> list2) {
        if (list1.size() != list2.size()) {
            return false;
        }
        
        for (int i = 0; i < list1.size(); i++) {
            if (list1.get(i) != list2.get(i)) {
                return false;
            }
        }
        
        return true;
    }
}
```

#### [965. Univalued Binary Tree](https://leetcode.com/problems/univalued-binary-tree/) Easy
Traversal
```java
class Solution {
    int val;
    boolean isUni = true;
    
    public boolean isUnivalTree(TreeNode root) {
        val = root.val;
        traverse(root);
        return isUni;
    }
    
    void traverse(TreeNode root) {
        if (root == null) return;
        
        if (root.val != val) {
            isUni = false;
        }
        traverse(root.left);
        traverse(root.right);
    }
    
}
```
Decomposition: try reverse thinking. the basic ides is to let the recursion go deeper and deeper.
Let the program continue until something goes wrong, instead of making sure it is going ok 
```java
class Solution {
    public boolean isUnivalTree(TreeNode root) {
        // base case
        // only when root.val == root.left.val and root.val == root.right.val this can be reached
        if (root == null) {
            return true;
        }
        
        // not equal
        // what i did wrong was that I return true when root.val == root.left.val
        // which will cause the problem to end prematurely
        // the right thing to do is do the reverse
        // let the problem continue until something is wrong
        if (root.left != null && root.val != root.left.val) {
            return false;
        }
        // not equal
        if (root.right != null && root.val != root.right.val) {
            return false;
        }
        
        return isUnivalTree(root.left) && isUnivalTree(root.right);
    }
}
```


#### [993. Cousins in Binary Tree](https://leetcode.com/problems/cousins-in-binary-tree/submissions/) Easy
Level traversal
```java
class Solution {
    public boolean isCousins(TreeNode root, int x, int y) {
        if (root == null) {
            return false;
        }
        
        return levelTraversal(root, x, y);
    }
    
    boolean levelTraversal(TreeNode root, int x, int y) {
        Queue<TreeNode> q = new LinkedList<>();
        TreeNode parentx = new TreeNode();
        TreeNode parenty = new TreeNode();
        q.offer(root);
        while (!q.isEmpty()) {
            int size = q.size();
            int sum = x + y;

            for (int i = 0; i < size; i++) {
                TreeNode cur = q.poll();
                if (cur.val == x) {
                    sum -= x;
                }
                if (cur.val == y) {
                    sum -= y;
                }
                
                if(cur.left != null) {
                    q.offer(cur.left);
                    if (cur.left.val == x) parentx = cur;
                    if (cur.left.val == y) parenty = cur;
                }
                if(cur.right != null) {
                    q.offer(cur.right);
                    if (cur.right.val == x) parentx = cur;
                    if (cur.right.val == y) parenty = cur;
                }
            }

            if (sum == 0 && parentx.val != parenty.val) {
                return true;
            }
        }
        
        return false;
        
    }
}
```

Traversal: store parent and compare values during traversal
```java
class Solution {
    TreeNode parentx = null;
    TreeNode parenty = null;
    int depthx = 0, depthy = 0;
    int x, y;
    
    public boolean isCousins(TreeNode root, int x, int y) {
        this.x = x;
        this.y = y;
        traverse(root, 0, null);
        if (depthx == depthy && parentx != parenty) {
            return true;
        }
        return false;
    }
    
    void traverse(TreeNode root, int depth, TreeNode parent) {
        if (root == null) {
            return;
        }
        
        if (root.val == x) {
            parentx = parent;
            depthx = depth;
        }
        
        if (root.val == y) {
            parenty = parent;
            depthy = depth;
        }
        
        traverse(root.left, depth + 1, root);
        traverse(root.right, depth + 1, root);
    }
}
```


#### [1379. Find a Corresponding Node of a Binary Tree in a Clone of That Tree](https://leetcode.com/problems/find-a-corresponding-node-of-a-binary-tree-in-a-clone-of-that-tree/) Easy
Traversal: wrote this solution myself
```java
class Solution {
    TreeNode result = null;
    
    public final TreeNode getTargetCopy(final TreeNode original, final TreeNode cloned, final TreeNode target) {    
        traverse(cloned, target);
        return result;
    }
    
    void traverse(TreeNode root, TreeNode target) {
        if (root == null || result != null) return;
        
        if (root.val == target.val) {
            result = root;
        }
        
        traverse(root.left, target);
        traverse(root.right, target);
    }
}
```

Decomposition
```java
class Solution {
    public final TreeNode getTargetCopy(final TreeNode original, final TreeNode cloned, final TreeNode target) {
        if (original == null) {
            return null;
        }
        
        if (original == target) {
            return cloned;
        }
        
        TreeNode left = getTargetCopy(original.left, cloned.left, target);
        if (left != null) {
            return left;
        }
        return getTargetCopy(original.right, cloned.right, target);
    }
}
```

#### [2236. Root Equals Sum of Children](https://leetcode.com/problems/root-equals-sum-of-children/) Easy
Too easy.

#### [226. Invert Binary Tree](https://leetcode.com/problems/invert-binary-tree/) Easy
Decomposition
```java
class Solution {
    // invert the tree and return the root node
    public TreeNode invertTree(TreeNode root) {
        if (root == null) return null;
        
        TreeNode left = invertTree(root.left);
        TreeNode right = invertTree(root.right);
        
        root.left = right;
        root.right = left;
        
        return root;
    }
}
```

#### [116. Populating Next Right Pointers in Each Node](https://leetcode.com/problems/populating-next-right-pointers-in-each-node/) Medium
Traversal
```java
class Solution {
    public Node connect(Node root) {
        if (root == null) return null;
        
        traverse(root.left, root.right);
        return root;
    }
    
    void traverse(Node node1, Node node2) {
        if (node1 == null || node2 == null) {
            return;
        }
        
        // link incoming two nodes
        node1.next = node2;
        
        /**
              1
                       imagine this is a trinary tree, with every two adjacent nodes being a child node  
          2   -  3  -> link incoming two nodes    
        4 - 5 - 6 - 7
link left tree(child1)  link right tree(child3)
      link left and right tree(child2)
        **/
        traverse(node1.left, node1.right);
        traverse(node2.left, node2.right);
        traverse(node1.right, node2.left);
    }
}
```

#### [114. Flatten Binary Tree to Linked List](https://leetcode.con/problems/flatten-binary-tree-to-linked-list/) Medium
Return value is void so flattening has to be done in place. So using a dummy node in traversal and returning it will not work here. Using a stack in traversal can do, though.
Decomposition
```java
class Solution {
    public void flatten(TreeNode root) {
        if (root == null) return;
        /**
                1
             2    5
           3   4    6

        backup   copy left to right    remove left
        5           1                     1
          6  ->  2     2            ->      2         -> flatten right sub-tree and bakcup tree -> link backup tree
               3   4 3   4                 3  4
        **/
        TreeNode right = root.right;
        root.right = root.left;
        root.left = null;
    
        flatten(root.right);
        flatten(right);
        
        while (root.right != null) {
            root = root.right;
        }
        
        root.right = right;
    }
}
```

#### [654. Maximum Binary Tree](https://leetcode.com/problems/maximum-binary-tree/) Medium
Decomposition
```java
class Solution {
    public TreeNode constructMaximumBinaryTree(int[] nums) {
        if (nums == null || nums.length == 0) return null;
        
        // 
        return build(nums, 0, nums.length - 1);
    }
    
    TreeNode build(int[] nums, int lo, int hi) {
        if (lo > hi) {
            return null;
        }
        
        int index = -1, max = Integer.MIN_VALUE;
        // hi could ibe in nums so here the sign is less than or equal to
        for (int i = lo; i <= hi; i++ ) {
            if (nums[i] > max) {
                max = nums[i];
                index = i;
            }
        }
        
        TreeNode root = new TreeNode(max);
        root.left = build(nums, lo, index - 1);
        root.right = build(nums, index + 1, hi);
        
        return root;
    }
}
```

#### [116. Populating Next Right Pointers in Each Node](https://leetcode.com/problems/populating-next-right-pointers-in-each-node/) Medium

```java
class Solution {
    public Node connect(Node root) {
        if (root == null) return null;
        
        traverse(root.left, root.right);
        return root;
    }
    
    void traverse(Node node1, Node node2) {
        if (node1 == null || node2 == null) {
            return;
        }
        
        // link incoming two nodes
        node1.next = node2;
        
        /**
              1
                       imagine this is a trinary tree, with every two adjacent nodes being a child node  
          2   -  3  -> link incoming two nodes    
        4 - 5 - 6 - 7
link left tree(child1)  link right tree(child3)
      link left and right tree(child2)
        **/
        traverse(node1.left, node1.right);
        traverse(node2.left, node2.right);
        traverse(node1.right, node2.left);
    }
}
```

#### [105. Construct Binary Tree from Preorder and Inorder Traversal](https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/) Medium
Decomposition: use preorder to determine root and inorder to find left sub-tree and right.
Understand how a tree looks like when flattened.
sub-tree
```java
/*
         root|    left      |  right
preorder   1  2  5  4  6  7  3  8  9
            left         root  right 
inorder    5  2  6  4  7  1  8  3  9
                     left     right root
postorder  5  6  7  4  2  8  9  3  1
*/
class Solution {
    HashMap<Integer, Integer> map = new HashMap<>();
    
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        // value : index
        for (int i = 0; i < inorder.length; i++) {
            map.put(inorder[i], i);
        }
        
        return build(preorder, 0, preorder.length - 1, inorder, 0, inorder.length - 1);
    }
    
        TreeNode build(int[] preorder, int preStart, int preEnd, int[] inorder, int inStart, int inEnd) {
            if (preStart > preEnd) {
                return null;
            }
            
            // get root node from preorder 
            int rootVal = preorder[preStart];
            TreeNode root = new TreeNode(rootVal);
            
            int rootInorderIndex = map.get(rootVal);
            int leftSize = rootInorderIndex - inStart;
            
            root.left = build(preorder, preStart + 1, preStart + leftSize, inorder, inStart, rootInorderIndex - 1);
            root.right = build(preorder, preStart + leftSize + 1, preEnd, inorder, rootInorderIndex + 1, inEnd);
        
            return root;
    }
}
```

#### [106. Construct Binary Tree from Inorder and Postorder Traversal](https://leetcode.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/) Medium
Decomposition
```java
/*
         root|    left      |  right
preorder   1  2  5  4  6  7  3  8  9
            left         root  right 
inorder    5  2  6  4  7  1  8  3  9
                     left     right root
postorder  5  6  7  4  2  8  9  3  1
*/
class Solution {
    HashMap<Integer, Integer> map = new HashMap<>();
    
    public TreeNode buildTree(int[] inorder, int[] postorder) {
        // value : index
        for (int i = 0; i < inorder.length; i++) {
            map.put(inorder[i], i);
        }
        
        return build(postorder, 0, postorder.length - 1, inorder, 0, inorder.length - 1);
    }
    
        TreeNode build(int[] postorder, int postStart, int postEnd, int[] inorder, int inStart, int inEnd) {
            if (inStart > inEnd) {
                return null;
            }
            
            // get root node from preorder 
            int rootVal = postorder[postEnd];
            TreeNode root = new TreeNode(rootVal);
            
            int rootInorderIndex = map.get(rootVal);
            int leftSize = rootInorderIndex - inStart;
            
            root.left = build(postorder, postStart, postStart + leftSize - 1, inorder, inStart, rootInorderIndex - 1);
            root.right = build(postorder, postStart + leftSize, postEnd - 1, inorder, rootInorderIndex + 1, inEnd);
        
            return root;
    }
}
```

#### [889. Construct Binary Tree from Preorder and Postorder Traversal](https://leetcode.com/problems/construct-binary-tree-from-preorder-and-postorder-traversal/) Medium
Decomposition
```java
class Solution {
    HashMap<Integer, Integer> map = new HashMap<>();
    
    public TreeNode constructFromPrePost(int[] preorder, int[] postorder) {
        for (int i = 0; i < postorder.length; i ++) {
            map.put(postorder[i], i);
        }
        
        return build(preorder, 0, preorder.length - 1, postorder, 0, postorder.length - 1);
    }
    
    TreeNode build(int[] preorder, int preStart, int preEnd, int[] postorder, int postStart, int postEnd) {
        if (preStart > preEnd) {
            return null;
        }
        
        if (preStart == preEnd) {
            return new TreeNode(preorder[preStart]);
        }
        
        int rootVal = preorder[preStart];
        int leftRootVal = preorder[preStart + 1];
        int leftRootIndex = map.get(leftRootVal);
        int leftSize = leftRootIndex - postStart + 1;
        
        TreeNode root = new TreeNode(rootVal);
        root.left = build(preorder, preStart + 1, preStart + leftSize, postorder, postStart, leftRootIndex);
        root.right = build(preorder, preStart + leftSize + 1, preEnd, postorder, leftRootIndex + 1, postEnd - 1);
        
        return root;
    }
}
```

#### [297. Serialize and Deserialize Binary Tree](https://leetcode.com/problems/serialize-and-deserialize-binary-tree/) Hard
To serialize a binary tree means flattening a tree by traversing the tree in a certain way.
Describe a tree in string.
Preorder
```java
public class Codec {
    String SEP = ","; // seperated by ,
    String NULL = "#";
    
    // Encodes a tree to a single string.
    public String serialize(TreeNode root) {
        StringBuilder result = new StringBuilder();
        traverse(root, result);
        return result.toString();
    }
    
    void traverse(TreeNode root, StringBuilder result) {
        if (root == null) {
            result.append(NULL).append(SEP);
            return;
        }
        
        result.append(root.val).append(SEP);
        traverse(root.left, result);
        traverse(root.right, result);
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        LinkedList<String> nodes = new LinkedList<>();
        for (String s : data.split(SEP)) {
            nodes.addLast(s);
        }
        return deserialize(nodes);
    }
    
    TreeNode deserialize(LinkedList<String> nodes) {
        //
        if (nodes.isEmpty()) {
            return null;
        }
        
        String firstItem = nodes.removeFirst();
        if (firstItem.equals(NULL)) {
            return null;
        }
        TreeNode root = new TreeNode(Integer.parseInt(firstItem));
        root.left = deserialize(nodes);
        root.right = deserialize(nodes);
        
        return root;
    }
}
```

#### [652. Find Duplicate Subtrees](https://leetcode.com/problems/find-duplicate-subtrees/) Medium
```java
class Solution {
    // sub-tree string representation: frequency
    HashMap<String, Integer> memo = new HashMap<>();
    LinkedList<TreeNode> results = new LinkedList<>();
    
    public List<TreeNode> findDuplicateSubtrees(TreeNode root) {
        traverse(root);
        return results;
    }
    
    String traverse(TreeNode root) {
        if (root == null) {
            return "#";
        }
        
        String left = traverse(root.left);
        String right = traverse(root.right);
        // postorder position
        String subtree = root.val + ',' + left + ',' + right;
        
        int freq = memo.getOrDefault(subtree, 0);
        if (freq == 1) {
            results.add(root);
        }
        
        memo.put(subtree, freq + 1);
        return subtree;
    }
}
```

BST starts here
------------------
1. smaller left and bigger right, sub-tree included. Binary search operations possible.
2. Inorder traversal will produce an ascending array. 

Inorder traversal for BST
#### [230. Kth Smallest Element in a BST](https://leetcode.com/problems/kth-smallest-element-in-a-bst/) Medium
Traveral. Not the best solution, though. Worst time complexity could reach n.
```java
class Solution {
    int result = -1;
    int rank = 0;
    
    public int kthSmallest(TreeNode root, int k) {
        traverse(root, k);
        return result;
    }
    
    void traverse(TreeNode root, int k) {
        if (root == null) {
            return;
        }
        

        traverse(root.left, k);
        // inorder position
        // get the tree node in ascending order
        rank++;
        if (rank == k) {
            result = root.val;
            return;
        }
        traverse(root.right, k);
    }
}
```

#### [538. Convert BST to Greater Tree](https://leetcode.com/problems/convert-bst-to-greater-tree/) Medium
```java
class Solution {
    int sum = 0;
    
    public TreeNode convertBST(TreeNode root) {
        traverse(root);
        return root;
    }
    
    void traverse(TreeNode root) {
        if (root == null) {
            return;
        }
        
        // the bigger value is processed first now
        // so here we are traversing a bst by descending order
        traverse(root.right);
        // the middle value is still in the middle
        sum += root.val;
        root.val = sum;
        traverse(root.left);
    }
}
```

Detect if a tree is BST
#### [98. Validate Binary Search Tree](https://leetcode.com/problems/validate-binary-search-tree/) Medium
In a bst, all left nodes are smaller than the root node, including all children nodes in the left sub-tree. So when traversing deep down, there has to be a way to let latter nodes know that there exists a limit to compare with.
```java
class Solution {
    public boolean isValidBST(TreeNode root) {
        return validate(root, null, null);
    }
    
    boolean validate(TreeNode root, TreeNode minNode, TreeNode maxNode) {
        if (root == null) {
            return true;
        }
        
        if (minNode != null && root.val <= minNode.val) {
            return false;
        }
        
        if (maxNode != null && root.val >= maxNode.val) {
            return false;
        }
        
        return validate(root.left, minNode, root) && validate(root.right, root, maxNode);
    }
}
```

Search in a BST
#### [700. Search in a Binary Search Tree](https://leetcode.com/problems/search-in-a-binary-search-tree/) Easy
Traversal
```java
class Solution {
    
    public TreeNode searchBST(TreeNode root, int val) {
        return traverse(root, val);
    }
    
    TreeNode traverse(TreeNode root, int val) {
        // whole tree traversed, still no luck
        if (root == null) {
            return null;
        }
        
        if (val == root.val) {
            return root;
        }

        if (val > root.val) {
            return traverse(root.right, val);
        }
        
        return traverse(root.left, val);
    }
}
```

Decomposition
```java
class Solution {
    
    public TreeNode searchBST(TreeNode root, int val) {
        // no luck
        if (root == null) {
            return null;
        }

        // val bigger, try right sub
        if (val > root.val) {
            return searchBST(root.right, val);
        }
        // val smaller, try left sub
        if (val < root.val) {
            return searchBST(root.left, val);
        }
        // found
        return root;
    }
}
```

Deletion in a BST
#### [450. Delete Node in a BST](https://leetcode.com/problems/delete-node-in-a-bst/) Medium
```java
class Solution {
    public TreeNode deleteNode(TreeNode root, int key) {
        if (root == null) {
            return null;
        }
        
        if (key == root.val) {
            if (root.left == null) {
                return root.right;
            }
        
            if (root.right == null) {
                return root.left;
            }
            
            root = switchLeftMaxAndRoot(root);
            // root = switchRightMinAndRoot(root);
        } else if (key > root.val) {
            root.right = deleteNode(root.right, key);
        } else if (key < root.val) {
            root.left = deleteNode(root.left, key);
        }
        
        return root;
    }
    
    
    TreeNode getMaxFromSubleft(TreeNode root) {
        while (root.right != null) {
            root = root.right;
        }
        return root;
    }
    
    // find the max in the left sub tree and remove it 
    TreeNode switchLeftMaxAndRoot(TreeNode root) {
        TreeNode maxSmaller = getMaxFromSubleft(root.left);
        root.left = deleteNode(root.left, maxSmaller.val);
        maxSmaller.left = root.left;
        maxSmaller.right = root.right;
        return maxSmaller;
    }
    
    TreeNode getMinFromSubright (TreeNode root) {
        while (root.left != null) {
            root = root.left;
        }
        return root;
    }
    
    // find the min in the right sub tree and remove it 
    TreeNode switchRightMinAndRoot (TreeNode root) {
        TreeNode minLarger = getMinFromSubright(root.right);
        root.right = deleteNode(root.right, minLarger.val);
        minLarger.left = root.left;
        minLarger.right = root.right;
        return minLarger;
    }
}
```

Build a BST
#### [96. Unique Binary Search Trees](https://leetcode.com/problems/unique-binary-search-trees/) Medium
Every node from 1 to n can be the root node. Then, the smaller nodes to its left consitute left subtree and the larger nodes to its right make its right subtree. The product sum between left and right subtrees is the result for current root. The logic applies to smaller trees as well. Thus we have a recursion definition. 
```java
class Solution {
    int[][] memo;
    
    public int numTrees(int n) {
        // starting from 1
        memo = new int[n + 1][n + 1];
        return count(1, n);
    }
    
    int count(int lo, int hi) {
        if (lo > hi) {
            return 1;
        }
        
        if (memo[lo][hi] != 0) {
            return memo[lo][hi];
        }
        
        int result = 0;
        for (int mid = lo; mid <= hi; mid++) {
            int leftCount = count(lo, mid - 1);
            int rightCount = count(mid + 1, hi);
            result += leftCount * rightCount;
        }
        
        memo[lo][hi] = result;
        
        return result;
    }
}
```

[95. Unique Binary Search Trees II](https://leetcode.com/problems/unique-binary-search-trees-ii/) Medium
Every number from 1 to n can be the root node, while every one of the former numbers can be its left subtree and every one of the latter numbers can be its right subtree. The logic applies to smaller trees. So we have a recursion definition.
```java
class Solution {
    public List<TreeNode> generateTrees(int n) {
        return build(1, n);
    }
    
    List<TreeNode> build(int lo, int hi) {
        List<TreeNode> res = new LinkedList<>();
        if (lo > hi) {
            res.add(null);
            return res;
        }
        
        for (int mid = lo; mid <= hi; mid++) {
            List<TreeNode> leftTree = build(lo, mid - 1);
            List<TreeNode> rightTree = build(mid + 1, hi);
            for (TreeNode leftRoot: leftTree) {
                for (TreeNode rightRoot: rightTree) {
                    // every root has different left and right subtrees
                    // so initiate root here, instead of outside the double loop
                    TreeNode root = new TreeNode(mid);
                    root.left = leftRoot;
                    root.right = rightRoot;
                    res.add(root);
                }
            }
        }
        
        return res;
    }
}
```

#### [99. Recover Binary Search Tree](https://leetcode.com/problems/recover-binary-search-tree/) Medium
Structure should not changed. So only values will be swapped.
```java
class Solution {
    TreeNode prev = new TreeNode(Integer.MIN_VALUE);
    TreeNode first = null, second = null;
    
    public void recoverTree(TreeNode root) {
        traverse(root);
        swapVals(first, second);
    }
    
    void traverse(TreeNode root) {
        if (root == null) {
            return;
        }
        
        traverse(root.left);
        // inorder position: ascending order
        if (root.val < prev.val) {
            if (first == null) {
                first = prev;
            }
            second = root;
        }
        
        prev = root;
        traverse(root.right);
    }
    
    void swapVals(TreeNode first, TreeNode second) {
        int temp = first.val;
        first.val = second.val;
        second.val = temp;
    }
}
```

#### [109. Convert Sorted List to Binary Search Tree](https://leetcode.com/problems/convert-sorted-list-to-binary-search-tree/) Medium
I don't know how to make use of a linked list, so I simply convert it into an array list and build from the result. Guess this is not a good solution in terms of efficiency.
```java
class Solution {
    public TreeNode sortedListToBST(ListNode head) {
        if (head == null) {
            return null;
        }
        
        List<Integer> nums = new ArrayList<>();
        while (head != null) {
            nums.add(head.val);
            head = head.next;
        }
        
        return build(nums, 0, nums.size() - 1);
    }
    
    TreeNode build(List<Integer> nums, int lo, int hi) {
        if (lo > hi) {
            return null;
        }
        
        int mid = (lo + hi) / 2;
        TreeNode root = new TreeNode(nums.get(mid));
        root.left = build(nums, lo, mid - 1);
        root.right = build(nums, mid + 1, hi);
        return root;
    }
}
```

Inorder traversal result is in ascending order. Make use of this bst feature.
```java
class Solution {
    public TreeNode sortedListToBST(ListNode head) {
        if (head == null) {
            return null;
        }
        
        int len = 0;
        for (ListNode n = head; n != null; n = n.next) {
            len++;
        }
        
        cur = head;
        return inorderBuild(0, len - 1);
    }
    
    
    ListNode cur;
    TreeNode inorderBuild(int lo, int hi) {
        if (lo > hi) {
            return null;
        }
        
        int mid = (lo + hi) / 2;
        TreeNode leftTree = inorderBuild(lo, mid - 1);
        // inorder position
        TreeNode root = new TreeNode(cur.val);
        cur = cur.next;
        TreeNode rightTree = inorderBuild(mid + 1, hi);
        root.left = leftTree;
        root.right = rightTree;
        return root;
    }
}
```

#### [449. Serialize and Deserialize BST](https://leetcode.com/problems/serialize-and-deserialize-bst/) Medium
Smaller left sub-tree and larger sub-tree. Make use of this feature to limit the scope in derserialization.
```java
public class Codec {
    String SEP = ",";
    String NULL = "#";

    // Encodes a tree to a single string.
    public String serialize(TreeNode root) {
        StringBuilder result = new StringBuilder();
        traverse(root, result);
        System.out.println(result.toString());
        return result.toString();
    }
    
    void traverse(TreeNode root, StringBuilder result) {
        if (root == null) {
            return;
        }

        // preorder position
        // root node will be the first, convenient
        // even it is a bst, inorder traversal result is still not enough to produce a tree without null pointers
        result.append(root.val).append(SEP);
        traverse(root.left, result);        
        traverse(root.right, result);
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        if (data == "") {
            return null;
        }
        
        LinkedList<Integer> nums = new LinkedList<>();
        for (String s : data.split(SEP)) {
            nums.offer(Integer.parseInt(s));
        }
        
        return deserialize(nums, Integer.MIN_VALUE, Integer.MAX_VALUE);
    }
    
    TreeNode deserialize(LinkedList<Integer> nums, int min, int max) {
        if (nums.isEmpty()) {
            return null;
        }
        
        int rootVal = nums.getFirst();
        if (rootVal > max || rootVal < min) {
            return null;
        }
        nums.removeFirst();
        TreeNode root = new TreeNode(rootVal);
        root.left = deserialize(nums, min, rootVal);
        root.right = deserialize(nums, rootVal, max);
        return root;
    }
    
}
```

#### [501. Find Mode in Binary Search Tree](https://leetcode.com/problems/find-mode-in-binary-search-tree/) Easy
Easy question? I don't think so.
```java
class Solution {
    List<Integer> mode = new ArrayList<>();
    TreeNode prev = null;
    // count for current node
    int curCount;
    // this will get updated along the way to help decide wheter a tree node is mode
    int maxCount;
    
    public int[] findMode(TreeNode root) {
        traverse(root);
        int[] results = new int[mode.size()];
        System.out.println(mode);
        for (int i = 0; i < results.length; i++) {
            results[i] = mode.get(i);
        }
        return results;
    }
    
    void traverse(TreeNode root) {
        if (root == null) {
            return;
        }
        
        traverse(root.left);
        
        // first tree node
        if (prev == null) {
            curCount = 1;
            maxCount = 1;
            mode.add(root.val);
        }  
        
        if (prev != null) {
            if (root.val == prev.val) {
                curCount++;
                if (curCount == maxCount) {
                    mode.add(root.val);
                } else if (curCount > maxCount) {
                    mode.clear();
                    maxCount = curCount;
                    mode.add(root.val);
                }
            }
            
            if (root.val != prev.val) {
                curCount = 1;
                if (curCount == maxCount) {
                    mode.add(root.val);
                }
            }
        }
        prev = root;
        traverse(root.right);
    }
}
```

#### [530. Minimum Absolute Difference in BST](https://leetcode.com/problems/minimum-absolute-difference-in-bst/submissions/) Easy
Solved it right away. This is easy, especially compared to the previoius one, which pretends to be easy.
```java
class Solution {
    TreeNode prev = null;
    int minDiff = Integer.MAX_VALUE;
    
    public int getMinimumDifference(TreeNode root) {
        traverse(root);
        return minDiff;
    }
    
    void traverse(TreeNode root) {
        if (root == null) {
            return;
        }
        
        traverse(root.left);
        if (prev != null) {
            int diff = root.val - prev.val;
            if (diff < minDiff) {
                minDiff = diff; 
            }
        }
        
        prev = root;
        traverse(root.right);
    }
}
```

#### [653. Two Sum IV - Input is a BST](https://leetcode.com/problems/two-sum-iv-input-is-a-bst/) Easy
Not a clever solution. Did not make use of bst features.
Traversal
```java
class Solution {
    List<Integer> otherHalves = new ArrayList<Integer>();
    boolean found = false;
    
    public boolean findTarget(TreeNode root, int k) {
        traverse(root, k);
        System.out.println(otherHalves);
        return found;
    }
    
    void traverse(TreeNode root, int k) {
        if (root == null) {
            return;
        }
        
        traverse(root.left, k);
        System.out.println(root.val);
        int other = k - root.val;
        if (otherHalves.indexOf(root.val) > -1) {
            found = true;
            return;
        }
        otherHalves.add(other);
        traverse(root.right, k);
    }
}
```
Decomposition
I am still not famaliar with the decomposition way of solving tree problems. But this is really elegant. Almost the same as find a value in a tree. The only different is that instead of finding the target value itself, here we are trying to find target value - root.value. Target value is seen as a sum here. A hashset is used to memorize all the nodes that have been visited. 
```java
class Solution {
    // a class level variable is often used in traversal
    // but here it is also used 
    HashSet<Integer> memo = new HashSet<>();
    public boolean findTarget(TreeNode root, int k) {
        // base cases
        // see if current node is seen in memo
        // still no luck
        if (root == null) {
            return false;
        }
        // gotcha
        if (memo.contains(k - root.val)) {
            return true;
        }

        // add root value every time a new node is seen 
        memo.add(root.val);
        
        // either in its left sub-tree or right sub-tree
        return findTarget(root.left, k) || findTarget(root.right, k);
    }
}
```

BST can be treated as a sorted array after inorder traversal. Use two pointers to narrow the scope.
Two pointers
```java
class Solution {
    public boolean findTarget(TreeNode root, int k) {
        List<Integer> nums = convert(root);
        int lo = 0, hi = nums.size() - 1;
        while (lo < hi) {
            int sum = nums.get(lo) + nums.get(hi);
            if (sum == k) {
                return true;
            } else if (sum > k) {
                hi--;
            } else if (sum < k) {
                lo++;
            }
        }
        return false;
    }
    
    List<Integer> convert(TreeNode root) {
        List<Integer> result = new ArrayList<>();
        
        if (root == null) {
            return result; 
        }
        
        result.addAll(convert(root.left));
        result.add(root.val);
        result.addAll(convert(root.right));
        
        return result;
    }
}
```

#### [669. Trim a Binary Search Tree](https://leetcode.com/problems/trim-a-binary-search-tree/) Medium
I am still trying to get the hang of depositional way of dealing with bst.
```java
class Solution {
    public TreeNode trimBST(TreeNode root, int low, int high) {
        if (root == null) {
            return null;
        }
        
        if (root.val < low) { 
            // cut current node and its left sub-tree
            return trimBST(root.right, low, high);
        }
        
        if (root.val > high) {
            // cut current node and its right sub-tree
            return trimBST(root.left, low, high);
        }
        
        root.left = trimBST(root.left, low, high);
        root.right = trimBST(root.right, low, high);
        
        return root;
    }
}
```

#### [701. Insert into a Binary Search Tree](https://leetcode.com/problems/insert-into-a-binary-search-tree/) Medium
Finally got one done! Easy piece.
```java
class Solution {
    public TreeNode insertIntoBST(TreeNode root, int val) {
        if (root == null) {
            return new TreeNode(val);
        }
        
        if (val > root.val) {
            root.right = insertIntoBST(root.right, val);
        }
        
        if (val < root.val) {
            root.left = insertIntoBST(root.left, val);
        }
        
        
        return root;
    }
}
```

#### [783. Minimum Distance Between BST Nodes](https://leetcode.com/problems/minimum-distance-between-bst-nodes/) Easy
```java
class Solution {
    int minDiff = Integer.MAX_VALUE;
    TreeNode prev = null;
    public int minDiffInBST(TreeNode root) {
        traverse(root);
        return minDiff;
    }
    
    void traverse(TreeNode root) {
        if (root == null) {
            return;
        }
        
        traverse(root.left);
        if (prev != null) {
            if (root.val - prev.val < minDiff) {
                minDiff = root.val - prev.val;
            }
        }
        prev = root;
        traverse(root.right);
    }
}
```

#### [938. Range Sum of BST](https://leetcode.com/problems/range-sum-of-bst/) Easy
```java
class Solution {
    int sum = 0;
    public int rangeSumBST(TreeNode root, int low, int high) {
        if (root == null) {
            return 0;
        }
        
        if (root.val < low) {
            return rangeSumBST(root.right, low, high);
        }
        
        if (root.val > high) {
            return rangeSumBST(root.left, low, high);
        }
        
        return root.val + rangeSumBST(root.left, low, high) + rangeSumBST(root.right, low, high);
    }
}
```

#### [1008. Construct Binary Search Tree from Preorder Traversal](https://leetcode.com/problems/construct-binary-search-tree-from-preorder-traversal/) Medium
Wrote it myself. Seems I am beginning to understand more and more about recursion and decomposition when dealing with BST.
```java
class Solution {
    public TreeNode bstFromPreorder(int[] preorder) {
        return build(preorder, 0, preorder.length - 1);
    }
    
    TreeNode build(int[] preorder, int lo, int hi) {
        if (lo > hi) {
            return null;
        }
        
        TreeNode root = new TreeNode(preorder[lo]);
        
        int mid = lo + 1;
        while (mid <= hi && preorder[mid] < preorder[lo]) {
            mid++;
        }
        
        // preorder[lo] is used
        // preorder[mid] is larger than the root val so use its previous value
        root.left = build(preorder, lo + 1, mid - 1);
        root.right = build(preorder, mid, hi);
        
        
        return root;
    }
}
```

#### [1038. Binary Search Tree to Greater Sum Tree](https://leetcode.com/problems/binary-search-tree-to-greater-sum-tree/) Medium
```java
class Solution {
    int sum = 0;
    public TreeNode bstToGst(TreeNode root) {
        traverse(root);
        return root;
    }
    
    void traverse(TreeNode root) {
        if (root == null) {
            return;
        }
        
        // visit right sub-tree first
        // this will result in descending order
        traverse(root.right);
        sum += root.val;
        root.val = sum;
        traverse(root.left);
    }
}
```

#### [173. Binary Search Tree Iterator](https://leetcode.com/problems/binary-search-tree-iterator/) Medium
Use stack to control flow and simulate recursion
```java
class BSTIterator {
    // first in last out
    private Stack<TreeNode> stk = new Stack<>(); 
    
    private void pushLeftBranch(TreeNode node) {
        while (node != null) {
            stk.push(node);
            node = node.left;
        }
    }
    
    public BSTIterator(TreeNode root) {
        pushLeftBranch(root);
    }
    
    public int next() {
        // left, root, right
        TreeNode node = stk.pop();
        pushLeftBranch(node.right);
        return node.val;
    }
    
    public boolean hasNext() {
        return !stk.isEmpty();
    }
}
```


#### [1305. All Elements in Two Binary Search Trees](https://leetcode.com/problems/all-elements-in-two-binary-search-trees/) Medium
My solution is not so efficient and takes up a lot of memeory. Almost brute force. Not so good.
```java
class Solution {
    
    public List<Integer> getAllElements(TreeNode root1, TreeNode root2) {
        List<Integer> list1 = build(root1);
        List<Integer> list2 = build(root2);
        List<Integer> result = new ArrayList<>();
        int i = 0, j = 0;
        while (i < list1.size() && j < list2.size()) {
            if (list1.get(i) < list2.get(j)) {
                result.add(list1.get(i));
                i++;
            } else {
                result.add(list2.get(j));
                j++;
            }
        }
        
        while (i < list1.size()) {
            result.add(list1.get(i));
            i++;
        }
        
        while (j < list2.size()) {
            result.add(list2.get(j));
            j++;
        }
        
        return result;
    }
    
    List<Integer> build(TreeNode root) {
        List<Integer> result = new ArrayList<Integer>();
        if (root == null) {
            return result;
        }
        
        result.addAll(build(root.left));
        result.add(root.val);
        result.addAll(build(root.right));
        
        return result;
    }
    
}
```

Use Iterator to control the flow
```java
class Solution {
    public List<Integer> getAllElements(TreeNode root1, TreeNode root2) {
        BSTIterator tree1 = new BSTIterator(root1);
        BSTIterator tree2 = new BSTIterator(root2);
        LinkedList<Integer> res = new LinkedList<>();
        while (tree1.hasNext() && tree2.hasNext()) {
            if (tree1.peek() < tree2.peek()) {
                res.add(tree1.next());
            } else {
                res.add(tree2.next());
            }
        }
        
        while (tree1.hasNext()) {
            res.add(tree1.next());
        }
        
        while (tree2.hasNext()) {
            res.add(tree2.next());
        }
        
        return res;
    }
}

class BSTIterator {
    private Stack<TreeNode> stk = new Stack<>(); 
    
    private void pushLeftBranch(TreeNode node) {
        while (node != null) {
            stk.push(node);
            node = node.left;
        }
    }
    
    public BSTIterator(TreeNode root) {
        pushLeftBranch(root);
    }
    
    public int peek() {
        return stk.peek().val;
    }
    
    public int next() {
        TreeNode node = stk.pop();
        pushLeftBranch(node.right);
        return node.val;
    }
    
    public boolean hasNext() {
        return !stk.isEmpty();
    }
}
```

#### [1373. Maximum Sum BST in Binary Tree](https://leetcode.com/problems/maximum-sum-bst-in-binary-tree/) Hard
Get max sum of sub binary search trees:
1. check if a root is bst
   1. left sub-tree is bst
   2. right sub-tree is bst
   3. root itself is bst
2. Get sum if root is bst
3. Update a global var whenver there is a bigger bst sum

Postorder, bst
Here a result array is used to store multiple states when leaving a new tree. This is new to me. Never expect you can do this in a recursion. 
```java
class Solution {
    int maxSum = 0;
    
    public int maxSumBST(TreeNode root) {
        traverse(root);
        return maxSum;
    }
    
    int[] traverse(TreeNode root) {
        if (root == null) {
            return new int[] {
                1, Integer.MAX_VALUE, Integer.MIN_VALUE, 0
            };
        }
        
        int[] left = traverse(root.left);
        int[] right = traverse(root.right);
        
        int[] res = new int[4];

        // postorder position
        // check if the current root if bst
        // left and right children are bst and root itself is bst
        if (isBST(left) && isBST(right) && validateRoot(root, left, right)) {
            res[0] = 1;
            res[1] = Math.min(left[1], root.val);
            res[2] = Math.max(right[2], root.val);
            res[3] = left[3] + right[3] + root.val;
            maxSum = Math.max(maxSum, res[3]);
        } else {
            res[1] = 0;
        }
        
        return res;
    }
        
    boolean isBST(int[] ans) {
        return ans[0] == 1;
    }
    
    boolean validateRoot (TreeNode root, int[] left, int[] right) {
        // larger than the max value in the left sub-tree
        // and smaller than the min value in the right sub-tree
        return root.val > left[2] && root.val < right[1];
    }
}
```

#### [1382. Balance a Binary Search Tree](https://leetcode.com/problems/balance-a-binary-search-tree/) Medium
Flattern and build
```java
class Solution {
    public TreeNode balanceBST(TreeNode root) {
        List<Integer> sortedNums = flatten(root);
        TreeNode newRoot = buildBST(sortedNums, 0, sortedNums.size() - 1);
        return newRoot;
    }
    
    List<Integer> flatten(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        if (root == null) {
            return res;
        }
        
        res.addAll(flatten(root.left));
        res.add(root.val);
        res.addAll(flatten(root.right));
        
        return res;
    }
    
    TreeNode buildBST(List<Integer> nums, int lo, int hi) {
        if (lo > hi) {
            return null;
        }
        
        int mid = (lo + hi) / 2;
        TreeNode root = new TreeNode(nums.get(mid));
        root.left = buildBST(nums, lo, mid - 1);
        root.right = buildBST(nums, mid + 1, hi);
        
        return root;
    }
}
```

BST finishes here
------------------