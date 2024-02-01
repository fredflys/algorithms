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

stack

```java
class Solution {
    public List<Integer> inorderTraversal(TreeNode root) {
        Stack<Pair<TreeNode, Integer>> stack = new Stack<>();
        Pair<TreeNode, Integer> stackTop = null;
        List<Integer> res = new LinkedList<>();
        for (stack.push(new Pair<>(root, 0)); !stack.isEmpty();) {
            stackTop = stack.pop();
            root = stackTop.getKey();
            if (root == null) continue;

            // left sub tree is already visited
            if (leftSubTreeVisisted(stackTop)) {
                // visit root
                res.add(root.val);
                // prepare to visit its right sub-tree
                stack.push(new Pair<>(root.right, 0));
                continue;
            }

            // left sub tree is not visited
            // then mark it as visited and prepare to visit its left sub-tree
            stack.push(new Pair<>(root, 1));
            stack.push(new Pair<>(root.left, 0));
        }

        return res;
    }

    private boolean leftSubTreeVisisted(Pair<TreeNode, Integer> pair) {
        return pair.getValue() > 0;
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

Non-recursive

```java
class Solution {
    private void preorder(TreeNode root, List<Integer> res) {
        if (root == null) {
            return;
        }

        Stack<TreeNode> stack = new Stack<>();
        for (stack.push(root); !stack.isEmpty();) {
            root = stack.pop();
            if (root != null) {
                res.add(root.val);
                stack.push(root.right);
                stack.push(root.left);
            }
        }
    }

    public List<Integer> preorderTraversal(TreeNode root) {

    }
}
```

#### [589. N-ary Tree Preorder Traversal](https://leetcode.com/problems/n-ary-tree-preorder-traversal/description/) Easy

```java
class Solution {
    public List<Integer> preorder(Node root) {
        List<Integer> res = new ArrayList<>();
        if (root != null) traverse(root, res);
        return res;
    }

    void traverse(Node root, List<Integer> res) {
        res.add(root.val);
        for (Node node: root.children) {
            if (node != null)
                traverse(node, res);
        }
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

stack

```java
class Solution {
    public List<Integer> postorderTraversal(TreeNode root) {
        Stack<Pair<TreeNode, Integer>> s = new Stack<>();
        List<Integer> res = new LinkedList<>();
        Pair<TreeNode, Integer> stackTop = null;
        for (s.push(new Pair<>(root, 0)); !s.isEmpty();) {
            stackTop = s.pop();
            root = stackTop.getKey();
            if (root == null) continue;

            int subTreeVisited = stackTop.getValue();

            // both sub trees are not visited
            // visit left sub tree
            if (subTreeVisited == 0) {
                s.push(new Pair<>(root, 1));
                s.push(new Pair<>(root.left, 0));
                continue;
            }

            // left sub-tree is visited
            // visit right subtree
            if (subTreeVisited == 1) {
                s.push(new Pair<>(root, 2));
                s.push(new Pair<>(root.right, 0));
                continue;
            }
            res.add(root.val);
        }
        return res;
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

#### [687. Longest Univalue Path](https://leetcode.com/problems/longest-univalue-path/submissions/) Medium

depth traversal

```java
class Solution {
    int res = 0;

    public int longestUnivaluePath(TreeNode root) {
        if (root == null) {
            return 0;
        }

        maxLen(root, root.val);
        return res;
    }

    // in the subtree, whose parent is root,
    private int maxLen(TreeNode root, int parentVal) {
        if (root == null) {
            return 0;
        }

        int left = maxLen(root.left, root.val);
        int right = maxLen(root.right, root.val);

        // still in the current subtree and children info is collected
        // the longest path that passes the current node
        res = Math.max(res, left + right);

        // unequal to parent value
        if (root.val != parentVal) {
            return 0;
        }

        // equal to parent value
        // leaving this subtree and return a value to its parent
        // no turning back is allowed, so a better path must be returned
        return Math.max(left, right) + 1;
    }
}
```

#### [1325. Delete Leaves With a Given Value](https://leetcode.com/problems/delete-leaves-with-a-given-value/) Medium

postorder traversal

```java
class Solution {
    public TreeNode removeLeafNodes(TreeNode root, int target) {
        if (root == null) {
            return null;
        }

        root.left = removeLeafNodes(root.left, target);
        root.right = removeLeafNodes(root.right, target);

        if (root.left == root.right && root.val == target) {
            return null;
        }

        return root;
    }
}
```

#### [814. Binary Tree Pruning](https://leetcode.com/problems/binary-tree-pruning/) Medium

postorder traversal, pruning from bottom up

```java
class Solution {
    public TreeNode pruneTree(TreeNode root) {
        if (root == null) {
            return null;
        }

        // no need to declare a variable and attach it to root node
        root.left = pruneTree(root.left);
        root.right = pruneTree(root.right);

        // prune 0-value leaf node
        if (root.left == root.right && root.val != 1) {
            return null;
        }

        return root;
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

#### [103. Binary Tree Zigzag Level Order Traversal](https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/) Medium

level traversal
I am still not familiar enough with level traversal. Instead of using queue, I use stack to do it, which is rather confusing. I also forgot how to loop through every loop.

```java
class Solution {
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        List<List<Integer>> results = new ArrayList<>();
        if (root == null) {
            return results;
        }

        Queue<TreeNode> q = new LinkedList<>();
        q.offer(root);
        boolean isLeftFirst = true;
        while (!q.isEmpty()) {
            int size = q.size();
            LinkedList<Integer> level = new LinkedList<>();
            for (int i = 0; i < size; i++) {
                TreeNode cur = q.poll();
                // change direction when building the level result
                if (isLeftFirst) {
                    level.addLast(cur.val);
                } else {
                    level.addFirst(cur.val);
                }

                add(q, cur.left);
                add(q, cur.right);

            }

            // change direction
            isLeftFirst = !isLeftFirst;
            // append level results
            results.add(level);
        }
        return results;
    }

    private void add(Queue<TreeNode> q, TreeNode node) {
        if (node != null) {
            q.offer(node);
        }
    }
}
```

#### [107. Binary Tree Level Order Traversal II](https://leetcode.com/problems/binary-tree-level-order-traversal-ii/)

Level traversal

```java
class Solution {
    public List<List<Integer>> levelOrderBottom(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        if (root == null) {
            return res;
        }

        Queue<TreeNode> q = new LinkedList<>();
        q.offer(root);
        while (!q.isEmpty()) {
            int size = q.size();
            LinkedList<Integer> level = new LinkedList<>();
            for (int i = 0; i < size; i++) {
                TreeNode node = q.poll();
                level.addLast(node.val);
                add(q, node.left);
                add(q, node.right);
            }
            res.add(0, level);
        }

        return res;
    }

    private void add (Queue<TreeNode> q, TreeNode root) {
        if (root != null) {
            q.offer(root);
        }
    }
}
```

#### [1161. Maximum Level Sum of a Binary Tree](https://leetcode.com/problems/maximum-level-sum-of-a-binary-tree/submissions/) Medium

level traversal

```java
class Solution {
    public int maxLevelSum(TreeNode root) {
        int maxSum = Integer.MIN_VALUE;
        int maxLevel = 0;
        int level = 0;
        Queue<TreeNode> q = new LinkedList<>();
        q.offer(root);
        while (!q.isEmpty()) {
            int size = q.size();
            int sum = 0;
            level++;
            for (int i = 0; i < size; i++) {
                TreeNode cur = q.poll();
                sum += cur.val;
                add(q, cur.left);
                add(q, cur.right);
            }
            // only update maxSum when it is strictly greater than current level sum
            if (sum > maxSum) {
                maxSum = sum;
                maxLevel = level;
            }
        }

        return maxLevel;
    }

    void add(Queue<TreeNode> q, TreeNode node) {
        if (node != null) {
            q.offer(node);
        }
    }
}
```

#### [429. N-ary Tree Level Order Traversal](https://leetcode.com/problems/n-ary-tree-level-order-traversal/submissions/) Medium

```java
class Solution {
    public List<List<Integer>> levelOrder(Node root) {
        Queue<Node> q = new LinkedList<>();
        List<List<Integer>> res = new ArrayList<>();
        if (root == null) {
            return res;
        }

        q.offer(root);
        while (!q.isEmpty()) {
            int size = q.size();
            List<Integer> level = new ArrayList<>();
            for (int i = 0; i < size; i++) {
                Node node = q.poll();
                level.add(node.val);
                for (Node child: node.children) {
                    add(q, child);
                }
            }
            res.add(level);
        }
        return res;
    }

    void add(Queue<Node> q, Node node) {
        if (node != null) {
            q.offer(node);
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

#### [331. Verify Preorder Serialization of a Binary Tree](https://leetcode.com/problems/verify-preorder-serialization-of-a-binary-tree/) Medium

deserialization

```java
class Solution {
    public boolean isValidSerialization(String preorder) {
        LinkedList<String> nodes = new LinkedList<>();
        for (String node: preorder.split(",")) {
            nodes.addLast(node);
        }

        // validated and nodes is exhausted, no nodes remain
        return validate(nodes) && nodes.isEmpty();
    }

    boolean validate(LinkedList<String> nodes) {
        // nodes is exhausts but does not stop at null sign
        // this is a problem
        if (nodes.isEmpty()) {
            return false;
        }

        String root = nodes.removeFirst();

        // null sign, sub-tree ends here, it is valid
        if (root.equals("#")) return true;

        // not #, this means it must have non-null children, go deeper
        return validate(nodes) && validate(nodes);
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

## BST starts here

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
Never wrap a recursion function with another one.

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

## BST finishes here

[113. Path Sum II](https://leetcode.com/problems/path-sum-ii/submissions/) Medium
Almost forgot one of the requirements: root to leaf sum

```java
class Solution {
    int target;
    List<List<Integer>> res;

    public List<List<Integer>> pathSum(TreeNode root, int targetSum) {
        target = targetSum;
        res = new ArrayList<>();
        if (root == null) {
            return res;
        }


        traverse(root, 0, new ArrayList<Integer>());
        return res;
    }

    private void traverse(TreeNode root, int sum, List<Integer> path) {
        if (root == null) {
            return;
        }

        sum += root.val;
        path.add(root.val);

        // root to leaf path
        // make sure it is leaf
        if (root.left == root.right && sum == target) {
            // path is a reference, will be modified later
            // so make a copy and store the copy
            res.add(new ArrayList<Integer>(path));
        }
        traverse(root.left, sum, path);
        traverse(root.right, sum, path);

        sum -= root.val;
        path.remove(path.size() - 1);
    }
}
```

#### (117. Populating Next Right Pointers in Each Node II)[https://leetcode.com/problems/populating-next-right-pointers-in-each-node-ii/] Medium

level traversal

```java
class Solution {
    public Node connect(Node root) {
        if (root == null) {
            return null;
        }

        Queue<Node> q = new LinkedList<>();
        q.offer(root);

        while (!q.isEmpty()) {
            int size = q.size();
            for (int i = 0; i < size; i++) {
                if (i < size - 1) {

                }
                Node cur = q.poll();

                // don't link the last on one level to the first on the next level
                // last item on a level needs no linking
                if (i <= size - 2) {
                    cur.next = q.peek();
                }
                add(q, cur.left);
                add(q, cur.right);
            }

        }

        return root;
    }

    void add(Queue<Node> q, Node node) {
        if (node != null) {
            q.offer(node);
        }
    }
}
```

#### [124. Binary Tree Maximum Path Sum](https://leetcode.com/problems/binary-tree-maximum-path-sum/submissions/) Hard

It took me an hour or so to fully grasp the solution. Seems it makes me understand recursion better. Below is a good explainer to this solution.
https://leetcode.cn/problems/binary-tree-maximum-path-sum/solution/shou-hui-tu-jie-hen-you-ya-de-yi-dao-dfsti-by-hyj8/
![123.png](https://s2.loli.net/2022/09/03/vPa6FEI4QHsmew1.png)

```java
class Solution {
    int maxSum = Integer.MIN_VALUE;

    public int maxPathSum(TreeNode root) {
        if (root == null) {
            return 0;
        }

        postorder(root);
        return maxSum;
    }

    int postorder (TreeNode root) {
        if (root == null) {
            return 0;
        }

        int left = postorder(root.left);
        int right = postorder(root.right);

        int innerMaxSum = root.val + left + right;
        maxSum = Math.max(maxSum, innerMaxSum);

        int outputSum = root.val + Math.max(left, right);

        return outputSum < 0 ? 0 : outputSum;
    }
}
```

#### [LintCode 595 · Binary Tree Longest Consecutive Sequence](https://www.lintcode.com/problem/595/description) Easy

Similar to leetcode 124

```java
public class Solution {
    int maxLength = 0;
    public int longestConsecutive(TreeNode root) {
        postorder(root);
        return maxLength;
    }

    int postorder(TreeNode root) {
        if (root == null) {
            return 0;
        }

        int left = postorder(root.left);
        int right = postorder(root.right);

        int leftLength = 1, rightLength = 1;
        if (root.left != null && root.left.val - root.val == 1) {
            leftLength = 1 + left;
        }

        if (root.right != null && root.right.val - root.val == 1) {
            rightLength = 1 + right;
        }

        int currentLength = Math.max(leftLength, rightLength);
        maxLength = Math.max(maxLength, currentLength);

        return currentLength;
    }
}
```

#### [LintCode 614 · Binary Tree Longest Consecutive Sequence II](https://www.lintcode.com/problem/614/) Medium

```java
public class Solution {
    int maxLength = 0;
    public int longestConsecutive2(TreeNode root) {
        postorder(root);
        return maxLength;
    }

    int[] postorder (TreeNode root) {
        int[] res = new int[2];
        if (root == null) {
            return res;
        }

        // res[0] stores the longest decreasing path from parent to child that passes through the root
        // res[1] stores the longest increasing path from parent to child that passes through the root
        res[0] = 1;
        res[1] = 1;

        int[] left = postorder(root.left);
        int[] right = postorder(root.right);

        if (root.left != null) {
            if (root.val - root.left.val == 1) {
                res[0] = left[0] + 1;
            } else if (root.left.val - root.val == 1) {
                res[1] = left[1] + 1;
            }
        }

        if (root.right != null) {
            if (root.val - root.right.val == 1) {
                // res[0] has been calculated above
                // here we choose the larger one because a path goes up and then down, not the same direction at the same time
                res[0] = Math.max(res[0], right[0] + 1);
            } else if(root.right.val - root.val == 1) {
                res[1] = Math.max(res[1], right[1] + 1);
            }
        }

        maxLength = Math.max(maxLength, res[0] + res[1] - 1);
        return res;
    }
}
```

#### [129. Sum Root to Leaf Numbers](https://leetcode.com/problems/sum-root-to-leaf-numbers/) Medium

```java
class Solution {
    List<String> res = new ArrayList<>();

    public int sumNumbers(TreeNode root) {
        traverse(root, "");

        int sum = 0;
        for (String s: res) {
            System.out.println(s);
            sum += Integer.parseInt(s);
        }

        return sum;
    }

    void traverse(TreeNode root, String s) {
        if (root == null) {
            return;
        }

        traverse(root.left, s + root.val);
        traverse(root.right, s + root.val);

        if (root.left == root.right) {
            res.add(s + root.val);
            return;
        }
    }
}
```

#### [199. Binary Tree Right Side View](https://leetcode.com/problems/binary-tree-right-side-view/) Medium

level traversal

```java
class Solution {
    List<Integer> res = new ArrayList<>();

    public List<Integer> rightSideView(TreeNode root) {
        if (root == null) {
            return res;
        }

        Queue<TreeNode> q = new LinkedList<>();
        q.offer(root);
        while (!q.isEmpty()) {
            int size = q.size();
            for (int i = 0; i < size; i++) {
                TreeNode cur = q.poll();
                // System.out.println(cur.val);
                if (i == 0) {
                    res.add(cur.val);
                }
                // right subtree first
                add(q, cur.right);
                add(q, cur.left);
            }
        }

        return res;
    }

    void add(Queue<TreeNode> q, TreeNode node) {
        if (node != null) {
            q.offer(node);
        }
    }
}
```

Preorder traversal

```java
class Solution {
    List<Integer> res = new ArrayList<>();
    int depth = 0;

    public List<Integer> rightSideView(TreeNode root) {
        if (root == null) {
            return res;
        }
        traverse(root);

        return res;
    }


    void traverse(TreeNode root) {
        if (root == null) {
            return;
        }

        // preorder position
        // add 1 to depth when entering the tree
        // because right subtree is visited first, only the first one to the right should be added
        depth++;
        if (res.size() < depth) {
            res.add(root.val);
        }

        traverse(root.right);
        traverse(root.left);
        // reset depth when leaving the sub-tree
        depth--;
    }

}
```

#### [662. Maximum Width of Binary Tree](https://leetcode.com/problems/maximum-width-of-binary-tree/) Medium

Level traversal, serialize nodes, binary heap, complete binary tree
left = root.id _ 2, right = root.id _ 2 + 1
![662[1].png](https://s2.loli.net/2022/09/05/bykzq3BdsJ2ApHx.png)

```java
class Solution {

    class SerialNode {
        TreeNode node;
        int id;

        public SerialNode (TreeNode node, int id) {
            this.node = node;
            this.id = id;
        }
    }

    public int widthOfBinaryTree(TreeNode root) {
        if (root == null) return 0;

        int max = 0;
        Queue<SerialNode> q = new LinkedList<>();
        q.offer(new SerialNode(root, 1));

        // variables used in the while loop
        int size, start, end, curId;
        TreeNode cur;
        SerialNode sn;

        while (!q.isEmpty()) {
            size = q.size();
            start = 0;
            end = 0;
            for (int i = 0; i < size; i++) {
                sn = q.poll();
                cur = sn.node;
                curId = sn.id;
                if (i == 0) {
                    start = curId;
                }
                if (i == size - 1) {
                    end = curId;
                }
                add(q, cur.left, curId * 2);
                add(q, cur.right, curId * 2 + 1);
            }

            max = Math.max(max, end - start + 1);
        }
        return max;
    }

    private void add (Queue<SerialNode> q, TreeNode node, int id) {
        if (node != null) {
            q.offer(new SerialNode(node, id));
        }
    }
}
```

traversal

```java
class Solution {
    public int widthOfBinaryTree(TreeNode root) {
        if (root == null) {
            return 0;
        }

        traverse(root, 1, 1);
        return max;
    }

    List<Integer> levelFirsts = new ArrayList<>();
    int max = 1;

    void traverse(TreeNode root, int id, int depth) {
        if (root == null) {
            return;
        }

        if (levelFirsts.size() == depth - 1) {
            levelFirsts.add(id);
        } else {
            max = Math.max(max, id - levelFirsts.get(depth - 1) + 1);
        }

        traverse(root.left, id * 2, depth + 1);
        traverse(root.right, id * 2 + 1, depth + 1);
    }
}
```

#### (1104. Path In Zigzag Labelled Binary Tree)[https://leetcode.com/problems/path-in-zigzag-labelled-binary-tree/] Medium

math, complete binary tree
from left to right count
1 1  
2 2^(2-1) 2^2 - 1  
3 2^(3-1) ..+1 ..+2 2^3 - 1
4 ......
zigzag count: (min + max) - label of left-to-right count will substitue original count with right-to-left count
This problem has more to do with math than with binary tree itself.

```java
class Solution {
    public List<Integer> pathInZigZagTree(int label) {
        int lastRow = getLastRow(label);
        List<Integer> res = getPath(label, lastRow);
        Collections.reverse(res);
        return res;
    }

    int getLastRow(int label) {
        int row = 1, rowStart = 1;
        // 1,2,4,8...
        // maximum value for every row is 2^row - 1
        while (rowStart * 2 <= label) {
            row++;
            rowStart *= 2;
        }
        return row;
    }

    List<Integer> getPath (int label, int row) {
        // if last row is even, it will be wrongly revered in the while loop
        if (row % 2 == 0) {
            label = reverseLabel(label, row);
        }
        List<Integer> path = new ArrayList<>();

        while (row > 0) {
            // even row number
            // from right to left, reversing is needed
            if (row % 2 == 0) {
                path.add(reverseLabel(label, row));
            } else {
                path.add(label);
            }
            // move to previous row
            row--;
            label >>= 1;
        }

        return path;
    }

    // change a left-to-right lable to a right-to-left one and vice versa
    int reverseLabel(int label, int row) {
        // 2^(row - 1) + 2^row - 1 - label
        return (1 << row - 1) + (1 << row) - 1 - label;
    }
}
```

More math and more concise. Math always gets me confused. I am still not so sure that I can write the solution by myself.

```java
class Solution {
    public List<Integer> pathInZigZagTree(int label) {
        List<Integer> path = new ArrayList<>();
        while (label >= 1) {
            path.add(label);
            label >>= 1;
            int depth = log(label);
            System.out.println(depth);
            int[] range = getLevelRange(depth);
            // zigzag, order changes every level, so reversing is always needed
            label = range[1] + range[0] - label;
        }

        Collections.reverse(path);
        return path;
    }

    int[] getLevelRange(int n) {
        int p = (int) Math.pow(2, n);
        return new int[]{p, p * 2 - 1};
    }

    int log(int x) {
        // 2^?  = x
        return (int) (Math.log(x) / Math.log(2));
    }
}
```

#### [1261. Find Elements in a Contaminated Binary Tree](https://leetcode.com/problems/find-elements-in-a-contaminated-binary-tree/) Medium

```java
class FindElements {
    TreeNode root;
    public FindElements(TreeNode root) {
        traverse(root, 0);
        this.root = root;
    }

    void traverse(TreeNode root, int val) {
        if (root == null) {
            return;
        }

        root.val = val;
        traverse(root.left, val * 2 + 1);
        traverse(root.right, val * 2 + 2);

    }

    public boolean find(int target) {
        return find(this.root, target);
    }

    boolean find(TreeNode root, int target) {
        if (root == null) {
            return false;
        }

        if (root.val == target) {
            return true;
        }

        return find(root.left, target) || find(root.right, target);
    }
}
```

#### [222. Count Complete Tree Nodes](https://leetcode.com/problems/count-complete-tree-nodes/) Medium

https://labuladong.github.io/algo/2/21/48/
complete binary tree
This is a solution that has a time compleity of O(n). Fails to meet the requirement.

```java
class Solution {
    public int countNodes(TreeNode root) {
        if (root == null) {
            return 0;
        }

        traverse(root, 1);
        return max;
    }

    int max = 1;
    void traverse(TreeNode root, int id) {
        if (root == null) {
            return;
        }

        if (id > max) {
            max = id;
        }

        traverse(root.left, id * 2);
        traverse(root.right, id * 2 + 1);
    }
}
```

Complete binary tree, perfect binary tree
Complete binary is somewhere between a normal binary tree and a perfect binary tree. At least there is one of its direct sub trees is a perfect tree.
![trees[1].png](https://s2.loli.net/2022/09/06/vQDaWTlCbNRI8Xd.png)
![1[1].jpg](https://s2.loli.net/2022/09/06/J95S1nHPE3lWw86.jpg)

```java
class Solution {
    public int countNodes(TreeNode root) {
        TreeNode left = root, right = root;
        int leftHeight = 0, rightHeight = 0;
        while (left != null) {
            left = left.left;
            leftHeight++;
        }
        while (right != null) {
            right = right.right;
            rightHeight++;
        }
        if (leftHeight == rightHeight) {
            return (int) Math.pow(2, leftHeight) - 1;
        }
        // this is how to calculate the nodes of a normal tree
        return 1 + countNodes(root.left) + countNodes(root.right);
    }
}
```

#### [337. House Robber III](https://leetcode.com/problems/house-robber-iii/) Medium

Focus on what a single node can do, instead of thinking about complicated cases. Focus.

```java
class Solution {
    Map<TreeNode, Integer> memo = new HashMap<>();
    public int rob(TreeNode root) {
        if (root == null) {
            return 0;
        }

        if (memo.containsKey(root)) {
            return memo.get(root);
        }

        int robThis = root.val
            + (
               root.left == null ? 0 :
               rob(root.left.left) + rob(root.left.right)
              )
            + (
                root.right == null ? 0 :
                rob(root.right.left) + rob(root.right.right)
              );
        int doNotRobThis = rob(root.left) + rob(root.right);
        int res = Math.max(robThis, doNotRobThis);
        memo.put(root, res);
        return res;
    }
}
```

Make use of return values of recursion functions to improve efficiency.

```java
class Solution {
    public int rob(TreeNode root) {
        int[] res = traverse(root);
        return Math.max(res[0], res[1]);
    }
    /*
    return an two-element array to store the maximum values for a root tree
    res[0] represents the max value when current house (node) is skipped
    res[1] represents the max value when current house (node) is robbed
    */
    int[] traverse(TreeNode root) {
        if (root == null) {
            return new int[]{0, 0};
        }

        int[] left = traverse(root.left);
        int[] right = traverse(root.right);
        // postorder position: maximum value after robbing or not robbing this house depends on its children house
        int rob = root.val + left[0] + right[0]; // current house is robbed, then its children house must be skipped
        int not_rob = Math.max(left[0], left[1]) + Math.max(right[0], right[1]); // current house is skipped, children houses now can be robbed or not, choose a better option

        return new int[]{not_rob, rob};
    }
}
```

#### (437. Path Sum III)[https://leetcode.com/problems/path-sum-iii/] Medium

Traversal, prefix sum

```java
class Solution {
    // according to constrains the sum could surpass integer range
    long presum;
    int target, res;
    Map<Long, Integer> map;
    public int pathSum(TreeNode root, int targetSum) {
        if (root == null) {
            return 0;
        }

        this.presum = 0;
        this.target = targetSum;
        this.res = 0;
        this.map = new HashMap<>();
        map.put(0L, 1);

        traverse(root);
        return res;
    }

    void traverse(TreeNode root) {
        if (root == null) {
            return;
        }

        // entering, set
        presum += root.val;
        res += map.getOrDefault(presum - target, 0);
        map.put(presum, map.getOrDefault(presum, 0) + 1);

        traverse(root.left);
        traverse(root.right);

        // leaving, reset
        map.put(presum, map.get(presum) - 1);
        presum -= root.val;
    }
}
```

Decomposition, two recursion functions

```java

class Solution {
    public int pathSum(TreeNode root, long targetSum) {
        if (root == null) {
            return 0;
        }

        int res = rootSum(root, targetSum);
        //  here is the trciky part, pay attention to the function used here
        //  the main funciton is also used as a recursive function
        //  this will make sure that every node will be treated as a root node once
        res += pathSum(root.left, targetSum);
        res += pathSum(root.right, targetSum);

        return res;
    }

    // this returns the desired path count when current node is a root
    private int rootSum(TreeNode root, long target) {
        int res = 0;

        if (root == null) {
            return res;
        }

        if (root.val == target) {
            res++;
        }

        res += rootSum(root.left, target - root.val);
        res += rootSum(root.right, target - root.val);

        return res;

    }
}
```

#### (508. Most Frequent Subtree Sum)[https://leetcode.com/problems/most-frequent-subtree-sum/submissions/] Medium

decomposition

```java
class Solution {

    Map<Integer, Integer> map = new HashMap<>();
    ArrayList<Integer> res = new ArrayList<>();
    int mostFrequent = 0;

    public int[] findFrequentTreeSum(TreeNode root) {
        if (root == null) {
            return new int[2];
        }

        sum(root);

        for (Map.Entry<Integer, Integer> entry: map.entrySet()) {
            if (entry.getValue() == mostFrequent) {
                res.add(entry.getKey());
            }
        }

        return convertListToArray(res);
    }

    int sum(TreeNode root) {
        if (root == null) {
            return 0;
        }

        int left = sum(root.left);
        int right = sum(root.right);

        int sum = root.val + left + right;
        map.put(sum, map.getOrDefault(sum, 0) + 1);
        if (map.get(sum) > mostFrequent) {
            mostFrequent = map.get(sum);
        }

        return sum;
    }

    int[] convertListToArray(List<Integer> list) {
        return list.stream().mapToInt(i -> i).toArray();
    }
}
```

#### [513. Find Bottom Left Tree Value](https://leetcode.com/problems/find-bottom-left-tree-value/) Medium

level traversal

```java
class Solution {
    int depth = 0;
    public int findBottomLeftValue(TreeNode root) {
        Queue<TreeNode> q = new LinkedList<>();
        q.offer(root);
        List<Integer> res = new ArrayList<>();
        while (!q.isEmpty()) {
            int size = q.size();
            for (int i = 0; i < size; i++) {
                TreeNode cur = q.poll();
                res.add(cur.val);
                add(q, cur.left);
                add(q, cur.right);
            }
            if (!q.isEmpty()) {
                res.clear();
            }
        }
        return res.get(0);
    }

    private void add(Queue<TreeNode> q, TreeNode node) {
        if (node != null) {
            q.offer(node);
        }
    }
}
```

I still don't fully understand preorder traversal. The first to reach the max depth will be the bottom left node.
traversal. Depth can also be counted in a vertical traversal.

```java
class Solution {
    int maxDepth = 0;
    int depth = 0;
    TreeNode res;
    public int findBottomLeftValue(TreeNode root) {
        traverse(root);
        return res.val;
    }

    void traverse(TreeNode root) {
        if (root == null) {
            return;
        }

        // preorder position
        depth++;
        if (depth > maxDepth) {
            maxDepth = depth;
            res = root;
        }
        traverse(root.left);
        traverse(root.right);
        depth--;
    }
}
```

#### [515. Find Largest Value in Each Tree Row](https://leetcode.com/problems/find-largest-value-in-each-tree-row/) Medium

Level traversal, pretty intuitive solution

```java
class Solution {
    public List<Integer> largestValues(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        if (root == null){
            return res;
        }

        Queue<TreeNode> q = new LinkedList<>();
        q.offer(root);
        int rowMax;
        while (!q.isEmpty()) {
            rowMax = Integer.MIN_VALUE;
            int size = q.size();
            for (int i = 0; i < size; i++) {
                TreeNode cur = q.poll();
                if (cur.val > rowMax) {
                    rowMax = cur.val;
                }
                add(q, cur.left);
                add(q, cur.right);
            }
            res.add(rowMax);
        }

        return res;
    }

    private void add(Queue<TreeNode> q, TreeNode node) {
        if (node != null) {
            q.offer(node);
        }
    }
}
```

preorder traversal. Still not famaliar with preorder traversal to get row info.

```java
class Solution {
    ArrayList<Integer> res = new ArrayList<>();

    public List<Integer> largestValues(TreeNode root) {
        if (root == null) {
            return res;
        }

        traverse(root,0);
        return res;
    }

    void traverse(TreeNode root, int depth) {
        if (root == null) {
            return;
        }

        // when first entering a level
        if (res.size() <= depth) {
            res.add(root.val);
        } else {
            res.set(depth, Math.max(res.get(depth), root.val));
        }

        traverse(root.left, depth + 1);
        traverse(root.right, depth + 1);
    }
}
```

#### [623. Add One Row to Tree](https://leetcode.com/problems/add-one-row-to-tree/) Medium

depth traversal

```java
class Solution {
    int newVal;
    int targetDepth;

    public TreeNode addOneRow(TreeNode root, int val, int depth) {
        if (depth == 1) {
            TreeNode newRoot = new TreeNode(val);
            newRoot.left = root;
            return newRoot;
        }

        newVal = val;
        targetDepth = depth;

        traverse(root, 1);
        return root;
    }

    void traverse(TreeNode root, int depth) {
        if (root == null) {
            return;
        }

        traverse(root.left, depth + 1);
        traverse(root.right, depth + 1);
        if (depth == targetDepth - 1) {
            addNewRow(root);
            return;
        }
    }

    void addNewRow(TreeNode root) {
        TreeNode newLeft = new TreeNode(newVal);
        TreeNode newRight = new TreeNode(newVal);
        TreeNode oldLeft = root.left;
        TreeNode oldRight = root.right;
        root.left = newLeft;
        root.right = newRight;
        newLeft.left = oldLeft;
        newRight.right = oldRight;
    }
}
```

#### [655. Print Binary Tree](https://leetcode.com/problems/print-binary-tree/) Medium

depth traversal

```java
class Solution {
    List<List<String>> res = new ArrayList<>();
    int depth;
    public List<List<String>> printTree(TreeNode root) {
        // depth = height + 1
        depth = getDepth(root);
        int row = depth;
        int column = (1 << depth) - 1; // (int) Math.pow(2, depth) - 1
        fillResWithEmptyStrings(res, row, column);
        System.out.println(column / 2);
        traverse(root, 0, (column - 1) / 2);
        return res;
    }

    int getDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }

        int left = getDepth(root.left);
        int right = getDepth(root.right);

        return Math.max(left, right) + 1;
    }

    void fillResWithEmptyStrings(List<List<String>> res, int row, int column) {
        for (int i = 0; i < row; i++) {
            List<String> currentRow = new ArrayList<>();
            for (int j = 0; j < column; j++) {
                currentRow.add("");
            }
            res.add(currentRow);
        }
    }

    void traverse(TreeNode root, int x, int y) {
        if (root == null) {
            return;
        }

        res.get(x).set(y, String.valueOf(root.val));
        traverse(root.left, x + 1, y - (1 << (depth - x - 1 - 1)));
        traverse(root.right, x + 1, y + (1 << (depth - x - 1 - 1)));
    }
}
```

#### [863. All Nodes Distance K in Binary Tree](https://leetcode.com/problems/all-nodes-distance-k-in-binary-tree/) Medium

BFS, level traversal that also involves parent node so the tree is treated as a graph here

```java
class Solution {
    Map<Integer, TreeNode> parent;
    List<Integer> res;

    public List<Integer> distanceK(TreeNode root, TreeNode target, int k) {
        parent = new HashMap<>();
        res = new ArrayList<>();
        traverse(root, null);
        // start from target, instead of root. This is a key difference.
        bfs(target, k);
        return res;

    }

    void traverse(TreeNode root, TreeNode parentNode) {
        if (root == null) {
            return;
        }

        parent.put(root.val, parentNode);
        traverse(root.left, root);
        traverse(root.right, root);
    }

    void bfs(TreeNode start, int k) {
        Queue<TreeNode> q = new LinkedList<>();
        Set<Integer> visited = new HashSet<>();
        q.offer(start);
        visited.add(start.val);
        int distance = 0;

        while (!q.isEmpty()) {
            int size = q.size();
            for (int i = 0; i < size; i++) {
                TreeNode cur = q.poll();

                // desired distance is reached
                if (distance == k) {
                    res.add(cur.val);
                }

                TreeNode parentNode = parent.get(cur.val);
                add(q, visited, parentNode);
                add(q, visited, cur.left);
                add(q, visited, cur.right);
            }

            distance++;
        }
    }

    void add(Queue<TreeNode> q, Set<Integer> visited, TreeNode node) {
        if (node != null && !visited.contains(node.val)) {
            visited.add(node.val);
            q.offer(node);
        }
    }
}
```

#### [1457. Pseudo-Palindromic Paths in a Binary Tree](https://leetcode.com/problems/pseudo-palindromic-paths-in-a-binary-tree/) Medium

depth traversal

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    int res;
    int[] count;

    public int pseudoPalindromicPaths (TreeNode root) {
        res = 0;
        count = new int[10];
        traverse(root);
        return res;
    }

    void traverse(TreeNode root) {
        if (root == null) {
            return;
        }

        // leaf node
        if (root.left == root.right) {
            count[root.val]++;
            if (isPalindrom(count)) {
                res++;
            }
            count[root.val]--;
        }

        count[root.val]++;
        traverse(root.left);
        traverse(root.right);
        count[root.val]--;
    }

    boolean isPalindrom(int[] count) {
        int odd = 0;
        for (int c: count) {
            if (c % 2 == 1) {
                odd++;
            }
        }

        if (odd <= 1) {
            return true;
        }

        return false;
    }
}
```

#### [865. Smallest Subtree with all the Deepest Nodes](https://leetcode.com/problems/smallest-subtree-with-all-the-deepest-nodes/) Medium

#### [1123. Lowest Common Ancestor of Deepest Leaves](https://leetcode.com/problems/lowest-common-ancestor-of-deepest-leaves/) Medium

Decomposition
Focus on depth instead of nodes. Focus on depth of sub-trees and their relationship. Focusing on nodes really got me nowhere. I still have this bad inclination to dive into details when dealing with binary tree problems. This will not lead me to a correct solution. Even in the best case, I can only can get a clumsy answer.

```java
class Solution {
    public TreeNode subtreeWithAllDeepest(TreeNode root) {
        if (root == null) return root;

        int leftDepth = getMaxDepth(root.left);
        int rightDepth = getMaxDepth(root.right);

        if (leftDepth == rightDepth) return root;

        if (leftDepth > rightDepth) {
            return subtreeWithAllDeepest(root.left);
        }
        return subtreeWithAllDeepest(root.right);
    }

    private int getMaxDepth(TreeNode root) {
        if (root == null) return 0;

        return Math.max(getMaxDepth(root.left), getMaxDepth(root.right)) + 1;
    }
}
```

#### [894. All Possible Full Binary Trees](https://leetcode.com/problems/all-possible-full-binary-trees/submissions/) Medium

Decomposition. Focus on what on node in the middle does and define the base case. Forget abou what its children or parent should do.
ref: https://leetcode.cn/problems/all-possible-full-binary-trees/solution/man-er-cha-shu-di-gui-xiang-jie-by-jalan/

```java
class Solution {
    List<TreeNode>[] memo;

    public List<TreeNode> allPossibleFBT(int n) {
        if (n % 2 == 0) {
            return new LinkedList<>();
        }

        memo = new LinkedList[n + 1];
        return buildTrees(n);
    }

    // build complete full trees with n nodes
    List<TreeNode> buildTrees(int n) {
        List<TreeNode> res = new LinkedList<>();

        // base case
        if (n == 1) {
            res.add(new TreeNode(0));
            return res;
        }
        if (memo[n] != null) {
            return memo[n];
        }

        for (int leftNodes = 1; leftNodes < n; leftNodes += 2) {
            int rightNodes = n - leftNodes - 1; // 1 means root node
            List<TreeNode> leftSubTrees = buildTrees(leftNodes);
            List<TreeNode> rightSubTrees = buildTrees(rightNodes);
            for (TreeNode left : leftSubTrees) {
                for (TreeNode right : rightSubTrees) {
                    TreeNode root = new TreeNode(0);
                    root.left = left;
                    root.right = right;
                    res.add(root);
                }
            }
        }

        memo[n] = res;
        return res;
    }
}
```

#### [958. Check Completeness of a Binary Tree](https://leetcode.com/problems/check-completeness-of-a-binary-tree/) Medium

complete binary tree, level traversal
For a complete binary tree, all that remains in level traversal must be consecutive null nodes.

```java
class Solution {
    public boolean isCompleteTree(TreeNode root) {
        Queue<TreeNode> q = new LinkedList<>();
        q.offer(root);
        boolean flag = false;
        while (!q.isEmpty()) {
            int size = q.size();
            for (int i = 0; i < size; i++) {
                TreeNode cur = q.poll();
                if (cur == null) {
                    flag = true;
                    continue;
                } else {
                    if (flag) {
                        return false;
                    }
                }
                q.offer(cur.left);
                q.offer(cur.right);
            }
        }

        return true;
    }
}
```

#### [919. Complete Binary Tree Inserter](https://leetcode.com/problems/complete-binary-tree-inserter/submissions/) Medium

level traversal

```java
class CBTInserter {
    private Queue<TreeNode> candidates = new LinkedList<>();
    private TreeNode root;

    // prepare candidates in intialization
    public CBTInserter(TreeNode root) {
        this.root = root;
        Queue<TreeNode> temp = new LinkedList<>();
        temp.offer(root);
        while (!temp.isEmpty()) {
            TreeNode cur = temp.poll();
            if (cur.left == null || cur.right == null) {
                candidates.offer(cur);
            }
            add(temp, cur.left);
            add(temp, cur.right);
        }
    }


    public int insert(int val) {
        TreeNode node = new TreeNode(val);
        TreeNode parent = candidates.peek();
        // as far left possible
        if (parent.left == null) {
            // parent right is still open, so it still counts as a candidate
            parent.left = node;
        } else {
            // no open positions available for this parent so remove it
            parent.right = node;
            candidates.poll();
        }

        // the newly added node will also a candidate
        candidates.offer(node);
        return parent.val;
    }

    public TreeNode get_root() {
        return root;
    }

    void add(Queue<TreeNode> q, TreeNode node) {
        if (node != null) {
            q.offer(node);
        }
    }
}
```

#### [951. Flip Equivalent Binary Trees](https://leetcode.com/problems/flip-equivalent-binary-trees/) Medium

decomposition

```java
class Solution {
    public boolean flipEquiv(TreeNode root1, TreeNode root2) {
        // both empty
        if (root1 == root2) {
            return true;
        }

        // either is empty but not both
        if (root1 == null || root2 == null) {
            return false;
        }

        if (root1.val != root2.val) {
            return false;
        }

        return (
            // unflipped
            flipEquiv(root1.left, root2.left) && flipEquiv(root1.right, root2.right)
        ) || (
            // flipped
            flipEquiv(root1.left, root2.right) && flipEquiv(root1.right, root2.left)
        );
    }

}
```

#### [968. Binary Tree Cameras](https://leetcode.com/problems/binary-tree-cameras/) Hard

```java
class Solution {
    public int minCameraCover(TreeNode root) {
        // if the root needs to taken care of
        // you'll have to set another one on the root because you have no parent to turn to
        return setCamera(root) == -1 ? count + 1 : count;
    }

    int count = 0;
    /*
    1: camera is set here
    0: covered but no camera here
    -1: not covered
    make use of the return value, it's a signal to its parent
    */
    int setCamera(TreeNode root) {
        if (root == null) {
            return 0;
        }

        int left = setCamera(root.left);
        int right = setCamera(root.right);

        // at least one child is not covered
        // a camera needs to be set here to cover them
        if (left == -1 || right == -1) {
            count++;
            // an camera is set here
            return 1;
        }

        // one child is already covered
        // so automatically the parent is taken care of
        if (left == 1 || right == 1) {
            return 0;
        }

        // let its parent take care of it
        return -1;
    }
}
```

#### [971. Flip Binary Tree To Match Preorder Traversal](https://leetcode.com/problems/flip-binary-tree-to-match-preorder-traversal/) Medium

```java
class Solution {
    List<Integer> res;
    int[] voyage;
    int index;
    public List<Integer> flipMatchVoyage(TreeNode root, int[] voyage) {
        res = new ArrayList<>();
        this.voyage = voyage;
        if (traverse(root)) {
            return res;
        }

        return Arrays.asList(-1);
    }

    boolean traverse(TreeNode root) {
        if (root == null) {
            return true;
        }

        if (root.val != voyage[index]) {
            return false;
        }

        index++;

        if (root.left != null && root.left.val != voyage[index]) {
            res.add(root.val);
            return traverse(root.right) && traverse(root.left);
        }

        return traverse(root.left) && traverse(root.right);
    }
}
```

#### [979. Distribute Coins in Binary Tree](https://leetcode.com/problems/distribute-coins-in-binary-tree/) Medium

postorder traversal
Finally got it done after 40 mins!

```java
class Solution {
    int count = 0;
    public int distributeCoins(TreeNode root) {
        traverse(root, null);
        return count;
    }

    void traverse(TreeNode root, TreeNode parent) {
        if (root == null) {
            return;
        }

        traverse(root.left, root);
        traverse(root.right, root);

        // borrow from parent
        while (parent != null && root.val < 1) {
            parent.val--;
            root.val++;
            count++;
        }

        // give to parent
        while (parent != null && root.val > 1) {
            root.val--;
            parent.val++;
            count++;
        }
    }
}
```

decomposition: focus on the root. How does it contribute to the balance of the current sub-tree? What is returned can not only be root itself, but also be something about the sub-tree. A single node is a sub-tree after all, which has no children, i.e. the base case.

```java
class Solution {
    public int distributeCoins(TreeNode root) {
        calculateBalance(root);
        return count;
    }

    int count = 0;
    int calculateBalance(TreeNode root) {
        if (root == null) {
            return 0;
        }

        int left = calculateBalance(root.left);
        int right = calculateBalance(root.right);
        int balance = left + right;

        if (root.val < 1) {
            // deficit: in need of one coin
            balance -= 1;
        } else {
            // maybe there is a surplus: keep one coin for itself and give the remaining coins
            // if root has one coin, then balance is 0, no deficit nor surplus
            balance += root.val - 1;
        }

        count += Math.abs(balance);
        return balance;
    }
}
```

#### [987. Vertical Order Traversal of a Binary Tree](https://leetcode.com/problems/vertical-order-traversal-of-a-binary-tree/) Hard

Despite a hard problem, the main idea of this problem is fairly easy. Get indices first, then sort new nodes and build the result. One step a time. A lot of technical details are involved. More of an engineering problem.

```java
class Solution {
    class IndexNode {
        int x, y;
        TreeNode node;
        public IndexNode (TreeNode node, int x, int y) {
            this.node = node;
            this.x = x;
            this.y = y;
        }
    }

    public List<List<Integer>> verticalTraversal(TreeNode root) {
        LinkedList<List<Integer>> res = new LinkedList<>();
        indexNodes(root, 0, 0);
        sortNodes(nodes);
        buildRes(nodes, res);
        return res;
    }

    List<IndexNode> nodes = new ArrayList<>();
    void indexNodes(TreeNode root, int x, int y) {
        if (root == null) {
            return;
        }

        nodes.add(new IndexNode(root, x, y));
        indexNodes(root.left, x + 1, y - 1);
        indexNodes(root.right, x + 1, y + 1);
    }

    void sortNodes(List<IndexNode> nodes) {
        Collections.sort(nodes, (IndexNode a, IndexNode b) -> {
            // compare node value only when rows and columns are both the same
            if (a.x == b.x && a.y == b.y) {
                return a.node.val - b.node.val;
            }
            // compare rows when columns are the same
            if (a.y == b.y) {
                return a.x - b.x;
            }
            return a.y - b.y;
        });
    }

    void buildRes(List<IndexNode> nodes, LinkedList<List<Integer>> res) {
        // used for nodes partition by column number
        int lastColumn = -1000;
        for (int i = 0; i < nodes.size(); i++) {
            IndexNode cur = nodes.get(i);
            if (cur.y != lastColumn) {
                res.addLast(new LinkedList<>());
                lastColumn = cur.y;
            }
            res.getLast().add(cur.node.val);
        }
    }
}
```

#### [988. Smallest String Starting From Leaf](https://leetcode.com/problems/smallest-string-starting-from-leaf/ Medium

path traversal
Forgot how to deal with a tree path problem. Append node when entering a sub-tree and remove node when leaving a sub-tree. This way all paths will be visited.

```java
class Solution {
    public String smallestFromLeaf(TreeNode root) {
        traverse(root);
        return res;
    }

    StringBuilder path = new StringBuilder();
    String res = null;
    void traverse(TreeNode root) {
        if (root == null) {
            return;
        }

        // leaf position: root to leaf
        if (root.left == root.right) {
            path.append((char) ('a' + root.val));
            // now leaf to root
            path.reverse();

            String pathStr = path.toString();
            if (res == null || res.compareTo(pathStr) > 0) {
                res = pathStr;
            }
            System.out.println(pathStr);

            // now root to leaf
            path.reverse();
            path.deleteCharAt(path.length() - 1);
            return;
        }

        path.append((char) ('a' + root.val));

        traverse(root.left);
        traverse(root.right);

        path.deleteCharAt(path.length() - 1);
    }
}
```

#### [998. Maximum Binary Tree II](https://leetcode.com/problems/maximum-binary-tree-ii/) Medium

decomposition

```java
class Solution {
    // insert val into the root tree and return the root node
    public TreeNode insertIntoMaxTree(TreeNode root, int val) {
        if (root == null) {
            return new TreeNode(val);
        }

        // val is greater in the current sub-tree
        // make it the root
        if (val > root.val) {
            TreeNode temp = root;
            root = new TreeNode(val);
            root.left = temp;
        } else {
            root.right = insertIntoMaxTree(root.right, val);
        }

        return root;
    }
}
```

#### [1026. Maximum Difference Between Node and Ancestor](https://leetcode.com/problems/maximum-difference-between-node-and-ancestor/) Medium

traversal

```java
class Solution {
    public int maxAncestorDiff(TreeNode root) {
        traverse(root);
        return maxDiff;
    }

    int maxDiff;
    List<Integer> ancestors = new ArrayList<>();
    void traverse(TreeNode root) {
        if (root == null) {
            return;
        }

        ancestors.add(root.val);
        for (Integer ancestor: ancestors) {
            if (ancestor == root.val) continue;
            int res = Math.abs(ancestor - root.val);
            if (res > maxDiff) {
                maxDiff = res;
            }
        }
        traverse(root.left);
        traverse(root.right);
        ancestors.remove(ancestors.size() - 1);

    }
}
```

decomposition: how to represent and convert the problem is a big problem for me.

```java
class Solution {
    public int maxAncestorDiff(TreeNode root) {
        getMinMax(root);
        return res;
    }

    int res = 0;
    int[] getMinMax(TreeNode root) {
        if (root == null) {
            return new int[]{Integer.MAX_VALUE, Integer.MIN_VALUE};
        }

        int[] leftMinMax = getMinMax(root.left);
        int[] rightMinMax = getMinMax(root.right);
        int rootMin = min(root.val, leftMinMax[0], rightMinMax[0]);
        int rootMax = max(root.val, leftMinMax[1], rightMinMax[1]);
        // update result
        // max value and min value may not have ancestor-child relationship so max - min will not deliver the expected answer
        res = max(res, rootMax - root.val, root.val - rootMin);

        return new int[]{rootMin, rootMax};
    }

    int min(int a, int b, int c) {
        return Math.min(Math.min(a, b), c);
    }

    int max(int a, int b, int c) {
        return Math.max(Math.max(a, b), c);
    }
}
```

#### [1080. Insufficient Nodes in Root to Leaf Paths](https://leetcode.com/problems/insufficient-nodes-in-root-to-leaf-paths/) Medium

decomposition: delete from the leaf level

```java
class Solution {
    public TreeNode sufficientSubset(TreeNode root, int limit) {
        if (root == null) {
            return null;
        }

        // at leaf node, if limit is not reached, it can bu safely deleted
        // info passed downwards: sum of the path
        if (root.left == root.right) {
            // delete
            if (root.val < limit) {
                return null;
            }
            return root;
        }

        TreeNode left = sufficientSubset(root.left, limit - root.val);
        TreeNode right = sufficientSubset(root.right, limit - root.val);
        // if left and root children are deleted, this means root can also be deleted
        // info passed upwards: whether children are both removed
        if (left == null && right == null) {
            return null;
        }

        root.left = left;
        root.right = right;

        return root;
    }
}
```

#### [1110. Delete Nodes And Return Forest](https://leetcode.com/problems/delete-nodes-and-return-forest/submissions/) Medium

decomposition

1. To delete a node, its parent should be notfiied (upwards) that the child is null.
2. To add a node to the forest, its parent should notify it (downwards) about whether the parent is already removed and itself needs to be removed.
   Recursion - Passing Info

- downwards by arguments
- upwards by return value

```java
class Solution {

    public List<TreeNode> delNodes(TreeNode root, int[] to_delete) {
        if (root == null) {
            return new LinkedList<>();
        }
        getDelSet(to_delete);
        delete(root, false);
        return res;
    }

    Set<Integer> delSet = new HashSet<>();
    void getDelSet(int[] to_delete) {
        for (int num: to_delete) {
            delSet.add(num);
        }
    }

    List<TreeNode> res = new LinkedList<>();
    TreeNode delete(TreeNode root, boolean hasParent) {
        if (root == null) {
            return null;
        }

        boolean needDeletion = delSet.contains(root.val);
        /*
        root will be added to the forest only when
        1) root itself is not deleted
        2) root has no parent because if root still has a parent its parent should instead be added

        this is where adding to forest takes place
        */
        if (!needDeletion && !hasParent) {
            res.add(root);
        }

        // if root will be deleted, then there is no parent
        // tell children that there is no parent
        root.left = delete(root.left, !needDeletion);
        root.right = delete(root.right, !needDeletion);

        // tell parent whether root will be deleted by returning either null or root itself
        // this is where deletion actually takes place
        return needDeletion ? null : root;
    }
}
```

#### [1145. Binary Tree Coloring Game](https://leetcode.com/problems/binary-tree-coloring-game/) Medium

There are three sub-trees after the first player colors a node: left, right and parent. If you block any of them and the chosen subtree has more nodes than do the other do, you can rest assured that you will win the game.

```java
class Solution {
    public boolean btreeGameWinningMove(TreeNode root, int n, int x) {
        if (root == null) {
            return false;
        }

        if (root.val == x) {
            int left = countNodes(root.left);
            int right = countNodes(root.right);
            int parent = n - left - right - 1;
            /*
            three blocking options:
            - color parent, then we have to make sure that parent sub-tree nodes are mothan left and right sub-trees plus x itself
            - ..
            - ..

            left > n - left - right - 1 + right + 1
                 > n - left
            left > n / 2
            */

            // return parent > n / 2 || left > n / 2 || right > n / 2
            return parent > left + right + 1 || left > parent + right + 1 || right > parent + left + 1;
        }


        return btreeGameWinningMove(root.left, n, x) || btreeGameWinningMove(root.right, n, x);
    }

    int countNodes(TreeNode root) {
        if (root == null) {
            return 0;
        }

        return 1 + countNodes(root.left) + countNodes(root.right);
    }
}
```

#### [1302. Deepest Leaves Sum](https://leetcode.com/problems/deepest-leaves-sum/) Medium

dfs
There are 10000 nodes at maximum and thus 10000 levels at maximum if there is only one node at every level. An array is used to store the sum of every level.

```java
class Solution {
    public int deepestLeavesSum(TreeNode root) {
        traverse(root);
        return leafSums[maxDepth];
    }

    int[] leafSums = new int[10000];
    int depth = 1;
    int maxDepth = 0;
    void traverse(TreeNode root) {
        if (root == null) {
            return;
        }

        depth++;
        if (root.left == root.right) {
            if (depth > maxDepth) {
                maxDepth = depth;
            }
            leafSums[depth] += root.val;
        }

        traverse(root.left);
        traverse(root.right);
        depth--;
    }
}
```

bfs, level traversal

```java
class Solution {
    public int deepestLeavesSum(TreeNode root) {
        Queue<TreeNode> q = new LinkedList<>();
        q.offer(root);

        int sum = 0;
        while (!q.isEmpty()) {
            int size = q.size();
            sum = 0;
            for (int i = 0; i < size; i++) {
                TreeNode cur = q.poll();
                sum += cur.val;
                add(q, cur.left);
                add(q, cur.right);
            }
        }
        return sum;
    }

    void add(Queue<TreeNode> q, TreeNode node) {
        if (node != null) {
            q.offer(node);
        }
    }
}
```

#### [1315. Sum of Nodes with Even-Valued Grandparent](https://leetcode.com/problems/sum-of-nodes-with-even-valued-grandparent/) Medium

traversal

```java
class Solution {
    public int sumEvenGrandparent(TreeNode root) {
        traverse(root);
        return sum;
    }

    List<Integer> path = new ArrayList<>();
    int sum = 0;
    void traverse(TreeNode root) {
        if (root == null) {
            return;
        }

        path.add(root.val);
        if (path.size() > 2 && path.get(path.size() - 1 - 2) % 2 == 0) {
            sum += root.val;
        }

        traverse(root.left);
        traverse(root.right);

        path.remove(path.size() - 1);
    }
}
```

a more straightforward solution

```java
class Solution {
    public int sumEvenGrandparent(TreeNode root) {
        traverse(root);
        return sum;
    }

    int sum = 0;

    void traverse(TreeNode root) {
        if (root == null) {
            return;
        }
        if (root.val % 2 == 0) {
            if (root.left != null) {
                if (root.left.left != null) {
                    sum += root.left.left.val;
                }
                if (root.left.right != null) {
                    sum += root.left.right.val;
                }
            }

            if (root.right != null) {
                if (root.right.left != null) {
                    sum += root.right.left.val;
                }
                if (root.right.right != null) {
                    sum += root.right.right.val;
                }
            }
        }

        traverse(root.left);
        traverse(root.right);
    }
}
```

#### [1339. Maximum Product of Splitted Binary Tree](https://leetcode.com/problems/maximum-product-of-splitted-binary-tree/) Medium

```java
class Solution {
    public int maxProduct(TreeNode root) {
        treeSum = getSum(root);
        getSum(root);
        return (int) (res % (1e9 + 7));
    }

    int treeSum;
    long res;
    int getSum(TreeNode root) {
        if (root == null) {
            return 0;
        }

        int left = getSum(root.left);
        int right = getSum(root.right);
        int rootSum = left + right + root.val;
        res = Math.max(res, (long) rootSum * (treeSum - rootSum));

        return rootSum;
    }
}
```

#### [1367. Linked List in Binary Tree](https://leetcode.com/problems/linked-list-in-binary-tree/) Medium

```java
class Solution {
    public boolean isSubPath(ListNode head, TreeNode root) {
        if (head == null) {
            return true;
        }

        if (root == null) {
            return false;
        }

        if (head.val == root.val) {
            if (check(head, root)) {
                return true;
            }
        }

        return isSubPath(head, root.left) || isSubPath(head, root.right);
    }

    boolean check(ListNode head, TreeNode root) {
        if (head == null) {
            return true;
        }

        if (root == null) {
            return false;
        }

        if (head.val == root.val) {
            return check(head.next, root.left) || check(head.next, root.right);
        }

        return false;
    }
}
```

#### [1372. Longest ZigZag Path in a Binary Tree](https://leetcode.com/problems/longest-zigzag-path-in-a-binary-tree/) Medium

Decomposition

```java
class Solution {
    public int longestZigZag(TreeNode root) {
        getPathLength(root);
        return res;
    }

    int res = 0;
    int[] getPathLength(TreeNode root) {
        if (root == null) {
            return new int[]{-1, -1};
        }

        int[] left = getPathLength(root.left);
        int[] right = getPathLength(root.right);
        /*
        left to right or  right to left
          o                  o
        o                      o
          o                  o

        recive info from children and return new info to its parent
        If root recives info from its left child,
        */
        /*
          $
        o
          o
        */
        int leftRight = left[1] + 1;
        /*
        $
          o
        o
        */
        int rightLeft = right[0] + 1;

        res = Math.max(res, Math.max(leftRight, rightLeft));

        return new int[]{leftRight, rightLeft};
    }
}
```

I also wrote a traversal solution. It can pass some example test cases, but the problem is that it runs too slow because there is recursion inside another recursion. The time complexity is too high and it will cause TLE. I'll paste the solution here just for reference.

```java
class Solution {
    public int longestZigZag(TreeNode root) {
        if (root == null) {
            return 0;
        }

        iterate(root);
        return max;
    }

    int max = 0;
    void traverse(TreeNode root, int direction, int count) {
        if (count > max) {
            max = count;
        }

        if (direction == 0 && root.right != null) {
            traverse(root.right, 1, count + 1);
        }

        if (direction == 1 && root.left != null) {
            traverse(root.left, 0, count + 1);
        }
    }

    void iterate(TreeNode root) {
        if (root == null) {
            return;
        }

        traverse(root, 0, 0);
        traverse(root, 1, 0);
        iterate(root.left);
        iterate(root.right);
    }
}
```

#### [1448. Count Good Nodes in Binary Tree](https://leetcode.com/problems/count-good-nodes-in-binary-tree/) Medium

Traversal
There is a for loop in every recursion so this solution runs very slowly.

```java
class Solution {
    public int goodNodes(TreeNode root) {
        traverse(root);
        return count;
    }

    int count = 0;
    int base;
    boolean flag;
    List<Integer> path = new ArrayList<>();
    void traverse(TreeNode root) {
        if (root == null) {
            return;
        }

        path.add(root.val);
        base = root.val;
        boolean flag = true;
        for (int i = 0; i < path.size() - 1; i++) {
            if (path.get(i) > base) {
                flag = false;
                break;
            }
        }
        if (flag) {
            count++;
        }
        traverse(root.left);
        traverse(root.right);
        path.remove(path.size() - 1);
    }
}
```

Actually, there is no need to compare every element in the current path. Simply compare the max value in the path with root val and you will know right away if it is a good node because as long as the max value is not greater than current node it is safe to say that no nodes in the path is greater.

```java
class Solution {
    public int goodNodes(TreeNode root) {
        traverse(root, root.val);
        return count;
    }

    int count = 0;

    void traverse(TreeNode root, int pathMax) {
        if (root == null) {
            return;
        }
        if (pathMax <= root.val) {
            count++;
        }

        pathMax = Math.max(pathMax, root.val);

        traverse(root.left, pathMax);
        traverse(root.right, pathMax);
    }
}
```

#### [1530. Number of Good Leaf Nodes Pairs](https://leetcode.com/problems/number-of-good-leaf-nodes-pairs/) Medium

ref: https://leetcode.cn/problems/number-of-good-leaf-nodes-pairs/solution/1530-hao-xie-zi-jie-dian-dui-de-shu-lian-wltu/
ref: https://leetcode.cn/problems/number-of-good-leaf-nodes-pairs/solution/di-gui-shu-zu-by-libins-82ab/

The main problem is how to calculate leaf to leaf distance.
leaf to leaf distance -> going from bottom up (child info returned to its parent, which collects such info from its children -> this leads to the post-order traversal) and through a common parent node -> calculate distance from leaf to other non-leaf nodes for both left and right sub-trees and summing these two distances will get a leaf-to-leaf distance

```java
class Solution {
    public int countPairs(TreeNode root, int distance) {
        desiredDistance = distance;
        getLeafDistances(root);
        return res;
    }

    int res;
    int desiredDistance;
    List<Integer> getLeafDistances(TreeNode root) {
        List<Integer> distances = new ArrayList<Integer>();
        // base cases
        if (root == null) {
            return distances;
        }

        if (root.left == root.right) {
            distances.add(0);
            return distances;
        }

        List<Integer> leftChildDistances = getLeafDistances(root.left);
        List<Integer> rightChildDistances = getLeafDistances(root.right);
        // post-order position
        calculateFromChildDistances(leftChildDistances, distances);
        calculateFromChildDistances(rightChildDistances, distances);

        // update good leaf nodes pairs count
        countPairs(leftChildDistances, rightChildDistances);

        return distances;
    }

    void calculateFromChildDistances(List<Integer> childDistances, List<Integer> currentDistances) {
        int size = childDistances.size();
        for (int i = 0; i < size; i++) {
            // plus child to current node, which is 1
            int currentDistance = childDistances.get(i) + 1;
            // pruning
            if (currentDistance <= desiredDistance) {
                currentDistances.add(currentDistance);
            }
        }
    }

    void countPairs(List<Integer> leftChildDistances, List<Integer> rightChildDistances) {
        for (int l: leftChildDistances) {
            for (int r: rightChildDistances) {
                // left child distance + right child distance + 2 (left to parent and right to parent)
                if (l + r + 2 <= desiredDistance) {
                    res++;
                }
            }
        }
    }
}
```

#### [1609. Even Odd Tree](https://leetcode.com/problems/even-odd-tree/) Medium

level traversal

```java
class Solution {
    public boolean isEvenOddTree(TreeNode root) {
        Queue<TreeNode> q = new LinkedList<>();
        q.offer(root);
        int level = -1;
        int size = 0;
        boolean even = true;
        TreeNode cur = null;
        while (!q.isEmpty()) {
            level++;
            size = q.size();
            even = level % 2 == 0;
            TreeNode pre = null;
            for (int i = 0; i < size; i++) {
                cur = q.poll();
                if (!checkEvenOdd(even, cur)) {
                    return false;
                }

                if (!checkOrder(even, pre, cur)) {
                    return false;
                }
                add(q, cur.left);
                add(q, cur.right);
                pre = cur;
            }
        }
        return true;
    }

    void add(Queue<TreeNode> q, TreeNode node) {
        if (node != null) {
            q.offer(node);
        }
    }

    boolean checkEvenOdd(boolean levelEven, TreeNode node) {
       if (levelEven && node.val % 2 == 0) {
           return false;
       }
       if (!levelEven && node.val % 2 == 1) {
           return false;
       }
       return true;
    }

    boolean checkOrder(boolean levelEven, TreeNode pre, TreeNode cur) {
       if (pre == null) {
           return true;
       }
       if (levelEven && cur.val <= pre.val) {
           return false;
       }
       if (!levelEven && cur.val >= pre.val) {
           return false;
       }
       return true;
    }
}
```

#### [2265. Count Nodes Equal to Average of Subtree](https://leetcode.com/problems/count-nodes-equal-to-average-of-subtree/) Medium

post-order traversal
This one is not hard. Collect tree sum and node count from children nodes and return the result to its parent.

```java
class Solution {
    public int averageOfSubtree(TreeNode root) {
        getSumAndNodeCount(root);
        return res;
    }

    int res = 0;
    // [sum, sub-tree node count]
    int[] getSumAndNodeCount(TreeNode root) {
        if (root == null) {
            return new int[2];
        }

        if (root.left == root.right) {
            res++;
            return new int[]{root.val, 1};
        }

        int[] left = getSumAndNodeCount(root.left);
        int[] right = getSumAndNodeCount(root.right);

        int treeSum = left[0] + right[0] + root.val;
        int nodeCount = left[1] + right[1] + 1;

        // update result if average is equal to root value
        if (root.val == treeSum / nodeCount) {
            res++;
        }

        return new int[]{treeSum, nodeCount};
    }
}
```

#### [2415. Reverse Odd Levels of Binary Tree](https://leetcode.com/problems/reverse-odd-levels-of-binary-tree/) Medium

This is a wrong solution. Inversion should be done in terms of the whole tree, instead of the sub-tree. This means that the root should be the symmetry axis.

```java
class Solution {
    public TreeNode reverseOddLevels(TreeNode root) {
        traverse(root);
        return root;
    }

    int depth = -1;
    void traverse(TreeNode root) {
        if (root == null) {
            return;
        }

        depth++;
        if (depth % 2 == 0 && root.left != root.right) {
            TreeNode newLeft = new TreeNode(root.left.val);
            TreeNode newRight = new TreeNode(root.right.val);

            newLeft.left = root.right.left;
            newLeft.right = root.right.right;
            newRight.left = root.left.left;
            newRight.right = root.left.right;
            root.right = newLeft;
            root.left = newRight;
        }
        // System.out.println(depth);
        traverse(root.left);
        traverse(root.right);
        depth--;
    }
}
```

preorder traversal
Left and right children are used as parameters, instead of root node. This is different.

```java
class Solution {
    public TreeNode reverseOddLevels(TreeNode root) {
        traverse(root.left, root.right, true);
        return root;
    }

    void traverse(TreeNode left, TreeNode right, boolean isOdd) {
        if (left == right) {
            return;
        }

        // this is where the swap happens
        if (isOdd) {
            int temp = left.val;
            left.val = right.val;
            right.val = temp;
        }

        traverse(left.left, right.right, !isOdd);
        traverse(left.right, right.left, !isOdd);
    }
}
```

level traversal

```java
class Solution {
    public TreeNode reverseOddLevels(TreeNode root) {
        Queue<TreeNode> q = new LinkedList<>();
        q.offer(root);
        boolean isOdd = true;
        while (!q.isEmpty()) {
            int sz = q.size();
            while (sz-- > 0) {
                TreeNode cur = q.poll();
                add(q, cur.left);
                add(q, cur.right);
            }

            // starting from the second level (index 1)
            // so it is an odd level
            if (isOdd && !q.isEmpty()) {
                int[] levelVals = new int[q.size()];
                int i = 0;
                for (TreeNode node: q) {
                    levelVals[i++] = node.val;
                }
                int j = levelVals.length - 1;
                for (TreeNode node: q) {
                    node.val = levelVals[j--];
                }
            }
            isOdd = !isOdd;
        }
        return root;
    }

    void add(Queue<TreeNode> q, TreeNode node) {
        if (node != null) {
            q.offer(node);
        }
    }
}
```

#### [Top View of Binary Tree](https://www.geeksforgeeks.org/problems/top-view-of-binary-tree/1?page=1&category=Tree&sortBy=submissions)

level traversal and mark columns, use a tree map to store the first node of each column

````java
```java
class Pair
{
    int hdistance;
    Node node;
    Pair (int distance, Node node)
    {
        this.hdistance = distance;
        this.node = node;
    }
}

class Solution
{
    //Function to return a list of nodes visible from the top view
    //from left to right in Binary Tree.
    static ArrayList<Integer> topView(Node root)
    {
        ArrayList<Integer> res = new ArrayList<>();
        if (root.left == null && root.right == null) {
            res.add(root.data);
            return res;
        }

        // column ->
        Map<Integer, Node> map = new TreeMap<>();

        Queue<Pair> candidates = new LinkedList<Pair>();
        // Pair = horizental distance -> node
        candidates.offer(new Pair(0, root));
        while (!candidates.isEmpty()) {
            Pair temp = candidates.poll();
            int hdistance = temp.hdistance;;
            Node node = temp.node;
            // only put the first node of each column
            if (!map.containsKey(hdistance)) {
                map.put(hdistance, node);
            }

            if (node.left != null) {
                candidates.offer(new Pair(hdistance - 1, node.left));
            }

            if (node.right != null) {
                candidates.offer(new Pair(hdistance + 1, node.right));
            }
        }

        for (Map.Entry<Integer, Node> entry: map.entrySet()) {
            res.add(entry.getValue().data);
        }
        return res;
    }
}
````

#### [Vertical Traversal of Binary Tree](https://www.geeksforgeeks.org/problems/print-a-binary-tree-in-vertical-order/1?page=1&category=Tree&sortBy=submissions)

```java
class Pair {
    int hd;
    Node node;
    public Pair(int hd, Node node) {
        this.hd = hd;
        this.node = node;
    }
}

class Solution
{
    //Function to find the vertical order traversal of Binary Tree.
    static ArrayList <Integer> verticalOrder(Node root)
    {
        ArrayList<Integer> res = new ArrayList<>();
        Map<Integer, List<Integer>> map = new TreeMap<>();
        Queue<Pair> q = new LinkedList<>();
        q.add(new Pair(0, root));
        while (!q.isEmpty()) {
            Node temp = q.peek().node;
            int hd = q.peek().hd;
            q.poll();
            map.putIfAbsent(hd, new ArrayList<>());
            map.get(hd).add(temp.data);
            if (temp.left != null) {
                q.offer(new Pair(hd - 1, temp.left));
            }
            if (temp.right != null) {
                q.offer(new Pair(hd + 1, temp.right));
            }
        }

        for (List<Integer> column: map.values()) {
            res.addAll(column);
        }

        return res;
    }
}
```

#### [Left View of Binary Tree](https://www.geeksforgeeks.org/problems/left-view-of-binary-tree/1?page=1&category=Tree&sortBy=submissions)

level traversal

```java
class Tree
{
    ArrayList<Integer> leftView(Node root)
    {
      ArrayList<Integer> res = new ArrayList<>();
      if (root == null) return res;
      Queue<Node> q = new LinkedList<Node>();
      q.add(root);
      while (!q.isEmpty()) {
          int count = q.size();
          List<Node> levelNodes = new ArrayList<>();
          for (int i = 0; i < count; i++) {
              if (q.peek().left != null) {
                  q.add(q.peek().left);
              }
              if (q.peek().right != null) {
                  q.add(q.peek().right);
              }
              levelNodes.add(q.remove());
          }
          if (levelNodes.size() > 0)
            res.add(levelNodes.get(0).data);
      }
      return res;
    }
}
```

#### [Level order traversal in spiral form](https://www.geeksforgeeks.org/problems/level-order-traversal-in-spiral-form/1?page=1&category=Tree&sortBy=submissions)

```java
class Spiral
{
    //Function to return a list containing the level order
    //traversal in spiral form.
    ArrayList<Integer> findSpiral(Node root)
    {
        ArrayList<Integer> res = new ArrayList<>();
        Queue<Node> q = new LinkedList<>();
        q.offer(root);
        boolean reversed = true;
        while (!q.isEmpty()) {
            int count = q.size();
            List<Integer> levelVals = new ArrayList<>();
            for (int i = 0; i < count; i++) {
                Node temp = q.poll();
                levelVals.add(temp.data);
                if (temp.left != null) q.offer(temp.left);
                if (temp.right != null) q.offer(temp.right);
            }
            if (reversed) {
                Collections.reverse(levelVals);
            }
            res.addAll(levelVals);
            reversed = !reversed;
        }

        return res;
    }
}
```

#### [Sum Tree](https://www.geeksforgeeks.org/problems/sum-tree/1?page=1&category=Tree&sortBy=submissions)

```java
class Solution
{
	boolean isSumTree(Node root)
	{
        if (root == null) return true;
        if (root.left == null && root.right == null) return true;
        return root.data == sumSubTree(root) && isSumTree(root.left) && isSumTree(root.right);
	}

    int sumSubTree(Node root) {
        if (root == null) return 0;
        int lval = 0, rval = 0;
        if (root.left != null) lval = root.left.data;
        if (root.right != null) rval = root.right.data;
        return lval + rval + sumSubTree(root.left) + sumSubTree(root.right);
    }
}

```

#### [Determine if Two Trees are Identical](https://www.geeksforgeeks.org/problems/determine-if-two-trees-are-identical/1?page=1&category=Tree&sortBy=submissions)

```java
class Solution
{
    //Function to check if two trees are identical.
	boolean isIdentical(Node root1, Node root2)
	{
	    if (root1 == null && root2 == null) return true;
	    if (root1 == null) return false;
	    if (root2 == null) return false;
	    if (root1.data != root2.data) return false;

	    return isIdentical(root1.left, root2.left) && isIdentical(root1.right, root2.right);
	}

}
```

#### [Mirror Tree](https://www.geeksforgeeks.org/problems/mirror-tree/1?page=1&category=Tree&sortBy=submissions)

```java
class Solution {
    // Function to convert a binary tree into its mirror tree.
    void mirror(Node node) {
        if (node == null) return;

        Node left = node.left;
        node.left = node.right;
        node.right = left;

        mirror(node.left);
        mirror(node.right);
    }
}
```
