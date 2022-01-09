```java
import java.util.LinkedList;
import java.util.Stack;
import java.util.Queue;

public class BinarySearchTreeDemo {
    public static void main(String[] args) {
        // 对第一种实现方式的测试
        int[] arr = {16,3,7,11,9,26,18,14,15};
        BinarySearchTree tree = new BinarySearchTree(arr[0]);
        for(int i = 1;i<arr.length;i++){
            tree.add(arr[i]);
        }
        tree.delete(7); 
        tree.infixTraversal();

        System.out.println("-------------------------------------");

        // 对第二种实现方式的测试
        BST<Integer> bst = new BST<>();
        for(int num: arr){
            bst.add(num);
        }
        System.out.println(bst);  
        System.out.println("floor for 15 inside the tree is " + bst.floor(17));
        System.out.println("ceiling for 13 inside the tree is " + bst.ceiling(13));
        System.out.println("--分隔线--");
        bst.infixOrder(); 
        System.out.println("--分隔线--");
        bst.preOrderNR();
        System.out.println("--分隔线--");
        bst.levelOrder();
        System.out.println("--分隔线--");
        bst.removeMin();
        bst.removeMax();
        bst.remove(16);
        bst.infixOrder();
    }
}    


/**
 * BinarySearchTree与BSTNode为一种实现方式，代码量较为臃肿，很多地方没有充分考虑递归的特性
 */
class BinarySearchTree{
    BSTNode root;

    public BinarySearchTree(int rootVal){
        root = new BSTNode(rootVal);
    }

	void add(int newVal){
        if(root == null){
            root = new BSTNode(newVal);
        }else{
            root.add(newVal);
        }
    }

    void infixTraversal(){
        if(root == null){
            System.out.println("Empty binary search tree.");
        } else {
            root.infixTraversal();
        }
    }

    BSTNode search(int val){
        if(root == null){
            System.out.println("Empty binary search tree.");
        }
        return root.search(val);
    }

    BSTNode searchParent(int val){
        if(root == null){
            System.out.println("Empty binary search tree.");
        }
        return root.searchParent(val);
    }

    /**
     * 删除节点主要面临的问题是：待删除节点如果有子节点，需要将子节点挂在父节点上，同时
     * 要确定待删除节点的子节点要挂到父节点的哪个位置，所以需要准备找出待删除节点和其父节点的方法。
     * 简而言之，就是要明确父节点--当前节点--子节点三者之间的关系，注意在使用父节点时要注意有没有父节点，要处理好特殊情况，否则会遇到空指针错误
     * 由此就有3种情况：叶子节点，单子树节点，双子树节点
     * @param val
     */
    void delete(int val){
        if(root == null){
            System.out.println("You cannot remove a node from an empty search tree.");
            return;
        }

        BSTNode node = search(val);
        if(node == null){
            System.out.println("The value is not found in this tree. Removal is not possible.");
            return;
        }
        BSTNode parent = searchParent(val);

        // 要删除的值落在叶子节点
        if(node.left == null && node.right == null){
            // 如果此时父节点为空，则说明当前树只有一个根节点，且要删除的就是根节点，只需将根节点置空即可
            if(parent == null){
                root = null;
                return;
            }

            if(parent.left == node){
                parent.left = null;
            } else {
                parent.right = null;
            }
        // 双子树节点
        } else if(node.left != null && node.right != null){
            // 删除后的替代节点有两个选择要么从其左子树中找最大值，要么从其右子树中找最小值
            // 因为只有这样的值换上去，才会满足平衡二叉树的要求，即当前节点皆大于左测子树的值并小于右侧子树的值
            BSTNode swapNode = node.left;
            while(swapNode.right != null){
                swapNode = swapNode.right;
            }
            System.out.println("Node to be swapped: " + swapNode);
            /**
             * 另一种方法，从右子树中找出最小的
             * BSTNode swapNode = node.right;
             * while(swapNode.left != null){
             *     swapNode = swapNode.left;
             * }
             */
            int swapVal = swapNode.val;
            delete(swapNode.val);
            node.val = swapVal;
        // 单子树节点
        } else {
            // 有左子节点
            if(node.left != null){
                // 如果没有父节点，则说明当前树只有一个根节点和一个子节点，此时只需将根节点指向子节点即可
                if(parent == null){
                    root = node.left;
                    return;
                }

                // 本身是父节点的左子节点
                if(parent.left == node){
                    parent.left = node.left;
                // 本身是父节点的右子节点
                } else {
                    parent.right = node.left;
                }
            // 有右子节点
            } else {
                if(parent == null){
                    root = node.right;
                    return;
                }

                // 本身是父节点的左子节点
                if(parent.left == node){
                    parent.left = node.right;
                // 本身是父节点的右子节点
                } else {
                    parent.right = node.right;
                }
            }
        }
        
    }


}

class BSTNode{
    int val;
    BSTNode left;
    BSTNode right;

    public BSTNode(int val){
        this.val = val;
    }

    void add(int newVal){
        // 小于当前节点则放在左枝
        if(newVal < this.val){
            // 左子节点如无，则置为左子节点，否则对左子节点递归执行插入方法
            if(this.left == null){
                this.left = new BSTNode(newVal);
            }else{
                this.left.add(newVal);
            }
        // 大于等于当前节点则放在右纸
        }else{
            // 与上同
            if(this.right == null){
                this.right = new BSTNode(newVal);
            }else{
                this.right.add(newVal);
            }
        }
    }

    BSTNode search(int val){
        // 找到时终止递归
        if(this.val == val){
            return this;
        }
        
        // 分别向左向右查找
        if(val < this.val){
            // 确保有左子节点才去查找
            if(this.left == null){
                // 没找到时终止递归
                return null;
            }
            // 向左递归
            return this.left.search(val);
        } else{
            // 确保有右子节点才去查找
            if(this.right == null){
                // 没找到时终止递归
                return null;
            }
            // 向右递归
            return this.right.search(val);
        } 
    }

    BSTNode searchParent(int val){
        // 找到时终止递归，左子节点或右子节点与要找的值相等，且相应子节点存在
        if( (this.left != null && this.left.val == val) 
            || (this.right != null && this.right.val == val) 
            ){
            return this;
        // 没找到就开始递归查找
        } else {
            // 向左查找
            if(val < this.val){
                // 没找到时终止递归
                if(this.left == null){
                    return null;
                }
                return this.left.searchParent(val);
            // 向右查找
            } else {
                // 没找到时终止递归
                if (this.right == null) {
                    return null;
                }
                return this.right.searchParent(val);
            }
        }
    }

    void infixTraversal(){
        if(this.left != null){
            this.left.infixTraversal();
        }
        System.out.println(this);
        if(this.right != null){
            this.right.infixTraversal();
        }
    }

    @Override
    public String toString() {
        return "Node: " + this.val;
    }
}


/**
 * BST为另外一种实现方式，充分考虑了递归的特性，代码更为简洁
 * @param <E>
 */
class BST<E extends Comparable<E>>{
    // 节点构造为内部私有类，用户不必了解搜索树内部的节点情况
    private class Node{
        public E e;
        public Node left, right;

        public Node(E e){
            this.e = e;
            left = null;
            right = null;
        }

        @Override
        public String toString() {
            return "Node: " + e;
        }
    }

    private Node root;
    private int size;

    public BST() {
        root = null;
        size = 0;
    }

    public int size(){
        return size;
    }

    public boolean isEmpty(){
        return size == 0;
    }

    public void add(E e){
        /** 原版使用递归函数的部分，需要额外判断根节点是否为空
         if(root == null){
            root = new Node(e);
            size++;
        } else {
            add(root, e);
        }
         */
        root = add(root, e);

    }
    
    /**
     * 向以node为根节点的树中插入元素e，递归算法
     * @param node 待插入子树的根节点，递归过程中是不断转换到更小的子树，定位到真正插入e的位置，所以需要这个参数
     * @param e 待插入元素e
     * @return 返回插入新节点后二分搜索树的根
     */
    private Node add(Node node, E e){
        
        /** 原版递归终止条件，返回值为空，缺点是较为臃肿，逻辑与调用时并不统一
        if(e.equals(node.e))
            return;
        else if(e.compareTo(node.e) < 0 && node.left == null){
            node.left = new Node(e);
            return;
        }
        else if(e.compareTo(node.e) > 0 && node.right == null){
            node.right = new Node(e);
            return; 
        }
        */

        // 递归中止条件，可以直接插入
        // 其实就是把空也当作是二叉树，这样逻辑就前后统一了
        if(node == null){
            size++;
            return new Node(e);
        }
            
        // 递归部分
        // 小于当前节点，则向左添加
        if(e.compareTo(node.e) < 0)
            // 挂接到原来根节点的左侧
            node.left = add(node.left, e);
        // 大于当前节点，向右添加
        // 等于0则什么都不做，等于在当前定义的搜索树中不会出现相等的元素
        else if(e.compareTo(node.e) > 0)
            node.right = add(node.right,e);

        // 返回新搜索树的根部
        return node;
    }

    public boolean contains(E e){
        return contains(root, e);
    }
    private boolean contains(Node node, E e){
        if(node == null)
            return false;
        
        if(e.compareTo(node.e) == 0)
            return true;
        else if(e.compareTo(node.e) < 0)
            return contains(node.left, e);
        else
            return contains(node.right, e);
    }

    // 层序遍历，即广度优先遍历，同一层先执行完操作，才会到下一层
    // 即先遍历到的先操作，属于先进先出的模型，可以使用队列来实现
    public void levelOrder(){
        Queue<Node> q = new LinkedList<>();
        q.add(root);
        while(!q.isEmpty()){
            Node current = q.remove();
            System.out.println(current.e);
            if(current.left != null)
                q.add(current.left);
            if(current.right != null)
                q.add(current.right);
        }
    }


    // 前序遍历非递归版本，借用栈来实现，因为是无论是前序中序还是后序，都是深度优先，有一个回退的过程
    // 对应的就是先遍历到的后出，即先进后出的模型，与栈一致
    // 而中序和后序的非递归实现更为复杂，后序尤甚
    public void preOrderNR(){
        Stack<Node> stack = new Stack<Node>();
        stack.push(root);
        while(!stack.isEmpty()){
            Node current = stack.pop();
            System.out.println(current.e);

            if(current.right != null)
                stack.push(current.right);
            if(current.left != null)
                stack.push(current.left);
        }
    }

    // 前序遍历是最为自然的一种遍历方式，符合树结构给我们的直觉印象
    public void preOrder(){
        preOrder(root);
    }
    // 前序遍历以node为根的二叉树
    private void preOrder(Node node){
        if(node == null)
            return;

        System.out.println(node.e);
        preOrder(node.left);
        preOrder(node.right);
    }

    // 中序遍历即左中右（小中大）的顺序，也就是按从小到大的顺序输出，这是相较于其它结构额外的优势，不做额外处理，就可以按升序排列
    public void infixOrder(){
        infixOrder(root);
    }

    private void infixOrder(Node node){
        if(node == null)
            return;
        infixOrder(node.left);
        System.out.println(node.e);
        infixOrder(node.right);
    }

    // Hibbard deletion
    /**
     * 删除共有三种情况：删除的节点只有左子树，只有右子树，或者左右子树皆有
     * 至于叶子节点，可以看作是只有左子树或者只有右子树，不必特殊处理
     * 对于只有左子树的节点，需要将待删除节点的左孩子当作新子树的根节点返回，并挂到待删除节点的父节点上
     * 对于只有右子树的节点，需要将待删除节点的右孩子当作新子树的根节点返回，并挂到待删除节点的父节点上
     * 对于有左右子树的节点，我们要做的是找到一个值取代待删除节点，最合适的自然是离待删节点最近的节点
     * 对此有两个选择，一个是左侧子树中的最大值，一个是右侧子树中的最小值，只有这两个节点中的任意一个
     * 顶替删除节点时，才能满足左侧子树都小于它，右侧子树都大于它的条件，不必额外做出改动
     */
    public void remove(E e){
        root = remove(root, e);
    }

    /**
     * 
     * @param node 待删除节点所在树的根节点
     * @param e 待删除的值
     * @return 删除以node为根节点的树中，元素为e的节点
     */
    private Node remove(Node node, E e){
        if(node == null)
            return null;
        
        // 还未找到e，向左或向右递归
         if(e.compareTo(node.e) < 0){
            node.left = remove(node.left, e);
            return node;
         }   
         else if(e.compareTo(node.e) > 0){
             node.right = remove(node.right,e);
             return node; 
         }
         // 真正找到e，开始删除
         else{ // e == node.ee
            // 待删除节点只有右子树
            if(node.left == null){
                Node rightNode = node.right;
                node.right = null;
                size--;
                return rightNode;
            }
            // 待删除节点只有左子树
            if(node.right == null){
                Node leftNode = node.left;
                node.left = null;
                size--;
                return leftNode;
            }

            // 待删除节点同时有左右子树
            // 注意这一种情况中不必再额外减小size，因为在removeMin中已经执行过了，其它步骤其实只是拿successor替代node

            // 先找到待删除节点的后继，即右子树中的最小节点
            Node successor = min(node.right);
            // 在右子树中删除该后继节点，并将新的根节点挂在后继节点的右侧
            successor.right = removeMin(node.right);
            successor.left = node.left;

            // 显式删除node根节点
            node.left = node.right = null;
            // 原本以node为根节点的子树，其根节点已经变为successor
            return successor;
         }
    }

    public E removeMin(){
        E ret = min();
        root = removeMin(root);
        return ret;
    }

    // 删除以node为根节点的树中的最小值，并返回删除最小值后的根节点
    private Node removeMin(Node node){
        // 递归中止，当前node已经是最小值
        // 要将当前node的右孩子返回，作为n删除node后子树的根节点
        if(node.left == null){
            Node rightNode = node.right;
            node.right = null;
            size--;
            return rightNode;
        }

        // 将返回的子树与node的左枝连接，这样才能确实改变原来的树结构，真正删除掉最小值
        node.left = removeMin(node.left);
        // 删除以node为根节点的树中的最小值后，node依然是根节点，因此返回该根节点
        return node;
    }

    public E min(){
        if(size == 0)
            throw new IllegalArgumentException("BST is empty.");
        return min(root).e;
    }
    
    private Node min(Node node){
        if(node.left == null)
            return node;
        return min(node.left);
    }

    public E removeMax(){
        E ret = max();
        root = removeMax(root);
        return ret;
    }

    private Node removeMax(Node node){
        if(node.right == null){
            Node leftNode = node.left;
            node.left = null;
            size--;
            return leftNode;
        }

        node.right = removeMax(node.right);
        return node;
    }

    public E max(){
        if(size == 0)
            throw new IllegalArgumentException("BST is empty.");
        return max(root).e;
    }
    
    private Node max(Node node){
        if(node.right == null)
            return node;
        return max(node.right);
    }

    // 找出树中比e大的值里最小的节点
    public Node ceiling(E e){
        if(size == 0)
            return null;
        if(e.compareTo(max()) > 0)
            return null;
        return ceiling(root, e);
    }

    private Node ceiling(Node node, E e){
        if(node == null)
            return null;
        
        if(e.compareTo(node.e) == 0)
            return node;
        else if(e.compareTo(node.e) > 0)
            return ceiling(node.right, e);
        else {
            Node temp = ceiling(node.left, e);
            if(temp == null)
                return node;
            return temp;
        }
    }

    //找出树中比e小的值里最大的节点
    public Node floor(E e){
        if(size == 0)
            return null;
        if(e.compareTo(min()) < 0)
            return null;
        return floor(root, e);
    }

    private Node floor(Node node, E e){
        if(node == null)
            return null;

        // e与当前节点相等就返回
        if(e.compareTo(node.e) == 0){
            return node;
        // e比当前节点小就继续向左找（floor的值比e还要小）
        } else if(e.compareTo(node.e) < 0)  {
            return floor(node.left,e);
        } else {

            Node temp = floor(node.right,e);
            if(temp == null)
                return node;
            else
                return temp;
        }

        
    }

    @Override
    public String toString(){
        StringBuilder sb = new StringBuilder();
        generateBSTString(root, 0, sb);
        return sb.toString();
    }

    private void generateBSTString(Node node, int depth, StringBuilder sb){
        if(node == null){
            sb.append(generateDepthString(depth) + "null\n");
            return;
        }

        sb.append(generateDepthString(depth) + node.e + "\n");
        generateBSTString(node.left, depth+1, sb);
        generateBSTString(node.right, depth+1, sb);
    }

    private String generateDepthString(int depth){
        StringBuilder sb = new StringBuilder();
        for(int i = 0; i < depth; i++){
            sb.append("-");
        }
        return sb.toString();
    }
}
```