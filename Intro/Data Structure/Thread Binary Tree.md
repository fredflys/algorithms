#### Thread Binary Tree

```java
public class ThreadBinaryTreeDemo {
    public static void main(String[] args) {
        ThreadNode aNode = new ThreadNode(1, "Alice");
        ThreadNode bNode = new ThreadNode(3, "Bob");
        ThreadNode cNode = new ThreadNode(6, "Cathy");
        ThreadNode dNode = new ThreadNode(8, "David");
        ThreadNode eNode = new ThreadNode(10, "Elizabeth");
        ThreadNode fNode = new ThreadNode(14, "Fred");
        /*
               1a       a
            3b    6c
          8d  10e 14f

        infixTraversal: 8 3 10 1 14 6
        preTraversal:   1 3 8 10 6 14
        ThreadInfix
                        8  -> 3
                        10 -< 3
                        10 -> 1
                        14 -< 1
                        14 -> 6
        ThreadPre
                        8  -< 3
                        8  -> 10
                        10 -< 8
                        10 -> 6
                        14 -< 6
        
        */
        aNode.setLeft(bNode);
        aNode.setRight(cNode);
        bNode.setLeft(dNode);
        bNode.setRight(eNode);
        cNode.setLeft(fNode);

        ThreadBinaryTree tree = new ThreadBinaryTree();
        tree.setRoot(aNode);
        // tree.preTraversal();
        // tree.setThreadInfix();
        // tree.traverseInfix();

        tree.setThreadPre();
        // System.out.println(eNode.getRight());
        tree.traversePre();
        // System.out.println(tree.getRoot());
        // System.out.println(fNode.getLeft());
        // System.out.println(fNode.getRight());

        System.out.println("-------");
        
    }

}

class ThreadNode extends TreeNode{
    // 0 indicates left/right sub-tree, whereas 1 means precursor/successor node
    private int leftType;
    private int rightType;
    private int no;
    private String name;
    private ThreadNode left;
    private ThreadNode right;

    public ThreadNode(int no, String name){
        super(no, name);
        this.no = no;
        this.name = name;
    }

    @Override
    public ThreadNode getLeft() {
        return this.left;
    }
    @Override
    public ThreadNode getRight() {
        return this.right;
    }
    public void setLeft(ThreadNode left) {
        this.left = left;
    }
    public void setRight(ThreadNode right) {
        this.right = right;
    }

    public int getLeftType() {
        return leftType;
    }
    public void setLeftType(int leftType) {
        this.leftType = leftType;
    }
    public int getRightType() {
        return rightType;
    }
    public void setRightType(int rightType) {
        this.rightType = rightType;
    }

}

class ThreadBinaryTree extends BinaryTree{
    private ThreadNode root;
    // reserve a pre-node, so you can use the current node in the next iteration
    // 当前的树是单向的，也就是要给某个节点设置前驱时，是无法直接取到的，要在当前轮次中存下当前节点，好在下一轮中当作前驱节点
    private ThreadNode pre;
    
    public void setRoot(ThreadNode root) {
        this.root = root;
    }

    @Override
    public ThreadNode getRoot() {
        return root;
    }

    public void traverseInfix(){
        ThreadNode current = root;
        while(current != null){
            // left sub-tree
            while(current.getLeftType() == 0){
                current = current.getLeft();
            }
            System.out.println(current);
            // root of current sub-tree    
            while(current.getRightType() == 1){
                current = current.getRight();
                System.out.println(current);
            }
            // right sub-tree
            current = current.getRight();
        }

    }

    public void traversePre(){
        ThreadNode current = root;
        while(current != null){
            // root
            System.out.println(current);
            // left sub-tree
            while(current.getLeftType() == 0){
                current = current.getLeft();
                System.out.println(current);
            }
            // right sub-tree
            // finishes after printing the root of the next sub-tree
            while(current.getRightType() == 1){
                current = current.getRight();
                System.out.println(current);
            }
            // to the left sub-tree again
            current = current.getLeftType() == 0 ? current.getLeft() : null;

        }
    }

    public void setThreadInfix(){
        this.setThreadInfix(root);
    }

    public void setThreadPre(){
        this.setThreadPre(root);
    }

    public void setThreadInfix(ThreadNode node){
        if(node == null){
            return;
        }
        // left sub-tree
        setThreadInfix(node.getLeft());
        // current node
        // --precursor 
        if(node.getLeft() == null){
            node.setLeft(pre);
            node.setLeftType(1);
        }
        // --successor
        if(pre != null && pre.getRight() == null){
            pre.setRight(node);
            pre.setRightType(1);
        }
        // key step: turn current node into the precursor to the next node 
        pre = node;
        
        //right sub-tree
        setThreadInfix(node.getRight());
    }

    public void setThreadPre(ThreadNode node){
        if(node == null){
            return;
        }
        // current node
        if(node.getLeft() == null){
            node.setLeft(pre);
            node.setLeftType(1);
        }
        if(pre != null && pre.getRight() == null){
            pre.setRight(node);
            pre.setRightType(1);
        }
        pre = node;
        // left sub-tree
        // 注意这里传入的参数必须有判断，否则就会陷入死循环，因为上面可能刚重新设置过left和right节点，而要处理的是真正的前驱和后继节点，而非线索化后的节点
        setThreadPre(node.getLeftType() == 0 ? node.getLeft() : null);
        // right sub-tree
        setThreadPre(node.getRightType() == 0 ? node.getRight() : null);
    }
}
```
