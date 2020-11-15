public class TreeDemo {
    public static void main(String[] args) {
        BinaryTree binaryTree = new BinaryTree();
        TreeNode nodeA = new TreeNode(1, "Alice");
        TreeNode nodeB = new TreeNode(2, "Bob");
        TreeNode nodeC = new TreeNode(3, "Catherine");
        TreeNode nodeD = new TreeNode(4, "David");

        binaryTree.setRoot(nodeA);
        nodeA.setLeft(nodeB); 
        nodeB.setRight(nodeC);
        nodeC.setLeft(nodeD);

        binaryTree.infixTraversal();

    }

    
}

class BinaryTree{
    private TreeNode root;

    public void setRoot(TreeNode root){
        this.root = root;
    }

    public void preTraversal(){
        if(this.root != null){
            this.root.preTraversal();
        }else{
            System.out.println("Empty binary tree.");
        }
    }

    public void infixTraversal(){
        if(this.root != null){
            this.root.infixTraversal();
        }else{
            System.out.println("Empty binary tree.");
        }
    }

    public void postTraversal(){
        if(this.root != null){
            this.root.postTraversal();
        }else{
            System.out.println("Empty binary tree.");
        }
    }
}

class TreeNode {
    private int no;
    private String name;
    private TreeNode left;
    private TreeNode right;

    public TreeNode(int no, String name) {
        this.no = no;
        this.name = name;
    }

    public void setLeft(TreeNode left) {
        this.left = left;
    }

    public void setRight(TreeNode right) {
        this.right = right;
    }

    public TreeNode getLeft() {
        return left;
    }

    public TreeNode getRight() {
        return right;
    }

    @Override
    public String toString(){
        return "TreeNode [no=" + this.no + ", name=" + name + "]";
    }

    public void preTraversal(){
        System.out.println(this);
        if(this.left != null){
            this.left.preTraversal();
        }
        if(this.right != null){
            this.right.preTraversal();
        }
    }

    public void infixTraversal(){
        if(this.left != null){
            this.left.infixTraversal();
        }
        System.out.println(this);
        if(this.right != null){
            this.right.infixTraversal();
        }
    }

    public void postTraversal(){
        if(this.left != null){
            this.left.infixTraversal();
        }
        if(this.right != null){
            this.right.infixTraversal();
        }
        System.out.println(this);
    }
}

