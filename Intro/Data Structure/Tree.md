```java
public class TreeDemo {
    public static void main(String[] args) {
        BinaryTree binaryTree = new BinaryTree();
        TreeNode nodeA = new TreeNode(1, "Alice");
        TreeNode nodeB = new TreeNode(2, "Bob");
        TreeNode nodeC = new TreeNode(3, "Catherine");
        TreeNode nodeD = new TreeNode(4, "David");

        binaryTree.setRoot(nodeA);
        nodeA.setLeft(nodeB); 
        nodeA.setRight(nodeC);
        nodeC.setRight(nodeD);
    
        binaryTree.infixTraversal();
        System.out.println(binaryTree.preSearch(4));
        // binaryTree.deleteNode(3);
        System.out.println("-------------");
        binaryTree.preTraversal();
    }
}

class BinaryTree{
    private TreeNode root;

    public void setRoot(TreeNode root){
        this.root = root;
    }
    public TreeNode getRoot() {
        return root;
    }
    
    public void preTraversal(){
        if(this.getRoot() != null){
            this.getRoot().preTraversal();
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
    
    public TreeNode preSearch(int no){
        System.out.println("Presearching......");
        TreeNode resultNode = null;
        if(this.root != null){
            resultNode = this.root.preSearch(no);
        }else{
            System.out.println("Empty binary tree.");
        }
        return resultNode;
    }
    
    public void deleteNode(int no){
        if(root != null){
            if(root.getNo() == no){
                root = null;
            }else{
                root.del(no);
            }
        }else{
            System.out.println("Emptry tree...nothing to delete");
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
    
    public int getNo() {
        return no;
    }
    public void setNo(int no) {
        this.no = no;
    }
    
    public TreeNode getLeft() {
        return left;
    }
    
    public TreeNode getRight() {
        return right;
    }
    
    @Override
    public String toString(){
        return "TreeNode [no=" + no + ", name=" + name + "]";
    }
    
    public void preTraversal(){
        System.out.println("Pre-traversing: " + this);
        //System.out.println("Test  " + this.getLeft());
        if(this.getLeft() != null){
            this.getLeft().preTraversal();
        }
        if(this.getRight() != null){
            this.getRight().preTraversal();
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
    
    public TreeNode preSearch(int no){
        System.out.println("--Comparison is done once...");
        if(this.no == no){
            return this;
        }else{
            TreeNode resultNode = null;
            if(this.left != null){
                resultNode = this.left.preSearch(no);
                if(resultNode != null){
                    return resultNode;
                }
            }
            if(this.right != null){
                resultNode = this.right.preSearch(no);
            }
            return resultNode;
    
        }
    }
    
    public TreeNode infixSearh(int no){
        TreeNode resultNode = null;
        if(this.left != null){
            resultNode = this.left.infixSearh(no);
            if(resultNode != null){
                return resultNode;
            }
        }
    
        if(this.no == no){
            return this;
        }
    
        if(this.right != null){
            resultNode = this.right.preSearch(no);
        }
        return resultNode;
    }
    
    public TreeNode postSearch(int no){
        TreeNode resultNode = null;
        if(this.left != null){
            resultNode = this.left.infixSearh(no);
            if(resultNode != null){
                return resultNode;
            }
        }
        if(this.right != null){
            resultNode = this.right.preSearch(no);
        }
        
        if(this.no == no){
            return this;
        }
    
        return resultNode;
    }
    
    public void del(int no){
        if(this.left != null && this.left.no == no){
            this.left = null;
            return;
        }
    
        if(this.right != null && this.right.no == no){
            this.right = null;
            return;
        }
    
        if(this.left != null){
            System.out.println("zzz");
            this.left.del(no);
        }
    
        if(this.right != null){
            this.right.del(no);
        }
    }
}
```
