#### AVL Tree

```java
public class AVLTreeDemo {
    public static void main(String[] args) {
        int[] arr = {16,3,7,11,9,26,18,14,15};
        AVLTree tree = new AVLTree();
        for(int i = 0;i<arr.length;i++){
            tree.add(arr[i]);
        }
        /**
         *  最终结果：
         *      11
         *     /  \
         *    7   18
         *   / \  / \
         *  3  9 15 26
         *      / \
         *     14 16
         */
        tree.infixTraversal();
        System.out.println("Root -> " + tree.root + ", Size -> " + tree.size);
        tree.remove(14);
        System.out.println("-----------------");
        tree.infixTraversal();
        
    }
}

class AVLTree{
    AVLNode root;
    int size = 0;

    public AVLTree(){
        
    }

    void add(int newVal){
        if(root == null){
            root = new AVLNode(newVal);
        }else{
            root.add(newVal);
        }
        size++;
    }

    void remove(int num){
        if(root == null){
            System.out.println("Empty tree. Removal is not possible.");
        } else {
            root.remove(num);
        }
    }

    void infixTraversal(){
        if(root == null){
            System.out.println("Empty tree.");
        }else{
            root.infixTraversal();
        }
    }

}

class AVLNode{
    int val;
    AVLNode left, right;

    public AVLNode(int val) {
        this.val = val;
    }
    
    int getHeight(){
        if(left == null && right == null){
            return 1;
        }
        int l = 0;
        int r = 0;

        if(left != null) l = left.getHeight();
        if(right != null) r = right.getHeight();
        
        if (l > r) return l+1;
        else return r+1;
    }

    int getBalanceFactor(){
        int l = 0;
        int r = 0;
        if(left != null) l = left.getHeight();
        if(right != null) r = right.getHeight();
        return l - r;
    }

    void add(int newVal){
        // 小于当前节点则放在左枝
        if(newVal < this.val){
            // 左子节点如无，则置为左子节点，否则对左子节点递归执行插入方法
            if(this.left == null){
                this.left = new AVLNode(newVal);
            }else{
                this.left.add(newVal);
            }
        // 大于等于当前节点则放在右枝
        }else{
            // 与上同
            if(this.right == null){
                this.right = new AVLNode(newVal);
            }else{
                this.right.add(newVal);
            }   
        }

        int bf = getBalanceFactor();

        // LL型
        if(bf > 1){
            // LR型(RR+LL)
            if(left != null && left.getBalanceFactor() < 0){
                System.out.println("Rotate for left child node (RR)" + " at " + this.left);
                left.leftRotate();
            } 
            System.out.println("Out of balance(LL) " + bf + " at " + this);
            rightRotate();
            return;
        } 

        // RR型
        if(bf < -1){
            // RL型(LL+RR)
            if(right != null && right.getBalanceFactor() > 0){
                System.out.println("Rotate for right child node (LL)" + " at " + this.right);
                right.rightRotate();
            }
            System.out.println("Out of balance(RR) " + bf + " at " + this);
            leftRotate();
            return;
        }

    }

    /** 
     *                    LL 
     *        o                              l
     *       / \                           /   \ 
     *      l   p         向右旋转        ll    no(o)
     *     / \       - - - - - - - ->    / \   / \
     *   ll   c                         a  b  c  p
     *   / \ 
     *  a   b
     *  o: 当前节点 no: 从当前节点构造的新节点 l: 当前节点的左孩子 ll: 当前节点的左孩子的左孩子 a,b,c,p：其它节点，按字母顺序值由小到大 
     *  右旋转不是把ll甩上去，而是把o甩下来，结果是把l当作根节点，代替o，其左子树维持不变，重新构造右子树，即
     *  加入从o构建的新节点，并改变c、p的位置。
     *  要注意的点是一定要创建新节点，来指代原节点。我一开始只做交换，结果只能是left和right节点指来指去，最后根本就加不进去
     *                    LR
     *        o                          o                        lr
     *       / \                        / \                      /  \
     *      l   p    向左旋转(RR)      lr  p    向右旋转(LL)    nl(l)  no(o)
     *     / \       ------>          / \      ------>         / \   / \  
     *    a  lr                     &(l)  c                   a  b  c  p
     *      / \                     / \
     *     b   c                   a   b
     * */ 
    void rightRotate(){
        // 构建新节点
        AVLNode node = new AVLNode(val);
        node.right = right;
        node.left = left.right;

        // 重置根节点为其左孩子，其左孩子为原来的左孙子，右孩子变为新节点（即从根节点构建出的新节点）
        val = left.val;
        left = left.left;
        right = node;
    }


    /** 
     *                    RR 
     *        o                              r
     *       / \                           /   \ 
     *      a   r     向左旋转 (y)         no(o) rr
     *         / \    - - - - - - - ->   / \   / \
     *        p   rr                    a   p q   s
     *           /  \ 
     *          q    s
     *  
     *  o: 当前节点 ~: 从当前节点构造的新节点 l: 当前节点的左孩子 ll: 当前节点的左孩子的左孩子 a,p,q,s：其它节点，按字母顺序值由小到大 
     *  右旋转不是把ll甩上去，而是把o甩下来，结果是把l当作根节点，代替o，其左子树维持不变，重新构造右子树，即
     *  加入从o构建的新节点，并改变c、p的位置。
     * 
     *                          RL
     *        o                          o                        rl
     *       / \                        / \                      /  \
     *      a   r        向右旋转(LL)  a   rl    向左旋转(RR)    no(o)  nr(r)  
     *         / \       ------>          / \      ------>      / \  / \  
     *        rl  s                      p  nr(r)               a  p  q  s
     *       / \                            / \
     *      p   q                          q   r
     * */ 
    void leftRotate(){
        AVLNode node = new AVLNode(val);

        node.left = left;
        node.right = right.left;

        val = right.val;
        left = node;
        right = right.right;
    }

    void removeMin(){
        AVLNode l = left;
        if(l.left == null){

        }
    }

    void remove(int num){
        if(num < val){
            if(left == null){
                return;
            }
            left.remove(num);
        }else if(num > val){
            if(right == null){
                return;
            }
            right.remove(num);
        }else{
            if(left == null){
                this.val = right.val;
                this.left = right.left;
                this.right = right.right;
                return;
            }
            if(right == null){
                this.val = left.val;
                this.left = left.left;
                this.right = right.right;
                return;
            }
        }
    }

    @Override
    public String toString() {
        return "Node: " + val;
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
}
```