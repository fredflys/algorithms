import java.lang.reflect.Array;

public class BinarySearchTreeDemo {
    public static void main(String[] args) {
        int[] arr = {7, 3, 10, 12, 5, 1, 9, 2};
        BinarySearchTree tree = new BinarySearchTree(arr[0]);
        for(int i = 1;i<arr.length;i++){
            tree.add(arr[i]);
        }
        
        tree.delete(7); 
        tree.infixTraversal();
    }    
}

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
     * 要确定待删除节点的子节点要挂到父节点的哪个位置。简而言之，就是要明确父节点--当前节点--子节点三者之间的关系
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
                // 本身是父节点的左子节点
                if(parent.left == node){
                    parent.left = node.left;
                // 本身是父节点的右子节点
                } else {
                    parent.right = node.left;
                }
            // 有右子节点
            } else {
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