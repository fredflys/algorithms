#### Array Binary Tree

```java
public class ArrBinaryTreeDemo {
    public static void main(String[] args) {
        int[] demoArr =  {1, 2, 3, 4, 5, 6, 7};    
        ArrBinaryTree demoTree = new ArrBinaryTree(demoArr);
        demoTree.preTraversal(0);
    }
}

class ArrBinaryTree{
    private int[] arr;
    
    public ArrBinaryTree(int[] arr){
        this.arr = arr;
    }

    public void preTraversal(int no){
        /**
         *     0
         *   1   2
         *  3 4 5 6
         *2n+1   2n+2    
         */
        if(arr == null || arr.length == 0){
            System.out.println("Empty array...");
        }

        System.out.println(arr[no]);
        if((2*no)+1 < arr.length){
            preTraversal((2*no)+1);
        }
        if((2*no)+2 < arr.length){
            preTraversal((2*no)+2);
        }
    }
}
```