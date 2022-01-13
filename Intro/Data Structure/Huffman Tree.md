#### Huffman Tree

```java
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class HuffmanTreeDemo {
    public static void main(String[] args) {
        int[] arr = {7,5,5,2,4};
        HuffmanTree tree = new HuffmanTree(arr);
        tree.prefixTraversal();
    }

}

class Hnode implements Comparable<Hnode> {
    int weight;
    Hnode left, right, parent;

    public Hnode(int weight){
        this.weight = weight;
    }
    

    public void setLeft(Hnode left) {
        this.left = left;
    }
    public void setRight(Hnode right) {
        this.right = right;
    }
    public Hnode getLeft() {
        return left;
    }
    public Hnode getRight() {
        return right;
    }
    public Hnode getParent() {
        return parent;
    }
    public void setParent(Hnode parent) {
        this.parent = parent;
    }


    @Override
    public String toString() {
        return "Hnode Weight: " + weight;
    }

    @Override
    public int compareTo(Hnode o) {
        // from lower to higher
        return this.weight - o.weight;
    }    


    public void prefixTraversal(){
        System.out.println(this);
        if(this.left != null){
            this.left.prefixTraversal();
        }
        if(this.right != null){
            this.right.prefixTraversal();
        }
    }
}

class HuffmanTree{
    List<Hnode> nodes;
    Hnode root;

    public HuffmanTree(int[] arr){
        nodes = new ArrayList<Hnode>();
        for(int weight: arr){
            nodes.add(new Hnode(weight));
        }

        while(nodes.size() > 1){
            Collections.sort(nodes);
            Hnode left = nodes.get(0);
            Hnode right = nodes.get(1);
            Hnode parent = new Hnode(left.weight + right.weight);
            parent.left = left;
            parent.right = right;
            // removal will not only delete the element at the specified position, but also shift subsequent elements to the left 
            nodes.remove(0);
            nodes.remove(0);
            nodes.add(parent);
        }
        root = nodes.get(0);
    }

    public void prefixTraversal(){
        root.prefixTraversal();      
    }
}
```