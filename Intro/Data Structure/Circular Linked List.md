```java
/* 
  约瑟夫环
*/


public class CircularLinkedListDemo {
    public static void main(String[] args) {
        CircularSingleLinkedList cl = new CircularSingleLinkedList();
        cl.addNodes(1);
        cl.show();
    }
}

class CircularSingleLinkedList{
    private Linknode first = new Linknode(-1);

    public void addNodes(int num){
        if(num < 1){
            System.out.println("Invlid");
            return;
        }

        Linknode current = null;
        for(int i =1;i <= num; i++){
            Linknode node = new Linknode(i);
            if(i==1){
                first = node;
                // first points to itself
                first.setNext(first);
                current = first;
            }else{
                // current points to the next
                current.setNext(node);
                // new node(now the last) points to the first
                node.setNext(first);
                // reset current to new node
                current = node;
            }
        }

    }

    public void show(){
        if(first == null){
            System.out.print("Empty circular linked list.");
            return;
        }

        Linknode current = first;
        System.out.printf("Current node: %d \n", current.getNo());
        while(current.getNext() != first ){
            current = current.getNext();
            System.out.printf("Current node: %d \n", current.getNo());
        }
    }
}


class Linknode{
    private int no;
    private Linknode next;
    public Linknode(int no){
        this.no = no;
    }
    public void setNext(Linknode next) {
        this.next = next;
    }
    public Linknode getNext() {
        return next;
    }
    public int getNo() {
        return no;
    }
}
```