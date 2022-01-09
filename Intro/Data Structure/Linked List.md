```java
/*
1.获取单链表的节点个数（不包括头节点） getLength
2.查找单链表中的倒数第k个节点 getNthToLastNode
3.反转单链表 reverse
4.逆序打印单链表 printReverse
*/
import java.util.Stack;


public class LinkedListDemo {
    public static void main(String[] args) {
        Node hero1 = new Node(1, "宋江", "及时雨");
        Node hero2 = new Node(2, "卢俊义", "玉麒麟");
        Node hero3 = new Node(3, "吴用", "智多星");
        Node hero4 = new Node(4, "林冲", "豹子头");
        Node hero5 = new Node(5, "A", "a");
        Node hero2b = new Node(2, "B", "b");

        LinkedList linkedlist = new LinkedList();
        linkedlist.insert(hero4);
        linkedlist.insert(hero2);
        linkedlist.insert(hero3);
        linkedlist.insert(hero1);
        linkedlist.insert(hero2);

        linkedlist.show();

        linkedlist.update(hero5);
        linkedlist.update(hero2b);

        //linkedlist.delete(1);
        linkedlist.show();
        System.out.printf("The linked list has %d nodes \n", linkedlist.getLength());

        System.out.println("Print the linked list from the bottom up.");
        linkedlist.printReverse();
        linkedlist.getNthToLastNode(2);
        linkedlist.reverse();
        linkedlist.show();
        
    }
}

class LinkedList{
    private Node head = new Node(0, "", "");
    
    // simply add a new node 
    public void add(Node node){
        Node current = this.head;

        while(current.next != null){
            current = current.next;
        }
        current.next = node;
    }

    public void insert(Node node){
        Node current = this.head;
        // when empty
        if(head.next == null){
            head.next = node;
            return;
        }
        // when not empty
        while(current.next != null){    
            if(current.next.no > node.no){
                // insert new node between current and its next node
                node.next = current.next; 
                current.next = node;
                return;
            }
            if(current.next.no == node.no){
                System.out.println("The same node is found in the list. No insertion needed.");
                return;
            }
            current = current.next;
        }
        // add
        current.next = node;
    }

    public void update(Node node){
        if(this.head == null){
            System.out.println("Empty linked list. Update aborted.");
            return;
        }
        Node current = head.next;
        while(current.next != null){
            if(current.no == node.no){
                // current = node 不可以这样写, node并没有指定next的值
                current.name = node.name;
                current.nickname = node.nickname;
                return;
            }
            current = current.next;
        }
        System.out.println("Node not found. Update aborted.");
        }

    public void delete(int no){
        // when empty
        if(head.next == null){
            System.out.println("Empty linked list. Deletion aborted.");
            return;
        }
        // when not empty
        Node current = this.head;
        while(current.next != null){
            // find the node right before the node to be deleted
            if(current.next.no == no){
                current.next = current.next.next;
                return;
            }
            current = current.next;
        }
        System.out.println("Not found. Deletion aborted.");
    }

    public void show(){
        if(head.next == null){
            System.out.println("Empty linkedlist. No items to be shown.");
            return;
        }

        Node current = this.head; 
        System.out.println("Below is the linked list:");
        while(current.next != null){
            current = current.next;
            System.out.println(current);
        }
    }


    public int getLength(){
        if(head.next == null){
            return 0;
        }
        Node current = head.next;
        int len = 1;
        while(current.next != null){
            len++;
            current = current.next;
        }
        return len;

    }

    public Node getNthToLastNode(int n){
        if(head.next == null){
            System.out.println("Empty linked list. Can't get the reversed nth node.");
            return null;
        }

        int len = this.getLength();
        // invalid case
        if(n > len && n <= 0){
            System.out.println("Invalid N.");
            return null;
        }
        
        Node current = head.next;
        int count = 1;
        while(count <= len - n){
            count++;
            current = current.next;
        }
    
        System.out.printf("The %d to last node is " + current + "\n", n);
        return current;
    }

    public void reverse(){
        /*
        新建空头，如有新链，遍历原链，依次从头部放入新链（与后部追加相反）
        如何从头部放入？
        -当前节点与新链已有节点相连
        -新头与当前节点相连
        最后将旧链的下一个指向新链的下一个
        */
        if(head.next == null || head.next.next == null){
            return;
        }

        Node current = head.next;
        Node next = null;
        Node reverseHead = new Node(0,"", "");

        while(current != null){
            next = current.next;
            // 当前节点与新链已有节点相连
            current.next = reverseHead.next;
            // 新头与当前节点相连
            reverseHead.next = current;
            current = next;
        }
        head.next = reverseHead.next;
    }

    public void printReverse(){
        if(head.next == null){
            System.out.println("Empty linked list. Nothing to print.");
        }

        Stack<Node> stack = new Stack();
        Node current = head;
        while(current.next != null){
            current = current.next;
            stack.add(current);
        }
        while(stack.size() > 0){
            System.out.println(stack.pop());
        }
    }
}

class Node {
    public int no;
    public String name;
    public String nickname;
    public Node next;

    public Node(int no, String name, String nickname){
        this.no = no;
        this.name = name;
        this.nickname = nickname;
    }

	@Override
    public String toString(){
        return "好汉: 排名=" + no + ", 名字=" + name + ", 绰号= " + nickname;
    }
}
```