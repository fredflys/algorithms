#### Hash Table

```java
import java.util.Scanner;

public class HashTableDemo{
    public static void main(String[] args) {
        HashTable hashtable = new HashTable(7);
        
        String input = " ";
        Scanner scanner = new Scanner(System.in);
        
        while(true){
            System.out.println("(a)dd: add a new employee");
            System.out.println("(l)ist: show employees");
            System.out.println("(s)earc: find an employee by id");
            System.out.println("(e)xit");
    
            input = scanner.next();
            switch(input){
                case "a":
                    System.out.print("id: ");
                    int id = scanner.nextInt();
                    System.out.print("name: ");
                    String name = scanner.next();
                    Employee emp = new Employee(id, name);
                    hashtable.add(emp);
                    break;
                case "l":
                    hashtable.list();
                    break;
                case "e":
                    scanner.close();
                    System.exit(0);
                case "s":
                    System.out.print("id: ");
                    int searchId = scanner.nextInt();
                    hashtable.search(searchId);
                default:
                    break;
            }
        }
    }
} 


class Employee{
    public int id;
    public String name;
    public Employee next;
    public Employee(int id, String name){
        this.id = id;
        this.name = name; 
    }
}

class EmpLinkedList{
    private Employee head;
    
    public void add(Employee emp){
        if(head == null){
            System.out.println("aaaaaaaaa");
            head = emp;
            return;
        }

        Employee cur = this.head;
        while(cur.next != null){
            cur = cur.next;
        }
        cur.next = emp;
    }

    public void list(){
        if(head == null){
            return;
        }
        Employee cur = head;
        // 注意这里不是判断cur.next，在这里卡过
        while(cur != null){
            System.out.printf("Current employee => id: %d, name: %s\n", cur.id, cur.name);
            cur = cur.next;
        }
    }

    public boolean search(int id){
        if(head == null){
            System.out.println("Not found.");
            return false;
        }
        Employee cur = head;
        while(cur != null){
            if(id == cur.id){
                System.out.printf("Employee found => id: %d, name: %s\n", cur.id, cur.name);
                return true;
            }
            cur = cur.next;
        }
        System.out.println("Not found.");
        return false;
    }

    public boolean delete(int id){
        if(head == null){
            System.out.println("Empty Employee Linked List...invalid search");
            return false;
        }
        Employee cur = head;
        while(cur.next != null){
            if(cur.next.id == id){
                System.out.println("Found and deleted......");
                cur.next = cur.next.next;
                return true;
            }
            cur = cur.next;
        }
        System.out.println("Employee not found......");
        return false;

    }
}

class HashTable{
    public EmpLinkedList[] arr;
    public int size;

    public HashTable(int size){
        this.arr = new EmpLinkedList[size];
        this.size = size;
        // 这里一定要初始化链表数组中的每个链表，否则里面存的都只是null，会有空指针错误
        for(int i=0;i<size;i++){
            arr[i] = new EmpLinkedList();
        }
            
    }

    public void add(Employee emp){
        int pos = getPos(emp.id);
        arr[pos].add(emp);
    }

    public boolean search(int id){
        int pos = getPos(id);
        return arr[pos].search(id);
    }

    public boolean delete(int id){
        if(search(id) == true){
            int pos = getPos(id);
            return arr[pos].delete(id);
        }
        return false;
    }

    public int getPos(int id){
        return id % size;
    }

    public void list(){
        System.out.println("LLL");
        for(EmpLinkedList l:arr){
            l.list();
        }       
    }
}
```