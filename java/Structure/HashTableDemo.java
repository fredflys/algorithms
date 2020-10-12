package Structure;

public class HashTableDemo {
    
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
    public Employee head;
    
    public void add(Employee emp){
        if(head == null){
            head = emp;
            return;
        }
        Employee cur = head;
        while(cur.next != null){
            cur = cur.next;
        }
        cur.next = emp;
    }

    public void list(){
        if(head == null){
            System.out.println("Empty Employee Linked List");
        }
        Employee cur = head;
        while(cur.next != null){
            System.out.printf("Current employee => id: %d, name: %s\n", cur.id, cur.name);
            cur = cur.next;
        }
    }

    public boolean search(int id){
        if(head == null){
            System.out.println("Empty Employee Linked List...nothing to search");
            return false;
        }
        Employee cur = head;
        while(cur.next != null){
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

}