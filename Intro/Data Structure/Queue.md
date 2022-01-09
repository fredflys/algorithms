```java
import java.util.Scanner;


public class QueueDemo{
    public static void main(String[] args){
        Queue demoq = new Queue(3);
        char key = ' ';
        Scanner scanner = new Scanner(System.in);
        boolean loop = true;
        while(loop){
            System.out.println("s(show): show the queue");
            System.out.println("e(exit): exit the program");
            System.out.println("p(push): push an item into the queue");
            System.out.println("o(pop): pop the rear item of the queue");
            System.out.println("k(peek): get a peek of the front item");
            key = scanner.next().charAt(0);
            switch (key){
                case 's':
                    demoq.show();
                    break;
                case 'p':
                    System.out.println("Please enter a number: ");
                    int item = scanner.nextInt();
                    demoq.push(item);
                    break;
                case 'o':
                    try {
                        int result = demoq.pop();
                        System.out.printf("The item popped out is %d.\n", result);
                    } catch (Exception e){
                        System.out.println(e.getMessage());
                    }
                    break;
                case 'k':
                    try {
                        int result = demoq.peek();
                        System.out.printf("The front item is %d.\n", result);
                    } catch (Exception e){
                        System.out.println(e.getMessage());
                    }
                    break;
                case 'e':
                    scanner.close();
                    loop = false;
                    break;
                default:
                    break;
            }
        }
    }


}

class Queue {
    private int maxSize;
    private int front;
    private int rear;
    private int[] arr;

    public Queue(int size) {
        maxSize = size;
        arr = new int[maxSize];
        front = -1;
        rear = -1;
    }

    public boolean isFull(){
        return rear == maxSize - 1;
    }

    public boolean isEmpty(){
        return rear == front;
    }

    public void push(int item){
        if(isFull()){
            System.out.println("There is no space for a new item because this queue is already full.");
            return;
        }
        rear ++;
        arr[rear] = item;
    }

    public int pop(){
        if(isEmpty()){
            throw new RuntimeException("No item can be popped out because this queue is empty.");
        }
        front++;
        return arr[front];
    }

    public void show(){
        if(isEmpty()){
            System.out.println("No item is to be shown because this queue is empty.");
            return;
        }
        for(int i = 0;i <arr.length;i++){
            System.out.printf("arr[%d] = %d\n", i, arr[i]);
        }
    }

    public int peek(){
        if(isEmpty()){
            throw new RuntimeException("No item is to be peeked because this queue is empty.");
        }
        return arr[front+1];
    }
}
```