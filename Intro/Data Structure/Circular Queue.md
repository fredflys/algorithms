```java
import java.util.Scanner;

public class CircularQueueDemo
{
    public static void main(String[] args){
        CircularQueue demoq = new CircularQueue(3);
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

class CircularQueue {
    private int maxSize;
    // use count to record how many slots are occupied
    private int count;
    // front points to the first element
    private int front;
    // rear points to the next slot to the last element
    private int rear;
    private int[] arr;

    public CircularQueue(int size) {
        // leave one slot for rear pointer
        /*
        when size == 3
        f     r
        0 1 2 3
        a b c -
        */
        maxSize = size + 1;
        arr = new int[maxSize];
        // front = 0;
        // rear = 0;
    }

    public boolean isFull(){
        return (rear + 1) % maxSize == front;
    }

    public boolean isEmpty(){
        return rear == front;
    }

    public void push(int item){
        if(isFull()){
            System.out.println("There is no space for a new item because this queue is already full.");
            return;
        }
        arr[rear] = item;
        rear++;
        rear = rear % maxSize;
        count++;
    }

    public int pop(){
        if(isEmpty()){
            throw new RuntimeException("No item can be popped out because this queue is empty.");
        }
        int frontVal = arr[front];
        front++;
        front = front % maxSize;
        count--;
        return frontVal;
    }

    public void show(){
        if(isEmpty()){
            System.out.println("No item is to be shown because this queue is empty.");
            return;
        }
        System.out.printf("front: %d\nrear: %d---\n", front, rear);
        // 这里卡了好久，i小于的不是count
        for(int i = front;i < (front+count);i++){
            System.out.printf("arr[%d] = %d\n", i % maxSize, arr[i % maxSize]);
        }
    }


    public int peek(){
        if(isEmpty()){
            throw new RuntimeException("No item is to be peeked because this queue is empty.");
        }
        return arr[front];
    }
}
```