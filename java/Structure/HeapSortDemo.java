package Structure;

import java.util.Arrays;

public class HeapSortDemo {
    public static void main(String[] args){
        int arr[] = {4,8,6,5,9};
        heapfifyIter(arr, 0, 5);
        System.out.println(Arrays.toString(arr));
    }

    public static void heapSort(int arr[]){
        buildHeap(arr);
        int length = arr.length;
        for(int i=length-1;i>=0;i--){
            swap(arr, i, 0);
            heapify(arr, 0,i);
        }
    }

    public static void buildHeap(int arr[]){
        int length = arr.length;
        int last = length - 1;
        int parent = (last - 1) / 2;
        for(int i = parent;i>=0;i--){
            heapify(arr, i, length);
        }
    }
    
    public static void heapify(int[] arr, int i, int length){
        if(i > length){
            return;
        }       

        int left = 2*i+1;
        int right = 2*i+2;
        int max = i;
        if(left < length && arr[left] > arr[max]){
            max = left;
        }
        if(right < length && arr[right] > arr[max]){
            max = right;
        }
        if(max != i){
            swap(arr, max, i);
            heapify(arr, max, length);
        }
    }

    public static void swap(int[] arr, int i, int j){
        int temp = arr[i];
        arr[i] = arr[j];    
        arr[j] = temp;
    }


    // non-recursive version
    public static void heapfifyIter(int[] arr, int i, int length){
        int left = i * 2 + 1;
        int temp = arr[i];
        for(int current = left;current < length;current = current * 2 + 1){
            if(current + 1 < length && arr[current+1] > arr[current]){
                current++;
            }
            if(arr[current] > arr[i]){
                arr[i] = arr[current];
                i = current;    
            }else{
                break;          
            }   
        }
        arr[i] = temp;        

    } 

}