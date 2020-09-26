import java.util.Arrays;

public class Sorting {
    public static void main(String[] args) throws Exception{
        int[] arr = {8,9,1, 7, 2, 3, 5, 4, 6, 0};
        // int[] res = bubble(arr);
        // System.out.printf("Sorted: %s\n", Arrays.toString(res));
        // res = selection(arr);
        // System.out.printf("Sorted: %s\n", Arrays.toString(res));
        // res = insertion(arr);
        // System.out.printf("Sorted: %s\n", Arrays.toString(res));
        // int[] shellArr = {8,9,1, 7, 2, 3, 5, 4, 6, 0};
        // res = shellBySwap(shellArr);
        // System.out.printf("Sorted: %s\n", Arrays.toString(res));
        // res = shellByInsertion(arr);
        // System.out.printf("Sorted: %s\n", Arrays.toString(res));
        // quickHelper(arr);
        // System.out.println(Arrays.toString(arr));
        // int[] res = mergeHelper(arr);
        // System.out.println(Arrays.toString(res));


        // testPerf();
    }

    public static void testPerf() throws Exception{
                // 生成测试数据
        int[] testArr = new int[100000];
        for(int i=0;i<testArr.length;i++){
            testArr[i] = (int)(Math.random() * 100000);
        }
        
        test("bubble", testArr);
        test("selection", testArr); 
        test("insertion", testArr);
        test("shellBySwap", testArr);      
        test("shellByInsertion", testArr); 
        test("quickHelper", testArr);
        test("mergeHelper", testArr);
        
    }

    public static void test(String funcName, int[] testArr) throws Exception{
        long start = System.currentTimeMillis();
        // 用反射调用方法
        Sorting.class.getMethod(funcName, int[].class).invoke(funcName, testArr);
        long end = System.currentTimeMillis();
        System.out.printf("Executing " + funcName + " sorting on a %d-item array takes %d" + " ms......\n", testArr.length, end - start);
    }




    // 冒泡排序 O(n**2)
    public static int[] bubble(int[] originalArr){
        // retain arr for next sorting
        int[] arr = originalArr.clone();
    
        int tempForSwap = 0;
        boolean swapped = false;
        /*
         value  2,3,1,5,7,4,6,8,10,9
         index  0 1 2 3 4 5 6 7  8 9
                  .              . 
         指针到length - 2位置的元素，length-1位置的元素即已归位
         内循环终止位置到1位置的元素，0位置的元素即已归位
        */
        for(int left=0;left<arr.length-2;left++){
            for(int cur=0;cur<arr.length-1-left;cur++){
                if(arr[cur] > arr[cur+1]){
                    swapped = true;
                    tempForSwap = arr[cur];
                    arr[cur] = arr[cur+1];
                    arr[cur+1] = tempForSwap;
                }
            }
    
            // 优化：如果上一轮内循环中，没有发生一次交换，则所有元素已然归位，可以提前结束
            if(!swapped){
                break;
            }else{
                swapped = false;
            }
        }
        return arr;
    }

    public static int[] selection(int[] originalArr){
        int[] arr = originalArr.clone();
        
        /*
        选择排序即从数组中选择出极值归位，再抛开极值，再从剩下的元素中选出极值归位，如此循环，直到最后一个元素归位，排序完成
        在start ~ n-1的位置区间内，将最小的值放在第start个位置，start从0开始
        */
        // int minIndex = 0;
        for(int start=0;start<arr.length;start++){
            // 每次开始当前轮次排序时，都假定起始位置即是最小值的位置
            int minIndex = start;
            for(int i=start+1;i<arr.length;i++){
                // 如果要逆序排列，只需要改<为>即可
                if(arr[minIndex] > arr[i]){
                    minIndex = i;
                }
            }
            if(start != minIndex){
                int min = arr[minIndex];
                arr[minIndex] = arr[start];
                arr[start] = min;
            }
            
        }
        return arr;
    }

    public static int[] insertion(int[] originalArr){
        int[] arr = originalArr.clone();
    
        /*
        选择排序的思想是视位置0的元素为有序数组，依次从右侧取元素，插入到该有序数组中，使其不断向右扩展，最终使得整个数组按顺序排列
        */
        for(int insertIndex=1;insertIndex<arr.length;insertIndex++){
            int insertVal = arr[insertIndex];
            int targetIndex = insertIndex - 1;
            /*
            while
            3 5 7  4 2 6  t--
                t  i
            3 5 7  7 2 6  t--
              t    i 
            3 5 5  7 2 6  t--
            t      i 

            3 4 5  7 2 6  t+1
              t
            */
            while(targetIndex >= 0 && insertVal < arr[targetIndex]){
                arr[targetIndex+1] = arr[targetIndex];  
                targetIndex--;
            }
            arr[targetIndex+1] = insertVal;
        }

        return arr;
    }

    public static int[] shellBySwap(int[] originalArr){

        int[] arr = originalArr.clone();
        // index 0 1 2 3 4 5 6 7 8 9
        //       8 9 1 7 2 3 5 4 6 0
        //       .         .          length/2 一组
        //       3 5 1 6 0 8 9 4 7 2
        //       . * . * . * . * . *  lenth/2/2 一组
        //       0 2 1 4 3 5 7 6 9 8
        //                            length/2/2/2 一组
 
        int tempForSwap;

        int length = arr.length;
        for(int stride = length/2;stride>0;stride/=2){
            for(int rightCur=stride; rightCur<length; rightCur++){
                for(int leftCur=rightCur-stride; leftCur>=0; leftCur-=stride){
                    if(arr[leftCur]>arr[leftCur+stride]){
                        int nextInGroup = leftCur+stride;
                        tempForSwap = arr[leftCur];
                        arr[leftCur] = arr[nextInGroup];
                        arr[nextInGroup] = tempForSwap;
                    }
                }

            }
        }
        return arr;
    }

    public static int[] shellByInsertion(int[] originalArr){
        // 希尔排序是对插入排序的改进

        int[] arr = originalArr.clone();
        
        int length = arr.length;
        for(int stride = length/2;stride>0;stride/=2){
            for(int cur=stride;cur<length;cur++){
                /*
                        0 1 2 3 4 5 6 7 8 9 
                        3 5 1 6 0 8 9 4 7 2
                       . * . * . * . * . *  lenth/2/2 一组
                       stride=2
                        cur=2
                         arr[2]<arr[0]     
                          // 假设就将原值放在原处，看成与不成
                          index=2
                          val=1
                           3 1 -> 3 3 -> 1 3
                        cur=3
                         arr[3]>arr[1]
                        cur=4
                         arr[4]<arr[2]
                          index=4
                          val=0
                           1 3 0 -> 1 3 3 -> 1 1 3 -> 0 1 3
                */
                if(arr[cur] < arr[cur - stride]){
                    int tartgetIndex = cur;
                    int insertVal = arr[cur];
    
                    while(tartgetIndex - stride >=0 && insertVal < arr[tartgetIndex - stride]){
                        arr[tartgetIndex] = arr[tartgetIndex - stride];
                        tartgetIndex -= stride;
                    }
                    arr[tartgetIndex] = insertVal;
                }
            }
        }
        return arr;
    }

    // 快速排序
    public static int[] quickHelper(int[] originalArr) {
        int[] arr = originalArr.clone();

        quick(arr, 0, arr.length-1);

        return arr;
    }

    public static void quick(int[] arr, int low, int high){
        int pivot;
        if(high>low){
            pivot = partition(arr, low, high);
            quick(arr, low, pivot-1);
            quick(arr, pivot+1, high);
        }
    }

    public static int partition(int[] arr, int low, int high){
        // 对冒泡排序的改进
        /*
        0 1 2 3 4 5 6 7 8 9
        wp                h
        3 2 1 5 7 4 6 8 9 0
        l                 r    
            l           r
            0           5	 
            r l 	    
        0	  3 
        */
        int pivotVal = arr[low];
        int left = low;
        int right = high;
        int tempForSwap;

        while(left<right){
            // 注意左右游标移动过程中不要越界
            while(arr[left] <= pivotVal && left<high){
                left++;
            }
            while(arr[right] > pivotVal && right>low){
                right--;
            }
            if(left<right){
                tempForSwap = arr[left];
                arr[left] = arr[right];
                arr[right] = tempForSwap;
            }
        }

        arr[low] = arr[right];
        arr[right] = pivotVal;

        return right;
    }

    // 归并排序
    public static int[] mergeHelper(int[] originalArr){
        int[] arr = originalArr.clone();

        int[] tempArr = new int[arr.length];
        divideForMerge(arr, 0, arr.length-1, tempArr);
        return arr;
    }

    public static void divideForMerge(int[] arr, int start, int end, int[] tempArr){
        // 分而治之
        if(start < end){
            int  mid = (start + end) / 2;
            divideForMerge(arr, start, mid, tempArr);
            divideForMerge(arr, mid + 1, end, tempArr);
            merge(arr, start, mid, end, tempArr);
        }
    }

    public static void merge(int[] arr, int start, int mid, int end, int[] tempArr){
        /*
        s     m            e         t
        4 5 7 8      1 2 3 6         
        l            r 
        --a
        l            . r             1 
        l              . r           1 2
        l                . r         1 2 3
        . l                r         1 2 3 4 5
          . l                r       1 2 3 4 5 6
        --b
            * 1              r       1 2 3 4 5 6 7 
              * 1            r       1 2 3 4 5 6 7 8

        */
        // merge次数: arr.length-1
        int leftCur = start;
        int rightCur = mid + 1;
        int tempCur = 0;

        // a 依次从左有序数组和右有序数组向临时数组转移 
        while(leftCur <= mid && rightCur <= end){
            if(arr[leftCur] <= arr[rightCur]){
                tempArr[tempCur++] = arr[leftCur++];
            }else{
                tempArr[tempCur++] = arr[rightCur++];
            }
        }

        // b 把上次转移剩下的一股脑转移到临时数组
        while(leftCur<=mid){
            tempArr[tempCur++] = arr[leftCur++];
        }
        while(rightCur<=end){
            tempArr[tempCur++] = arr[rightCur++];
        }

        // 从临时数组往原数组转移
        tempCur = 0;
        int arrCur = start;
        while(arrCur <= end){
            arr[arrCur++] = tempArr[tempCur++]; 
        }


    }




}






