import java.util.Arrays;

public class Search {
    public static void main(String[] args) {
        int[] arr = {0,1,1,3,4,5,6,7,5,9};
        System.out.println(binaryRecur(arr, 0, arr.length-1, 5));
        System.out.println(fibSearch(arr, 5));
        System.out.println(anotherFibSearch(arr, 5));
    }

    public static int sequential(int[] arr, int val){
        int res=-1;
        for(int cur=0;cur<arr.length;cur++){
            if(arr[cur] == val){
                res = cur;
            }
        }
        return res;
    }

    // 二分搜索非递归版本
    public static int binary(int[] arr,  int val){
        /*
        因为回过头去放慢镜头，二分查找的过程就是一个 维护 low 的过程： 
        low从0起始。只在中位数遇到确定小于目标数时才前进，并且永不后退。
        low一直在朝着第一个目标数的位置在逼近。知道最终到达。
        */
        int low = 0, high = arr.length-1;
        while(low<=high){
            int mid = low + (high - low) / 2;
            if(val < arr[mid]) high = mid -1;
            if(val > arr[mid]) low = mid + 1;
            if(val == arr[mid]) return mid;
        }
        return low;
    }

    // 二分搜索-递归
    public static int binaryRecur(int[] orderedArr, int low, int high, int val){
        if(low > high){
            return -1; 
        }
 
        // (low+high)/2 = (low+low+high-low)2=low+(high-low)/2
        // 避免low + high的结果超过Integer的最大值
        int mid = low + (high - low) / 2; 
        if(val < orderedArr[mid]){
            return binaryRecur(orderedArr, low, mid-1,val);
        }else if (val > orderedArr[mid]){
            return binaryRecur(orderedArr, mid+1, high, val);
        }else{
            return mid;
        }
    }


    public static int interpolation(int[] arr, int low, int high, int val){
        if(low > high || val < arr[low] || val > arr[high]){
            return -1;
        }

        int mid =  low + (high - low) * (val - arr[low]) / (arr[high] - arr[low]);
        if(val > arr[mid]){
            return interpolation(arr, mid + 1, high, val);
        }else if(val < arr[mid]){
            return interpolation(arr, low, mid - 1, val);
        }else{
            return mid;
        }

    }


    // 生成第个n位置上的斐波那契数
    public static int fib(int n){
        int pre = 1;
        int post = 1;
        int res = 1;
        for(int i =3;i<=n;i++){
            res = pre + post;
            pre = post;
            post = res;
        }
        return res;
    }

    public static int fibSearch(int[] arr, int val){
        int low = 0;
        int high = arr.length-1;
        int n = 0;
        int mid;
        /*
              l                                       h
        index 0  1  2  3  4  5  6  7  8   9   10  11  12 
        arr   10 22 30 44 56 58 60 70 100 110 130 0   0
              -  -  -  -  -  -  -  -              130 130              
                                   m
                           n
        index  1 2 3 4 5 6 7  8  9  10 11
        fibarr 1 1 2 3 5 8 13 21 34 55 89
        */
        while(fib(n) < arr.length){
            n++;
        }
        // 扩展原数组，长度为当前斐波那契数，好在接下来按照fib(n-1)和fib(n-2)进行分组
        int[] temp = Arrays.copyOf(arr, fib(n));
        // 将后面添加的元素都置为原数组的末尾值，否则就不符合有序数组这一要求了（扩展的元素默认都为0）。0
        for(int i = arr.length;i<temp.length;i++){
            temp[i] = arr[high];
        }
        while(low<=high){
            mid = low + fib(n-1) - 1;
            if(val < temp[mid]){
                high = mid - 1;
                n--;
            }else if (val > temp[mid]){
                low = mid + 1;
                n -= 2;
            }else{
                if(mid <= high){
                    return mid;
                } else {
                    return high;
                }
            }
        }
        // if not found
        return -1;
    }

    public static int anotherFibSearch(int[] arr, int val){
        int n=0;
        int offset = -1;
        int mid=0;
        while(fib(n) < arr.length){
            n++;
        }
        while( fib(n) > 1){
            mid = offset + fib(n-2);
            if( mid > arr.length-1){
                mid = arr.length -1;
            }
            if( val > arr[mid]){
                n -= 1;
                offset = mid;
            }else if( val < arr[mid]){
                n -= 2;
            }else{
                return mid;
            }
        }

        if(arr[offset+1] == val){
            return offset+1;
        }

        return -1;

    }

}


