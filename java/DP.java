import java.util.HashMap;

import javax.lang.model.util.ElementScanner14;

public class DP {
    public static void main(String[] args) {
        DPAlgo dp = new DPAlgo();
        DPAlgo.Fib dpFib = dp.new Fib();
        System.out.println(dpFib.fib(9));
        System.out.println(dpFib.fibMemo(9, new HashMap<Integer, Integer>()));
        System.out.println(dpFib.fibBottomUp(9));
        System.out.println(dpFib.fibYABottomUp(9));


        DPAlgo.CoinChange dpcc = dp.new CoinChange();
        System.out.println(dpcc.recur(27));
        System.out.println(dpcc.dp(new int[]{2,5,7},27));

        DPAlgo.UniquePaths dpup = dp.new UniquePaths();
        System.out.println(dpup.dp(4,5));

        DPAlgo.JumpGame dpjg = dp.new JumpGame();
        System.out.println(dpjg.dp(new int[]{2,3,1,1,4}));
        System.out.println(dpjg.dp(new int[]{3,2,1,0,4}));
    }
}

class DPAlgo{
    /**
     * 适用
     * 	计数
			多少种方式从A到B
			多少种方式选出k个数使得和是sum
		求极值
			从A到B的最大数字和
			最长上升子序列长度
		求存在性
			石子游戏，先手是否必胜
            能不能选出k个数使得和是sum
     * 通用步骤
     *  确定状态
			开数组
			最后一步
			子问题
		转移方程
        初始条件和边界情况
        计算顺序 	
     * 
     */
    class Fib{
        /**
         * 最直观的递归解法，但时间复杂度较高，约为2^n
         * @param n
         * @return 返回n位置的斐波那契数
         */
        int fib(int n){
            if(n <= 2)
                return 1;
            else
                return fib(n-1) + fib(n-2);
        }

        /**
         * 优化技巧：使用memorization优化原来的递归算法，fib(n)只在第一次调用时才会递归， 所以此时的复杂度为n
         * @param n
         * @param memo 存储对应位置上的斐波那契数，如果查到，就无需计算
         * @return
         */
        int fibMemo(int n, HashMap<Integer, Integer> memo){
            int f;
            if(memo.containsKey(n))
                f = memo.get(n);

            if(n <= 2)
                f = 1;
            else
                f = fibMemo(n-1, memo) + fibMemo(n-2, memo);
                memo.put(n, f);
                return f;
        }
        /**
         * 另一种思考方式，由自顶向下转为自底向上，时间复杂度与memorization相同，只是换了一种方式写而已
         * 这种写法可以更直观地看出时间复杂度就是线性，而且可以看出每次计算其实只需要保存之前两次的计算结果即可，由此有下一种优化
         * @param n
         * @return
         */
        int fibBottomUp(int n){
            int[] memo = new int[n+1];
            int f;
            for(int i=1;i < memo.length; i++){
                if(i <= 2)
                    f = 1;
                else 
                    f = memo[i-1] + memo[i-2];
                memo[i] = f;
            }
            return memo[n];
        }

        /**
         * 还是自底而上，但是在保持线性阶段的同时，减少了空间使用
         * @param n
         * @return
         */
        int fibYABottomUp(int n){
            int previous = 1;
            int beforePrevious = 1;
            int f = 2;
            for(int i=3;i < n; i++){
                beforePrevious = previous;
                previous = f;
                f = previous + beforePrevious;
            }
            return f;
        }
    }
    
    class CoinChange{
        /**
         * 有A种硬币，求组成M元的最小硬币个数
         */

        /**
         * 时间复杂度很大，与计算斐波那契数的递归解法类似
         * @param x 要兑换的金额
         * @return 达到兑换金额的最少硬币组合个数
         */
        double recur(int x){
            if(x == 0)
                return 0;
            double res = 1.0 / 0.0; // 无穷大
            if(x >= 2)
                res = Math.min(recur(x-2)+1, res);
            if(x >= 5)
                res = Math.min(recur(x-5)+1, res);
            if(x >= 7)
                res = Math.min(recur(x-7)+1, res);
            return res;
        }

        /**
         * 动态规划问题分为4步
         *  确定状态
         *      f(x) = 组成x金额的最小硬币数
         *      最后一步：无论如何组合，最小组成序列中的最后一枚硬币一定是A中的任意元素，那么就有A.length种可能，最后结果是f(x-a0) + 1....f(x-an) + 1中的最小值
         *      子问题：最小的子问题是凭出0元，最少需要0次，即f[0] = 0
         *  转移方程
         *      min(f[M-a0] + 1, f[M-a1] + 1....,f[M-an] + 1)
         *  初始条件和边界情况
         *      f[0] = 0
         *      拼不出Y，则f[Y]=正无穷
         *  计算顺序
         *      f[0],f[1],f[2]...确保计算f(m),小于m的金额都已经计算
         * @param A 硬币种类的数组
         * @param M 最好要拼出的金额
         * @return 组成M的最小硬币数
         */
        int dp(int[] A, int M){
            // 开数组，存储对应金额的最下硬币组成数
            int[] f = new int[M + 1];
            int n = A.length;
            // 初始条件
            f[0] = 0;

            int i,j;
            for(i = 1; i<= M; i++){
                // 初始化f[i]为无穷大，表示拼不出来
                f[i] = Integer.MAX_VALUE;
                // 状态转移：尝试每一种组合可能
                for(j = 0; j < n; j++){
                    // 保证数组不越界，以及在拼不出来时，不至于出现无穷大加1的情况（在Java里，无穷大加1会变成负数）
                    if(i >= A[j] && f[i - A[j]] != Integer.MAX_VALUE)
                        f[i] = Math.min(f[i - A[j]] + 1, f[i]);
                }
            }

            if(f[M] == Integer.MAX_VALUE)
                f[M] = -1;
            
            return f[M];
        }
    }

    /**
     * 计数型问题
     * 给定m行n列的网格，机器人从左上角(0,0)出发，每次只能朝着下或朝右走一步，问有多少种不同方式走到右下角
     */
    class UniquePaths{
        /**
         * 状态
         * 走到右下角(m-1,n-1)的最后一步要么是从左侧向右走过来的(m-2,n-1)，要么是从上向下走过来的(m-1,n-2)
         * 假设有X种方式走到(m-2,n-1)，有Y种方式走到(m-1,n-2)，二者相加，既无重复，也无遗漏，X+Y就是走到(m-1,n-1)的所有可能路径数
         * 这样问题规模缩小，右下角转化为两个顶点之和
         * 设f[i][j]为机器人有多少种方式从左上角走到(i,j)
         * 转移方程
         * f[i][j] = f[i-1][j] + f[i][j-1]
         * 初始条件和边界情况
         * f[0][0] = 1, i=0或者j=0时只能向着一个方向前进。第一行(i=0)只能向右走，第一列(j=0)只能向下走
         * 计算顺序
         * 计算第0行，f[0][1]...f[0][n-1]
         * 计算第1行，f[1][0]...f[1][n-1]
         * ...
         * f[m-1][m-1]
         * 时间和空间复杂度
         * O(MN)
         * @return
         */
        int dp(int m, int n){
            int[][] f = new int[m][n];
            int i,j;
            for(i=0;i<m;i++){ // 从上到下
                for(j=0;j<n;j++){ // 从左到右
                    if(i==0 || j==0)
                        // 初始化写在了内部
                        f[i][j] = 1;
                    else
                        f[i][j] =f[i-1][j] + f[i][j-1];
                }
            }
            return f[m-1][n-1];
        }
    }

    /**
     * 可行性问题
     * 有n块石头在x轴的0,1,...n-1位置
     * 一只青蛙在石头0，想跳到石头n-1
     * 如果青蛙在第1块石头上，最多可以向右跳ai
     * 问青蛙能否跳到石头n-1
     * 
     */
    class JumpGame{
        /**
         * 确定状态
         * 跳到n-1石头上的最后一步一定是从i跳过来的，i<n-1
         * 青蛙得能跳到i，而且能从i调过来，即最后一步不超过跳跃的最大距离n-1-i <= ai
         * 子问题就变成了解青蛙能不能跳到石头i，f[j]表示青蛙能否跳到石头j
         * 转移方程
         *           f[j]    =    OR 0<=i<j        (f[i]              AND   i + a[i] >= j)
         * 青蛙能不能跳到石头j  枚举上一个跳到的石头i  青蛙能不能跳到石头i      最后一步距离不能超过ai  
         *                    OR表示枚举过程中只要有一个满足条件就是true  AND表示后面这两个条件必须同时满足
         * 初始条件
         * f[0] = true，因为青蛙一开始就是石头0上
         * 计算顺序
         * 从左到右，答案是f[n-1]
         * 时间和空间复杂度
         * O(n**2)   O(N)
         * @return
         */
        boolean dp(int[] A){
            int n = A.length;
            boolean[] f = new boolean[n];
            // 初始化
            f[0] = true;

            int i,j;
            // j表示最后位置，从1开始（即从青蛙起始位置的下一个点开始）
            for(j=1;j<n;j++){
                f[j] = false;
                //  i是上一块石头
                for(i=0;i<j;i++) {
                    // 从上一块石头的最后一跳
                    if(f[i] && i + A[i] >= j){
                        f[j] = true;
                        // 只要能从某一块石头跳过来，就说明没问题，因此这里直接跳出内层循环
                        break;
                    }
                }
            }

            return f[n-1];
        }
    }
}