```java
import java.util.HashMap;

import java.util.Arrays;

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

        DPAlgo.HouseRobber dphr = dp.new HouseRobber();
        System.out.println(dphr.dp(new int[]{2,7,9,3,1}));
        System.out.println(dphr.dpCircle(new int[]{1,2,3,1}));

        DPAlgo.MaxProfit dpmp = dp.new MaxProfit();
        System.out.println(dpmp.dp(new int[]{3,3,5,0,0,3,1,4}));
        System.out.println(dpmp.dp(new int[]{1,2,3,4,5}));
        System.out.println(dpmp.dpCooldown(new int[]{1,2,3,0,2})); // 3

        DPAlgo.Knapsack dpks = dp.new Knapsack();
        System.out.println(dpks.zeroone(new int []{2,3,4,5,9}, new int[]{3,4,5,8,10}, 20));
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

    class HouseRobber{
        /**
         * 时间序列
         * leetcode 198
         * 你是一个职业盗贼，锁定了一条大街准备今晚作案，街上每栋房子里都有固定量的财务。
         * 但是相邻的房子之间有报警器，一旦两个相邻的房子在同一晚被盗，就会触发报警器。
         * 现已知一系列非负整数表示每个房子里的财务数目，请计算在不触发报警器的情况下，你今晚可以盗取的最大值。
         * e.g: [2,7,9,3,1] => 12
         * 思路：
         * 无论最终结果如何，最后一栋房子要么抢，要么不抢，只有两种可能。且是求最大值，结果就是这两种可能中的较大者。
         * 以f[i][0]表示第i-th个房子不抢劫的最大收益，f[i][1]表示第i-th个房子抢劫时的最大收益
         * 则结果第i个房子的最大收益就是max(f[i][0], f[i][1])
         * 其中：
         *      f[i][1] = f[i-1][0] + A[i] 第i个房子抢了，前一个就不能抢了，结果就是前一个房子不抢时的最大收益加上抢劫当前房子的收益
         *      f[i][0] = max(f[i-1][0], f[i-1][1]) 第i个房子既然不抢了，那前一个房子就可抢可不抢了，收益就是这两种可能中较大的那个
         * 初始值有两个，一个是第一个房子不抢的收益（0），另一个是第一个房子抢劫时的收益（A[0]）
         */
        int dp(int[] A){
            int n = A.length;
            int[][] f = new int[n][2];
            
            f[0][0] = 0;
            f[0][1] = A[0];

            int i;
            for(i=1;i<n;i++){
                f[i][0] = Math.max(f[i-1][0],f[i-1][1]);
                f[i][1] = f[i-1][0] + A[i];
            }

            int res = Math.max(f[n-1][0], f[n-1][1]);
            return res;
        }

        /**
         * 变体
         * leetcode 213
         * 给一圈首尾相连的房子，还是相邻的房子不能抢，问抢劫的最大收益。
         * 首尾既然相连，则首尾也遵循条件，即首尾也至少有一个不能抢
         * 此时有两种情况
         *      第一个房子不抢，那么对于A[1]->A[n-1]就是一个独立的House Robbery问题
         *      最后一个房子不抢，那么对于A[0]->A[n-2]就是一个独立的House Robbery问题
         * 取二者中的较大者即可。当然，也有可能前一个和最后一个都不抢，但已经包含在上面两种方式中了
         */
        int dpCircle(int[] A){
            int n = A.length;
            if(n == 0) return 0;
            else if(n == 1) return A[0];

            int[] firstNotRobbed = new int[n-1];
            int[] lastNotRobbed = new int[n-1];
            
            firstNotRobbed = Arrays.copyOfRange(A, 1, n);
            lastNotRobbed = Arrays.copyOfRange(A, 0, n-1);
            
            return Math.max(dp(firstNotRobbed), dp(lastNotRobbed));
        }
    }

    class MaxProfit{
        /**
         * leetcode 123
         * 给定一个数组，它的第 i 个元素是一支给定的股票在第 i 天的价格。
        设计一个算法来计算你所能获取的最大利润。你最多可以完成 两笔 交易。
        注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
        e.g. [3,3,5,0,0,3,1,4] => 6

        我们要关注的只是每一天可能的状态。一天的只可能有5种状态，用f[i][j]表示第i天第j种状态的最大收益：
            0.什么也没买（恒为0）这种情况的收益恒为0，所以也可以完全忽略
                dp[i][0] = d[i-1][0] // 当天与前一天收益一致
            1.有第1次买的股票（第一次买过）
                -A[i]      // 要么是当天买的
                dp[i-1][1] // 要么是延续以前已经买过的状态 
            2.没有第1次买的股票（买卖了一次）
                dp[i-1][1] + A[i] // 要么是今天卖的，收益就是以前第1次有股票的收益加上卖出的收益
                dp[i-1][2]  // 要么是延续以前已经卖过1次的状态
            3.有第2次买的股票（第二次买过）
                dp[i-1][2] - A[i] // 要么是今天买的，收益就是以前第1次卖出股票的收益加上第2次买股票的支出
                dp[i-1][3] // 要么是延续前一天已经第2次购买的状态
            4.没有第2次买的股票（买卖了两次）
                dp[i-1][3/ + A[i] // 要么是今天卖的，收益是以前第2次持有的收益加上第2次卖出的收益
                dp[i-1][4] // 要么是延续前一天已经第2次卖出的状态
        初始化
            0.没有操作，收益自然是0
            1.买入要花钱，收益是A[0]
            2.卖出操作一定是为了收获利润，最差就是多少钱买入多少钱卖出，即收益为0，小于0的就没必要考虑这种情况了，因为肯定比全程什么都不做要差劲
            3.不管是第几次，买入就要花钱，收益是A[0]
            4.同2
         */
        int dp(int[] A){
            // int n = A.length;
            // int[][] f = new int[n][5];

            // f[0][0] = 0;
            // f[0][1] = -A[0];
            // f[0][3] = -A[0];

            // int i;
            // for(i=1;i<n;i++){
            //     f[i][0] = f[i-1][0];
            //     f[i][1] = Math.max(f[i-1][0] - A[i], f[i-1][1]);
            //     f[i][2] = Math.max(f[i-1][1] + A[i], f[i-1][2]);
            //     f[i][3] = Math.max(f[i-1][2] - A[i], f[i-1][3]);
            //     f[i][4] = Math.max(f[i-1][3] + A[i], f[i-1][4]);
            // }
            
            // // 有股票而没有卖出肯定收益不会是最高的，所以最后的结果不用考虑第1次有股票和第2次有股票的情况
            // int res = Math.max(f[n-1][2], f[n-1][4]);
            // return res;

            int n = A.length;
            int[][] f = new int[n][4];

            f[0][0] = -A[0];
            f[0][2] = -A[0];

            int i;
            for(i=1;i<n;i++){
                f[i][0] = Math.max(- A[i], f[i-1][0]);
                f[i][1] = Math.max(f[i-1][0] + A[i], f[i-1][1]);
                f[i][2] = Math.max(f[i-1][1] - A[i], f[i-1][2]);
                f[i][3] = Math.max(f[i-1][2] + A[i], f[i-1][3]);
            }
            
            // 有股票而没有卖出肯定收益不会是最高的，所以最后的结果不用考虑第1次有股票和第2次有股票的情况
            int res = Math.max(f[n-1][1], f[n-1][3]);
            return res;
        }

        /**
         * leetcode 309 
        给定一个整数数组，其中第 i 个元素代表了第 i 天的股票价格 。​
        设计一个算法计算出最大利润。在满足以下约束条件下，你可以尽可能地完成更多的交易（多次买卖一支股票）:
        你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
        卖出股票后，你无法在第二天买入股票 (即冷冻期为 1 天)。
         *      分析状态而不是动作
         *      0.持有
         *          0->0 昨天就持有，今天继续持有：f[i-1][0]
         *          2->0 昨天不是冷静期，今天买入：f[i-1][2] - A[i]
         *      没持有
         *          1.处于冷冻期，所以未持有
         *              昨天持股，今天卖出：f[i-1][0] + A[i]
         *          2.非冷冻期
         *              昨天就是非冷冻期，延续昨天状态，今天还是没买：f[i-1][2]
         *              昨天是冷冻期：f[i-1][1]
         *              
         *  初始化
         *      第0天要么买，要么不买，持股状态下就是-A[0]，不持股就是0
         */
        int dpCooldown(int A[]){
            int n = A.length;
            if(n<=1) return 0;

            int[][] f = new int[n][3];

            f[0][0] = -A[0];
            f[0][1] = 0;
            f[0][2] = 0;

            int i;
            for(i=1;i<n;i++){
                f[i][0] = Math.max(f[i-1][0], f[i-1][2] - A[i]);
                f[i][1] = f[i-1][0] + A[i];
                f[i][2] = Math.max(f[i-1][1], f[i-1][2]);
            }

            // 最后一天还持有，肯定不是最大收益，不必关注
            // 我常常把这里的n写成i，对照半天才能发现错误，实在是太粗心了
            int res = Math.max(f[n-1][1], f[n-1][2]);
            return res;
        }
    }

    class WiggleSequence{
        /**
         * 如果连续数字之间的差严格地在正数和负数之间交替，则数字序列称为摆动序列。第一个差（如果存在的话）可能是正数或负数。少于两个元素的序列也是摆动序列。
            例如， [1,7,4,9,2,5] 是一个摆动序列，因为差值 (6,-3,5,-7,3) 是正负交替出现的。相反, [1,4,7,2,5] 和 [1,7,4,5,5] 不是摆动序列，第一个序列是因为它的前两个差值都是正数，第二个序列是因为它的最后一个差值为零。
            给定一个整数序列，返回作为摆动序列的最长子序列的长度。 通过从原始序列中删除一些（也可以不删除）元素来获得子序列，剩下的元素保持其原始顺序。
            
            0.以当前元素结尾上升（当前元素要小于前一个元素）A[i] < A[i-1]
                前一个元素结尾下降再加上1 f[i-1][0] + 1
            以当前元素结尾下降（当前元素要大于前一个元素）
                前一个元素结尾上升再加上1
         */
        int dq(int[] A){
            int n = A.length;
            return 0;
        }
    }

    class Knapsack{
        /**
         * 给定一组多个（[公式]）物品，每种物品都有自己的重量（[公式]）和价值（[公式]），
         * 在限定的总重量/总容量（[公式]）内，选择其中若干个（也即每种物品可以选0个或1个），
         * 设计选择方案使得物品的总价值最高。
         * @param weights 物品重量数组
         * @param prices 物品价格数组
         * @param capacity 背包实际承重
         * @return 包内物品的最大价值
         */
        int zeroone(int[] weights, int[] prices, int capacity){
            // f[i][j]表示前i个商品放在负重为j的背包的最大价值
            // 最后要返回的是f[weight.length][capacity]，所以开数组时都要加1
            // f[0][j] 表示背包负重为j时，没有商品可放的最大价值，结果为0
            // f[i][0] 表示背包负重为0时，放入前i件物品的最大价值，结果为0
            // 可以看出这里的i，j都是对应实际上是第几件商品，多大的重量，因此用到weights时索引要减去1才能找到对应的值
            int[][] f = new int[weights.length+1][capacity+1];

            // 延伸部分：定义一个数组来存储背包内物品达到最大价值时的物品装入情况
            int[][] how = new int[weights.length+1][capacity+1];

            // 第k件商品，当前背包容量c 
            int k,c;
            // k最大值是5
            for(k=1;k<weights.length+1;k++){
                for(c=1;c<capacity+1;c++){
                    // 如果第k件商品的重量大于当前背包重量，那k就放不进去，看前k-1商品放在c容量的背包里的最大价值
                    // k从1开始，但要从weights的第0个（第1件商品）开始取
                    if(weights[k-1] > c)
                        f[k][c] = f[k-1][c];
                    else{
                    // 如果放得进去，则可放可不放，取其中较大者
                    // 涉及到weights和prices数组的地方都要k都要减去1
                        int bagged = f[k-1][c-weights[k-1]] + prices[k-1];
                        int otherwise = f[k-1][c];
                        if(bagged > otherwise){
                            f[k][c] = bagged; 
                            how[k][c] = 1;
                         }
                         else
                            f[k][c] = otherwise;
                        // f[k][c] = Math.max(f[k-1][c], f[k-1][c-weights[k-1]] + prices[k-1]);
                    }
                }
            }

            // for(int i=0;i<weights.length+1;i++){
            //     for(int j=0;j<capacity+1;j++){
            //         System.out.print(f[i][j] + " ");
            //     }
            //     System.out.println();
            // }
            int j = capacity;
            int i = weights.length;
            while(i>0 && j>0){
                if(how[i][j] == 1){
                    System.out.printf("Item %d is added.\n", i);
                    j -= weights[i-1];
                }
                i--;
            }

            return f[weights.length][capacity];
        }
    }
}
```