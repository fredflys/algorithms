#### N Queens Puzzle

```java
import java.util.List;
import java.util.ArrayList;

public class nQueensPuzzle {
    public static void main(String[] args) {
        nQueens puzzle = new nQueens(4);
        // public static List<int[]> result;
        // puzzle.place(0);
        for(int[] placements:puzzle.result){
            for(int i:placements){
                System.out.printf("%d ", i);
            }
            System.out.println();
        }

        System.out.println("-----------");
        nQueensYetAnother puzzleYA = new nQueensYetAnother(4);
        for(List<Integer> placements: puzzleYA.result){
                System.out.println(placements);
            }
        
    }

}

class nQueens{
    /**
     * 在nxn的棋盘上，放上n个皇后，且使其互不攻击（相互不在一行、一列、一对角线上），问有几种摆法？
     */
    public int queens;
    public List<int[]> result;
    // 对应一种可能的放置方式，其中索引表示行，值表示列，如[1,3,0,2]就表示第0个皇后放在第0行第1列，以此类推
    public int[] colPlacements;

    public nQueens(int queens){
        this.queens = queens;
        this.colPlacements = new int[queens];
        this.result = new ArrayList<int[]>();
        //
        this.place(0);
    }

    // 在第n行放入第n个皇后
    public void place(int n){
        // 是否最后一个皇后已经摆完，是则得到一种解法，入结果列表
        if(n == queens){
            // colPlacements会不断被复用，如果将其添加到result中，由于下面的for循环，那结果只会是queesn个queens-1元素
            // 须将当前colPlacements复制一份，放入result中
            int[] copy = colPlacements.clone();
            result.add(copy);
            return;
        }
        // 尝试该行中的每一个位置（列）
        for(int i=0;i<queens;i++){
            colPlacements[n] = i;
            if(isValid(n)){
                // 如果合适，就开始摆下一行。直到摆到最后一行为止
                place(n+1);
            }
        }
    }

    public boolean isValid(int n){
        // 要与每一个已经摆放过的皇后都比较一遍，方能确定是否合适
        for(int i=0;i<n;i++){
            // 第n个皇就在第n行，因此不必考虑是否在一行，先判断是否在同一列
            if(colPlacements[i] == colPlacements[n] 
            /* 如两皇后在对角线上，依此对角线可作一直角等腰三角形，其底和高相等，即二皇后所在行之相差等于列之相差
            . . c .
            . . . .
            a . . . 如左图a、c两个皇后在对角线上，二者行差为2，高差为2
            */
               || Math.abs(n-i) == Math.abs(colPlacements[n] - colPlacements[i])){
                return false;
            }
        }
        return true;
    }
}



class nQueensYetAnother {
    public List<List<Integer>> result; 

    public nQueensYetAnother(int n) {
        result = nQueens(n);        
    }

    public static List<List<Integer>> nQueens(int n){
        List<List<Integer>> result = new ArrayList<>();
        solveNQueens(n, 0, new ArrayList<Integer>(), result);
        return result;
    }

    public static void solveNQueens(int n, int row, List<Integer> colPlacements, List<List<Integer>> result){
        // goal 是否已经放了最后一个皇后 
        if(row == n){
            result.add(new ArrayList<>(colPlacements));
        }else{
            for(int col=0;col<n;col++){
                // choice - choose a placement
                colPlacements.add(col);
                // constraint - check if it's valid
                if(isValid(colPlacements)){
                    solveNQueens(n, row+1, colPlacements, result);
                }
                // 这步有些不太理解
                colPlacements.remove(colPlacements.size()-1);
            }
        }
    }

    public static boolean isValid(List<Integer> colPlacements){
        int rowId = colPlacements.size()-1;
        for(int i=0;i<rowId;i++){
            int diff = Math.abs(colPlacements.get(i) - colPlacements.get(rowId));
            if(diff == 0 || diff == rowId - i){
                return false;
            }
        }
        return true;
    }
}
```
