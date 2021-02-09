public class SparseArray {
    public static void main(String[] args){
        // 创建二维数组
        int chessBoard[][] = new int[11][11];
        chessBoard[1][2] = 1;
        chessBoard[2][3] = 2;

        int rows = chessBoard.length;
        int columns = chessBoard[0].length;
        int sum = 0;
        // 打印棋盘，并统计有多少非0值，在创建稀疏数组时会用到
        for(int[] row: chessBoard){
            for(int column: row){
                System.out.printf("%d\t", column);
                // System.out.println(column);
                if(column != 0){
                    sum += 1;        
                }
            }
            System.out.println();
        }


        // 从二维数组转稀疏数组
        // 根据shape创建稀疏数组
        int sparseArray[][] = new int[sum+1][3];
        sparseArray[0][0] = rows;
        sparseArray[0][1] = columns;
        sparseArray[0][2] = sum;
        // 第0行已经存储了二维数组的行、列、非0值个数，因此从第二行开始放入数据
        int count = 1;
        for(int i = 0;i < rows; i++){
            for(int j = 0;j < columns; j++){
                if (chessBoard[i][j] != 0){
                    sparseArray[count][0] = i;
                    sparseArray[count][1] = j;
                    sparseArray[count][2] = chessBoard[i][j];
                    count += 1;
                }
            }
        }
        
        System.out.println();
        for(int i =0; i < sparseArray.length;i++){
            System.out.printf("%d\t%d\t%d\t\n", sparseArray[i][0], sparseArray[i][1], sparseArray[i][2]);
        }
    
        // 将稀疏数组转回二维数组
        int originalArray[][] = new int[sparseArray[0][0]][sparseArray[0][1]];
        // 从第二行开始迭代
        for(int i = 1;i < sparseArray.length;i++){
            int[] currentRow = sparseArray[i];
            originalArray[currentRow[0]][currentRow[1]] = currentRow[2];
        }

        System.out.println(originalArray[1][0]);


    }

}