#### Sparse Array

```java
public class SparseArray {
    public static void main(String[] args){
        // ������ά����
        int chessBoard[][] = new int[11][11];
        chessBoard[1][2] = 1;
        chessBoard[2][3] = 2;

        int rows = chessBoard.length;
        int columns = chessBoard[0].length;
        int sum = 0;
        // ��ӡ���̣���ͳ���ж��ٷ�0ֵ���ڴ���ϡ������ʱ���õ�
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


        // �Ӷ�ά����תϡ������
        // ����shape����ϡ������
        int sparseArray[][] = new int[sum+1][3];
        sparseArray[0][0] = rows;
        sparseArray[0][1] = columns;
        sparseArray[0][2] = sum;
        // ��0���Ѿ��洢�˶�ά������С��С���0ֵ��������˴ӵڶ��п�ʼ��������
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
    
        // ��ϡ������ת�ض�ά����
        int originalArray[][] = new int[sparseArray[0][0]][sparseArray[0][1]];
        // �ӵڶ��п�ʼ����
        for(int i = 1;i < sparseArray.length;i++){
            int[] currentRow = sparseArray[i];
            originalArray[currentRow[0]][currentRow[1]] = currentRow[2];
        }

        System.out.println(originalArray[1][0]);
    }
}
```