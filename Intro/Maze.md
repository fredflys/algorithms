```java
public class MazeDemo {
    public static void main(String[] args) {
        Maze maze = new Maze(8,7);
        maze.show();
        maze.findPath(1,1);
        maze.show();
    }
}

class Maze{
    public int row;
    public int col;
    public int[][] map;

    public Maze(int row, int col){
        this.row = row;
        this.col = col;
        this.map = new int[row][col];

        // build walls
        for(int i=0;i<col;i++){
            map[0][i] = 1;
            map[row-1][i] = 1;
        }
        for(int j=0;j<row;j++){
            map[j][0] = 1;
            map[j][col-1] = 1;
        }

        // build blocks
        this.map[3][1] = 1;
        this.map[3][2] = 1;
    }

    public void show(){
        System.out.println("The maze is as below: ");
        for(int r=0;r<row;r++){
            for(int c=0;c<col;c++){
                System.out.print(map[r][c] + " "); 
            }
            System.out.println();
        }
    }

    // set exit at map[row-2][col-2]
    // 0 means untravelled, 1 means blocked, 2 means path, 3 means travelled yet leading nowhere
    public boolean findPath(int startRow, int startCol){
        // boolean means whether to try out 

        // if already at the exit
        if(map[row-2][col-2] == 2){
            return true;
        }else{
            // still untravelled
            if(map[startRow][startCol] == 0){
                // suppose it's the path
                map[startRow][startCol] = 2;
                // try moving down
                if(findPath(startRow+1, startCol)){
                    return true;
                // try moving right
                } else if(findPath(startRow, startCol+1)){
                    return true;
                // try moving up
                } else if(findPath(startRow-1, startCol)){
                    return true;
                // try moving left
                } else if(findPath(startRow, startCol-1)){
                    return true;
                } else{
                    map[startRow][startCol] = 3;
                    return false;
                }
            // already travelled, no need to try
            } else {
                return false;
            }
        }
    }
}


/*
backtrack
    1.what choice are we making
        for every row which column to place the queen in
    2.what are the constraints of the problem

    3.what is our goal

*/
```
