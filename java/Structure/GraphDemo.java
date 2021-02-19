import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Queue;
import java.util.LinkedList;

public class GraphDemo {
    public static void main(String[] args) {
        int num = 8;
        // String[] vertexes = {"A", "B", "C", "D", "E"};
        int[] vertexes = {1,2,3,4,5,6,7,8};
        /**
         *     1
         *   /   \
         *  2     3
         * / \   / \
         * 4 5  6 - 7
         * \ /
         *  8
         * dfs: 1 2 4 8 5 3 6 7 
         * bfs: 1 2 3 4 5 6 7 8
         */
        Graph<Integer> graph = new Graph<Integer>(num);
        
        for(int v: vertexes){
            graph.insert(v);
        }
        
        graph.createEdge(0,1,1);
        graph.createEdge(0,2,1);
        graph.createEdge(1,3,1);
        graph.createEdge(1,4,1);
        graph.createEdge(3,7,1);
        graph.createEdge(4,7,1);
        graph.createEdge(2,5,1);
        graph.createEdge(2,6,1);
        graph.createEdge(5,6,1);        
    
        graph.show();
        System.out.println("Depth First Search: ");
        graph.dfs();
        System.out.println("\nBreadth First Search: ");
        graph.bfs();
    }
}

class Graph<E extends Comparable<E>>{
    private ArrayList<E> vertexes;
    private int[][] edges;
    private int edgeNum;

    public Graph(int n){
        edges = new int[n][n];
        vertexes = new ArrayList<E>(n);
        edgeNum = 0;
    }

    public void insert(E e){
        vertexes.add(e);
    }

    /**
     * 为图内任意两个顶点之际建立关系，由于是无向图，需要给v1到v2、v2到v1都赋上一样的权值
     * @param v1 顶点1的下标
     * @param v2 顶点2的下标
     * @param weight 两者之间的权值，0表示两点不相通，1表示两者相通
     */
    public void createEdge(int v1, int v2, int weight){
        edges[v1][v2] = weight;
        edges[v2][v1] = weight;
        edgeNum++;
    }

    public int getVertexNum(){
        return vertexes.size();
    }

    public int getEdgeNum(){
        return edgeNum;
    }

    public E getVertex(int index){
        return vertexes.get(index);
    }

    public int getEdge(int v1, int v2){
        return edges[v1][v2];
    }

    public int[] getEdges(int v1){
        return edges[v1];
    }

    public void show(){
        for(int[] edge: edges){
            System.out.println(Arrays.toString(edge));
        }
    }

    // 深度优先遍历
    public void dfs(){
        List<Integer> visisted = new ArrayList<Integer>();
        // 从根节点开始遍历，当前尚未访问到任何节点
        dfs(0, visisted);
    }
    /**
     * 
     * @param index 当前要遍历的节点索引
     * @param visited 已经遍历过的节点索引集合
     */
    private void dfs(int index, List<Integer> visited){
        if(visited.contains(index))
            return;

        // do something on current vertex
        System.out.print(vertexes.get(index) + " ");
        visited.add(index);

        int[] edges = getEdges(index);

        for(int i=1;i<edges.length;i++){
            if(edges[i] > 0){
                dfs(i, visited);
            }
        }
    }

    // 广度优先遍历
    public void bfs(){
        Queue<Integer> unvisited = new LinkedList<Integer>();
        List<Integer> visited = new ArrayList<Integer>();

        bfs(0, unvisited, visited);
    }

    /**
     * 
     * @param index 当前要访问节点的下标
     * @param unvisited 尚未范围的节点队列
     * @param visited 已经访问过的节点列表
     */
    private void bfs(int index, Queue<Integer> unvisited, List<Integer> visited){
        // 递归终止条件，已经访问过的节点无需再次访问
        if(visited.contains(index)){
            return;
        }
        
        // 访问index下标的节点，并加入已访问节点的列表
        System.out.print(vertexes.get(index) + " ");
        visited.add(index);

        // 取出与节点的相连的其它节点
        int[] edges = getEdges(index);
        for(int i=0;i<edges.length;i++){
            if(edges[i] > 0){
                unvisited.add(i);
            }
        }

        // 递归部分，依次访问队列中的节点，直到队列清空
        while(!unvisited.isEmpty()){
            bfs(unvisited.poll(), unvisited, visited);
        }
    }


}