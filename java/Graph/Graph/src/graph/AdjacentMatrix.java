package graph;
import java.io.File;
import java.util.ArrayList;
import java.util.Scanner;

public class AdjacentMatrix {
	// vertices
	private int V;
	// edges
	private int E;
	// Space: O(V^2)
	private int[][] adj;
	
	public AdjacentMatrix (String filename) {
		File file = new File(filename);
		try (Scanner scanner = new Scanner(file)) {
			V = scanner.nextInt();
			if (V < 0) {
				throw new IllegalArgumentException("Vertices must be non-negative.");
			}
			adj = new int[V][V];
			if (E < 0) {
				throw new IllegalArgumentException("Edges must be non-negative.");
			}
			E = scanner.nextInt();
			for (int i = 0; i < E; i++) {
				int u = scanner.nextInt();
				int v = scanner.nextInt();
				validateVertex(u);
				validateVertex(v);
				
				// detect loop
				if (u == v) {
					throw new IllegalArgumentException("A loop is detected. Only simple graph is allowed.");
				}
				
				// detect parallel edge
				if (adj[u][v] == 1) {
					throw new IllegalArgumentException("A parallel edge is detected. Only simple graph is allowed.");
				}
				
				adj[u][v] = 1;
				adj[v][u] = 1;
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(String.format("V = %d E = %d\n", V, E));
		for (int i = 0; i < V; i++) {
			for (int j = 0; j < V; j++) {
				sb.append(String.format("%d ", adj[i][j]));
			}
			sb.append("\n");
		}
		
		return sb.toString();
	}
	
	public int V () {
		return V;
	}
	
	public int E () {
		return E;
	}
	
	// Time: O(1)
	public boolean hasEdge (int u, int v) {
		validateVertex(u);
		validateVertex(v);
		return adj[u][v] == 1;
	}
	
	// Time: O(V)
	public ArrayList<Integer> adj (int u) {
		validateVertex(u);
		ArrayList<Integer> adjacentEdges = new ArrayList<>(); 
		for (int i = 0; i < V; i++) {
			if (adj[u][i] == 1) {
				adjacentEdges.add(i);
			}
		}
		return adjacentEdges;
	}
	
	public int getDegree(int u) {
		return adj(u).size();
	}
	
	private void validateVertex(int v) {
		if (v < 0 || v > V) {
			throw new IllegalArgumentException(String.format("Vertex %s is invalid.", v));
		}
	}

	
	public static void main(String[] args) {
		AdjacentMatrix adj = new AdjacentMatrix("g.txt");
		System.out.println(adj);
	}
}
