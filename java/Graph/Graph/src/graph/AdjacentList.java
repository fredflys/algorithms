package graph;

import java.io.File;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.Scanner;

public class AdjacentList {
	// vertices
	private int V;
	// edges
	private int E;
	// Space: O(V + E)
	// use hash set or tree set instead of linked list   to reduce time
	private LinkedList<Integer>[] adj;
	
	// build graph: O(E*V) if parallel edges have to be detected
	public AdjacentList (String filename) {
		File file = new File(filename);
		try (Scanner scanner = new Scanner(file)) {
			V = scanner.nextInt();
			if (V < 0) {
				throw new IllegalArgumentException("Vertices must be non-negative.");
			}
			adj = new LinkedList[V];
			for (int i = 0; i < V; i++) {
				adj[i] = new LinkedList<Integer>();
			}
			
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
				if (adj[u].contains(v)) {
					throw new IllegalArgumentException("A parallel edge is detected. Only simple graph is allowed.");
				}
				
				adj[u].add(v);
				adj[v].add(u);
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(String.format("V = %d E = %d\n", V, E));
		for (int u = 0; u < V; u++) {
			sb.append(String.format("%d: ", u));
			for (int v: adj[u]) {
				sb.append(String.format("%d ", v));
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
	
	// Time: O(degree(V))
	public boolean hasEdge (int u, int v) {
		validateVertex(u);
		validateVertex(v);
		return adj[u].contains(v);
	}
	
	// Time: O(degree(V))
	public LinkedList<Integer> adj (int u) {
		validateVertex(u);
		return adj[u];
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
		AdjacentList adj = new AdjacentList("g.txt");
		System.out.println(adj);
	}
}
