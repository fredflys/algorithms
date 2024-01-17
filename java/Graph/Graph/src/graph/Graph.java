package graph;

import java.io.File;
import java.util.TreeSet;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Scanner;

public class Graph {

	// vertices
	private int V;
	// edges
	private int E;
	private TreeSet<Integer>[] adj;
	private int connectComponentCount = 0;
	
	// build graph O(ElogV) 
	public Graph (String filename) {
		File file = new File(filename);
		try (Scanner scanner = new Scanner(file)) {
			V = scanner.nextInt();
			if (V < 0) {
				throw new IllegalArgumentException("Vertices must be non-negative.");
			}
			adj = new TreeSet[V];
			for (int i = 0; i < V; i++) {
				adj[i] = new TreeSet<Integer>();
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
	// hide implementation details
	public Iterable<Integer> adj (int u) {
		validateVertex(u);
		return adj[u];
	}
	
	public int getDegree(int u) {
		validateVertex(u);
		return adj[u].size();
	}
	
	private void validateVertex(int v) {
		if (v < 0 || v > V) {
			throw new IllegalArgumentException(String.format("Vertex %s is invalid.", v));
		}
	}
	
	/*
	 * DFS
	 */
	// improvement: change boolean array to int array to store more info
	// private boolean[] visited;
	private int[] visited;
	private ArrayList<Integer> orderDFS = new ArrayList<>();
	public Iterable<Integer> dfs() {
		visited = new int[V];
		for (int i = 0; i < V; i++) {
			visited[i] = -1;
		}
		for (int v = 0; v < V; v++) {
			if (visited[v] == -1) {
				dfs(0);
				++connectComponentCount;
			}
		}
		return orderDFS;
	}
	
	public void dfs(int u) {
		visited[u] = connectComponentCount;
		// preorder position
		orderDFS.add(u);
		for (int v: adj(u)) {
			if (visited[v] == -1) {
				dfs(v);
			}
		}
		// postorder position
		// orderDFS.add(u);
	}
	
	public int getConnecetedComponentCount() {
		if (connectComponentCount == 0) {
			dfs();
		}
//		System.out.println(Arrays.toString(visited));
		return connectComponentCount;
	}
	
	public boolean isConnected(int u, int v) {
		validateVertex(u);
		validateVertex(u);
		return visited[u] == visited[v];
	}
	
	public ArrayList<Integer>[] components() {
		ArrayList<Integer>[] res = new ArrayList[connectComponentCount];
		for (int i = 0; i < connectComponentCount; i++) {
			res[i] = new ArrayList<>();
		}
		
		for (int v = 0;  v < V; v++) {
			res[visited[v]].add(v);
		}
		
		return res;
	}
	
	/*
	 * get single source path
	 * */
	int[] pred;
	private void dfs(int u, int parent) {
		visited[u] = 1;
		pred[u] = parent;
		for (int v: adj(u)) {
			if (pred[v] == -1) {
				dfs(v, u);
			}
		}
	}
	
	public Iterable<Integer> getPath(int s, int t) {
		validateVertex(s);
		validateVertex(t);
		
		pred = new int[V];
		for (int i = 0; i < pred.length; i++) {
			pred[i] = -1;
		}
		dfs(s, s);
		
		List<Integer> res = new ArrayList<>();
		if (pred[t] == -1) {
			return res;
		}
		
		// find the path from backwards as stored is parent info
		for (int cur = t; cur != s; cur = pred[cur]) {
			res.add(cur);
		}
		res.add(s);
		Collections.reverse(res);
		return res;
	}
	
	/*
	 * detect cycle
	 * */
	public boolean detectCycle() {
		visited = new int[V];
		for (int i = 0; i < V; i++) {
			visited[i] = -1;
		}
		
		for (int v = 0; v < V; v++) {
			if (visited[v] == -1) {
				if (dfsCycle(v, v)) {
					return true;
				}
			}
		}
		return false;
	}
	
	private boolean dfsCycle(int u, int parent) {
		visited[u] = 1;

		for (int v: adj(u)) {
			if (visited[v] == -1) {
				if (dfsCycle(v, u)) {
					return true;	
				}
			} else if (v != parent) {
				return true;
			}
		}
		return false;
	}
	
	/*
	 * check if graph is a bipartite
	 * */
	private int[] colors;
	public boolean isBipartite() {
		colors = new int[V];
		for (int i = 0; i < V; i++) {
			colors[i] = -1;
		}
		
		for (int v = 0; v < V; v++) {
			if (colors[v] == -1) {
				if (!dfsBipartite(v, 0)) {
					return false;
				}
			}
		}
		return true;
	}
	
	// coloring
	private boolean dfsBipartite(int u, int color) {
		colors[u] = color;
		
		for (int v: adj(u)) {
			// not visited
			if (colors[v] == -1) {
				if (!dfsBipartite(v, color % 1)) return false;
				continue;
			}
			
			// already visited
			if (colors[v] != color % 1) {
				return false;
			}
		}
		return true;
	}
	
	public static void main(String[] args) {
		Graph adj = new Graph("g.txt");
		System.out.println(adj.dfs());
//		System.out.println(adj.getConnecetedComponentCount());
		System.out.println(adj.isConnected(1, 2));
		System.out.println(adj.getPath(0, 6));
		System.out.println(adj.detectCycle());
		System.out.println(adj.isBipartite());
	}

}
