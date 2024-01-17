## Graph
cosists of
- vertx, edge

category
- whether directed | weighted

Tricky
- loop, parallel edges
- simple graph - no loop nor parallel edges

connected or not connected
= any connected sub-graph is a connected component
- a tree is a acylic connected graph and every node can be its root
- a spanning tree from a connected graph, with all the vertices and its edges being a subset of the graph (minimus edges to keep the graph still connected [V - 1]).
- only a connected graph has a spanning tree
- a graph, whether connected or not, has a spanning forest 
- degree (for unweighted undirect graph) is the amount of a vertex's adjacent edges

sparse and dense graph
- focus in mainly on sparse graph

representation
- adjacency matrix
    - A[i][j] = 1 means i is connected to j
    - use 2 or more to represent parallel edges
    - diagnoal elements are zeros for simple graph
    - diagonal symmetry for undirected graph
    - high space complexity (O(V^2)) 
- adjacency list
  - build graph : O(E * V), whenever inserting an edge, the node's edges have to be iterated through to check whether it already exists
  - improvement: use tree set (logV) or hash set(1) instead of linked list to reduce time
  - tree set can be used to retain order and save more space
  - 1 < logN < N, logN is more like O(1) than O(N)
![](https://s2.loli.net/2022/12/02/Up7iY9v2DlCVRxA.png)

Traversal
DFS
  - compared with tree traversal, a node might be visited multiple times 
  - applications
    - count connected components
    - single source path
    - cycle detection
    - bipartite graph

Other topics
- graph isomorphism - NP problem 
  ![](https://s2.loli.net/2022/12/04/JcByI3ar1OfZqg2.png)
- Planar graph
  ![](https://s2.loli.net/2022/12/04/gk4Ofj89Hawhy6n.png)

#### [133. Clone Graph ](https://leetcode-cn.com/problems/clone-graph/)<span style="color:orange">Medium</span>

Steps:

1. Find all the nodes by BFS
2. Clone nodes
3. Clone edges

```java
class Solution {
    public Node cloneGraph(Node node) {
        if(node == null)
            return null;
        
        // get ready: find all the original nodes
        List<Node> nodes = findNodesByBFS(node);
        // clone all the nodes
        Map<Node, Node> mappingToCloned = copyNodes(nodes);
        // copy every edge from original nodes to cloned nodes
        copyEdges(nodes, mappingToCloned);
        // get a new node from mapping
        return mappingToCloned.get(node);
    }
    
    public List<Node> findNodesByBFS(Node node) {
        Queue<Node> candidates = new LinkedList<>();
        Set<Node> visited = new HashSet<>();
        // whenever a new candidate arrives, never forget to mark it as visited
        candidates.offer(node);
        visited.add(node);
        while (!candidates.isEmpty()) {
            Node current = candidates.poll();
            for (Node neighbor: current.neighbors) {
                // already visited so skip it
                if (visited.contains(neighbor))
                    continue;
                candidates.offer(neighbor);
                visited.add(neighbor);
            }
        }
        
        return new ArrayList<>(visited);
    }
    
    public Map<Node, Node> copyNodes(List<Node> nodes) {
        Map<Node, Node> mapping = new HashMap<>();
        for (Node node: nodes) {
            // clone by creating a new node with the value from the original node
            // now the new node has no neighbors
            mapping.put(node, new Node(node.val));
        }
        return mapping;
    }
    
    // copying edges means adding neighbors to every cloned node
    public void copyEdges(List<Node> nodes, Map<Node, Node> mapping) {
        for (Node original: nodes) {
            Node cloned = mapping.get(original);
            for (Node neighbor: original.neighbors) {
                Node clonedNeighbor = mapping.get(neighbor);
                cloned.neighbors.add(clonedNeighbor);
            }
        }
    }
}
```

#### [127. Word Ladder](https://leetcode-cn.com/problems/word-ladder/) <span style="color:red">Hard</span>

BFS, Shortest path 

Every word in the word list can be considered as a node, whose possible transformations in the word list can be taken as neighbors. The difference is that this time neighbors need to be found by a certain rule.

``` python
class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        wordList = set(wordList)
        candidates = collections.deque([beginWord])
        visited = [beginWord]
        
        distance = 1
        while candidates:
            distance += 1
            
            size = len(candidates)
            for _ in range(size):
                candidate = candidates.popleft()
                
                for similar_word in self.find_similar_words(candidate, wordList):
                    # early exit 
                    if similar_word == endWord:
                    	return distance
                    if similar_word in visited:
                        continue
                    candidates.append(similar_word)
                    visited.append(similar_word)
        return 0
    
    def find_similar_words(self, word, wordList):
        similar_words = []
        for i in range(len(word)):
            for char in 'abcdefghijklmnopqrstuvwxyz':
                if char == word[i]:
                    continue
                new_word = word[:i] + char + word[i + 1:]
                if new_word in wordList:
                    similar_words.append(new_word)
        return similar_words
```

double-end BFS
![](https://s2.loli.net/2022/06/10/wWoMxDyiuk8LApv.png)
This solution will not pass all test cases. Really cann't see where is wrong.

```python
from collections import deque
chars = 'abcdefghijklmnopqrstuvwxyz'

class Solution:
    def ladderLength(self, start, end, wordList):
        if end not in wordList:
            return 0
        
        wordSet = set(wordList)
        wordSet.add(start)
        # wordSet.add(end)
        
        similar_words = self.build_graph(wordSet)
        print(similar_words)
        
        forward_candidates = deque([start])
        forward_visited = set([start])
        backward_candidates = deque([end])
        backward_visited = set([end])
        
        distance = 1
        while forward_candidates and backward_candidates:
            # add to distance first
            distance += 1
            if self.explore(similar_words, forward_candidates, forward_visited, backward_visited):
                return distance
            distance += 1
            if self.explore(similar_words, backward_candidates, backward_visited, forward_visited):
                return distance
        return 0
        
    def build_graph(self, wordSet):
        def get_similar_words(word, wordSet):
            similar_words = set()
            for i in range(len(word)):
                for char in chars:
                    if char == word[i]:
                        continue
                    next_word = word[:i] + char + word[i + 1:]
                    if next_word in wordSet:
                        similar_words.add(next_word)
            return similar_words
            
        graph = {}
        for word in wordSet:
            graph[word] = get_similar_words(word, wordSet)
        return graph
    
    def explore(self, graph, current_candidates, current_visited, opposite_visited):
        for _ in range(len(current_candidates)):
            word = current_candidates.popleft()
            for next_word in graph[word]:
                if next_word in current_visited:
                    continue
                if next_word in opposite_visited:
                    return True 
                current_candidates.append(next_word)
                current_visited.add(next_word)
            return False
```

#### [752. Open the Lock](https://leetcode.com/problems/open-the-lock/) Medium
double-end BFS
```java
class Solution {
    String start = "0000";
    
    public int openLock(String[] deadends, String target) {
        return bfs(target, deadends);
    }
    
    String moveUp(String s, int slot) {
        char[] ch = s.toCharArray();
        // character instead of integer
        if (ch[slot] == '9') {
            ch[slot] = '0';
        } else {
            ch[slot] += 1;   
        }
        return new String(ch);
    }
    
    String moveDown(String s, int slot) {
        char[] ch = s.toCharArray();
        if (ch[slot] == '0') {
            ch[slot] = '9';
        } else {
            ch[slot] -= 1;   
        }
        return new String(ch);
    }
    
    Set<String> convertToSet(String[] deadends) {
        Set<String> deads = new HashSet<>();
        for(String dead: deadends) deads.add(dead);
        return deads;
    }
    
    int bfs(String target, String[] deadends) {
        Set<String> q1 = new HashSet<>();
        Set<String> q2 = new HashSet<>();
        Set<String> deads = convertToSet(deadends);
        Set<String> visited = new HashSet<>();
        
        q1.add(start);
        q2.add(target);
        int step = 0;
        
        while (!q1.isEmpty() && !q2.isEmpty()) {
            Set<String> candidates = new HashSet<>();
            
            int size = q1.size();
            for (String cur: q1) {
                
                // target is reached
                if (q2.contains(cur)) {
                    return step;
                }
                
                if (deads.contains(cur)) {
                    continue;
                }
                
                visited.add(cur);
                
                for (int j = 0; j < start.length(); j++) {
                    String up = moveUp(cur, j);
                    String down = moveDown(cur, j);
                    System.out.println(down);
                    if (!visited.contains(up)) {
                        candidates.add(up);
                    }
                    
                    if (!visited.contains(down)) {
                        candidates.add(down);
                    }
                }
            }
            step++;
            q1 = q2;
            q2 = candidates;
        }
        
        return -1;
    }
}
```

#### [773. Sliding Puzzle](https://leetcode.com/problems/sliding-puzzle/) Hard
BFS. Think carefully about how to represent a problem.
```java
class Solution {
    
    public int slidingPuzzle(int[][] board) {
        String start = buildStartString(board);
        String target = "123450";
        return bfs(start, target);
    }
    
    // represent the board as a string
    // this will make things a lot more easier
    // didn't think of this before
    String buildStartString(int[][] board) {
        StringBuilder start = new StringBuilder();
        int m = 2, n = 3;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                start.append(board[i][j]);
            }
        }
        return start.toString();
    }
    
    int[][] getMoves() {
        int[][] moves = new int[][]{
            {1, 3}, // 0 is at 0 (board[0][0]), can only move right to 1 -> board[0][1], or move down to 3 -> board[1][0]
            {0, 2, 4}, // 0 is at 1 (board[0][1], can move left to board[0][0], right to board[0][2] and down to board[1][1])
            {1, 5}, // 0 is at 2 (board[0][2]), can move left to board[0][1] and down to board[1][2]
            {0, 4}, // 0 is at 3 (board[1][0]), can move up to board[0][0] and right to board[1][1]
            {1, 3, 5}, // 0 is at 4 (board[1][2]), can move up to board[0][1], left to board[1][0] and right to board[1][2]
            {2, 4} // 0 is at 5 (board[1][2]), can move eith up to board[0][2] or left to board[1][1`]
        };
        return moves;
    }
    
    int findZeroPosition(String s) {
        int index = 0;
        for (; s.charAt(index) != '0'; index++);
        return index;
    }
    
    // move zero to possible places
    String swap(char[] chars, int i, int j) {
        char temp = chars[i];
        chars[i] = chars[j];
        chars[j] = temp;
        return new String(chars);
    }
    
    int bfs(String start, String target) {
        Queue<String> q = new LinkedList<>();
        Set<String> visited = new HashSet<>();
        int[][] moves = getMoves();
        int step = 0;
        q.offer(start);
        visited.add(start);

        
        while (!q.isEmpty()) {
            int size = q.size();
            for (int i = 0; i < size; i++) {
                String cur = q.poll();
                if (target.equals(cur)) {
                    return step;
                }
                
                int zeroIndex = findZeroPosition(cur);
                for (int moveTo: moves[zeroIndex]) {
                    String newBoard = swap(cur.toCharArray(), moveTo, zeroIndex);
                    if (!visited.contains(newBoard)) {
                        q.offer(newBoard);
                        visited.add(newBoard);
                    }
                }
                
            }
            step++;
        }   
        return -1;
    }   
}
```


#### [126. Word Ladder II](https://leetcode-cn.com/problems/word-ladder-ii/) <span style="color:red">Hard</span>
BFS
```java
public class Solution {
	class Node {
		int distance;
		List<String> path;
		Node(int distance, List<String> path) {
			this.distance = distance;
			this.path = path;
        }
    }
	
    public List<List<String>> findLadders(String beginWord, String endWord, List<String> wordList) {
        List<List<String>> results = new ArrayList<>();
        if (wordList == null || wordList.size() == 0) {
            return results;
        }
        
        Set<String> wordDict = new HashSet<>(wordList);
        Set<String> visited = new HashSet<>();
        Queue<Node> queue = new LinkedList<>();
        queue.offer(buildNode(beginWord, 0, new ArrayList<String>()));
        Integer shortestDist = null;
        
        while (!queue.isEmpty()) {
            // Record what we meet at this level
            Set<String> thisLevelVisited = new HashSet<>();
            int size = queue.size();
            for (int k = 0; k < size; k++) {
                Node node = queue.poll();
				List<String> path = node.path;
				String word = path.get(path.size() - 1);
				int distance = node.distance;
                
                // If this level has a distance greater than shortest distance, we don't need to consider it any more
                if (shortestDist != null && distance > shortestDist) {
                    continue;
                }
                
                // If we find the endWord, then we have the shortest distance
                if (word.equals(endWord)) {
                    if (shortestDist == null) {
                        shortestDist = distance;
                    }
                    results.add(path);
                    continue;
                }
                
				for (String nextWord: wordDict) {
					if (!isNeighbor(word, nextWord) || visited.contains(nextWord)) {
						continue;
					}
					
                    queue.offer(buildNode(nextWord, distance, new ArrayList<String>(path)));
                    thisLevelVisited.add(nextWord); 	
				}
            }
            visited.addAll(thisLevelVisited);
        }
        return results;
    }
	
	private Node buildNode(String word, int distance, List<String> path) {
		path.add(word);
		return new Node(distance + 1, path);
	}
    
	private boolean isNeighbor(String a, String b) {
        int diff = 0;
        for (int i = 0; i < a.length() && diff < 2; i++) {
            if (a.charAt(i) != b.charAt(i)) {
                diff++;
            }
        }
        return diff == 1;
    }
}
```
BFS + DFS
DFS will go deeper and deeper and along the way a path can be stored. What's not good about DFS is that it will not stop until it reaches a dead end, which means a lot of needless attempts. But this can be fixed with BFS because a path is extended layer by layer in BFS, which means roundabout paths can be skipped. So in essence, BFS is employed to prune the tree for DFS to walk through.
![](https://s2.loli.net/2022/02/12/dBEA3NC5LZKUFXa.png)
```java
class Solution {
    public List<List<String>> findLadders(String beginWord, String endWord, List<String> wordList) {
        List<List<String>> results = new ArrayList<>();
        if (wordList == null || !wordList.contains(endWord)) {
           return results; 
        }
        
        Set<String> wordDict = new HashSet<>(wordList);
        wordDict.add(beginWord);
        // distance from begin word to another word
        Map<String, Integer> distanceTo = new HashMap<>();
        // a word to other words that can be reached without extra steps
        Map<String, List<String>> nearWordsTo = new HashMap<>();
        for (String word : wordDict) {
            nearWordsTo.put(word, new ArrayList<String>());
        }
        
        bfs(nearWordsTo, distanceTo, beginWord, endWord, wordDict);
        
        // there is no way to arrive at end word
        if (distanceTo.get(endWord) == null) {
            return results;
        }
        dfs(results, new ArrayList<String>(), beginWord, endWord, nearWordsTo, 
        return results;
    }
    
    private void bfs(Map<String, List<String>> nearWordsTo, Map<String, Integer> distanceTo, String beginWord, String endWord, Set<String> wordDict) {
        // begin word to begin word requires no moevement
        distanceTo.put(beginWord, 0);
        Queue<String> queue = new LinkedList<String>();
        // add begin word for initialization
        queue.offer(beginWord);
        
        while(!queue.isEmpty()) {
            String word = queue.poll();
            for (String nextWord: wordDict) {
                if (!isNeighbor(word, nextWord)) {
                    continue;
                }
                
                if (!distanceTo.containsKey(nextWord) || distanceTo.get(nextWord) == distanceTo.get(word) + 1) {
                    nearWordsTo.get(word).add(nextWord);
                }
                
                if (!distanceTo.containsKey(nextWord)) {
                    distanceTo.put(nextWord, distanceTo.get(word) + 1);
                    queue.offer(nextWord);
                }
            }
        }
    }
    
    private void dfs(List<List<String>> results, List<String> path, String word, String endWord, Map<String, List<String>> nearWordsTo, int minLen) {
        if (path.size() == minLen + 1) {
            return;
        }
        
        path.add(word);
        if (word.equals(endWord)) {
            results.add(new ArrayList<String>(path));
        }
        else {
            for (String nextWord : nearWordsTo.get(word)) {
                dfs(results, path, nextWord, endWord, nearWordsTo, minLen);
            }
        }

        path.remove(path.size() - 1);
    }
    
    private boolean isNeighbor(String a, String b) {
        int diff = 0;
        for (int i = 0; i < a.length() && diff < 2; i++) {
            if (a.charAt(i) != b.charAt(i)) {
                diff++;
            }
        }
        return diff == 1;
    }
}
```

#### [1625. Lexicographically Smallest String After Applying Operations](https://leetcode.com/problems/lexicographically-smallest-string-after-applying-operations/description/) Medium
```java
class Solution {
    String res = null;

    public String findLexSmallestString(String s, int a, int b) {
        res = s;
        Queue<String> candidates = new LinkedList<>();
        Set<String> visited = new HashSet<>();
        String candidate = null;
        visited.add(s);
        for (candidates.add(s); !candidates.isEmpty();) {
            s = candidates.poll();
            candidate = addUpEvenDigits(s, a);
            addCandidate(candidates, visited, candidate);
            candidate = reverseDigits(s, b);
            addCandidate(candidates, visited, candidate);
        }
        return res;
    }

    String addUpEvenDigits(String source, int added) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < source.length(); i++) {
            sb.append(
                (i & 1) == 0 ?
                    source.charAt(i):
                    (char) ((source.charAt(i) - '0' + added) % 10 + '0')
            );
        }
        return sb.toString();
    }

    String reverseDigits(String source, int pos) {
        return source.substring(source.length() - pos) + source.substring(0, source.length() - pos);
    }

    void addCandidate(Queue<String> candidates, Set<String> visited, String candidate) {
        System.out.println(candidate);
        if (visited.add(candidate)) {
            if (res.compareTo(candidate) > 0) res = candidate;
            candidates.add(candidate);
        }
    }
}
```

#### [200. Number of Islands ](https://leetcode-cn.com/problems/number-of-islands/)<span style="color:orange">Medium</span>

BFS on a matrix to get number of connected blocks

```python
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        islands = 0
        
        # empty matrix or invalid data 
        if not grid or not grid[0]:
            return islands
        visited = set()
        
		# possible directions: right, left, up, down
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        
        def bfs_search(row, column):
            candidates = collections.deque([ (row, column) ])
            visited.add( (row, column) ) 
            
            while candidates:
                row, column = candidates.popleft()
				# move in four possible directions
                for row_delta, column_delta in directions:
                    next_row = row + row_delta
                    next_column = column + column_delta
                    if not is_valid(next_row, next_column):
                        continue
                    candidates.append((next_row, next_column))
                    visited.add((next_row, next_column))
                    
        def is_valid(row, column):
            # out of limits
            if not (0 <= row < len(grid) and 0 <= column < len(grid[0])):
                return False
            # already visited
            if (row, column) in visited:
                return False
            # water
            if grid[row][column] == "0":
                return False
            return True
        
        # walk through the space
        for row in range(len(grid)):
            for column in range(len(grid[0])):
                # search starts only when current position is island and not visited
                if grid[row][column] == "1" and (row, column) not in visited:
                    bfs_search(row, column)
                    islands += 1
		
        return islands 
```

#### [611 · Knight Shortest Path - LintCode](https://www.lintcode.com/problem/611/)

BFS on a matrix

```python
"""
Definition for a point.
class Point:
    def __init__(self, a=0, b=0):
        self.x = a
        self.y = b
"""
class Solution:
    """
    @param grid: a chessboard included 0 (false) and 1 (true)
    @param source: a point
    @param destination: a point
    @return: the shortest path 
    """
    def shortestPath(self, grid, source, destination):
        moves = [
            (-1,  2), (1,   2), # up left, up right
            (-1, -2), (1,  -2), # down left, down right
            (-2,  1), (-2, -1), # left up, left down
            (2,   1), (2,  -1), # right up, right down
        ]

        candidates = collections.deque([(source.x, source.y)])
        # current position -> move count
        visited = {(source.x, source.y) : 0}

        def is_valid(x, y):
            # out of limits
            if x < 0 or x >= len(grid) or y < 0 or y >= len(grid[0]):
                return False
            # position already occupied
            if grid[x][y] == 1:
                return False
            return True

        while candidates:
            x, y = candidates.popleft()
            # arrives at destination
            if (x, y) == (destination.x, destination.y):
                return visited[(x, y)]
            # move in all possible directions
            for dx, dy in moves:
                next_x, next_y = x + dx, y + dy
                if (next_x, next_y) in visited:
                    continue
                if not is_valid(next_x, next_y):
                    continue
                candidates.append((next_x, next_y))
                visited[(next_x, next_y)] = visited[(x, y)] + 1

        return -1
```

double-end BFS
fixed start and end point
```python
from collections import deque

moves = [
    (-1,  2), (1,   2), # up left, up right
    (-1, -2), (1,  -2), # down left, down right
    (-2,  1), (-2, -1), # left up, left down
    (2,   1), (2,  -1), # right up, right down
]

class Solution:
    def shortest_path(self, grid, source, destination):
        if not grid or not grid[0]:
            return -1
        if (source.x, source.y) == (destination.x, destination.y):
            return 0

        forward_candidates, forward_visited, backward_candidates, backward_visited = self.prepare(source, destination)

        distance = 0
        # if no candidates are found on one side
        while forward_candidates and backward_candidates:
            distance += 1
            # explore from the start and see if matched from backward
            if self.explore(grid, forward_candidates, forward_visited, backward_visited):
                return distance        
            distance += 1
            # explore from the end and see if matched from forward
            if self.explore(grid, backward_candidates, backward_visited, forward_visited):
                return distance
        return -1

    def prepare(self, start, end):
        fc = deque([(start.x, start.y)])
        fv = set([(start.x, start.y)])
        bc = deque([(end.x, end.y)])
        bv = set([(end.x, end.y)])
        return fc, fv, bc, bv

    def explore(self, grid, current_candidates, current_visited, opposite_visited):
        for _ in range(len(current_candidates)):
            x, y = current_candidates.popleft()
            for dx, dy in moves:
                new_x, new_y = x + dx, y + dy
                if not self.is_valid(grid, current_visited, new_x, new_y):
                    continue
                if (new_x, new_y) in opposite_visited:
                    return True
                current_candidates.append((new_x, new_y))
                current_visited.add((new_x, new_y))
        return False

    
    def is_valid(self, grid, visited, x, y):
        # out of limits
        if x < 0 or x >= len(grid) or y < 0 or y >= len(grid[0]):
            return False
        # position already occupied
        if grid[x][y] == 1:
            return False
        return (x, y) not in visited
```

#### [207. Course Schedule](https://leetcode-cn.com/problems/course-schedule/) <span style="color:orange">Medium</span>

The solution is almost identical to that to 210. Simply returns true if courses_taken == numCourses.

#### [210. Course Schedule II](https://leetcode-cn.com/problems/course-schedule-ii/) <span style="color:orange">Medium</span>

BFS, Topological sort

```python
class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        # index : prerequisite num, element: the open courses after you go to the prerequisite class
        next_courses = [ [] for _ in range(numCourses)]

        # every class has how many prerequistes, initialized to 0 for every class
        # the equivalent in graph is called in-degree
        pre_courses_count = [0] * numCourses
        for next, pre in prerequisites:
            next_courses[pre].append(next)
            pre_courses_count[next] += 1

        # courses that can be taken
        candidates = collections.deque()
        # initialize candidates
        for course in range(numCourses):
            # if the course has no prerequisites, then that's the course we should take first
            if pre_courses_count[course] == 0:
                candidates.append(course)

        courses_taken = []

        while candidates:
            # take a course
            course = candidates.popleft()
            # save the course
            courses_taken.append(course)
            # look through what can be taken next
            for next_course in next_courses[course]:
                # since the course is already taken, ever next course has one less prerequisite
                pre_courses_count[next_course] -= 1
                # if the next course has no prerequisites, then it can be taken. So add it to the candidates
                if pre_courses_count[next_course] == 0:
                    candidates.append(next_course)
        
        # if the courses taken is not enough, it means it's impossible to take all courses
        if len(courses_taken) != numCourses:
            return []
        return courses_taken
```

#### [444. Sequence Reconstruction](https://leetcode.com/problems/sequence-reconstruction/) <span style="color:orange">Medium</span>

BFS, Topological sort

```python
class Solution:
    """
    @param org: a permutation of the integers from 1 to n
    @param seqs: a list of sequences
    @return: true if it can be reconstructed only one or false
    """
    def sequenceReconstruction(self, org, seqs):
        def build_graph(seqs):
            graph = {}
            for seq in seqs:
                for node in seq:
                    if node not in graph:
                        graph[node] = set()
            
            for seq in seqs:
                for i in range(1, len(seq)):
                    # from sqe[i - 1] to seq[i]
                    graph[seq[i - 1]].add(seq[i])

            return graph

        def get_indegrees(graph):
            indegrees = {
                node : 0
                for node in graph
            }
            for node in graph:
                for neighbor in graph[node]:
                    indegrees[neighbor] += 1
            return indegrees

        def topo_sort(graph):
            indegrees = get_indegrees(graph)

            candidates = collections.deque([])
            for node in graph:
                if indegrees[node] == 0:
                    candidates.append(node)

            topo_order = []
            while candidates:
                # having only sequence means there are never more than one candidate in the queue
                if len(candidates) > 1:
                    return False
                
                node = candidates.popleft()
                topo_order.append(node)
                for neighbor in graph[node]:
                    indegrees[neighbor] -= 1
                    if indegrees[neighbor] == 0:
                        candidates.append(neighbor)
                
            return topo_order

        graph = build_graph(seqs)
        topo_order = topo_sort(graph)
        print(graph, topo_order, org)
        return topo_order == org
```

#### [269. Alien Dictionary ](https://leetcode-cn.com/problems/alien-dictionary/)<span style="color:red">Hard</span>

BFS, Topological sort

```python
from heapq import heappush, heappop, heapify
class Solution:
    """
    @param words: a list of words
    @return: a string which is correct order
    """
    def alienOrder(self, words):
        def build_graph(words):
            # char that comes first -> char that comes after the key
            graph = {}
            for word in words:
                for char in word:
                    graph[char] = set()
            for i in range(len(words) - 1):
                current_word = words[i]
                next_word = words[i + 1]
                shorter_length = min(len(current_word), len(next_word))
                # do not overstep
                for j in range(shorter_length):
                    # if the char in the same position is equal, move on to compare the next character
                    if current_word[j] != next_word[j]:
                        # the character in the current word is prior to that of the next word 
                        graph[current_word[j]].add(next_word[j])
                        break
                    
                    if j == shorter_length - 1 and len(current_word) > len(next_word):
                        return None
            return graph

        def get_indegrees(graph):
            indegrees = {node : 0 for node in graph}
            for node in graph:
                for neighbor in graph[node]:
                    indegrees[neighbor] += 1
            return indegrees

        def topo_sort(graph):
            indegrees = get_indegrees(graph)
            candidates = [node for node in graph if indegrees[node] == 0]
            # like priority queue in Java. It works like a queue, the difference is that it always puts ""
            heapify(candidates)
            topo_order = ""
            while candidates:
                node = heappop(candidates)
                topo_order += node
                for neighbor in graph[node]:
                    indegrees[neighbor] -= 1
                    if indegrees[neighbor] == 0:
                        heappush(candidates, neighbor)
            
            if len(topo_order) == len(graph):
                return topo_order
            return ""

        graph = build_graph(words)
        if not graph:
            return ""
        return topo_sort(graph)
```

#### [1203. Sort Items by Groups Respecting Dependencies](https://leetcode.com/problems/sort-items-by-groups-respecting-dependencies/description/) Hard
reference: https://leetcode.cn/problems/sort-items-by-groups-respecting-dependencies/solution/dao-xu-lu-lu-zhe-ti-by-xyzza-1ah1/
topological sort
```java
class Solution {
    public int[] sortItems(int n, int m, int[] group, List<List<Integer>> beforeItems) {
        // topological sort on groups and items
        // order should be kept among groups and items as well

        // assign new group numbers to ungrouped items
        for (int i = 0; i < group.length; i++) {
            if (group[i] == -1) group[i] = m++;
        }

        // initialization  
        int[] groupIndegrees = new int[m];
        int[] itemIndegrees = new int[n];
        List<Integer>[] groupAdjacency = new ArrayList[m];
        List<Integer>[] itemAdjacency = new ArrayList[n];
        int index = 0;
        while (index < m + n) {
            if (index < m) groupAdjacency[index] = new ArrayList<>();
            if (index < n) itemAdjacency[index] = new ArrayList<>();
            index++;
        }

        // fill in group adjacency list and group indegrees
        for (int i = 0; i < group.length; i++) {
            for (int beforeItem: beforeItems.get(i)) {
                // beforeItem comes before i and they do not belong to the same group
                // then the group where beforeItem comes from comes before the group where i comes from
                if (group[beforeItem] != group[i]) {
                    groupAdjacency[group[beforeItem]].add(group[i]);
                    groupIndegrees[group[i]]++;
                }
            }
        }

        // tp sort on groups
        List<Integer> sortedGroups = tpSort(groupAdjacency, groupIndegrees);
        if (sortedGroups.size() == 0) return new int[0];

        // fill in item adjacency and item indegrees
        for (int j = 0; j < n; j++) {
            for (int beforeItem: beforeItems.get(j)) {
                itemAdjacency[beforeItem].add(j);
                itemIndegrees[j]++;
            }
        }

        List<Integer> sortedItems = tpSort(itemAdjacency, itemIndegrees);
        if (sortedItems.size() == 0) return new int[0];

        // build results from sorted groups and sorted items
        HashMap<Integer, List<Integer>> groupedItems = new HashMap<>();
        for (Integer item: sortedItems) {
            groupedItems.computeIfAbsent(group[item], key -> new ArrayList()).add(item);
        }

        int[] res = new int[n];
        int resIndex = 0;
        for (Integer groupId: sortedGroups) {
            List<Integer> itemsFromGroup = groupedItems.getOrDefault(groupId, new ArrayList<>());
            for (Integer item: itemsFromGroup) res[resIndex++] = item;
        }

        return res;
    }

    List<Integer> tpSort(List<Integer>[] adjacency, int[] indegrees) {
        List<Integer> res = new ArrayList<>();
        Queue<Integer> queue = new LinkedList<>();
        int n = indegrees.length;

        for (int i = 0; i < n; i++) {
            if (indegrees[i] == 0) queue.offer(i);
        }

        while (!queue.isEmpty()) {
            Integer item = queue.poll();
            res.add(item);
            for (int next : adjacency[item]) {
                indegrees[next]--;
                if (indegrees[next] == 0) queue.offer(next);
            }
        }

        return res.size() == n ? res : new ArrayList<>();
    }
}
```

#### [816 · Traveling Salesman Problem - LintCode](https://www.lintcode.com/problem/816/)

NP Problem, shortest path while traversing all nodes

##### 1. Permutation Style DFS 

A correct solution but will run into TLS because of low efficiency.

```python
class Result:
    def __init__(self):
        # can contain traveling path if needed
        self.min_cost = float("inf")

class Solution:
    def minCost(self, n, roads):
        result = Result()
        roads_costs = self.build_graph(roads, n)
        self.dfs(1, n, set([1]), 0, roads_costs, result)
        return result.min_cost

    def build_graph(self, roads, cities_count):
        graph = {}
        # A -> B : cost
        # initialization
        for city_from in range(1, cities_count + 1):
            graph[city_from] = {}
            for city_to in range(1, cities_count + 1):
                graph[city_from][city_to] = float("inf")
        
        # assign real costs
        for city_a, city_b, cost in roads:
            graph[city_a][city_b] = min(graph[city_a][city_b], cost)
            graph[city_b][city_a] = min(graph[city_b][city_a], cost)
        
        return graph


    def dfs(self, city, cities_count, visited, cost, graph, result):
        # stops when all cities are visited
        if len(visited) == cities_count:
            result.min_cost = min(result.min_cost, cost)
            return

        for next_city in graph[city].keys():
            if next_city in visited:
                continue
            
            visited.add(next_city)
            self.dfs(next_city, cities_count, visited, cost + graph[city][next_city], graph, result)
            visited.remove(next_city)
```

##### 2. Optimal Prunning
```python
class Result:
    def __init__(self):
        self.min_cost = float("inf")

class Solution:
    def minCost(self, n, roads):
        result = Result()
        roads_costs = self.build_graph(roads, n)
        self.dfs(1, n, set([1]), [1], 0, roads_costs, result)
        return result.min_cost

    def build_graph(self, roads, cities_count):
        graph = {}
        # A -> B : cost
        # initialization
        for city_from in range(1, cities_count + 1):
            graph[city_from] = {}
            for city_to in range(1, cities_count + 1):
                graph[city_from][city_to] = float("inf")
        
        # assign real costs
        for city_a, city_b, cost in roads:
            graph[city_a][city_b] = min(graph[city_a][city_b], cost)
            graph[city_b][city_a] = min(graph[city_b][city_a], cost)
        
        return graph


    def dfs(self, city, cities_count, visited, path, cost, graph, result):
        if len(visited) == cities_count:
            result.min_cost = min(result.min_cost, cost)
            return

        for next_city in graph[city].keys():
            # there is no need to stay put
            if next_city in visited:
                continue
            
            if self.has_shorter_path(path, graph, next_city):
                continue
            
            visited.add(next_city)
            path.append(next_city)
            self.dfs(next_city, cities_count, visited, path, cost + graph[city][next_city], graph, result)
            path.pop()
            visited.remove(next_city)

    def has_shorter_path(self, path, graph, new_city):
        # starting point is fixed so start from the second city
        for i in range(1, len(path)):
            end_city = path[-1]
            current_city = path[i]
            previous_city = path[i - 1]
            natrual_path = graph[previous_city][current_city] + graph[end_city][new_city]
            try_path = graph[previous_city][end_city] + graph[current_city][new_city]
            if try_path < natrual_path:
                return True
        return False
```


##### 3. State Compression Dynamic Programming (DP)**

TIme Complexity: $O(2^n*n)$

Say the salesman start from 1 and arrives at 4 and along the way we pass by 2 and 3, so he has the routes below:

1 --> 2 --> 3 --> 4 or 1 --> 3 --> 2 --> 4

If start and end are fixed, the path that has a lower cost is the best route for now and  it can be taken as the locally optimal solution. The process could repeat itself until the end. The problem now is converted into a combination problem.

$dp[start, b, c, ... , end][end]$ This matrix stored the min cost when the salesman start and then pass by b, c... and arrives at end.

examples: 

$dp[1, 2][2] = dp[1, 1] + costs[1][2]$: the lowest cost from 1 to 2 is the lowest cost from 1 to 1 plus the cost from 1 to 2

$dp[1,2,3,4][4] = min(dp[1, 2, 3][3] + costs[3][4] , dp[1,2,3][2] + costs[2][4])$ :  

- 1 --> 2 --> 3 --> 4 the cost starting from 1, passing by 2 and 3, arriving at 3 plus the cost from 3 to 4
- 1 --> 3 --> 2 -- >4 the cost starting from 1, passing by 2 and 3, arriving at 2 (i. e. from 3 to 2) plus the cost from 2 to 4

But an array can not be used as indexing so it is represented by a sequence of binary numbers. For example, 1 --> 3 --> 5 is 10101 and $2^0 + 2^2 + 2^4 = 21$.  

```python
class Solution:
    def minCost(self, n, roads):
        # costs[a][b] means how much is the cost going from a to b
        costs = self.build_graph(roads, n)
        routes_count = 1 << n # 2 ^ n combinations
        dp = [
            [float('inf')] * (n + 1)   # n + 1 possible ends (first city is 1 not 0 so here is n + 1 instead of n)
            for _ in range(routes_count) # 2 ^ n - 1 possible routes
        ]
        dp[1][1] = 0
        for route in range(routes_count):
            # start is fixed as 1 so at least the salesman arrives at 2
            for end in range(2, n + 1):
                compressed_end = 1 << (end - 1)
                # AND: only both
                # skip if end is not in the route
                if route & compressed_end == 0:
                    continue
                # XOR: either one but not both
                # remove end from the route 
                previous_route = route ^ compressed_end
                for last_passed in range(1, n + 1):
                    compressed_last_passed = 1 << (last_passed - 1)
                    # skip if l
                    if previous_route & compressed_last_passed == 0:
                        continue
                    dp[route][end] = min(dp[route][end], dp[previous_route][last_passed] + costs[last_passed][end])
        return min(dp[routes_count - 1])
        
    def build_graph(self, roads, cities_count):
        graph = {}
        # A -> B : cost
        # initialization
        for city_from in range(1, cities_count + 1):
            graph[city_from] = {}
            for city_to in range(1, cities_count + 1):
                graph[city_from][city_to] = float("inf")
        
        # assign real costs
        for city_a, city_b, cost in roads:
            graph[city_a][city_b] = min(graph[city_a][city_b], cost)
            graph[city_b][city_a] = min(graph[city_b][city_a], cost)
        
        return graph
```

##### 4. Randomization/Genetic/Simulated Annealing Algorithm

I don't understand how it works so leave this space empty for now.

#### [17. Letter Combinations of a Phone Number](https://https://leetcode.com/problems/letter-combinations-of-a-phone-number/) <span style="color:orange">Medium</span>
Combination, DFS
e.g. 23 => "abc", "def"
              ""
0        a      b        c
1  ad ae af  bd be bf cd ce cf
```java
class Solution {
    public static final String[] KEYBOARD = {
        "",  // 0
        "",  // 1 
        "abc",
        "def",
        "ghi",
        "jkl",
        "mno",
        "pqrs",
        "tuv",
        "wxyz"
    };

    public List<String> letterCombinations(String digits) {
        List<String> combinations = new ArrayList<String>();
        if (digits.length() == 0 || digits == null) {
            return combinations;
        }

        dfs(digits, 0, "", combinations);
        return combinations;
    }

    public void dfs(String digits, int index, String combination, List<String> combinations) {
        if (index == digits.length()) {
            combinations.add(combination);
            return;
        }

        int digit = digits.charAt(index) - '0';
        String letters = KEYBOARD[digit];
        for (int i = 0; i < letters.length(); i++) {
            // backtrack implicitly
            dfs(digits, index + 1, combination + letters.charAt(i), combinations);
        }
    }
}
```

#### [79. Word Search](https://leetcode-cn.com/problems/word-search/) <span style="color:orange">Medium</span> 
```python
class Solution:
    DIRECTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    def exist(self, board: List[List[str]], word: str) -> bool:
        if not board or not board[0]:
            return []
        
        # early exit
        # if the board has not enough characters, the word cannot be found
        board_counter = self.count_board(board)
        if not self.possible_on_board(word, board_counter):
                return False

        found = False
        for i in range(len(board)):
            for j in range(len(board[0])):
                if self.dfs(word, 0, board, (i, j), set()):
                    found = True
                    break
            if found:
                break
        return True if found else False
    
    def count_board(self, board):
        counter = collections.Counter()
        for i in range(len(board)):
            for j in range(len(board[0])):
                counter[board[i][j]] += 1
        return counter
    
    def possible_on_board(self, word, board_counter):
        word_counter = collections.Counter(word)
        for char in word_counter:
            if word_counter[char] > board_counter[char]:
                return False
        return True

    def dfs(self, word, index, board, position, visited):
        if index == len(word):
            return True

        # out of board boundary
        if not self.on_board(position, board):
            return False
        # already visited
        if position in visited:
            return False

        x, y = position
        # no match
        if board[x][y] != word[index]:
            return False

        visited.add(position)
        for _x, _y in self.DIRECTIONS:
            if self.dfs(word, index + 1, board, (x + _x, y + _y),  visited):
                return True
        visited.remove(position)

    def on_board(self, position, board):
        return 0 <= position[0] < len(board) and 0 <= position[1] < len(board[0])
```

#### [212. Word Search II](https://leetcode-cn.com/problems/word-search-ii/) <span style="color:red">Hard</span> 
If Path Exists, DFS
```python
class Solution:
    # up, down, left, right
    DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        if not board or not board[0]:
            return []
        words = set(words)
        answers = [] 
        board_counter = self.count_board(board)
        for word in words:
            if not self.possible_on_board(word, board_counter):
                continue
            
            found = False
            for i in range(len(board)):
                for j in range(len(board[0])):
                    if self.search(word, 0, board, (i, j), set()):
                        answers.append(word)
                        found = True
                        # break from inner loop
                        break
                # break from outer loop
                if found:
                    break
        return answers

    # count how many times each letter is used   
    def count_board(self, board):
        counter = collections.Counter()
        for i in range(len(board)):
            for j in range(len(board[0])):
                counter[board[i][j]] += 1
        return counter
    
    # if not enough letters exist on the board, the word cannot be found
    def possible_on_board(self, word, board_counter):
        word_counter = collections.Counter(word)
        for char in word_counter:
            if word_counter[char] > board_counter[char]:
                return False
        return True
    
    def search(self, word, index, board, position, visited):
        # successful exit: every char is matched and the word is found
        if index == len(word):
            return True

        if position in visited:
            return False
        
        # pruning: stop when moving outside the board
        if not self.on_board(position, board):
            return False 
        
        if board[position[0]][position[1]] == word[index]:
            visited.add(position)
            index += 1
            for _x, _y in self.DIRECTIONS:
                if self.search(word, index, board, (position[0] + _x, position[1] + _y), visited):
                    return True
            visited.remove(position)
        return False
    
    def on_board(self, position, board):
        return 0 <= position[0] < len(board) and 0 <= position[1] < len(board[0])
```
another way of doing dfs, but this method will fail due to TLE. It runs too slow even after pruning.
```python
class Solution:
    # up, down, left, right
    DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # another way of doing dfs, but this method will fail due to TLE
    def findWords2(self, board: List[List[str]], words: List[str]) -> List[str]:
        if not board or not board[0]:
            return []

        word_set = set(words)
        prefix_set = self.get_prefix_set(words)
        result_set = set()
        for x in range(len(board)):
            for y in range(len(board[0])):
                self.dfs(x, y, board[x][y], board, word_set, prefix_set, set([(x, y)]), result_set)
        return list(result_set)


    def get_prefix_set(self, words):
        prefix_set = set()
        for word in words:
            for i in range(len(word)):
                prefix_set.add(word[:i + 1])
        return prefix_set

    def dfs(self, x, y, word, board, word_set, prefix_set, visited,  result_set):
        # pruning: early exit
        if not self.is_prefix(word, prefix_set):
            return

        # catch valid words
        if word in word_set:
            if word in prefix_set:
                prefix_set.remove(word)
            result_set.add(word)
        
        # try all four directions
        for delta_x, delta_y in self.DIRECTIONS:
            _x = x + delta_x
            _y = y + delta_y

            # not on board? skip the move 
            if not self.on_board((_x, _y), board):
                continue
            # already visited? skip the move
            if (_x, _y) in visited:
                continue
            
            visited.add((_x, _y))
            self.dfs(_x, _y, word + board[_x][_y], board, word_set, prefix_set, visited, result_set)
            visited.remove((_x, _y))
        
    def is_prefix(self, word, prefix_set):
        if word in prefix_set:
            return True
        for prefix in prefix_set:
            if word in prefix:
                return True
        return False
```

#### [433. Minimum Genetic Mutation](https://leetcode.com/problems/minimum-genetic-mutation/) Medium
```java
class Solution {
    private static String GENES = "ACGT";
    
    public int minMutation(String start, String end, String[] bank) {
        int steps = 0;
        
        Queue<String> q = new LinkedList<>();
        HashSet<String> visited = new HashSet<>();
        List<String> repo = Arrays.asList(bank);
        q.add(start);
        visited.add(start);
        
        while (!q.isEmpty()) {
            // candidates queue will grow later
            // so q.size() is actually dynamic
            // if i is initialized as 0 and condition is set to i < q.size()
            // this will turn out not as expected beacuse size will get bigger
            // but initialization statement will only be executed once
            // so i can be initialized as q.size()
            for (int i = q.size(); i > 0 ; i--) {
                String s = q.poll();
                if (s.equals(end)) return steps;
                mutate(s, q, repo, visited);
            }
            steps++;
        }
        
        return -1;
    }
    
    void mutate(String source, Queue<String> candidates, List<String> repo, HashSet<String> visited) {
        char[] chars = source.toCharArray();
        String candidate;
        for (int i = 0; i < chars.length; i++) {
            char original = chars[i];
            
            for (int j = 0; j < GENES.length(); j++) {
                if (original == GENES.charAt(j)) continue;
                chars[i] = GENES.charAt(j);
                candidate = new String(chars);
                if (!visited.contains(candidate) && repo.contains(candidate)) {
                    candidates.add(candidate);
                    visited.add(candidate);
                }
            }
            
            // revert it back
            chars[i] = original;
        }
    }
}
```

#### [947. Most Stones Removed with Same Row or Column](https://leetcode.com/problems/most-stones-removed-with-same-row-or-column/) Medium
DFS
```java
class Solution {
    public int removeStones(int[][] stones) {
        HashSet<Pair<Integer, Integer>> visited = new HashSet<>();
        int res = stones.length;
        for (int i = 0; i < stones.length; i++) {
            if (!beenVisited(visited, stones[i][0], stones[i][1])) {
                dfs(stones, visited, stones[i][0], stones[i][1]);
                // the best possible shot is that still 1 stone is left
                // beczuse you always need one stone to remove another stone
                // note that the result is decremented in the main loop, not in the recursive dfs loop
                // if one point leads to the rest of points, that's the best shot
                res--;
            }
        }
        return res;
    }

    void dfs(int[][] stones, HashSet<Pair<Integer, Integer>> visited, int x, int y) {
        visited.add(getCoordinate(x, y));
        for (int i = 0; i < stones.length; i++) {
            if (!beenVisited(visited, stones[i][0], stones[i][1])) {
                if (x == stones[i][0] || y == stones[i][1]) 
                    dfs(stones, visited, stones[i][0], stones[i][1]);
            }
        }
    }

    boolean beenVisited(HashSet<Pair<Integer, Integer>> visited, int x, int y) {
        return visited.contains(getCoordinate(x, y));
    }

    Pair<Integer, Integer> getCoordinate(int x, int y) {
        return new Pair<Integer, Integer>(x, y);
    }
}
```
Union Find
```java
```

#### [1584. Min Cost to Connect All Points](https://leetcode.com/problems/min-cost-to-connect-all-points/description/) Medium
union find
connect all points -> graph -> find the minimum spanning tree and calculate the total weight of the tree
```java
class Solution {
    int[] parent;
    int[] size;

    public int minCostConnectPoints(int[][] points) {
        // reprent a graph using an edge list, with first and second element denoting verticex and the third element denoting the manhattan distance
        List<int[]> edges = new ArrayList<>();
        int n = points.length;
        parent = new int[n];
        size = new int[n];
        for (int i = 0; i < n; i++) {
            parent[i] = i;
            size[i] = 1;
            for (int j = 0; j < i; j++) {
                edges.add(new int[] {
                    i, j, getDistance(points[i], points[j])
                });
            }
        }
        // sort all edges by the weight, i.e. the distance
        Collections.sort(edges, (a, b) -> (a[2] - b[2]));
        
        int res = 0;
        // for the minimum spanning tree, there's only n - 1 edges
        for (int i = 0, j = 0; i < edges.size() && j < n - 1; i++) {
            int[] edge = edges.get(i);
            if (merge(edge[0], edge[1])) {
                j++;
                res += edge[2];
            }
        }
        return res;
    }

    int getDistance(int[] p, int[] q) {
        return Math.abs(p[0] - q[0]) + Math.abs(p[1] - q[1]);
    }

    int find(int x) {
        if (parent[x] != x) {
            parent[x] = find(parent[x]);
        }
        return parent[x];
    }

    // merge two sets
    boolean merge(int p, int q) {
        p = find(p);
        q = find(q);
        // already connected
        if (p == q) {
            return false;
        }

        if (size[p] > size[q]) {
            parent[q] = p;
            size[p] += size[q];
        } else {
            parent[q] = p;
            size[q] += size[p];
        }
        return true;
    }
```

#### [1489. Find Critical and Pseudo-Critical Edges in Minimum Spanning Tree](https://leetcode.com/problems/find-critical-and-pseudo-critical-edges-in-minimum-spanning-tree/description/) Hard
```java
class Solution {
    private int[] parent;
    private int[] size;
    private int[][] edges;
    private static int INF = 1000000;

    public List<List<Integer>> findCriticalAndPseudoCriticalEdges(int n, int[][] inputEdges) {
        initialize(n);
        edges = new int[inputEdges.length][4];
        for (int i = 0; i < edges.length; i++) {
            for (int j = 0; j < 3; j++) {
                edges[i][j] = inputEdges[i][j];
            }
            edges[i][3] = i;
        }
        Arrays.sort(edges, (a, b) -> (a[2] - b[2]));
        // default mst, ignore no edges
        final int defaultMstWeight = mst(n, -1);
        List<Integer> criticalEdges = new ArrayList<>();
        List<Integer> noncriticalEdges = new ArrayList<>();
        List<List<Integer>> resEdges = new ArrayList<>();

        for (int i = 0; i < edges.length; i++) {
            initialize(n);
            if (mst(n, i) > defaultMstWeight) {
                criticalEdges.add(edges[i][3]); 
            } else {
                initialize(n);
                merge(edges[i][0], edges[i][1]);
                if (mst(n - 1, i) + edges[i][2] == defaultMstWeight) {
                    noncriticalEdges.add(edges[i][3]);
                }
            }
        }

        resEdges.add(criticalEdges);
        resEdges.add(noncriticalEdges);
        return resEdges;
    }   

    void initialize(int n) {
        parent = new int[n];
        size = new int[n];
        for (int i = 0; i < n; i++) {
            parent[i] = i;
            size[i] = 1;
        }
    }

    int find(int x) {
        if (parent[x] != x) {
            parent[x] = find(parent[x]);
        }
        return parent[x];
    }

    boolean merge(int p, int q) {
        p = find(p);
        q = find(q);
        if (p == q) {
            return false;
        }

        if (size[p] > size[q]) {
            parent[q] = p;
            size[p] += size[q];
        } else {
            parent[p] = q;
            size[q] += size[p];
        }
        return true;
    }

    int mst(int n, int ignoredEdgeIndex) {
        int res = 0, mstEdges = 0;
        for (int i = 0; mstEdges < n - 1 && i < edges.length; i++) {
            if (i == ignoredEdgeIndex) continue;

            if (merge(edges[i][0], edges[i][1])) {
                mstEdges++;
                res += edges[i][2];
            }
        }

        if (mstEdges == n - 1) {
            return res;
        }
        return INF;
    }
}
```

#### [1631. Path With Minimum Effort](https://leetcode.com/problems/path-with-minimum-effort/description/) Medium
shortest path
```java
class Solution {
    private int[][] moves = {
        {-1, 0}, {1, 0},
        {0, -1}, {0, 1}
    };
    private int m, n;

    public int minimumEffortPath(int[][] heights) {
        m = heights.length;
        n = heights[0].length;
        boolean[][] visited = new boolean[m][n];
        // {cost, x, y}
        PriorityQueue<int[]> q = new PriorityQueue<>((a, b) -> Arrays.compare(a, b));
        
        // add starting point to queue: starting from (0, 0) and cost is zero  
        for (q.add(new int[] {0, 0, 0}); !q.isEmpty(); ) {
            final int[] item = q.poll();
            int x = item[1], y = item[2], cost = item[0];
            
            // already marked, should be skipped
            if (visited[x][y]) {
                continue;
            }

            visited[x][y] = true;
            
            // reach the ending point
            if (x == m - 1 && y == n - 1) {
                return item[0];
            }

            for (int i = 0; i < moves.length; i++) {
                int newx = moves[i][0] + x, newy = moves[i][1] + y;
                
                if (!isValid(newx, newy)) {
                    continue;
                }

                q.add(
                    new int[] {
                        Math.max(cost, Math.abs(heights[x][y] - heights[newx][newy])),
                        newx,
                        newy
                    }
                );
            }   
        }

        return -1;
    }

    private boolean isValid(int x, int y) {
        return (x >= 0 && x < m) && (y >= 0 && y < n);
    }
}
```

#### [1368. Minimum Cost to Make at Least One Valid Path in a Grid](https://leetcode.com/problems/minimum-cost-to-make-at-least-one-valid-path-in-a-grid/) Hard
shortest path 
```java
class Solution {
    private int[][] moves = {
        {0, 1}, {0, -1},
        {1, 0}, {-1, 0}
    };
    int m, n;
    
    public int minCost(int[][] grid) {
        m = grid.length;
        n = grid[0].length;
        int[][] costs = new int[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                costs[i][j] = -1;
            }
        }

        PriorityQueue<int[]> q = new PriorityQueue<>((a, b) -> Arrays.compare(a, b));

        for (q.add(new int[] {0, 0, 0}); costs[m - 1][n - 1] < 0;) {
            int[] item = q.poll();
            int cost = item[0], x = item[1], y = item[2];
            if (costs[x][y] > -1) {
                continue;
            }

            costs[x][y] = cost;
            for (int i = 0; i < moves.length; i++) {
                int newx = x + moves[i][0], newy = y + moves[i][1];
                if (!isValid(newx, newy)) {
                    continue;
                }

                int directionId = i + 1;
                q.add(new int[] {
                    // no cost will be added if direction marker from the previous postion is  
                    cost + (grid[x][y] == directionId ? 0 : 1),
                    newx,
                    newy
                });
            }
        }

        return costs[m - 1][n - 1];
    }

    private boolean isValid(int x, int y) {
        return (x >= 0 && x < m) && (y >= 0 && y < n);
    }
}
```