## Graph

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

## 