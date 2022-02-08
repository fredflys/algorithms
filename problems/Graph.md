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


#### [212. Word Search II](https://leetcode-cn.com/problems/word-search-ii/) <span style="color:red">Hard</span> 
Permutation DFS
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