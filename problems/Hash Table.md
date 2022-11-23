## Hash Table

#### [242. Valid Anagram](https://leetcode-cn.com/problems/valid-anagram/) <span style="color:green">Easy</span>

Easy! FIgured it out right away! Now I see I do make progress.

```java
class Solution {
    public boolean isAnagram(String s, String t) {
        // if lengths are equal, they cannot be anagrams
        // change not-equal to less-than and you have the solution to problem 383
        if(s.length() != t.length())
            return false;
        
        int[] count = new int[26];
        for(int i = 0; i < s.length(); i++){
            count[s.charAt(i) - 'a']++;
        }
        for(int j = 0; j < t.length(); j++){
            // improvement: there is no need to walk through count another time. Comparing while deducting.
            if(--count[t.charAt(j) - 'a'] < 0)
                return false;
        }
       
        return true;
    }
}
```

```python
	return sorted(s) == sorted(t)
```

#### [383. Ransom Note](https://leetcode-cn.com/problems/ransom-note/) <span style="color:green">Easy</span> 

Solution almost identical as that to problem 242. 

```python
# another solution
a = Counter(ransomNote) 
b = Counter(magazine)
# get the intersection set
return (a & b) == a
```

#### [49. Group Anagrams](https://leetcode-cn.com/problems/group-anagrams/) <span style="color:orange">Medium</span>

```python
# Passs, but running is slow
class Solution(object):
    def groupAnagrams(self, strs):
        """
        :type strs: List[str]
        :rtype: List[List[str]]
        """
        sorts = []
        result = []
        for s in strs:
            _ = sorted(s)
            if _ not in sorts:
                sorts.append(_)
                result.append([])
            result[sorts.index(_)].append(s)
        return result
        
```

```java
// java 8 new API stream， used to organize collections
class Solution {
    public List<List<String>> groupAnagrams(String[] strs) {
        return new ArrayList<>(Arrays.stream(strs)
            .collect(
                Collectors.groupingBy(str -> {
                    char[] array = str.toCharArray();
                    Arrays.sort(array);
                    return new String(array);
                    })
            )
            .values()
        );
    }
}

```

```java
// hashmap
class Solution{
    public List<List<String>> groupAnagrams(String[] strs) {
        int length = strs.length;
        // use hash map to group results
        Map<String, List<String>> map = new HashMap<>();
        for(String str: strs){
            char[] chars = str.toCharArray();
            Arrays.sort(chars);
            // used sorted string as the key
            String key = String.valueOf(chars);
            // if not contained in the map, create a empty array to as a holder
            if(!map.containsKey(key)){
                map.put(key, new ArrayList<>());
            }
            map.get(key).add(str);
        }
        return new ArrayList<>(map.values());
    }
}
```

#### [349. Intersection of Two Arrays](https://leetcode-cn.com/problems/intersection-of-two-arrays/) <span style="color:green">Easy</span>

```python
class Solution(object):
    def intersection(self, nums1, nums2):
        map = {}
        result = []
        for num in nums1:
            if map.get(num):
                map[num] += 1
            else:
                map[num] = 1
        
        for num in nums2:
            if map.get(num) and num not in result:
                result.append(num)
        
        return result
```

#### [350. Intersection of Two Arrays II]( https://leetcode-cn.com/problems/intersection-of-two-arrays-ii/) <span style="color:green">Easy</span>

```python
class Solution(object):
    def intersect(self, nums1, nums2):
        map1 = {}
        map2 = {}
        result = []
        for num in nums1:
            if map1.get(num):
                map1[num] += 1
            else:
                map1[num] = 1

        for num in nums2:
            if map2.get(num):
                map2[num] += 1
            else:
                map2[num] = 1
        
        for key in map1.keys():
            if key in map2.keys():
                result = result + [key] * (map1[key] if map1[key] < map2[key] else map2[key])

        return result
```

#### [202. Happy Number](https://leetcode-cn.com/problems/happy-number/) <span style="color:green">Easy</span>

```python
class Solution(object):
    def isHappy(self, n):
        sums = []
        while n != 1:
            n = self.convert(n)
            if n in sums:
                return False
            sums.append(n)
        return True

    def convert(self, n):
        sum = 0
        for digit in str(n):
            digit = int(digit)
            sum += digit * digit
        return sum
```

#### [1. Two Sum](https://leetcode-cn.com/problems/two-sum/) <span style="color:green">Easy</span> **

```python
class Solution(object):
    def twoSum(self, nums, target):
        for i in range(len(nums)):
            other = target - nums[i]
            if other in nums and i != nums.index(other):
                return [i, nums.index(other)]
        return []
```

```python
class Solution(object):
    def twoSum(self, nums, target):
        hashset = set()
        for num in nums:
            if target - num in hashset:
                return num, target - number
           	hashset.add(num)
        return []
```

```python
# two-pointers
class Solution(object):
    def twoSum(self, nums, target):
        if not nums:
            return []
        
        # save index and number at the same time
        # because original index will be lost after sort
        # e.g. [3,3] 6 ans: [0,1] instead of [0,0] 
        num_tuples = [
            (num, index)
            for index, num in enumerate(nums)
        ]
        num_tuples.sort()
        left, right = 0, len(nums) - 1
        while left < right:
            if num_tuples[left][0] + num_tuples[right][0] > target:
                right -= 1
            elif num_tuples[left][0] + num_tuples[right][0] < target:
                left += 1
            else:
                return num_tuples[left][1], num_tuples[right][1]
        return []
```



#### [454. 4Sum II](https://leetcode-cn.com/problems/4sum-ii/) <span style="color:orange">Medium</span>

```python
# official solution
class Solution(object):
    def fourSumCount(self, nums1, nums2, nums3, nums4):
        counter = collections.Counter(i + j for i in nums1 for j in nums2)
        ans = 0
        for i in nums3:
            for j in nums4:
                if -(i+j) in counter:
                    ans += counter[-(i+j)]
        return ans
```

```java
class Solution {
    public int fourSumCount(int[] A, int[] B, int[] C, int[] D) {
        int ans = 0;
		Map<Integer, Integer> counter = new HashMap<Integer, Integer>();
        for(int a: A){
            for(int b: B){
                counter.put(a + b, counter.getOrDefault(a + b, 0) + 1)
            }
        }
        for(int c: C){
            for(int d: D){
                if(counter.containsKey(-c-d)){
                    ans += counter.get(-c-d);
                }
            }
        }
        return ans;
    }
}
```

#### [15. 3Sum](https://leetcode-cn.com/problems/3sum/) <span style="color:orange">Medium</span>

```python
# my solution not accepted because of low efficiency
class Solution(object):
    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        counter = collections.Counter()
        result = []
        for i in range(len(nums) - 1):
            for j in range(i + 1,len(nums)):
                twosum = nums[i] + nums[j] 
                if -twosum in nums and nums.index(-twosum) != i and nums.index(-twosum) != j:
                    candidate = sorted([nums[i], nums[j], - nums[i] - nums[j]])
                    if candidate not in result:
                        result.append(candidate)
        return result
```

```javascript
// two-pointer after sort
var threeSum = function(nums) {
  if (nums.length < 3) {
    return [];
  }
  // 从小到大排序
  const arr = nums.sort((a,b) => a-b);
  // 最小值大于 0 或者 最大值小于 0，说明没有无效答案
  if (arr[0] > 0 || arr[arr.length - 1] < 0) {
    return [];
  }
  const n = arr.length;
  const res = [];
  for (let i = 0; i < n; i ++) {
    // 如果当前值大于 0，和右侧的值再怎么加也不会等于 0，所以直接退出
    if (nums[i] > 0)
      return res;
	  
    // 当前循环的值和上次循环的一样，就跳过，避免重复值
    if (i > 0 && nums[i] === nums[i - 1])
      continue;

    // 双指针
    let l = i + 1;
    let r = n - 1;
    while(l < r) {
      const _sum = nums[i] + nums[l] + nums[r];
      if (_sum > 0)
        r --;
      if (_sum < 0) 
        l ++;
      if (_sum === 0) {
        res.push([nums[i], nums[l], nums[r]]);
        // 跳过重复值
        while(l < r && nums[l] === nums[l + 1])
          l ++;
        // 同上
        while(l < r && nums[r] === nums[r - 1])
          r --;
        l ++;
        r --;
      }
    }
  }
  return res;
};
```

#### [18. 4Sum](https://leetcode-cn.com/problems/4sum/) <span style="color:orange">Medium</span>

```python
# two pointer
class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        nums.sort()
        n = len(nums)
        results = []

        if not nums or n < 4:
            return results
		# stops when c is 1 from the last
        for a in range(n - 3):
            if a > 0 and nums[a] == nums[a - 1]:
                continue
            for b in range(a + 1, n - 2):
                if b > a + 1 and nums[b] == nums[b - 1]:
                    continue
                c = b + 1
                d = n - 1
                while c < d:
                    sum = nums[a]+nums[b]+nums[c]+nums[d]
                    if sum < target:
                        c += 1
                    elif sum > target:
                        d -= 1
                    else:
                        results.append([nums[a], nums[b], nums[c], nums[d]])
                        # avoid duplicates
                        while c < d and nums[c + 1] == nums[c]:
                            c += 1
                        while c < d and nums[d - 1] == nums[d]:
                            d -= 1
                        c += 1
                        d -= 1
        return results
```

```java
// backtrack and dfs
class Solution {
    List<List<Integer>> ans = new ArrayList<>();
    List<Integer> list = new ArrayList<>();
    int cur = 0;
    public List<List<Integer>> fourSum(int[] nums, int target) {
        Arrays.sort(nums);
        dfs(nums, target,0);
        return ans;
    }
    void dfs(int[] nums, int target, int begin){
		// condition where recursion stops
        if(list.size() == 4){
            if(cur == target){
                ans.add(new ArrayList<>(list));
            }
            return;
        }

        for(int i = begin; i < nums.length; i++ ){
            // cutting branches
			// remaining candidates are not enough
			if(nums.length - i  < 4 - list.size())
				return;
			// current item is equal to the previous one
            if(begin != i && nums[i - 1] == nums[i])
				continue;
			// current sum plus the least possible combination of remaining values is still larger than target   
            if(i < nums.length - 1 && cur + nums[i] > target - (3 - list.size()) * nums[i + 1])		
                return;
			// current sum plus the largest possible combination of remaining values is still less then target
            if(i < nums.length - 1 && cur + nums[i] < target - (3 - list.size()) * nums[nums.length - 1])
				continue;
			
            cur += nums[i];
            list.add(nums[i]);
            dfs(nums, target, i + 1);
            list.remove(list.size() - 1);
            cur -= nums[i];
        }
    }
}
```

#### [LintCode 685 · First Unique Number in Data Stream](https://www.lintcode.com/problem/685/) Medium
```java
public class Solution {
    /**
     * @param nums: a continuous stream of numbers
     * @param number: a number
     * @return: returns the first unique number
     */
    public int firstUniqueNumber(int[] nums, int number) {
        if (nums == null || nums.length == 0) {
            return -1;
        }

        Map<Integer, Boolean> numUniqueMap = new HashMap<>();
        for (int num: nums) {
            // unique: not in map  
            numUniqueMap.put(num, !numUniqueMap.containsKey(num));
            if (num == number) {
                break;
            }
        }

        if (!numUniqueMap.containsKey(number)) {
            return -1;
        }

        // as long as the map contains the number, there must be an answer.
        // there is no need to break when num == number
        // since hash map is unordered, loop the nums to get the first unique number
        for (int num : nums) {
            if (numUniqueMap.get(num)) {
                return num;
            }
        }

        return -1;
    }
}
```
#### [LintCode 960 · First Unique Number in Data Stream II](https://www.lintcode.com/problem/960/) Medium
```java
public class DataStream {
    private class ListNode {
        int val;
        ListNode next;
        ListNode (int val) {
            this.val = val;
        }
        ListNode (int val, ListNode next) {
            this.val = val;
            // a one-way linked node
            this.next = next;
        }
    }

    // a linked list is used to save all unique nodes
    // with a dummy node, the head can be treated just as other nodes
    private ListNode dummy;
    private ListNode tail;
    // a map is used to store a previous node to every node, which can be quite helpful when reorder the linked list 
    private Map<Integer, ListNode> numToPrevNode;
    private Set<Integer> duplicates;

    public DataStream(){
        dummy = new ListNode(0);
        tail = dummy;
        numToPrevNode = new HashMap<>();
        duplicates = new HashSet<>();
    }
    /**
     * @param num: next number in stream
     * @return: nothing
     */
    public void add(int num) {
        // only unique numbers can be added
        // so if the incoming number is a duplicate (already happened 2 twos), ignore it
        if (duplicates.contains(num)) {
            return;
        }

        // already came once, then now it's the second time, so remove it from the linked list and mark it as a duplicate
        if (numToPrevNode.containsKey(num)) {
            remove(num);
            duplicates.add(num);
            return;
        }

        // coming in for the first time
        addToTail(num);
    }

    private void remove(int num) {
        ListNode prev = numToPrevNode.get(num);
        prev.next = prev.next.next;
        numToPrevNode.remove(num);
 
        /**
        e.g.
           prev   tail 
        ... 7 --> 10 --> null
        after removal:
           prev
        ... 7 --> null
        **/
        if (prev.next == null) {
            tail = prev;
        } else {
            // update numToPrev map
            numToPrevNode.put(prev.next.val, prev);
        }
    }

    /**
     * @return: the first unique number in stream
     */
    public int firstUnique() {
        // no unique numbers or no numbers at all
        if (dummy.next == null) {
            return -1;
        }
        // the first node in the 
        return dummy.next.val;
    }

    private void addToTail(int num) {
        ListNode node = new ListNode(num);
        tail.next = node;
        numToPrevNode.put(num, tail);
        // update the tail
        tail = node;
    }
}
```

#### [380. Insert Delete GetRandom O(1)](https://leetcode.com/problems/insert-delete-getrandom-o1/) Medium
```java
import java.util.Random;

class RandomizedSet {
    ArrayList<Integer> nums;
    HashMap<Integer, Integer> numToIndex;
    Random rand;

    public RandomizedSet() {
        // store values
        nums = new ArrayList<>();
        // val -> its position in nums
        numToIndex = new HashMap<>();
        rand = new Random();
    }
    
    public boolean insert(int val) {
        boolean found = exists(val);
        if (!found) {
            nums.add(val);
            numToIndex.put(val, nums.size() - 1);
        }
        // insertion is possible only when the num is not found
        return !found;
    }
    
    public boolean remove(int val) {
        boolean found = exists(val);
        if (found) {
            int pos = numToIndex.get(val);
            if (pos != nums.size() - 1) {
                moveLastNumToPos(pos);
            }

            numToIndex.remove(val);
            nums.remove(nums.size() - 1);
        }
        
        // deletion is possible only when the num is found
        return found;
    }
    
    public int getRandom() {
        int randomIndex = rand.nextInt(nums.size());
        return nums.get(randomIndex);
    }
    
    private boolean exists(int val) {
        return numToIndex.containsKey(val);
    }
    
    private void moveLastNumToPos(int pos) {
        int lastNum = nums.get(nums.size() - 1);
        nums.set(pos, lastNum);
        numToIndex.put(lastNum, pos);
    }
}
```
#### [381. Insert Delete GetRandom O(1) - Duplicates allowed](https://leetcode.com/problems/insert-delete-getrandom-o1-duplicates-allowed/) Hard
```java
class RandomizedCollection {
    
    ArrayList<Integer> nums;
    // since duplicates are allowed, a number has multiple positions now
    HashMap<Integer, Set<Integer>> numToIndexes;
    Random rand;

    public RandomizedCollection() {
        nums = new ArrayList<>();
        numToIndexes = new HashMap<>();
        rand = new Random();
    }
    
    public boolean insert(int val) {
        boolean isMissing = missing(val);
        if (isMissing) {
            numToIndexes.put(val, new HashSet<>());
        }
        nums.add(val);
        numToIndexes.get(val).add(nums.size() - 1);
        return isMissing;
    }
    
    public boolean remove(int val) {
        if (missing(val)) {
            return false;
        }
        
        // get one position and remove it
        int position = numToIndexes.get(val).iterator().next();
        numToIndexes.get(val).remove(position);
        if (position < nums.size() - 1) {
            moveLastNumToPos(position);
        }
        
        nums.remove(nums.size() - 1);
        
        // be sure to clean up 
        if (missing(val)) {
            numToIndexes.remove(val);
        }
        return true;
    }
    
    public int getRandom() {
        int randomIndex = rand.nextInt(nums.size());
        return nums.get(randomIndex);
    }
    
    private boolean missing(int val) {
        // the num is never inserted or it was once inserted but now has only 0 occurance
        return !numToIndexes.containsKey(val) || numToIndexes.get(val).isEmpty();
    }
    
    private void moveLastNumToPos(int pos) {
        int lastNum = nums.get(nums.size() - 1);
        nums.set(pos, lastNum);
        numToIndexes.get(lastNum).remove(nums.size() - 1);
        numToIndexes.get(lastNum).add(pos);
    }
}
```
#### [146. LRU Cache](https://leetcode.com/problems/lru-cache/) Medium
```python
class LinkedNode:
    def __init__(self, key=None, value=None, next=None):
        self.key = key
        self.value = value
        self.next = next
        
class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.dummy = LinkedNode()
        self.tail = self.dummy
        self.key_to_prev = {}

    def get(self, key: int) -> int:
        if key not in self.key_to_prev:
            return -1
        
        self.move_to_tail(key)
        return self.tail.value

    def put(self, key: int, value: int) -> None:
        # update
        if key in self.key_to_prev:
            self.move_to_tail(key)
            return
        
        # add
        key_node = LinkedNode(key, value)
        self.place_on_tail(key_node)
        
        # remove first node if overloaded
        if self.size() > self.capacity:
            self.pop_head()
        
    def move_to_tail(self, key):
        # previous node
        prev = self.key_to_prev[key]
        # key node
        key_node = prev.next
        
        # key node is on the tail
        if key_node == self.tail:
            return
        
        self.link_prev_to_next(prev, key_node)
        self.place_on_tail(key_node)
    
    # remove first node
    def pop_head(self):
        head = self.dummy.next
        del self.key_to_prev[head.key]
        self.link_prev_to_next(self.dummy, head)
    
    def size(self):
        return len(self.key_to_prev)
    
    # link previous node to next node
    def link_prev_to_next(self, prev_node, current_node):
        next_node = current_node.next
        prev_node.next = next_node
        self.key_to_prev[next_node.key] = prev_node
    
    # place a node on tail
    def place_on_tail(self, node):
        self.key_to_prev[node.key] = self.tail
        self.tail.next = node
        self.tail = node
```
Ordered dict
```java
class LRUCache {
    
    Map<Integer, Integer> map = new LinkedHashMap<Integer, Integer>();
    int cap;
    
    public LRUCache(int capacity) {
        cap = capacity;
    }
    
    public int get(int key) {
        if(!map.containsKey(key)) {
            return -1;
        }
        int val = map.remove(key);
        map.put(key, val);
        return val;
    }
    
    public void put(int key, int value) {
        // update
        if(map.containsKey(key)) {
            map.remove(key);
            map.put(key, value);
            return;
        }

        // add
        map.put(key, value);
        // if overloaded
        if(map.size() > cap) {
            map.remove(map.entrySet().iterator().next().getKey());
        }
    }
}
```

## Heap
Heap is a complete binary tree.
![](https://s2.loli.net/2022/02/16/dUt4lZWLpfRXFsI.png)
![](https://s2.loli.net/2022/02/16/xhb2lm1d7T9GLFZ.png)

#### [973. K Closest Points to Origin](https://leetcode.com/problems/k-closest-points-to-origin/) Medium
max heap - reversed min heap
```python
class Solution:
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        heap = []
        results = []
        for point in points:
            dist = self.get_distance(point)
            # longest distance will now be the shortest distance
            heapq.heappush(heap, (-dist, point[0], point[1]))
            # when lenght exceeds k, then shortest(actually the furest point) distance will be popped
            if len(heap) > k:
                heapq.heappop(heap)
        while len(heap) > 0:
            _, x, y = heapq.heappop(heap)
            results.append([x, y])
            
        return results
        
    def get_distance(self, point):
        return point[0] ** 2 + point[1] ** 2
```

#### [2196. Create Binary Tree From Descriptions](https://leetcode.com/problems/create-binary-tree-from-descriptions/submissions/) Medium
```java
class Solution {
    public TreeNode createBinaryTree(int[][] descriptions) {
        Map<Integer, TreeNode> map = new HashMap<>();
        Set<Integer> children = new HashSet<>();
        buildTree(descriptions, map, children);
        int rootVal = getRootVal(descriptions, children);
        return map.getOrDefault(rootVal, null);
    }
    
    void appendChild(TreeNode parent, TreeNode child, int isLeft) {
        if (isLeft == 1) {
            parent.left = child;
        } else {
            parent.right = child;
        }
    }
    
    TreeNode buildNode(int val, Map<Integer, TreeNode> map) {
        TreeNode node = map.getOrDefault(val, new TreeNode(val));
        map.put(val, node);
        return node;
    }

    void buildTree(int[][] descriptions, Map<Integer, TreeNode> map, Set<Integer> children) {
        for (int[] desc: descriptions) {
            TreeNode parent = buildNode(desc[0], map);
            TreeNode child = buildNode(desc[1], map);
            appendChild(parent, child, desc[2]);
            // collect children values
            children.add(desc[1]);
        }
    }

    int getRootVal(int[][] descriptions, Set<Integer> children) {
        int rootVal = -1;
        for (int[] desc: descriptions) {
            // if the desc value is never found in children, this means it is the root
            if (!children.contains(desc[0])) {
                rootVal = desc[0];
                return rootVal;
            }
        }
        return -1;
    }
}
```

#### [13. Roman to Integer](https://leetcode.com/problems/roman-to-integer/) Easy
```java
class Solution {
    public int romanToInt(String s) {
        Map<Character, Integer> map = new HashMap<>();
        map.put('I', 1);
        map.put('V', 5);
        map.put('X', 10);
        map.put('L', 50);
        map.put('C', 100);
        map.put('D', 500);
        map.put('M', 1000);
        
        int i = s.length() - 1;
        int current = 0;
        int prev = 0;
        int sum = 0;
        while (i >= 0) {
            current = map.get(s.charAt(i));
            if (current < prev) {
                sum -= current;
            } else {
                sum += current;
            }
            prev = current;
            i--;
        }

        return sum;
    }
}
```

#### [12. Integer to Roman](https://leetcode.com/problems/integer-to-roman/) Medium
Very intuitive solution. Depsite its ugly look, it runs pretty fast.
```java
class Solution {
    public String intToRoman(int num) {
        StringBuilder res = new StringBuilder();
        while (num != 0) {
            if (num >= 1000) {
                res.append("M");
                num -= 1000;
                continue;
            }
            
            if (num >= 900) {
                res.append("CM");
                num -= 900;
                continue;
            }
            
            if (num >= 500) {
                res.append("D");
                num -= 500;
                continue;
            }
            
            
            if (num >= 400) {
                res.append("CD");
                num -= 400;
                continue;
            }
            
            if (num >= 100) {
                res.append("C");
                num -= 100;
                continue;
            }
            
            if (num >= 90) {
                res.append("XC");
                num -= 90;
                continue;
            }
            
            if (num >= 50) {
                res.append("L");
                num -= 50;
                continue;
            }
            
            if (num >= 40) {
                res.append("XL");
                num -= 40;
                continue;
            }
            
            if (num >= 10) {
                res.append("X");
                num -= 10;
                continue;
            }
            
            if (num >= 9) {
                res.append("IX");
                num -= 9;
                continue;
            }
            
            if (num >= 5) {
                res.append("V");
                num -= 5;
                continue;
            }
            
            if (num >= 4) {
                res.append("IV");
                num -= 4;
                continue;
            }
            
            if (num >= 1) {
                res.append("I");
                num -= 1;
                continue;
            }
        }
        
        return res.toString();
    }
}
```
Tweaked a little lit and it now looks mmuch condense.
```java
class Solution {
    public String intToRoman(int num) {
        // use two arrays instead of a hashmap here
        int[] nums = new int[]{1000,900,500,400,100,90,50,40,10,9,5,4,1};
        String[] romans = new String[]{"M","CM","D","CD","C","XC","L","XL","X","IX","V","IV","I"};
        StringBuilder res = new StringBuilder();
        for (int i = 0; i < nums.length; i++) {
            while (num >= nums[i]) {
                res.append(romans[i]);
                num -= nums[i];
            }
        }
        return res.toString();
    }
}
```

#### [170. Two Sum III - Data structure design](https://leetcode.com/problems/two-sum-iii-data-structure-design/) Easy
User HashMap to save occurances (e.g. 2,4,4,5, 4 shows up two times, if the occurance is not noted down, a target 8 can never be reached)
Typically a data structure problems involves trade-offs between different methods based upon how frequently each method is going to be used
```java
public class TwoSum {
    // number : counter
    private HashMap<Integer, Integer> counter;

    public TwoSum() {
        counter = new HashMap<Integer, Integer>();
    }

    /**
     * @param number: An integer
     * @return: nothing
     * @time: O(1)
     */
    public void add(int number) {
        counter.put(number, counter.getOrDefault(number, 0) + 1);
    }

    /**
     * @param value: An integer
     * @return: Find if there exists any pair of numbers which sum is equal to the value.
     * @time: O(n)
     */
    public boolean find(int target) {
         for (Integer num: counter.keySet()) {
             int other = target - num;
             int desiredCount = other == num ? 2 : 1;
             if (counter.getOrDefault(other, 0) >= desiredCount) {
                 return true;
             }
         }
         return false;
    }
}
```