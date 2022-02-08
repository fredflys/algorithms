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

#### [438. Find All Anagrams in a String](https://leetcode-cn.com/problems/find-all-anagrams-in-a-string/) <span style="color:orange">Medium</span>

```java
// A, but running is slow
class Solution {
    public List<Integer> findAnagrams(String s, String p) {
        int sLen = s.length();
        int pLen = p.length();
        char[] chars = p.toCharArray();
        Arrays.sort(chars);
        ArrayList result = new ArrayList();
        for(int i = 0;i <= sLen - pLen; i++){
            char[] subs = s.substring(i, i + pLen).toCharArray();
            Arrays.sort(subs);
            if(Arrays.equals(chars, subs)){
                result.add(i);
            }
        }
        return result;
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