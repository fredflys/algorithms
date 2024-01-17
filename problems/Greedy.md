#### [1431. Kids With the Greatest Number of Candies](https://leetcode.com/problems/kids-with-the-greatest-number-of-candies/description/) Easy
```java
class Solution {
    public List<Boolean> kidsWithCandies(int[] candies, int extraCandies) {
        int n = candies.length;
        
        int max = 0;
        for (int c: candies) {
            if (c > max) max = c;
        }

        List<Boolean> res = new LinkedList<>();
        for (int i = 0; i < n; i++) {
            if (candies[i] + extraCandies >= max) {
                res.add(true);
            } else {
                res.add(false);
            }
        }
        return res;
    }
}
```

#### [1403. Minimum Subsequence in Non-Increasing Order](https://leetcode.com/problems/minimum-subsequence-in-non-increasing-order/description/) Easy
order doesn't matter
```java
class Solution {
    public List<Integer> minSubsequence(int[] nums) {
        Arrays.sort(nums);
        int n = nums.length;
        
        int total = 0;
        for (int num: nums) {
            total += num;
        }

        List<Integer> res = new LinkedList<>();
        for (int i = n - 1, desiredSum = 0, remainingSum = total; desiredSum <= remainingSum; i--) {
            desiredSum += nums[i];
            remainingSum -= nums[i];
            res.add(nums[i]);
        }
        return res;
    }
}
```

#### [1221. Split a String in Balanced Strings](https://leetcode.com/problems/split-a-string-in-balanced-strings/description/) Easy
```java
class Solution {
    public int balancedStringSplit(String s) {
        int res = 0;
        int n = s.length();
        int balance = 0;
        for (char c: s.toCharArray()) {
            if (c == 'L') balance += 1;
            if (c == 'R') balance -= 1;
            // now you can cut
            if (balance == 0) res += 1;
        }
        return res;
    }
}
```

#### [1518. Water Bottles](https://leetcode.com/problems/water-bottles/description/) Easy
```java
class Solution {
    public int numWaterBottles(int numBottles, int numExchange) {
        int res = 0;
        for (int finished = 0, empty = 0; finished < numBottles; finished++) {
            res++;
            if (++empty == numExchange) {
                res++;
                // finished the drink you get for numExchange empty bottles
                // you will get another empty bottle
                empty = 1;
            }
        }
        return res;
    }
}
```

#### [860. Lemonade Change](https://leetcode.com/problems/lemonade-change/) Easy
```java
class Solution {
    public boolean lemonadeChange(int[] bills) {
        int n = bills.length;
        int[] change = new int[2];
        for (int i = 0; i < n; i++) {
            if (bills[i] == 5) {
                change[0]++;
                continue;
            }

            if (bills[i] == 10) {
                if (change[0] == 0) return false;
                change[0]--;
                change[1]++;
                continue;
            }

            if (bills[i] == 20) {
                for (int required = 15; required > 0;) {
                    if (change[1] > 0 && required > 10) {
                        required -= 10;
                        change[1]--;
                        continue;
                    }

                    if (change[0] > 0) {
                        required -= 5;
                        change[0]--;
                        continue;
                    }

                    return false;
                }
            }
        }
        return true;
    }
}
```

#### [1024. Video Stitching](https://leetcode.com/problems/video-stitching/description/) Medium
```java
class Solution {
    public int videoStitching(int[][] clips, int time) {
        int n = clips.length;
        Arrays.sort(clips, (a, b) -> {
            return a[0] == b[0] ? (a[1] - b[1]) : (a[0] - b[0]);
        });

        int res = 0, right = 0;
        // [left, right] is the current stitching range, which is empty at first
        // starting from first clip
        // every time a new clip is used, start from the new clip's end and try to stretch the range
        for (int left = right, i = 0; right < time && i < n; left = right, ++res) {
            // desired range (time) is still not covered and there are still clips left
            // and the starting point of the to-be-used clip falls in the stiched range
            // if all conditions are met, try to stitch in the new clip and stretch the range   
            for(; right < time && i < n && left >= clips[i][0]; right = Math.max(right, clips[i++][1]));
            // after the stretch, right still remains put
            // this means the range cannot be enlarged
            if (right == left) {
                return -1;
            }
        }

        return right >= time ? res : -1;
    }
}
```

#### [1296. Divide Array in Sets of K Consecutive Numbers](https://leetcode.com/problems/divide-array-in-sets-of-k-consecutive-numbers/) Medium
```java
class Solution {
    public boolean isPossibleDivide(int[] nums, int k) {
        int n = nums.length;
        if (n % k != 0) {
            return false;
        }

        // initialization
        TreeMap<Integer, Integer> map = new TreeMap<>();
        for (int num: nums) {
            map.put(num, map.getOrDefault(num, 0) + 1);
        }

        for (int i = n / k; i > 0; i--) {
            int num = map.firstKey();
            for (int j = 0; j < k; j++, num++) {
                int count = map.getOrDefault(num, 0);
                // required but not found in map
                if (count == 0) return false;
                // still have some left, update is needed
                if (--count > 0) map.put(num, count);
                // used up, needs to be removed from map
                if (count == 0) map.remove(num);
            }
        }

        return true;
    }
}
```

#### [1520. Maximum Number of Non-Overlapping Substrings](https://leetcode.com/problems/maximum-number-of-non-overlapping-substrings/) Hard
```java
class Solution {
    public List<String> maxNumOfSubstrings(String s) {
        int[] firstAppearance = new int[26];
        int[] lastAppearance = new int[26];
        Arrays.fill(firstAppearance, -1);

        // store first and last positions for every character 
        int n = s.length();
        for (int i = 0; i < n; i++) {
            int code = getCode(s, i);
            if (firstAppearance[code] < 0) {
                firstAppearance[code] = i;
            }
            lastAppearance[code] = i;
        }

        LinkedList<String> res = new LinkedList<>();
        for (int i = 0, lastSubStringEnd = -1; i < n; i++) {
            int code = getCode(s, i);

            // not the first position of the character, can be skipped
            if (firstAppearance[code] != i) continue;

            int currentSubStringEnd = lastAppearance[code];
            for (int j = i + 1; j <= currentSubStringEnd; j++) {
                int charInSubStringCode = getCode(s, j);
                // for every character in the substring, all occurances should be included
                if (firstAppearance[charInSubStringCode] >= i) {
                    currentSubStringEnd = Math.max(currentSubStringEnd, lastAppearance[charInSubStringCode]);
                } else {
                    // stretch the range if necessary to cover all occurances of the current character 
                    currentSubStringEnd = -1;
                }
            }

            if (currentSubStringEnd >= 0) {
                    // overlapped
                    // prefer the current substring because it is a substring of the last substring
                    if (lastSubStringEnd >= i) {
                        res.removeLast();
                    }
                    res.add(s.substring(i, currentSubStringEnd + 1));
                    lastSubStringEnd = currentSubStringEnd;
            }
        }

        return res;
    }

    private int getCode(String s, int index) {
        return s.charAt(index) - 'a';
    }
}
```

#### [1383. Maximum Performance of a Team](https://leetcode.com/problems/maximum-performance-of-a-team/) Hard
```java
class Solution {
    public int maxPerformance(int n, int[] speed, int[] efficiency, int k) {
        int[][] workers = new int[n][2];
        for (int i = 0; i < n; i++) {
            workers[i] = new int[] {efficiency[i], speed[i]};
        }

        // descending based on efficiency to ensure that the newly added worker has the minimum efficiency
        Arrays.sort(workers, (a, b) -> (b[0] - a[0]));
        PriorityQueue<Integer> candidateSpeeds = new PriorityQueue<>(k);
        long teamPerf = 0, teamSpeed = 0;
        for (int i = 0; i < n; i++) {
            if (candidateSpeeds.size() == k) {
                // kick out the worker with the least speed and leave some space for the new worker
                teamSpeed -= candidateSpeeds.remove();
            }

            // add a new worker, this work will have the least efficiency as the array is descending based on efficiency
            candidateSpeeds.add(workers[i][1]);
            teamSpeed += workers[i][1];

            teamPerf = Math.max(teamPerf, teamSpeed * workers[i][0]);
        }

        return (int) (teamPerf % 1000000007);
    }
}
```

