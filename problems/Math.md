#### [7. Reverse Integer](https://leetcode.com/problems/reverse-integer/) Easy
modulo operation is used to retrieve the digits one by one.
build a number from highest digit
```java
class Solution {
    public int reverse(int x) {
        int res = 0;

        while (x != 0) {
            if (res < Integer.MIN_VALUE / 10 || res > Integer.MAX_VALUE / 10) {
                return 0;
            }
            
            // get the last digit
            int digit = x % 10;
            res = res * 10 + digit;
            // remove the last digit
            x /= 10;
        }

        return res;
    }
}
```
#### [9. Palindrome Number](https://leetcode.com/problems/palindrome-number/) Easy
Stupid solution of mine.
```java
class Solution {
    public boolean isPalindrome(int x) {
        if (x < 0) {
            return false;
        }
        
        List<Integer> digits = new ArrayList<>();
        int a;
        int b;
        while (x != 0) {
            a = x / 10;
            b = x % 10;
            digits.add(b);
            x = a;
        }
        
        int i = 0;
        int j = digits.size() - 1;
        while (i < j) {
            if (digits.get(i++) != digits.get(j--)) {
                return false;
            }
        }
        
        return true;
    }
}
```
build a number from highest digits
```java
class Solution {
    public boolean isPalindrome(int x) {
        if (x < 0) return false;
        
        int backup = x;
        int reversed = 0;
        while (x > 0) {
            int last_digit = x % 10;
            x /= 10;
            reversed = reversed * 10 + last_digit;
        }
        return reversed == backup;
    }
}
```
#### [8. String to Integer (atoi)](https://leetcode.com/problems/string-to-integer-atoi/) Medium
Many corner cases around the boundary. Tried and failed many times until all corner cases were covered.
Build number from highest digits.
```java
class Solution {
    public int myAtoi(String s) {
        String input = s.trim();
        boolean isNegative = false;
        if (input.length() == 0) return 0;

        int res = 0;
        char[] chars = input.toCharArray();
        for (int i = 0; i < chars.length; i++) {
            if (i == 0 && chars[i] == '-') {
                isNegative = true;
                continue;
            }

            if (i == 0 && chars[i] == '+') {
                continue;
            }

            if (!Character.isDigit(chars[i])) break;
        
            if (res == Integer.MAX_VALUE / 10 && chars[i] - 48 > 7) {
                return isNegative ? Integer.MIN_VALUE : Integer.MAX_VALUE;
            }
            if (res > Integer.MAX_VALUE / 10) {
                return isNegative ? Integer.MIN_VALUE : Integer.MAX_VALUE;
            }

            res = res * 10 + chars[i] - 48;
        }

        return isNegative ? res * -1 : res;
    }
}
```
#### [914. X of a Kind in a Deck of Cards](https://leetcode.com/problems/x-of-a-kind-in-a-deck-of-cards/) Easy
Euclidean algorithm, gcd
```java
class Solution {
    public boolean hasGroupsSizeX(int[] deck) {
        int[] counter = new int[10001];
        for (int card: deck) {
            counter[card]++;
        }
        
        int g = 0;
        for (int i = 0; g != 1 && i < counter.length; i++) {
            g = gcd(g, counter[i]);
        }
        return g > 1;
    }
    
    int gcd(int x, int y) {
        return y > 0 ? gcd(y, x % y) : x;
    }
}
```
#### [1250. Check If It Is a Good Array](https://leetcode.com/problems/check-if-it-is-a-good-array/) Hard
Bézout's identity, Extended Euclidean algorithm, gcd
```java
class Solution {
    public boolean isGoodArray(int[] nums) {
        int dividend = nums[0], remainder;
        for (int num: nums) {
            while (num > 0) {
                remainder = dividend % num;
                dividend = num;
                num = remainder;
            }
        }
        return dividend == 1;
    }
}
```
#### [223. Rectangle Area](https://leetcode.com/problems/rectangle-area/) Medium
```java
```
#### [1201. Ugly Number III](https://leetcode.com/problems/ugly-number-iii/) Medium
```java
```

#### [372. Super Pow](https://leetcode.com/problems/super-pow/) Medium
(a * b) % k = (a % k)(b % k) % k
![](https://s2.loli.net/2022/11/20/cCFV6QyAz3dbWtP.png)
Euler's theorem, recursive, exponentiation
```java
class Solution {
    int mod = 1337;

    public int superPow(int a, int[] b) {
        return superPow(a, b, b.length);
    }

    private int superPow(int base, int[] exponents, int len) {
        if (len == 0) return 1;

        int last = exponents[--len];
        
        int former = powerModulo(base, last);
        int latter = powerModulo(superPow(base, exponents, len), 10);
        return (former * latter) % mod;
    }

    private int powerModulo(int base, int exponent) {
        // a^0 = 1
        if (exponent == 0) return 1;
        base = base % mod;

        // odd
        if (exponent % 2 == 1) {
            return (base * powerModulo(base, exponent - 1)) % mod;
        }

        // even
        int sub = powerModulo(base, exponent / 2);
        return (sub * sub) % mod;
    }
}
```

#### [50. Pow(x, n)](https://leetcode.com/problems/powx-n/) Medium
recursive, exponentiation
```java
class Solution {
    public double myPow(double x, int exponent) {
        if (exponent == 0) return 1;
        
        if (exponent == Integer.MIN_VALUE) {
            // make it not min value by increment
            return myPow(1 / x, -(exponent + 1)) / x;
        }
        
        if (exponent < 0) {
            return myPow(1 / x, -exponent);
        }
        
        // odd
        if (exponent % 2 == 1) {
            return x * myPow(x, exponent - 1);
        }
        
        double sub = myPow(x, exponent / 2);
        return sub * sub;
    }
}
```
recursive
```java
class Solution {
    private double pow(double x, long y) {
        if (y == 1) {
            return x;
        }

        double res = pow(x, y / 2);
        res *= res;
        return (y % 2 == 1) ? (res * x) : res; 
    }

    public double myPow(double x, int n) {
        if (n == 0) {
            return 1;
        }

        long m = n;
        return m < 0 ? (1. / pow(x, -m)) : pow(x, m);
    }
}
```
stack
```java
class Solution {
    private double pow(double x, long y) {
        if (y == 1) {
            return x;
        }

        double res = pow(x, y / 2);
        res *= res;
        return (y % 2 == 1) ? (res * x) : res; 
    }

    public double myPow(double x, int n) {
        if (n == 0) {
            return 1;
        }

        long m = n;
        return m < 0 ? (1. / pow(x, -m)) : pow(x, m);
    }
}
```

#### [1227. Airplane Seat Assignment Probability](https://leetcode.com/problems/airplane-seat-assignment-probability/) Medium
combinatorial mathematics, probability
case 1: 
the first person sitting in seat 1, then everyone will be in place, the n-th person will sit in seat n, no doubt
f(n) = 1
case 2:
the first person sitting in seat n, it is then impossible for the n-th person to sit in n-th seat
f(n) = 0
case 3:
the first persin sitting in i-th seat (1 < i < n), then passengers from 2 to i - 1 are still able to sit in the right seats and seat i is also taken, that leaves n-th passenger n - (i - 2 + 1) possible seats
f(n) = f(n - i + 1)
comine these cases:
f(1) = 1 (n = 1)
f(n) = ( 1 + 0 + f(n - 1) + f(n - 2) + .. f(2) ) / n
     = (f(1) + f(2) + ... + f(n - 1)) (n > 1)

a. n * f(n) = f(1) + f(2) + ... + f(n - 1) (n > 1)
b. (n - 1) * f(n - 1) = f(1) + f(2) + ... + f(n - 2) (n > 2)
a - b =
n * f(n) - (n - 1) * f(n - 1) = f(n - 1)
n * f(n) = (n - 1) * f(n - 1) + f(n - 1)
f(n) = f(n - 1) (n > 2)

Another point of view
f(n) = 1/n                                    -> 1st person picks his own seat
    + 1/n * 0                                 -> 1st person picks last one's seat
	+ (n-2)/n * (                            ->1st person picks one of seat from 2nd to (n-1)th
        1/(n-2) * f(n-1)                   -> 1st person pick 2nd's seat
        1/(n-2) * f(n-2)                  -> 1st person pick 3rd's seat
        ......
        1/(n-2) * f(2)                     -> 1st person pick (n-1)th's seat
	)
	
=> f(n) = 1/n * ( f(n-1) + f(n-2) + f(n-3) + ... + f(1) )

Now, you can easily get
f(1) = 1
f(2) = 1/2
f(3) = 1/2
...

```java
class Solution {
    public double nthPersonGetsNthSeat(int n) {
        return n == 1 ? 1 : .5;
    }
}  
```

#### [1467. Probability of a Two Boxes Having The Same Number of Distinct Balls](https://leetcode.com/problems/probability-of-a-two-boxes-having-the-same-number-of-distinct-balls/) Hard
dfs, muliplication principle, probability

valid cases: two boxes have the same amount of balls
good cases:  a valid case and two boxes have the samo amount of distinct colors
references: [Counting Problems](https://math.libretexts.org/Courses/Truckee_Meadows_Community_College/TMCC%3A_Precalculus_I_and_II/Under_Construction_test2_11%3A_Sequences_Probability_and_Counting_Theory/Under_Construction%2F%2Ftest2%2F%2F11%3A_Sequences%2C_Probability_and_Counting_Theory%2F%2F11.5%3A_Counting_Principles)
```java
class Solution {
    int[] box1;
    int[] box2;
    Map<Double, Double> memo = new HashMap<>();
    double good = 0;
    double valid = 0;
    int[] balls;

    public double getProbability(int[] balls) {
        this.balls = balls;
        box1 = new int[balls.length];
        box2 = new int[balls.length];
        dfs(0);
        return good / valid;
    }

    void dfs(int i) {
        // there are spare balls left, continue allocation
        if (i != balls.length) {
            // allocating balls of color i 
            for (int j = 0; j <= balls[i]; j++) {
                box1[i] = j;
                box2[i] = balls[i] - j;
                dfs(i + 1);
                // backtrack
                box1[i] = 0;
                box2[i] = 0;
            }
            return;
        }

        // all balls are put into boxes now
        
        // two boxes should have equal amount of balls
        // this is not a valid permutation
        if (countBalls(box1) != countBalls(box2))
            return;

        double box1Permutations = countDistinctPermutations(box1);
        double box2Permutations = countDistinctPermutations(box2);
        int box1Colors = countColors(box1);
        int box2Colors = countColors(box2);
        valid += box1Permutations * box2Permutations;

        // this is the permutation we seek: tow boxes that have same amount of distinct colors  
        if (box1Colors == box2Colors) {
            good += box1Permutations * box2Permutations;
        }
    }

    int countBalls(int[] box) {
        int count = 0;
        for (int balls: box) {
            count += balls;
        }
        return count;
    }

    int countColors(int[] box) {
        int count = 0;
        for (int balls: box) {
            if (balls > 0) count++;
        }
        return count;
    }

    double factorial(double balls) {
        // use memorization to reduce calculations
        if (memo.containsKey(balls)) {
            return memo.get(balls);
        }

        if (balls == 0) {
            return 1;
        }

        double res = balls * factorial(balls - 1);
        memo.put(balls, res);
        return res;
    }

    double countDistinctPermutations(int[]  box) {
        double repeatedBallsPermutations = 1.0;
        int totalBalls = 0;
        for (int balls: box) {
            if (balls > 1) {
                repeatedBallsPermutations *= factorial((double) balls);
            }
            totalBalls += balls;
        }
        // cancel out repeated permutations, e.g. 1 2 2
        return factorial(totalBalls) / repeatedBallsPermutations;
    }
}
```

#### [172. Factorial Trailing Zeroes](https://leetcode.com/problems/factorial-trailing-zeroes/description/) Medium
trailing zeros -> 2 * 5 = 10 -> how many pairs of 2 and 5 -> every even number has factor 2, 5's multiple has factor 5
-> 2s are far more than 5s -> how many 5s the input number can be divided into
```java
class Solution {
    public int trailingZeroes(int n) {
        int res = 0;
        int divisor = 5;
        while (divisor <= n) {
            res += n / divisor;
            divisor *= 5;
        }
        return res;
    }
}
```