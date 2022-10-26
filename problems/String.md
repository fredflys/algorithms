## String

#### [344. Reverse String](https://leetcode-cn.com/problems/reverse-string/) <span style="color:green">Easy</span>

#### [541. Reverse String II](https://leetcode-cn.com/problems/reverse-string-ii/) <span style="color:green">Easy</span>

#### [151. Reverse Words in a String](https://leetcode-cn.com/problems/reverse-words-in-a-string/) <span style="color:orange">Medium</span>

#### [28. Implement strStr()](https://leetcode-cn.com/problems/implement-strstr/) <span style="color:green">Easy</span>

```java
// Brute force n**2
public int strStr(String source, String target){
    if (target == null || target.equals(""))
        return -1;
    
    // limit possible start points
    for(int i = 0; i < source.length() - target.length() + 1; i++){
        // default flag is true, only set it to false when unequal chars are found
        boolean equal = true;
        for(int j = 0; j < target.length(); j++){
            if(source.charAt(i + j) != target.charAt(j)){
                equal = false;
                break;
            }
        }
        
        if(equal)
            return i;
    }
    
    return -1;
}
```

Rabin-Karp: use hash code to speed comparisons

<img src="https://s2.loli.net/2021/12/19/YLfja3DWJAXv4nQ.png" alt="image-20211219225340858" style="zoom: 33%;" /> 

```java
public int BASE = 1000000;
public int strStr(String source, String target){
	if (source == null || target == null || target.equals(""))
        return -1;
    
    int targetLen = target.length();
    int sourceLen = source.length();
    if(sourceLen < targetLen)
        return -1;
    
    int power = 1;
    for(int i = 0; i < targetLen; i++)
        power = (power * 31) % BASE;
    
    // map target string to hash code 
    int targetHashed = 0;
    for(int i = 0; i < targetLen; i++)
        targetHashed = (targetHashed * 31 + target.charAt(i)) % BASE;
    
    int hashed = 0;
    for(int i = 0; i < sourceLen; i++){
        hashed = (hashed * 31 + source.charAt(i)) % BASE;
        // current length is less than target length, hashing continues
        if(i < targetLen - 1)
            continue;
        
        // current length longer than target length, remove the starting point from current window
        if(i >= targetLen){
            hashed = hashed - (source.charAt(i - targetLen) * power) % BASE;
            if(hashed < 0)
                hashed += BASE;
        }
        
        // double check is required in case of hash collision
        if(hashed == targetHashed){
			if(source.substring(i - targetLen + 1, i + 1).equals(target))
                return i - targetLen + 1;
        }
    }
    
    return -1;
}
```

#### [459. Repeated Substring Pattern](https://leetcode-cn.com/problems/repeated-substring-pattern/) <span style="color:green">Easy</span>

#### [5. Longest Palindromic Substring](https://leetcode.com/problems/longest-palindromic-substring/) <span style="color:orange">Medium</span>

```python
# brute force with nested loop
class Solution:
    def longestPalindrome(self, s: str) -> str:
        if not s:
            return None

        length = len(s)
        # possible substring length from full length to 1 
        for possibleLength in range(length, 0, -1):
            for start in range(length - possibleLength + 1):
                if self.is_palindrome(s, start, start + possibleLength - 1):
                    return s[start : start + possibleLength]
        return ""
    
    def is_palindrome(self, s, left, right):
        while left < right and s[left] == s[right]:
            left += 1
            right -= 1
        return left >= right

```

Expanding from a middle line

<img src="https://s2.loli.net/2021/12/20/ZB91efyuRP5TiU8.png" alt="image-20211220220032016" style="zoom:33%;" /> 

```java
// build a palindrome from the bottom up instead of checking every possibility 
public String longestPalindrome(String s){
    if(s == null)
        return null;
    
    String longest = "";
    for(int i = 0; i < s.length(); i++){
        String oddPalindrome = getPalindromeFrom(s, i, i);
        if(longest.length() < oddPalindrome.length())
            longest = oddPalindrome;
        
        String evenPalindrome = getPalindromeFrom(s, i, i + 1);
        if(longest.length() < evenPalindrome.length())
            longest = evenPalindrome;
    }
    return longest;
}

private String getPalindromeFrom(String s, int left, int right){
    while(left >= 0 && right < s.length()){
        if(s.charAt(left) != s.charAt(right))
            break;
        left--;
        right++;
    }
    return s.substring(left + 1, right);
}
```

Dynamic Programming

s[i : j] is a palindrome only if 

- s[i + 1 : j - 1] is a palindome

- s[i] = s[j]

```java
public String longestPalindrome(String s){
    if(s == null && s.equals(""))
        return "";
    
    int n = s.length();
    boolean[][] isPalindrome = new boolean[n][n];
   	
    // matrix initialization
    int longest = 1, start = 0;
    for(int i = 0; i < n; i++){
    	// base case: a single char is always a palindrome
        isPalindrome[i][i] = true;
        if(i < n - 1){
            // base case: two chars as a palindrome if s[i] == s[i + 1]
            isPalindrome[i][i + 1] = s.charAt(i) == s.charAt(i + 1);
            if(isPalindrome[i][i + 1]){
                start = i;
                longest =2;
            }
        }
    }
    // since whether s[i : j] is a palindrome depends on s[i + 1: j - 1], start the iteration from i = n - 1
    for(int i = n - 1;i >= 0; i--){
        // two chars are treated as a base case so j starts from i + 2
        for(int j = i + 2; j < n; j++){
            isPalindrome[i][j] = isPalindrome[i + 1][j - 1] && s.charAt(i) == s.charAt(j);
            if(isPalindrome[i][j] && j - i + 1 > longest){
                start = i;
                longest = j - i + 1;
            }
        }
    }
    
    return s.substring(start, start + longest);
}
```

#### [125. Valid Palindrome](https://leetcode-cn.com/problems/valid-palindrome/) <span style="color:green">Easy</span>

```python
class Solution:
    def isPalindrome(self, s):
        left, right = 0, len(s) - 1
        while left < right:
            while left < right and not self.is_valid(s[left]):
                left += 1
            while left < right and not self.is_valid(s[right]):
                right -= 1
            if left < right and s[left].lower() != s[right].lower():
                return False
            left += 1
            right -= 1
        return True
    def is_valid(self, char):
        return char.isdigit() or char.isalpha()
```

#### [680. Valid Palindrome II](https://leetcode-cn.com/problems/valid-palindrome-ii/) <span style="color:green">Easy</span>

```java
class Solution {
    class Pair {
        int left, right;
        public Pair(int left, int right){
            this.left = left;
            this.right = right;
        }
    }

    public boolean validPalindrome(String s) {
        if(s == null)
            return false;

        Pair pair = findDIfference(s, 0, s.length() - 1);
        if(pair.left >= pair.right)
            return true;

        return isPalindrome(s, pair.left + 1, pair.right) || isPalindrome(s, pair.left, pair.right - 1);   
    }

	private Pair findDIfference(String s, int left, int right){
        while(left < right && s.charAt(left) == s.charAt(right)){
            left++;
            right--;
        }
        return new Pair(left, right);
    }
    
    private boolean isPalindrome(String s, int left, int right) {
        Pair pair = findDIfference(s, left, right);
        return pair.left >= pair.right;
    }
}
```

#### [344. Reverse String](https://leetcode.com/problems/reverse-string/) Easy
two pointers
```java
class Solution {
    public void reverseString(char[] s) {
        int l = 0;
        int r = s.length - 1;
        while (l < r) {
            char temp = s[l];
            s[l] = s[r];
            s[r] = temp;
            l++;
            r--;
        }
    }
}
```

#### [541. Reverse String II](https://leetcode.com/problems/reverse-string-ii/) Easy
two pointers
```java
class Solution {
    public String reverseStr(String s, int k) {
        int start = 0;
        int end = 2 * k;
        char[] chars = s.toCharArray();
        int n = chars.length;
        while (end <= n) {
            reverse(chars, start, start + k - 1);
            
            end += 2 * k;
            start += 2 * k;
        }
        
        // remaining chars less than k
        // reverse them all
        if (n - start > 0 && n - start < k) {
            reverse(chars, start, n - 1);
            return new String(chars);
        }
        
        // remaining chars less than 2k
        // reverse first k from start
        if (n - start > 0 && n - start < 2 * k) {
            reverse(chars, start, start + k - 1);
            return new String(chars);
        }
        
        return new String(chars);
    }

    public void reverse(char[] s, int l, int r) {
        while (l < r) {
            char temp = s[l];
            s[l] = s[r];
            s[r] = temp;
            l++;
            r--;
        }
    }
}
```

#### [557. Reverse Words in a String III](https://leetcode.com/problems/reverse-words-in-a-string-iii/) Easy
```java
class Solution {
    public String reverseWords(String s) {
        char[] chars = s.toCharArray();
        int l = 0;
        int r = 0;
        int n = chars.length;
        while (r < n) {
            while (r < n && chars[r] != ' ') {
                r++;
            }
            while (l < n && chars[l] == ' ') {
                l++;
            }
            reverse(chars, l, r - 1);
            l = r;
            r++;
        }
        
        return new String(chars);
    }
    
    public void reverse(char[] s, int l, int r) {
        if (r >= s.length) {
            return;
        }
         while (l < r) {
            swap(s, l, r);
            l++;
            r--;
        }
    }
    
    void swap(char[] s, int i, int j) {
        char temp = s[i];
        s[i] = s[j];
        s[j] = temp;
    }
}
```

#### [917. Reverse Only Letters](https://leetcode.com/problems/reverse-only-letters/) Easy
when using incrementing index, never forget to check whether it is out boundary.
```java
class Solution {
    public String reverseOnlyLetters(String s) {
        char[] chars = s.toCharArray();
        int l = 0;
        int r = chars.length - 1;
        while (l < r) {
            // be careful not to cross the upper bound
            while (l < chars.length && !isLetter(chars[l])) {
                l++;
            }
            
            // be careful not to cross the lower bound
            while (r > -1 && !isLetter(chars[r])) {
                r--;
            }
            
            if (l < r) {
                swap(chars, l, r);
                l++;
                r--;
            }
        }
        
        return new String(chars);
    }
    
    boolean isLetter(char c) {
        return (c >= 97 && c <= 122) || (c >= 65 && c <= 90);
    }
    
    void swap(char[] s, int i, int j) {
        char temp = s[i];
        s[i] = s[j];
        s[j] = temp;
    }
}
```

#### [2000. Reverse Prefix of Word](https://leetcode.com/problems/reverse-prefix-of-word/) Easy
two pointers
```java
class Solution {
    public String reversePrefix(String word, char ch) {
        if (word.indexOf(ch) < 0) {
            return word;
        }
        
        char[] chars = word.toCharArray();
        int l = 0;
        while (chars[l] != ch) {
            l++;
        }
        reverse(chars, 0, l);
        return new String(chars);
    }
    
    public void reverse(char[] s, int l, int r) {
        while (l < r) {
            swap(s, l, r);
            l++;
            r--;
        }
    }
    
    void swap(char[] s, int i, int j) {
        char temp = s[i];
        s[i] = s[j];
        s[j] = temp;
    }
}
```

#### [1662. Check If Two String Arrays are Equivalent](https://leetcode.com/problems/check-if-two-string-arrays-are-equivalent/) Easy
```java
class Solution {
    public boolean arrayStringsAreEqual(String[] word1, String[] word2) {
        StringBuilder sb = new StringBuilder();
        
        
        for (String s: word1) {
            sb.append(s);
        }
        
        for (String s: word2) {
            if (sb.length() == 0) {
                return false;
            }
            
            if (sb.indexOf(s) == 0) {
                sb.delete(0, s.length());
            }
        }
        
        return sb.length() == 0;
    }
}
```

#### [151. Reverse Words in a String](https://leetcode.com/problems/reverse-words-in-a-string/) Medium
```java
class Solution {
    public String reverseWords(String s) {
        StringBuilder sb = new StringBuilder();
        
        for (String item: reverse(s).split(" ")) {
            if (item.trim() == "") {
                continue;
            }

            sb.append(reverse(item).trim());
            sb.append(" ");
        }
        
        return sb.delete(sb.length() - 1, sb.length()).toString();
    }
    
    String reverse(String s) {
        return new StringBuilder(s).reverse().toString();
    }
}
```