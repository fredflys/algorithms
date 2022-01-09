```java
public class StringMatch {
    public static void main(String[] args) {
        String text = "bccbabababacaca";
        String pattern = "ababaca";
        Match match = new Match();
        int resBF = match.bruteForce(text.toCharArray(), pattern.toCharArray());
        int resKMP = match.KMP(text.toCharArray(), pattern.toCharArray());
        System.out.println("Brute Force: Matching starts at " + resBF + " of the original text.");
        System.out.println("KMP: Matching starts at " + resKMP + " of the original text.");
    }
}

class Match{
    public int bruteForce(char[] text, char[] pattern){
        int textLength = text.length;
        int patLength = pattern.length;
        // 暴力匹配就是要尝试每一种可能，总共有textLength-patLength种可能
        for(int i = 0;i <= textLength-patLength; i++){
            int k = 0;
            // 每种可能都从pattern的第0个位置开始向后尝试
            while(k < patLength && text[i+k] == pattern[k])
                k++;
            // 如果能尝试完pattern的所有位置
            if(k == patLength)
                return i;
        }
        return -1;
    }

    // 暴力匹配另外一种实现
    public int bruteForceYA(char[] text, char[] pattern){
        int textLength = text.length;
        int patLength = pattern.length;

        int textIndex = 0;
        int patIndex = 0;
        while(textIndex < textLength && patIndex < patLength){
            if(text[textIndex] == pattern[patIndex]){
                textIndex++;
                patIndex++;
            }else{
                // 不匹配则从上次匹配的最后位置的下一个再次开始（当前匹配到的位置-已经匹配的长度+1）
                textIndex = textIndex - patIndex + 1;
                // 此时又重新开始匹配，模式串从0开始
                patIndex = 0;
            }
        }

        if(patIndex == patLength)
            return textIndex - patIndex;
        else
            return -1;
    }


    public int KMP(char[] text, char[] pattern){
        int textLength = text.length;
        int patLength = pattern.length;
        if(textLength == 0) return 0;
        
        int[] fail = computeFail(pattern);
        int textIndex = 0;
        int patIndex = 0;
        while(textIndex < textLength){
            if(text[textIndex] == pattern[patIndex]){
                if(patIndex == patLength - 1)
                    return textIndex - patLength + 1;
                textIndex++;
                patIndex++;
            } else if (patIndex > 0){
                /**
                 * 不匹配则回退到前一个位置最大公共前缀的后一个位置，继续开始匹配
                 * f[5-1] = 2
                 * 0 1 2 3 4 5
                 * A B C A B C
                 * A B C A B B
                 *       A B C A B B
                */
                patIndex = fail[patIndex - 1];
            } else
                textIndex++;
        }
        return -1;
    }

    private int[] computeFail(char[] pattern) {
        int length = pattern.length;
        int[] fail = new int[length];

        int right = 1;
        int left = 0;
        while(right < length){
            if(pattern[right] == pattern[left]){
                fail[right] = left + 1;
                left++;
                right++;
            } else if (left > 0)
                // 没匹配上就要从上一次匹配成功的后一个位置开始
                // 即前一个字符在前缀表中对应的位置
                left = fail[left-1];
            else
                // 没匹配上，左指针又指向了0，说明当前位置
                right++;
        }
        return fail;
    }   
}
```