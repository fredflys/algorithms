#### Stack

```java
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.ArrayList;
import java.util.Stack; 

public class StackDemo{
    public static void main(String[] args) {
        /*
        IntStack IntStack = new IntStack(5);
        boolean loop = true;
        String input = "";
        Scanner scanner = new Scanner(System.in);

        while(loop){
            System.out.println("(s)how to show the IntStack");
            System.out.println("(p)ush to push a new item");
            System.out.println("p(op) to pop an item from IntStack top");
            System.out.println("(e)xit to leave");

            input = scanner.next();
            switch (input){
                case "s":
                    IntStack.show();
                    break;
                case "p":
                    System.out.println("Enter the value to be pushed:");
                    int val = scanner.nextInt();
                    IntStack.push(val);
                    break;
                case "o":
                    try {
                        int topVal = IntStack.pop();
                        System.out.printf("Top value is %d.\n", topVal);
                    } catch(Exception e){
                        System.out.print(e.getMessage());
                    }
                    break;
                case "e":
                    scanner.close();
                    loop = false;
                    break;
                default:
                    break;
            }

        }
        System.out.println("Bye......"); 
        System.out.printf("\n%d %d %d %d", '+'+0,'-'+0,'*'+0,'/'+0);
        */
        String aRePolishString = "3 4 + 5 * 6 - 2 /";
        String aPolishString = "- * + 3 4 5 6";
        String aInfixString = "3+2*6-4";
        System.out.println(calInfixExpr(aInfixString));
        System.out.println(calRePolishExpr(aRePolishString));
        System.out.println(calPolishExpr(aPolishString));

        String infixExpr = "1.2 +((2+3.3)*4)-5";
        System.out.println(convertInfixToPostfix(infixExpr));
    }

    // infix to postfix
    // 1+((2+3)*4)-5 -> 1 2 3 + 4 * + 5 -
    public static List convertInfixToPostfix(String expr){
        // conver infix expr into a list
        /*
        初始化s1、s2两个栈
        遍历中缀表达式列表
            数字
                入s2
            符号
                左括号
                    入s1
                右括号
                    直到s1栈顶为(
                        持续弹出s1栈顶，结果放入s2
                    如果s1栈顶为(, 则弹出s1栈顶
                否则
                    直到s1为空，而且当前符号优先级大于s2栈顶符号的优先级
                        弹出s1栈顶，结果放入s2
                    当前符号入s1
        直到s1栈空
            弹出s1栈顶，放入s2
        返回s2
        */ 

        // to suppress type safety warning
        @SuppressWarnings("unchecked")
        List<String> exprList = convertInfixExprToList(expr.replace(" ","")); 

        // parse infix expression list
        Stack<String> operatorStack = new Stack<String>();
        List<String> postfixExprList = new ArrayList<String>();
        for(String s:exprList){
            // if numeric
            if(!isOperator(s.charAt(0))){
                postfixExprList.add(s);
            }else{// if not numeric


                if(s.equals("(")){
                    operatorStack.push(s);
                }
                // to remove brackets
                else if(s.equals(")")){
                    while(!operatorStack.peek().equals("(")){
                        postfixExprList.add(operatorStack.pop());
                    }
                    operatorStack.pop();
                }
                else{
                    while(operatorStack.size() != 0  
                        && getOperatorRank(s.charAt(0)) <= getOperatorRank(operatorStack.peek().charAt(0))
                        ){
                            postfixExprList.add(operatorStack.pop());
                         }
                    operatorStack.push(s);
                    }
                }
            }

        while(operatorStack.size() != 0){
            postfixExprList.add(operatorStack.pop());
        }

        return postfixExprList;
    }

    public static List convertInfixExprToList(String expr){
        List<String> exprList = new ArrayList<String>(); 
        int cur = 0;
        String concat;
        char ch;
        do{
            ch = expr.charAt(cur);
            if(isOperator(ch)){
                exprList.add(""+ch);
                // don't forget to move the cursor 
                cur++;
            }else{
                concat = "";
                // when number is double-digit
                while(cur < expr.length() && !isOperator(ch=expr.charAt(cur))){
                    concat += ch;
                    cur++;
                }
                exprList.add(concat);
            }
        
        }while(cur < expr.length());
        return exprList;
    }

    // postfix
    public static int calRePolishExpr(String expr){
        String[] aStrings =  expr.split(" ");
        IntStack calIntStack = new IntStack(aStrings.length);
        for(String i: aStrings){
            int val;
            try{
                // numeric
                val = Integer.parseInt(i);
            }catch(Exception e){
                // if operator
                val = i.charAt(0);
            }
            if(isOperator(val)){
                int num1 = calIntStack.pop();
                int num2 = calIntStack.pop();
                val = calculate(num1, num2, val);
            }
            calIntStack.push(val);
        }
        return calIntStack.pop();
    }

    // prefix
    public static int calPolishExpr(String expr){
    String[] aStrings = expr.split(" ");    
    List<String> aList = Arrays.asList(aStrings);
    // list is reversed in place, so no need to attach to a new varibale
    Collections.reverse(aList);
    IntStack calIntStack = new IntStack(aStrings.length);
    for(String i: aList){
        int val;
        try{
            // numeric
            val = Integer.parseInt(i);
        }catch(Exception e){
            // if operator
            val = i.charAt(0);
        }
        if(isOperator(val)){
            int num1 = calIntStack.pop();
            int num2 = calIntStack.pop();
            // note that num1 - num2
            val = calculate(num2, num1, val);
        }
        calIntStack.push(val);
    }
    return calIntStack.pop();
}

    // infix
    public static int calInfixExpr(String expr){
        int maxSize = expr.length();
        IntStack numberIntStack = new IntStack(maxSize);
        IntStack operatorIntStack = new IntStack(maxSize);
        int num1 = 0;
        int num2 = 0;
        int operator;
        int result;
        for(int cur=0; cur<maxSize;cur++){
            char ch = expr.substring(cur, cur+1).charAt(0);
            // current char is an operator
            if(isOperator(ch)){
                if(!operatorIntStack.isEmpty()){
                    // when oper IntStack is not empty, check if the top is ranked higher         
                    if(getOperatorRank(ch) <= getOperatorRank(operatorIntStack.peek()) ){
                        num1 = numberIntStack.pop();
                        num2 = numberIntStack.pop();
                        operator = operatorIntStack.pop();
                        result = calculate(num1, num2, operator); 
                        numberIntStack.push(result);
                    }
                }
                // always push the operator onto the IntStack 
                operatorIntStack.push(ch);
            }
            // current char is numeric
            else{
                // convert char into int -- refer to ASCII
                numberIntStack.push(ch - 48);
            }
        
        }

        while(!operatorIntStack.isEmpty()){  
            num1 = numberIntStack.pop();
            num2 = numberIntStack.pop();
            operator =  operatorIntStack.pop();
            result = calculate(num1, num2, operator); 
            numberIntStack.push(result);
        }

        int finalResult = numberIntStack.pop();
        //System.out.printf("The expression %s is evaluated to be %d", expr, finalResult);
        return finalResult;
    }

    public static int getOperatorRank(int val){
        if(val == '*' || val == '/'){
            return 1;
        }else if(val == '+' || val == '-'){
            return 0;
        }else{
            return -1;
        }
    }
    
    public static boolean isOperator(int val){
        return val == '+' || val =='-' || val == '*' || val == '/' || val == '(' || val == ')';
    }

    public static int calculate(int num1, int num2, int operator){
        int result = 0;
        switch (operator){
            case '+':
                result = num1 + num2;
                break;
            case '-':
                result = num2 - num1;
                break;
            case '*':
                result = num1 * num2;
                break;    
            case '/':
                result = num2 / num1;
                break;
            default:
                break;
        }
        return result; 
    }
}


class IntStack{
    private int maxSize;
    private int[] IntStack;
    private int top = -1;

    public IntStack(int size){
        maxSize = size;
        IntStack = new int[size];
    }

    public boolean isFull(){
        return top == maxSize - 1;
    }

    public boolean isEmpty(){
        return top == -1;
    }
    
    public void push(int val){
        if(isFull()){
            System.out.println("Full IntStack. Cannot push another value.");
        }

        top++;
        IntStack[top] = val;
    }

    public int pop(){
        if(isEmpty()){
            throw new RuntimeException("Empty IntStack. No item to be popped.");
        }

        int val = IntStack[top];
        top--;
        return val;
    }
    
    public int peek(){
        return IntStack[top];
    }

    public void show(){
        if(isEmpty()){
            System.out.println("Empty IntStack. Nothing to be shown.");
        }

        for(int i = top; i >= 0; i--){
            System.out.printf("IntStack[%d] = %d\n", i, IntStack[i]);
        }
    }
}
```