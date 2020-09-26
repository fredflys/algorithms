
        生成测试数据
        int[] testArr = new int[100000];
        for(int i=0;i<testArr.length;i++){
            testArr[i] = (int)(Math.random() * 100000);
        }

        test("bubble", testArr);
        test("selection", testArr); 
        test("insertion", testArr);
        test("shellBySwap", testArr);      
        test("shellByInsertion", testArr); 
        test("quickHelper", testArr);
        test("mergeHelper", testArr);