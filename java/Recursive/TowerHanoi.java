public class TowerHanoi {
    public static void main(String[] args) {
        move(2, "A", "B", "C");
    }

    /**
     * 把所有圆盘从起始位置挪动到指定位置。挪动n个圆盘，就和挪动2个圆盘没有差别。将n个圆盘拆成上面的n-1个和下面1个
     * 两部分。要做的只是先把n-1个挪到空柱上，把1个挪动目标位置，再把n-1个挪到目标位置，完成。
     * @param n 要移动的圆盘数量
     * @param fr 圆盘所在的起始位置
     * @param to 要求到达的位置
     * @param spare 用来暂时放置圆盘的额外空柱
     */
    static void move(int n, String fr, String to, String spare){
        if(n == 1)
            printMove(fr, to);
        else{
            // 把上面n-1个圆盘，从起始柱挪到额外的空柱
            move(n-1, fr, spare, to);
            // 把最下面的1个圆盘，从起始柱挪到目标柱
            move(1, fr, to, spare);
            // 把上面n-1个圆盘，从额外的空柱挪到目标柱
            move(n-1, spare, to, fr);
        }
    }

    /**
     * 
     * @param fr 圆盘所在位置
     * @param to 圆盘目标处
     */
    static void printMove(String fr, String to){
        System.out.println("Move from " + fr + " to " + to);
    }

}