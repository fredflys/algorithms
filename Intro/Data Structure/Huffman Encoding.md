#### Huffman Encoding

```java
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.w3c.dom.TypeInfo;

public class HuffmanEncodingDemo {
    public static void main(String[] args) {
        // ---------------------------编码与解码字符串---------------------------------
        String line = "i like like like java do you like a java";
        byte[] bytes = line.getBytes();
        HuffmanEncoding encoding = new HuffmanEncoding(bytes);
        byte[] encodingBytes = encoding.getEncodingBytes();
        byte[] decodedBytes = encoding.getDecodedBytes();
        System.out.println("Result after compression: " + Arrays.toString(encodingBytes) + ".\nLength reduced from "
                + bytes.length + " to " + encodingBytes.length + ".\nCompression rate is "
                + (float) (bytes.length - encodingBytes.length) / bytes.length + ".");
        System.out.println("Decoded bytes from the result is: " + new String(decodedBytes) + "\nMatched: "
                + Arrays.equals(decodedBytes, bytes));

        // ----------------------------压缩与解压文件---------------------------------
        System.out.println("-----------------------------------");

        String zipSrcPath = "..//..//original.bmp";
        String zipDstPath = "..//..//zipped";
        HuffmanEncoding fileEncoding = new HuffmanEncoding(zipSrcPath, zipDstPath);
        System.out.println("File zipped successfully...");
        String unzipDstPath = "..//..//unzipped.bmp";
        fileEncoding.unzipFile(zipDstPath, unzipDstPath);
        System.out.println("File unzipped successfully...");
    }

}

class HuffmanEncoding {
    // character -> tree path e.g. 32 -> 01, 97 -> 100..
    // 0 indicates left, 1 indicates right
    private Map<Byte, String> mapping = new HashMap<Byte, String>();
    private ENode treeRoot;
    private List<ENode> nodes;
    private byte[] encodingBytes;
    private byte[] decodedBytes;

    public HuffmanEncoding(byte[] bytes) {
        // 从字符串的字节构建节点
        createNodes(bytes);
        // 根据节点构建霍夫曼树
        createHuffmanTree(nodes);
        // 从霍夫曼树生成每个字节映射的字符串编码
        buildCodesForChildNodes(treeRoot, "", new StringBuilder());
        // 从字符串编码回构字节，压缩完成
        encode(bytes);
        // 解码
        decode(mapping, encodingBytes);
    }

    public HuffmanEncoding(String srcPath, String dstPath) {
        zipFile(srcPath, dstPath);
    }

    // --------------------------压缩-------------------------------------
    // e.g. [ ENode[data=32, weight=9] ... ]
    // prepare nodes from which a Huffman tree will be built
    private void createNodes(byte[] bytes) {
        ArrayList<ENode> nodes = new ArrayList<ENode>();

        Map<Byte, Integer> map = new HashMap<>();
        // get occurance of each character
        for (Byte b : bytes) {
            // get a wrapper for int so it can be compared to null
            Integer count = map.get(b);
            if (count == null) {
                map.put(b, 1);
            } else {
                map.put(b, ++count);
            }
        }
        // turn every distinct character into a node
        for (Map.Entry<Byte, Integer> entry : map.entrySet()) {
            ENode node = new ENode(entry.getKey(), entry.getValue());
            nodes.add(node);
        }
        this.nodes = nodes;
    }

    private void createHuffmanTree(List<ENode> nodes) {
        while (nodes.size() > 1) {
            Collections.sort(nodes);
            ENode left = nodes.get(0);
            ENode right = nodes.get(1);
            ENode parent = new ENode(null, left.weight + right.weight);
            parent.setLeft(left);
            parent.setRight(right);
            nodes.remove(left);
            nodes.remove(right);
            nodes.add(parent);
        }
        this.treeRoot = nodes.get(0);
    }

    /*
     *
     * @param node: current node
     * 
     * @param path: 0(left), 1(right), null(root)
     * 
     * @param previousCodeBuilder: code built before (from which new code will be
     * built)
     */
    private void buildCodesForChildNodes(ENode node, String path, StringBuilder previousCodeBuilder) {
        StringBuilder currentCodeBuilder = new StringBuilder(previousCodeBuilder);
        currentCodeBuilder.append(path);
        // if node is a leaf node, put node and code into the mapping hashmap
        // base case for the recursive calls
        if (node.data != null) {
            mapping.put(node.data, currentCodeBuilder.toString());
        } else {
            // if node is not a leaf node, run method recursively build codes for child
            // createNodes
            // left branch
            buildCodesForChildNodes(node.left, "0", currentCodeBuilder);
            // right branch
            buildCodesForChildNodes(node.right, "1", currentCodeBuilder);
        }
    }

    /*
     * @param: 待压缩的字符串字节数组 result: 压缩后的字节数组
     */
    private void encode(byte[] bytes) {
        // convert every character into its mapping value and concatenate e.g.
        // 110101010...
        StringBuilder mapString = new StringBuilder();
        for (byte b : bytes) {
            mapString.append(mapping.get(b));
        }
        // compress mapping value string to bytes array(8 bit for a byte)
        int bytesLength = (mapString.length() + 7) / 8; // remainder is discarded
        byte[] encodingBytes = new byte[bytesLength];
        // every time a byte is put into encodingBytes, pos will be added by 1
        int pos = 0;
        // a byte has 8 bits
        for (int i = 0; i < mapString.length(); i += 8) {
            String currentByteStr;
            // avoid index-out-of-bounds error
            if (i + 8 > mapString.length()) {
                // right to the end
                currentByteStr = mapString.substring(i);
            } else {
                currentByteStr = mapString.substring(i, i + 8);
            }
            encodingBytes[pos++] = (byte) Integer.parseInt(currentByteStr, 2);
        }
        this.encodingBytes = encodingBytes;
    }

    private void zipFile(String srcPath, String dstPath) {
        // 原文件读取流
        FileInputStream inputStream = null;
        // 压缩文件输出流
        FileOutputStream outputStream = null;
        // 压缩文件要按对象块输出，以区分压缩后的字节以及还原时要用到的编码表
        ObjectOutputStream objectOutputStream = null;
        try {
            inputStream = new FileInputStream(srcPath);
            outputStream = new FileOutputStream(dstPath);
            objectOutputStream = new ObjectOutputStream(outputStream);

            byte[] fileBytes = new byte[inputStream.available()];
            inputStream.read(fileBytes);

            createNodes(fileBytes);
            createHuffmanTree(nodes);
            buildCodesForChildNodes(treeRoot, "", new StringBuilder());
            encode(fileBytes);

            // 写入编码后的原文件字节
            objectOutputStream.writeObject(encodingBytes);
            // 写入
            objectOutputStream.writeObject(mapping);
        } catch (Exception e) {
            System.out.println(e.getMessage());
        } finally {
            // 关闭输入流和输出流
            try {
                inputStream.close();
                outputStream.close();
                objectOutputStream.close();
            } catch (Exception e) {
                System.out.println(e.getMessage());
            }
        }

    }

    void unzipFile(String srcPath, String dstPath) {
        FileInputStream inputStream = null;
        FileOutputStream outputStream = null;
        ObjectInputStream objectInputStream = null;
        try {
            inputStream = new FileInputStream(srcPath);
            objectInputStream = new ObjectInputStream(inputStream);
            outputStream = new FileOutputStream(dstPath);

            byte[] fileBytes = (byte[]) objectInputStream.readObject();

            Map<Byte, String> storedMapping = (HashMap<Byte, String>)objectInputStream.readObject();
            decode(storedMapping, fileBytes);
            outputStream.write(decodedBytes);
        } catch (Exception e) {
            System.out.println(e.getMessage());
        } finally {
            try {
                inputStream.close();
                objectInputStream.close();
                outputStream.close();
            } catch (Exception e) {
                System.out.println(e.getMessage());
            }
        }
    }

    // ------------------------------解压----------------------------------
    private String byteToBinaryString(byte byt, boolean isLastByte){
        // 借用Integer中的方法，把byte转为bit，但Integer负数有32位
        int bytInt = (int) byt;
        // 如果byt不是最后一个byt，正数需要对256做或运算以补位（如int 1会被转成1，而不是0...1），负数做了也不受影响
        // 如果byt是负数，需要截断，只取最后8位（此时是补码）
        if(!isLastByte){ 
            // 256对应二进制是1 0000 0000，正数与之做位或运算，结果会补为9位，负数做位或运算，结果不受影响 
            bytInt |= 256;
        }
        String binaryString = Integer.toBinaryString(bytInt);

        // 只要不是最后一个byte，都只取结果的最后8位
        if(!isLastByte){
            return binaryString.substring(binaryString.length() - 8);
        }else{
            return binaryString;
        }
    }

    private void decode(Map<Byte, String> mapping, byte[] bytes){
        StringBuilder decodedBitString = new StringBuilder();
        for(int i=0;i<bytes.length;i++){
            boolean isLastByte = (i == bytes.length - 1);
            decodedBitString.append(byteToBinaryString(bytes[i], isLastByte));
        }

        // 翻转mapping的键值，以便从二进制变为字符，e.g. 110 -> 32
        Map<String, Byte> reversedMapping = new HashMap<String, Byte>();
        for(Map.Entry<Byte, String> entry: mapping.entrySet()){
            reversedMapping.put(entry.getValue(), entry.getKey());
        }

        List<Byte> decodedBytesList = new ArrayList<Byte>();
        /*
        遍历位字符串，遇到能构建出reversedMap的key，就截断并取出，e.g. 110010...
        难点在于Huffman编码非是定长，因此无法按照特定步长截取，步长可能每次都不一样
        在外层循环之内，需要尝试每次移动一位，判断当前i+forward位是否是有效的编码，下次循环开始前，i要停在有效编码的最后一位
        */
        for(int i=0;i<decodedBitString.length();){
            boolean isMatched = false;
            Byte currentByte = null;
            int forward = 1;

            while(!isMatched){
                String tempKey = decodedBitString.substring(i, i + forward);
                currentByte = reversedMapping.get(tempKey);
                if(currentByte == null){
                    forward++;
                }else{
                    isMatched = true;
                }
            }
            decodedBytesList.add(currentByte);
            // 跳到当前byte的中止位置 
            i += forward;
        }
        
        // 把list中的数据放入byte数组中（变list为数组）
        byte[] decodedBytes = new byte[decodedBytesList.size()];
        for(int i=0;i<decodedBytes.length;i++){
            decodedBytes[i] = decodedBytesList.get(i);
        }
        this.decodedBytes = decodedBytes;
    }

    public byte[] getEncodingBytes() {
        return encodingBytes;
    }
    public byte[] getDecodedBytes() {
        return decodedBytes;
    }

}

class ENode implements Comparable<ENode>{
    Byte data; // byte for a character, e.g. a => 97
    int weight; // occurance for a character
    ENode left;
    ENode right;
    
    public ENode(Byte data, int weight){
        this.data = data;
        this.weight = weight;
    }

    @Override
    public int compareTo(ENode o) {
        // from lower to higher
        return this.weight - o.weight;
    }

    @Override
    public String toString() {
        return "Node data: " + data + " weight: " + weight;
    }

    public void preTraversal() {
        System.out.println(this);
        if(this.left != null){
            this.left.preTraversal();
        }
        if(this.right != null){
            this.right.preTraversal();
        }
    }

    public void setLeft(ENode left) {
        this.left = left;
    }
    public void setRight(ENode right) {
        this.right = right;
    }
    public ENode getLeft() {
        return left;
    }
    public ENode getRight() {
        return right;
    }
}
```