import java.io.*;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Random;
import java.util.Scanner;

/**
 * Created by nhan on 14/4/16.
 */
public class Generator {
    private HashSet<String> explored;
    private static final int  NUM_OF_ENCODED = 7;
    private Random rand;

    public Generator() {
        explored = new HashSet<String>();
        rand = new Random();
    }

    public String convertToStr(int[] nums) {
        StringBuilder sb = new StringBuilder();
        for (int n: nums) {
            sb.append(n);
            sb.append(',');
        }

        return sb.substring(0, sb.length()-1);
    }

    public static NextState decodeState(String encoded) {
        String[] strs = encoded.split(",");
        int[] nums = new int[NUM_OF_ENCODED];
        int[][] fields = new int[NextState.ROWS][NextState.COLS];
        for (int i = 0; i < strs.length; i++) {
            nums[i] = Integer.parseInt(strs[i]);
        }

        // Decode the nums by shifting bits
        int bits = 0;
        int[] tops = new int[NextState.COLS];
        int t;
        for (int i = 0; i < NextState.ROWS - 1; i++) {
            t = 0;
            for (int j = 0; j < NextState.COLS; j++) {
                int num = bits / 32;
                fields[i][j] = nums[num] & 1;
                nums[bits / 32] >>= 1;
                bits++;
                if (fields[i][j] == 1) {
                    tops[j] = i + 1;
                }
            }
        }

        int nextPiece = nums[NUM_OF_ENCODED-1] & ((1 << 3) - 1);

        // Checking validity of the state
        int maxHeight = 0;
        for (int j = 0; j < NextState.COLS; j++) {
            if (tops[j] > maxHeight) maxHeight = tops[j];
        }

        // Checking if there is a row with all empty or all non-empty
        boolean valid;
        for (int i = 0; i < maxHeight; i++) {
            valid = false;
            for (int j = 0; j < NextState.COLS - 1; j++) {
                if (fields[i][j] != fields[i][j+1]) valid = true;
            }

            if (!valid) return null;
        }

        // Check if nextPiece is valid
        if (nextPiece >= NextState.N_PIECES) return null;

        NextState s = new NextState();
        s.setNextPiece(nextPiece);
        s.setFieldDeep(fields);
        s.setTopDeep(tops);

        return s;
    }

    /**
     * The state is encoded into a string. The string will contains integer (32-bit)
     * separated by commas. There will be 7 integers (224 bits) to represent a complete
     * state. The first 200 LSB bits will represent the status of the cells. The next 3 LSB
     * represent the next piece.
     * @return the encoded string of a complete state
     */
    public String generateUniqueState() {
        String encodedStr = "";
        int[] encodedNums = new int[NUM_OF_ENCODED];

        do {
            for (int i = 0; i < NUM_OF_ENCODED; i++) {
                encodedNums[i] = rand.nextInt();
            }

            encodedStr = convertToStr(encodedNums);
        } while (explored.contains(encodedStr) || !isValid(encodedStr));

        return encodedStr;
    }

    public boolean isValid(String str) {
        return decodeState(str) != null;
    }

    public void generate(int limit, String fName) {
        boolean append = readStates(fName);
        ArrayList<String> newStates = new ArrayList<String>();
        String s;

        for (int i = 0; i < limit; i++) {
            s = generateUniqueState();
            newStates.add(s);
            explored.add(s);
        }

        writeStates(fName, append, newStates);
    }

    public boolean readStates(String fName) {
        boolean append;
        try (BufferedReader br = new BufferedReader(new FileReader(fName))) {
            Scanner sc = new Scanner(br);
            while(sc.hasNext()) {
                explored.add(sc.nextLine());
            }
            append = true;
        } catch (IOException e) {
            append = false;
        }

        return append;
    }

    public void writeStates(String fName, boolean append, ArrayList<String> newStates) {
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(fName, append))) {
            for (String s: newStates) {
                bw.write(s);
                bw.newLine();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        int limit = (args.length >= 1) ? Integer.parseInt(args[0]) : 100;
        String fName = (args.length >= 2) ? args[1] : "states.txt";

        Generator g = new Generator();
        g.generate(limit, fName);
//        Generator.decodeState("1141230911,-654591384,1287972206,-1601558924,-1582006779,-370877823,-609776290");


    }
}
