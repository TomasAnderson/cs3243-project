import java.io.*;
import java.util.*;


//class Learner {
//	
//	private double[] weights;
//	public static final int K = 8;
//	private static final String NEWFILENAME = "newWeight.txt";
//
//	public Learner(String[] args) {
//		int noThread = Integer.valueOf(args[0]);
//		readWeightFile(args[1]);
//	}
//	
//	public void learn () {
//		System.out.println("Which algorithm do you want to use");
//		System.out.println("1.LPSI");
//		Scanner sc = new Scanner(System.in);
//		
//		switch (sc.nextInt()) {
//		case 1:
//			LSPI lspi = new LSPI(weights);
//			weights = lspi.learn();
//			
//		//TODO: more algorithm	
//			
//		}
//		
//		writeWeightFile();
//	}
//	
//	private void readWeightFile(String fileName) {
//		weights = new double[K];
//		try (BufferedReader br = new BufferedReader(new FileReader(fileName))) {
//			Scanner sc = new Scanner(br);
//			int i = 0;
//			while(sc.hasNext()) {
//				weights[i] = Double.parseDouble(sc.nextLine());
//				i++;
//			}
//		} catch (IOException e) {
//			System.out.println("Invalid Filename."
//					+ "1. Re-enter file name 2. generate random weight"); 
//			Scanner sc = new Scanner(System.in);
//			if(Integer.valueOf(sc.nextLine()) == 1) {
//				System.out.println("Original weight file:");
//				readWeightFile(sc.nextLine());
//			}
//			else {
//				for (int i = 0; i < K; i++) {
//					weights[i] = new Random().nextInt();
//				}
//			}
//		}
//	}
//	
//	private void writeWeightFile() {
//		try (BufferedWriter bw = new BufferedWriter(new FileWriter(NEWFILENAME))) {
//			for (Double weight: weights) {
//				bw.write(weight.toString());
//				bw.newLine();
//			}
//			System.out.println("final weight written to " + NEWFILENAME);
//		} catch (IOException e) {
//			e.printStackTrace();
//		}
//	}
//}
//
//
//class LSPI {
//
////	private static double[] weights = new double[] {
////			-18632.774652174616,
////			6448.762504425676,
////			-29076.013395444257,
////			-36689.271441668505,
////			-16894.091937650956,
////			-8720.173920864327,
////			-49926.16836221889,
////			-47198.39106032252
////			
////	};
//		
//	private static double[] weights;
//	
//	FeatureFunction ff = new FeatureFunction();
//	
//	int limit = 100000;
//	public static final int COLS = 10;
//	public static final int ROWS = 21;
//	public static final int N_PIECES = 7;
//	public static final int K = 8;
//	private static int LOST_REWARD = -1000000;
//	private static final double GAMMA = 0.9; //can be changed
//	private static final double EPS = 0.0005;
//	private static final double P = 1.0/N_PIECES;
//	private static final String PROCESS = "/20";
//	
//
//	public LSPI(double[] w) {
//		weights = Arrays.copyOf(w, w.length);
//	}
//
//	private int pickMove(NextState s, double[] w) {		
//		
//		int bestMove=0, currentMove;
//		double bestValue = Double.NEGATIVE_INFINITY, currentValue=0.0;
//		NextState ns = new NextState();
//
//		for (currentMove=0;currentMove < s.legalMoves().length; currentMove++) {
//		    ns.copyState(s);
//			ns.makeMove(currentMove);
//			
//			if (ns.hasLost()) continue; 
//			
//			currentValue = ff.computeValueOfState(ns, w);
//
//			if (currentValue > bestValue) {
//				bestMove = currentMove;
//				bestValue = currentValue;
//			}
//		}
//		return bestMove;
//		
//	}
//	
//	private double getR(NextState ns, NextState nns, int action) {
//		if(nns.hasLost())
//			return P * LOST_REWARD;
//		else 
//			return P * (nns.getRowsCleared()-ns.getRowsCleared());
//	}
//	
//	public double[] learn() {
//	
//		NextState s = new NextState();
//		double[] prevWeight = Arrays.copyOf(weights, weights.length);
//		NextState ns = new NextState();
//		NextState nns = new NextState();
//		
//		int count = 0;
//		
//		//stop when weights converge or count reduce to 0
//		//don't need to compare for the first iteration
//		while ((diff(weights,prevWeight)>EPS || count == 0) && count < 20) {			
//			prevWeight = Arrays.copyOf(weights, weights.length);
//			System.out.println(Arrays.toString(weights));
//			System.out.println(count+PROCESS); //print out current stage
//			weights = updateWeights(s, weights, ns, nns);
//			count++;
//		}
//		for (int i = 0; i < K; i++) {
//			weights[i] = weights[i] < 0 ? weights[i] : -weights[i];
//		}
//		weights[1] = - weights[1];
//		System.out.println("Final Weight: " + Arrays.toString(weights));
//		return weights;
//	}
//	
//	private double[] updateWeights(NextState s, double[] w, NextState ns, NextState nns) {
//		double reward = 0;
//		double[][] A = new double[K][K];
//		
//		for(int j = 0; j < K; j++) {
//			A[j][j]=1.0/100000; //corresponding to the number of state
//		}
//		double[][] B = new double[K][1];
//		Generator gen = new Generator();
//		
//		for(int i = 0; i < limit; i++) {
//
//            do {
//                s = Generator.decodeState(gen.generateUniqueState());
//            } while (s == null);
//			
//			//to get summation of all the possible action and nextStates
//			for(int action = 0; action < s.legalMoves().length; action++) {
//
//				ns.copyState(s);
//				ns.makeMove(action);
//				
//				if(!ns.hasLost()) {
//					reward = 0;
//					double[][] phi1 = new double[K][1];
//					double[][] phi2 = new double[K][1];
//					double[][] phiSum = new double[K][1];
//					phi1 = Matrix.convertToColumnVector(ff.computeFeatureVector(ns));
//
//					//calculate summation of all the possibilities
//					for(int piece = 0; piece < N_PIECES; piece++) {
//						ns.setNextPiece(piece);
//						nns.copyState(ns);
//						nns.makeMove(pickMove(nns, w));
//						
//						phi2 = Matrix.convertToColumnVector(ff.computeFeatureVector(nns));
//						phiSum = Matrix.matrixAdd(phiSum, phi2);
//						reward += getR(ns, nns, action);
//					}
//					
//					//find numerator
//					//As both GAMMA and P is constant
//					double[][] tempSum = Matrix.multiplyByConstant(phiSum, GAMMA*P);					
//					double[][] transposed = Matrix.transpose(Matrix.matrixSub(phi1, tempSum));
//					double[][] numerator = Matrix.matrixMulti(A, phi1);					
//					numerator = Matrix.matrixMulti(numerator, transposed);
//					numerator = Matrix.matrixMulti(numerator, A);
//					
//					//find denominator
//					double[][] temp = Matrix.matrixMulti(transposed, A);
//					temp = Matrix.matrixMulti(temp, phi1);
//					//temp is a 1*1 array
//					double denominator = 1.0 + temp[0][0];
//					
//					A = Matrix.matrixSub(A, Matrix.multiplyByConstant(numerator, 1.0 / denominator));
//					B = Matrix.matrixAdd(B, Matrix.multiplyByConstant(phi1, reward));
//				}
//			}	
//		}
//		
//		w = Matrix.convertToArray(Matrix.matrixMulti(A, B));
//		return w;
//	}
//
//	private double diff(double[] w, double[] prevWeight) {
//		int diff = 0;
//		for (int i = 0; i < w.length; i++) {
//			diff += (w[i] - prevWeight[i])*(w[i] - prevWeight[i]);
//		}
//		
//		return diff;
//	}
//
//	private void generateRandomState(NextState s) {
//		int [][] fields = new int[ROWS][COLS];
//		int [] tops = new int[COLS];
//		for(int j = 0; j < ROWS-1; j++)
//			for(int k = 0; k < COLS; k++) {
//				if(Math.random()*2 >= 1) {
//					fields[j][k] = 1;
//					tops[k]= j + 1;
//				}
//				else 
//					fields[j][k] = 0;
//			}
//		s.setFieldDeep(fields);
//		s.setTopDeep(tops);
//		s.setNextPiece((int) Math.random()*N_PIECES);
//	}
//
//}
//
//
//
////Author:https://github.com/ngthnhan/Tetris/blob/final/src/PlayerSkeleton.java
//class Matrix {
//
//	/**
//	 * Performs matrix multiplication.
//	 * @param A input matrix
//	 * @param B input matrix
//	 * @return the new matrix resultMatrix
//	 */
//	public static double [][] matrixMulti(double [][] A, double [][]B){
//		int aRows = A.length;
//		int aCols = A[0].length;
//		int bRows = B.length;
//		int bCols = B[0].length;
//		if(aCols != bRows){
//			throw new IllegalArgumentException("The first matrix's rows is not equal to the second matrix's columns, cannot perform matrix multiplication");
//		}
//		else{
//			double [][] resultMatrix = new double [aRows][bCols];
//			for (int i = 0; i < aRows; i++) {
//				for (int j = 0; j < bCols; j++) {
//					resultMatrix[i][j] = 0.00000;
//				}
//			}
//			for (int i = 0; i < aRows; i++) {
//				for (int j = 0; j < bCols; j++) {
//					for (int k = 0; k < aCols; k++) {
//						resultMatrix[i][j] += A[i][k] * B[k][j];
//					}
//				}
//			}
//			return resultMatrix;
//		}
//	}
//
//	/**
//	 * returns the transpose of the input matrix M
//	 */
//	public static double [][] transpose(double [][] M){
//		int mRows = M.length;
//		int mCols = M[0].length;
//		double [][] resultMatrix = new double [mCols][mRows];
//		for(int i = 0; i < mRows; i++){
//			for(int j = 0; j < mCols; j++){
//				resultMatrix[j][i] = M[i][j];
//			}
//		}
//		return resultMatrix;
//	}
//
//	/**
//	 * creates positiv or negativ matrix addition of matrix A relative matrix B based of character c
//	 * @param A input matrix
//	 * @param B additon matrix
//	 * @param c either '-' or '+'
//	 * @return output matrix of this addition
//	 */
//	public static double[][] matrixAddition(double[][] A, double[][] B, char c) {
//		int aRows = A.length;
//		int aCols = A[0].length;
//		int bRows = B.length;
//		int bCols = B[0].length;
//		if (aRows != bRows || aCols !=bCols ) {
//			throw new IllegalArgumentException("both input matrix needs to be in the same format");
//		}
//		double [][] resultmatrix = new double [aRows][aCols];
//		for ( int i = 0 ; i < aRows ; i++ ) {
//			for (int j = 0; j < aCols; j++) {
//				if(c== '+'){
//					resultmatrix[i][j] = A[i][j] +  B[i][j];
//				}
//				else if (c=='-'){
//					resultmatrix[i][j] = A[i][j] -  B[i][j];
//				}
//				else{
//					throw new IllegalArgumentException("character input can only be '-' or '+'");
//				}
//			}
//		}
//		return resultmatrix;
//	}
//
//	//Matrix addition. A add B
//	public static double[][] matrixAdd(double[][] A, double[][] B) {
//		return matrixAddition(A,B,'+');
//	}
//
//	//Matrix substitution. A minus B
//	public static double[][] matrixSub(double[][] A, double[][] B) {
//		return matrixAddition(A,B,'-');
//	}
//
//
//	/**
//	 * Creates the submatrix of a given position of the input matrix M
//	 * @param M input matrix
//	 * @param exclude_row excluding row
//	 * @param exclude_col excluding column
//	 * @return the new matrix resultMatrix
//	 */
//	public static double [][] createSubMatrix(double [][] M, int exclude_row, int exclude_col) {
//		int mRows = M.length;
//		int mCols = M[0].length;
//		double[][] resultMatrix = new double[mRows - 1][mCols - 1];
//		int resultMatrixRow = 0;
//
//		for (int i = 0; i < mRows; i++) {
//			//excludes the aaa row
//			if (i == exclude_row) {
//				continue;
//			}
//			int resultMatrixCol = 0;
//			for (int j = 0; j < mCols; j++) {
//				//excludes the aaa column
//				if (j == exclude_col){
//					continue;
//				}
//				resultMatrix[resultMatrixRow][resultMatrixCol] = M[i][j];
//				resultMatrixCol+=1;
//			}
//			resultMatrixRow+=1;
//		}
//		return resultMatrix;
//	}
//
//	/**
//	 * Calculate the determinant of the input matrix
//	 * @param M input matrix
//	 * @return the determinant
//	 * @throws IllegalArgumentException
//	 */
//	public static double determinant(double [][] M) throws IllegalArgumentException {
//		int aRows = M.length;
//		int aCols = M[0].length;
//		double sum = 0.0;
//
//		if (aRows!=aCols) {
//			throw new IllegalArgumentException("matrix need to be square.");
//		}
//		else if(aRows ==1){
//			return M[0][0];
//		}
//		if (aRows==2) {
//			return (M[0][0] * M[1][1]) - ( M[0][1] * M[1][0]);
//		}
//		// breaks down larger matrix into smaller Submatrix
//		// calculates their determinant by recursion
//		for (int j=0; j<aCols; j++) {
//			sum += placeSign(0,j) * M[0][j] * determinant(createSubMatrix(M, 0, j));
//		}
//		return sum;
//	}
//
//	/**
//	 * Checks if the place sign is positive or negative
//	 */
//	private static double placeSign(int i, int j) {
//		if((i+j)%2 ==0 ){
//			return 1.0;
//		}
//		return -1.0;
//	}
//
//	/**
//	 * function creating the Adjugate of a matrix
//	 * @param M input matrix
//	 * @return the Adjugate matrix called resultMatrix
//	 * @throws IllegalArgumentException
//	 */
//	public static double [][] matrixAdjugate(double[][] M) throws IllegalArgumentException{
//		int mRows = M.length;
//		int mCols = M[0].length;
//		double [][] resultMatrix = new double [mRows][mCols];
//
//		for (int i=0;i<mRows;i++) {
//			for (int j=0; j<mCols;j++) {
//				// i j is reversed to get the transpose of the cofactor matrix
//				resultMatrix[j][i] = placeSign(i,j)* determinant(createSubMatrix(M, i, j));
//			}
//		}
//		return resultMatrix;
//	}
//
//
//	/**
//	 * Add constant c to every element in the matrix M
//	 */
//	public static double[][] multiplyByConstant(double[][] M, double c) {
//		int mRows = M.length;
//		int mCols = M[0].length;
//		double [][] resultMatrix = new double [mRows][mCols];
//
//		for(int i = 0; i < mRows; i++){
//			for(int j = 0; j < mCols; j++){
//				resultMatrix[i][j] = c*M[i][j];
//			}
//		}
//		return resultMatrix;
//	}
//
//	/**
//	 * Return the Inverse of the matrix
//	 */
//	public static double [][] matrixInverse(double [][] M) throws IllegalArgumentException {
//		double det = determinant(M);
//		if(det==0){
//			throw new IllegalArgumentException("The determinant is Zero, the matrix doesn't have an inverse");
//		}
//		return (multiplyByConstant(matrixAdjugate(M), 1.0/det));
//	}
//
//	public static double [][] convertToRowVector(double[] singleArray){
//		double[][] rowVector = new double[1][singleArray.length];
//		for(int i=0;i<singleArray.length;i++) {
//			rowVector[0][i]=singleArray[i];
//		}
//		return rowVector;
//	}
//
//	public static double [][] convertToColumnVector(double[] singleArray){
//		double[][] columnVector = new double[singleArray.length][1];
//		for(int i=0;i<singleArray.length;i++) {
//			columnVector[i][0]=singleArray[i];
//		}
//		return columnVector;
//	}
//
//	public static double[] convertToArray(double[][] myMatrix){
//		double[] myArray = new double[myMatrix.length];
//		for(int i=0;i<myMatrix.length;i++) {
//			myArray[i]=myMatrix[i][0];
//		}
//		return myArray;
//	}
//
//}
//
//
///**
// * Created by nhan on 14/4/16.
// */
//class Generator {
//    private HashSet<String> explored;
//    private static final int  NUM_OF_ENCODED = 7;
//    private Random rand;
//
//    public Generator() {
//        explored = new HashSet<String>();
//        rand = new Random();
//    }
//
//    public String convertToStr(int[] nums) {
//        StringBuilder sb = new StringBuilder();
//        for (int n: nums) {
//            sb.append(n);
//            sb.append(',');
//        }
//
//        return sb.substring(0, sb.length()-1);
//    }
//
//    public static NextState decodeState(String encoded) {
//        String[] strs = encoded.split(",");
//        int[] nums = new int[NUM_OF_ENCODED];
//        int[][] fields = new int[NextState.ROWS][NextState.COLS];
//        for (int i = 0; i < strs.length; i++) {
//            nums[i] = Integer.parseInt(strs[i]);
//        }
//
//        // Decode the nums by shifting bits
//        int bits = 0;
//        int[] tops = new int[NextState.COLS];
//        int t;
//        for (int i = 0; i < NextState.ROWS - 1; i++) {
//            t = 0;
//            for (int j = 0; j < NextState.COLS; j++) {
//                int num = bits / 32;
//                fields[i][j] = nums[num] & 1;
//                nums[bits / 32] >>= 1;
//                bits++;
//                if (fields[i][j] == 1) {
//                    tops[j] = i + 1;
//                }
//            }
//        }
//
//        int nextPiece = nums[NUM_OF_ENCODED-1] & ((1 << 3) - 1);
//
//        // Checking validity of the state
//        int maxHeight = 0;
//        for (int j = 0; j < NextState.COLS; j++) {
//            if (tops[j] > maxHeight) maxHeight = tops[j];
//        }
//
//        // Checking if there is a row with all empty or all non-empty
//        boolean valid;
//        for (int i = 0; i < maxHeight; i++) {
//            valid = false;
//            for (int j = 0; j < NextState.COLS - 1; j++) {
//                if (fields[i][j] != fields[i][j+1]) valid = true;
//            }
//
//            if (!valid) return null;
//        }
//
//        // Check if nextPiece is valid
//        if (nextPiece >= NextState.N_PIECES) return null;
//
//        NextState s = new NextState();
//        s.setNextPiece(nextPiece);
//        s.setFieldDeep(fields);
//        s.setTopDeep(tops);
//
//        return s;
//    }
//
//    /**
//     * The state is encoded into a string. The string will contains integer (32-bit)
//     * separated by commas. There will be 7 integers (224 bits) to represent a complete
//     * state. The first 200 LSB bits will represent the status of the cells. The next 3 LSB
//     * represent the next piece.
//     * @return the encoded string of a complete state
//     */
//    public String generateUniqueState() {
//        String encodedStr = "";
//        int[] encodedNums = new int[NUM_OF_ENCODED];
//
//        do {
//            for (int i = 0; i < NUM_OF_ENCODED; i++) {
//                encodedNums[i] = rand.nextInt();
//            }
//
//            encodedStr = convertToStr(encodedNums);
//        } while (explored.contains(encodedStr) || !isValid(encodedStr));
//
//        return encodedStr;
//    }
//
//    public boolean isValid(String str) {
//        return decodeState(str) != null;
//    }
//
//    public void generate(int limit, String fName) {
//        boolean append = readStates(fName);
//        ArrayList<String> newStates = new ArrayList<String>();
//        String s;
//
//        for (int i = 0; i < limit; i++) {
//            s = generateUniqueState();
//            newStates.add(s);
//            explored.add(s);
//        }
//
//        writeStates(fName, append, newStates);
//    }
//
//    public boolean readStates(String fName) {
//        boolean append;
//        try (BufferedReader br = new BufferedReader(new FileReader(fName))) {
//            Scanner sc = new Scanner(br);
//            while(sc.hasNext()) {
//                explored.add(sc.nextLine());
//            }
//            append = true;
//        } catch (IOException e) {
//            append = false;
//        }
//
//        return append;
//    }
//
//    public void writeStates(String fName, boolean append, ArrayList<String> newStates) {
//        try (BufferedWriter bw = new BufferedWriter(new FileWriter(fName, append))) {
//            for (String s: newStates) {
//                bw.write(s);
//                bw.newLine();
//            }
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
//    }
//
//    public static void main(String[] args) {
//        int limit = (args.length >= 1) ? Integer.parseInt(args[0]) : 100;
//        String fName = (args.length >= 2) ? args[1] : "states.txt";
//
//        Generator g = new Generator();
//        g.generate(limit, fName);
////        Generator.decodeState("1141230911,-654591384,1287972206,-1601558924,-1582006779,-370877823,-609776290");
//
//
//    }
//}
//
//
class NextState extends State {
    private State originalState;
    private int turn = 0;
    private int cleared = 0;

    private int action = 0;

    //each square in the grid - int means empty - other values mean the turn it was placed
    private int[][] field = new int[ROWS][COLS];
    private int[] top = new int[COLS];

    private int[][][] 	pBottom;
    private int[][] 	pHeight;
    private int[][][]	pTop;

    NextState(State s) {
        this.pBottom = State.getpBottom();
        this.pHeight = State.getpHeight();
        this.pTop = State.getpTop();

        copyState(s);
    }

    NextState() {
        this.pBottom = State.getpBottom();
        this.pHeight = State.getpHeight();
        this.pTop = State.getpTop();

        this.turn = 0;
        this.cleared = 0;

        this.lost = false;

        this.field = new int[ROWS][COLS];
        this.top = new int[COLS];
    }

    //random integer, returns 0-6
    private int randomPiece() {
        return (int)(Math.random()*N_PIECES);
    }


    public void copyState(State s) {
        originalState = s;
        this.nextPiece = s.getNextPiece();
        this.lost = s.lost;
        for (int i = 0; i < originalState.getField().length; i++) {
            field[i] = Arrays.copyOf(originalState.getField()[i], originalState.getField()[i].length);
        }

        top = Arrays.copyOf(originalState.getTop(), originalState.getTop().length);
        turn = originalState.getTurnNumber();
        cleared = originalState.getRowsCleared();
        action = -1;
    }

    public State getOriginalState() { return originalState; }

    public int getRowsCleared() { return cleared; }

    public int[][] getField() { return field; }

    public void setFieldDeep(int[][] newField) {
        for (int i = 0; i < newField.length; i++) {
            this.field[i] = Arrays.copyOf(newField[i], newField[i].length);
        }
    }

    public int[] getTop() { return top; }

    public void setTopDeep(int[] newTop) {
    	
        this.top = Arrays.copyOf(newTop, newTop.length);
        
    }

    public int getAction() { return action; }

    public int getTurnNumber() { return turn; }

    public int getNextPiece() { return this.nextPiece; }

    public void setNextPiece(int next) { this.nextPiece = next; }

    public void makeMove(int move) {
        action = move;
        makeMove(legalMoves[nextPiece][move]);
    }

    public boolean makeMove(int orient, int slot) {
        turn++;
        //height if the first column makes contact
        int height = top[slot]-pBottom[nextPiece][orient][0];
        //for each column beyond the first in the piece
        for(int c = 1; c < pWidth[nextPiece][orient];c++) {
            height = Math.max(height,top[slot+c]-pBottom[nextPiece][orient][c]);
        }

        //check if game ended
        if(height+pHeight[nextPiece][orient] >= ROWS) {
            lost = true;
            return false;
        }

        //for each column in the piece - fill in the appropriate blocks
        for(int i = 0; i < pWidth[nextPiece][orient]; i++) {

            //from bottom to top of brick
            for(int h = height+pBottom[nextPiece][orient][i]; h < height+pTop[nextPiece][orient][i]; h++) {
                field[h][i+slot] = turn;
            }
        }

        //adjust top
        for(int c = 0; c < pWidth[nextPiece][orient]; c++) {
            top[slot+c]=height+pTop[nextPiece][orient][c];
        }

        //check for full rows - starting at the top
        for(int r = height+pHeight[nextPiece][orient]-1; r >= height; r--) {
            //check all columns in the row
            boolean full = true;
            for(int c = 0; c < COLS; c++) {
                if(field[r][c] == 0) {
                    full = false;
                    break;
                }
            }
            //if the row was full - remove it and slide above stuff down
            if(full) {
                cleared++;
                //for each column
                for(int c = 0; c < COLS; c++) {

                    //slide down all bricks
                    for(int i = r; i < top[c]; i++) {
                        field[i][c] = field[i+1][c];
                    }
                    //lower the top
                    top[c]--;
                    while(top[c]>=1 && field[top[c]-1][c]==0)	top[c]--;
                }
            }
        }

        return true;
    }

}

/**
 * Created by zhouyou on 7/3/17.
 */
class FeatureFunction {
    private static final int NUM_OF_FEATURE = 8;
    public static final int F1 	= 0; // Landing height
    public static final int F2 	= 1; // Rows clear
    public static final int F3 	= 2; // Row transition
    public static final int F4 	= 3; // Col transition
    public static final int F5 	= 4; // Num of holes
    public static final int F6 	= 5; // Well sum
    public static final int F7	= 6; // Empty cells below some filled cell in the same column
    public static final int F8	= 7; // Average height of columns


    public double computeValueOfState(NextState s, double[] weights) {
        double[] featureVector = computeFeatureVector(s);
        double result=0;
        for (int i=0;i<weights.length;i++) {
            result += featureVector[i]*weights[i];
        }
        return result;
    }

    public double[] computeFeatureVector(NextState s) {
        double[] features = new double[NUM_OF_FEATURE];
        features[F1] = feature1(s);
        features[F2] = feature2(s);
        features[F3] = feature3(s);

        int[] features45Return = features457(s);
        features[F4] = features45Return[0];
        features[F5] = features45Return[1];
        features[F7] = features45Return[2];

        double[] features6Return = features68(s);
        features[F6] = features6Return[0];
        features[F8] = features6Return[1];
        return features;
    }


    /**
     * feature functions
     */
    private double feature1(NextState s) {
        int[][] legalMoves = s.getOriginalState().legalMoves();
        int action = s.getAction();
        int orient = legalMoves[action][State.ORIENT];
        int slot = legalMoves[action][State.SLOT];
        int piece = s.getOriginalState().getNextPiece();

        double height = -1;
        for (int i=0, col=slot; i<s.getOriginalState().getpWidth()[piece][orient];i++,col++) {
            height = Math.max(height, s.getOriginalState().getTop()[col] - s.getOriginalState().getpBottom()[piece][orient][i]);
        }
        return height + s.getOriginalState().getpHeight()[piece][orient] / 2.0;
    }


    private double feature2(NextState s) {
        return s.getRowsCleared() - s.getOriginalState().getRowsCleared() + 1;
    }

    private double feature3(NextState s) {
        int transCount = 0;
        int[][] field = s.getField();

        for (int i = 0; i < State.ROWS - 1; i++) {
            if (field[i][0] == 0) transCount++;
            if (field[i][State.COLS - 1] == 0) transCount++;
            for(int j=1;j<State.COLS;j++) {
                if (isDifferent(field[i][j], field[i][j-1])) {
                    transCount++;
                }
            }
        }
        return transCount;
    }

    public int[] features457(State s) {
        int[][] field = s.getField();
        int[] top = s.getTop();
        // Feature 4 result:
        int columnTransitions = 0;
        // Feature 5 result:
        int holes = 0;
        int gaps = 0;
        boolean columnDone = false;
        // Traverse each column
        for (int i = 0; i < State.COLS; i++) {
            // Traverse each row until the second highest
            for (int j = 0; j < State.ROWS - 1; j++) {
                // Feature 4: Count any differences in adjacent rows
                if (isDifferent(field[j][i], field[j+1][i]))
                    columnTransitions++;
                // Feature 5: Count any empty cells directly under a filled cell
                if ((field[j][i] == 0) && (field[j+1][i] > 0))
                    holes++;
                if ((field[j][i] == 0) && j<top[i])
                    gaps++;
                // Break if rest of column is empty
                if(j >= top[i])
                    columnDone = true;
            }
            if(columnDone)
                continue;
        }
        int[] results = {columnTransitions, holes, gaps};
        return results;
    }

    public double[] features68(State s) {
        int[] top = s.getTop();
        double cumulativeWells = 0, total=0;

        for (int i = 0; i < State.COLS; i++){
            total += top[i];
            // Feature 6:
            // Make sure array doesn't go out of bounds
            int prevCol = i == 0 ? State.ROWS : top[i - 1];
            int nextCol = i == State.COLS - 1 ? State.ROWS : top[i + 1];
            // Find depth of well
            int wellDepth = Math.min(prevCol, nextCol) - top[i];
            // If number is positive, there is a well. Calculate cumulative well depth
            if(wellDepth > 0)
                cumulativeWells += wellDepth * (wellDepth + 1) / 2;
        }
        total = ((double)total)/State.COLS;
        double[] results = {cumulativeWells, total};
        return results;
    }



    /**
     * Utility functions
     */
    private boolean isDifferent(int cellA, int cellB) {
        boolean cellAFilled = cellA != 0;
        boolean cellBFilled = cellB != 0;

        return cellAFilled != cellBFilled;
    }
}

public class PlayerSkeleton{
	private FeatureFunction featureFunction;
	private static double[] weights = new double[] {
			-18632.774652174616,
			6448.762504425676,
			-29076.013395444257,
			-36689.271441668505,
			-16894.091937650956,
			-8720.173920864327,
			-49926.16836221889,
			-47198.39106032252
			
			//8 feature functions
	};

	private NextState nextstate;

	public PlayerSkeleton() {
		featureFunction = new FeatureFunction();
		nextstate = new NextState();
	}

	public int pickMove(State s, int[][] legalMoves) {

		int bestMove=0, currentMove;
		double bestValue = Double.NEGATIVE_INFINITY, currentValue=0.0;

		for (currentMove=0;currentMove < legalMoves.length; currentMove++) {
		    nextstate.copyState(s);
			nextstate.makeMove(currentMove);
			currentValue = featureFunction.computeValueOfState(nextstate, weights);

			if (nextstate.hasLost()) continue; 
			
			if (currentValue > bestValue) {
				bestMove = currentMove;
				bestValue = currentValue;
			}
		}
		return bestMove;
	}
	
	public static void main(String[] args) {
		State s = new State();
		//new TFrame(s);
		PlayerSkeleton p = new PlayerSkeleton();
		while(!s.hasLost()) {
			s.makeMove(p.pickMove(s,s.legalMoves()));
			//s.draw();
			//s.drawNext(0,0);
			try {
				Thread.sleep(0);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		System.out.println("You have completed "+s.getRowsCleared()+" rows.");
	}
	
}
