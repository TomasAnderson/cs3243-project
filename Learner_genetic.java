import java.util.Arrays;

public class Learner_genetic {

	private static double[] weights = new double[] {
			-18632.774652174616,
			6448.762504425676,
			-29076.013395444257,
			-36689.271441668505,
			-16894.091937650956,
			-8720.173920864327,
			-49926.16836221889,
			-47198.39106032252
			
	};
	static FeatureFunction ff = new FeatureFunction();
	
	static int limit = 100000;
	static int stateLength;
	public static final int COLS = 10;
	public static final int ROWS = 21;
	public static final int N_PIECES = 7;
	private static final int K = 8;
	private static int LOST_REWARD = -1000000;
	private static double GAMMA = 0.9;
	private static double EPS = 0.0001;
	private static double P = 1.0/7;
	
	
	
	private static int pickMove(NextState s, double[] w) {		
		
		int bestMove=0, currentMove;
		double bestValue = Double.NEGATIVE_INFINITY, currentValue=0.0;
		NextState ns = new NextState();

		for (currentMove=0;currentMove < s.legalMoves().length; currentMove++) {
		    ns.copyState(s);
			ns.makeMove(currentMove);
			currentValue = ff.computeValueOfState(ns, w);

			if (currentValue > bestValue) {
				bestMove = currentMove;
				bestValue = currentValue;
			}
		}
		return bestMove;
		
	}
	
	private static double getReward(NextState ns, NextState nns, int action) {
		
		//parameters??
		if(nns.hasLost())
			return P * LOST_REWARD;
		else 
			return P * (nns.getRowsCleared()-ns.getRowsCleared());
	}
	
	public static void main() {
	
		NextState s = new NextState();
		System.out.println(s.getTurnNumber());
		double[] w = Arrays.copyOf(weights, weights.length);
		double[] prevWeight = Arrays.copyOf(w, w.length);
		
		double[][] A = new double[K][K];
		for(int i = 0; i < 8; i++) {
			A[i][i]=0.00001;
		}
		
		double[][] B = new double[K][1];
		int [][] fields = new int[ROWS][COLS];
		int [] tops = new int[COLS];
		
		for(int i = 0; i < limit; i++) {
			generateRandomState(s, fields, tops);
			System.out.println(s.getTurnNumber());
			
			double[][] phi1, phi2 = new double[K][1];
			double[][] phiSum = new double[K][1];
			double reward = 0;
			NextState ns = new NextState();
			NextState nns = new NextState();
			
			for(int action = 0; action < s.legalMoves().length; action++) {
				ns.copyState(s);
				ns.makeMove(action);
				if(!ns.hasLost()) {
					phi1 = Matrix.convertToColumnVector(ff.computeFeatureVector(ns));
					
					for(int piece = 0; piece < N_PIECES; piece++) {
						ns.setNextPiece(piece);
						nns.copyState(ns);
						nns.makeMove(pickMove(nns, w));
						phi2 = Matrix.convertToColumnVector(ff.computeFeatureVector(nns));
						phiSum = Matrix.matrixAdd(phiSum, phi2);
						reward += getReward(ns, nns, action);
					}
					double[][] temp = Matrix.multiplyByConstant(phiSum, GAMMA*P);
					double[][] transposed = Matrix.transpose(Matrix.matrixSub(phi1, temp));
					double[][] numerator = Matrix.matrixMulti(A, phi1);
					numerator = Matrix.matrixMulti(numerator, transposed);
					numerator = Matrix.matrixMulti(numerator, A);
					temp = Matrix.matrixMulti(transposed, A);
					temp = Matrix.matrixMulti(temp, phi1);
					double denominator = 1.0 + temp[0][0];
					A = Matrix.matrixSub(A, Matrix.multiplyByConstant(numerator, 1.0 / denominator));
					B = Matrix.matrixAdd(B, Matrix.multiplyByConstant(phi1, reward));
				}
			}
			w = Matrix.convertToArray(Matrix.matrixMulti(A, B));
		}
	}

	private static void generateRandomState(NextState s, int[][] fields, int[] tops) {
		for(int j = 0; j < ROWS; j++)
			for(int k = 0; k < COLS; k++) {
				if(Math.random()*2 >= 1) {
					fields[j][k] = 1;
					tops[k]= j + 1;
				}
				else 
					fields[j][k] = 0;
			}
		s.setFieldDeep(fields);
		s.setTopDeep(tops);
		s.setNextPiece((int) Math.random()*N_PIECES);
	}

}
//
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

