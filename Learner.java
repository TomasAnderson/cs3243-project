import java.io.*;
import java.nio.channels.FileChannel;
import java.util.*;

import org.jgap.Chromosome;
import org.jgap.Configuration;
import org.jgap.FitnessFunction;
import org.jgap.Gene;
import org.jgap.Genotype;
import org.jgap.IChromosome;
import org.jgap.Population;
import org.jgap.impl.DefaultConfiguration;
import org.jgap.impl.DoubleGene;

public class Learner {
	
	private double[] weights;
	public static final int K = 8;
	private static final String NEWFILENAME = "newWeight.txt";

	public Learner(String[] args) {
		int noThread = Integer.valueOf(args[0]);
		readWeightFile(args[1]);
	}
	
	public void learn () {
		System.out.println("Which algorithm do you want to use");
		System.out.println("1.LPSI");
		System.out.println("2. Genetic");
		Scanner sc = new Scanner(System.in);
		
		switch (sc.nextInt()) {
		case 1:
			LSPI lspi = new LSPI(weights);
			weights = lspi.learn();
		case 2:
			Genetic g = new Genetic();
			try {
				g.start();
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			
		//TODO: more algorithm	
			
		}
		
		writeWeightFile();
	}
	
	private void readWeightFile(String fileName) {
		weights = new double[K];
		try (BufferedReader br = new BufferedReader(new FileReader(fileName))) {
			Scanner sc = new Scanner(br);
			int i = 0;
			while(sc.hasNext()) {
				weights[i] = Double.parseDouble(sc.nextLine());
				i++;
			}
		} catch (IOException e) {
			System.out.println("Invalid Filename."
					+ "1. Re-enter file name 2. generate random weight"); 
			Scanner sc = new Scanner(System.in);
			if(Integer.valueOf(sc.nextLine()) == 1) {
				System.out.println("Original weight file:");
				readWeightFile(sc.nextLine());
			}
			else {
				for (int i = 0; i < K; i++) {
					weights[i] = new Random().nextInt();
				}
			}
		}
	}
	
	private void writeWeightFile() {
		try (BufferedWriter bw = new BufferedWriter(new FileWriter(NEWFILENAME))) {
			for (Double weight: weights) {
				bw.write(weight.toString());
				bw.newLine();
			}
			System.out.println("final weight written to " + NEWFILENAME);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}


class LSPI {

//	private static double[] weights = new double[] {
//			-18632.774652174616,
//			6448.762504425676,
//			-29076.013395444257,
//			-36689.271441668505,
//			-16894.091937650956,
//			-8720.173920864327,
//			-49926.16836221889,
//			-47198.39106032252
//			
//	};
		
	private static double[] weights;
	
	FeatureFunction ff = new FeatureFunction();
	
	int limit = 10000;
	public static final int COLS = 10;
	public static final int ROWS = 21;
	public static final int N_PIECES = 7;
	public static final int K = 8;
	private static int LOST_REWARD = -1000000;
	private static final double GAMMA = 0.9; //can be changed
	private static final double EPS = 0.0005;
	private static final double P = 1.0/N_PIECES;
	private static final String PROCESS = "/20";
	

	public LSPI(double[] w) {
		weights = Arrays.copyOf(w, w.length);
	}

	private int pickMove(NextState s, double[] w) {		
		
		int bestMove=0, currentMove;
		double bestValue = Double.NEGATIVE_INFINITY, currentValue=0.0;
		NextState ns = new NextState();

		for (currentMove=0;currentMove < s.legalMoves().length; currentMove++) {
		    ns.copyState(s);
			ns.makeMove(currentMove);
			
			if (ns.hasLost()) continue; 
			
			currentValue = ff.computeValueOfState(ns, w);

			if (currentValue > bestValue) {
				bestMove = currentMove;
				bestValue = currentValue;
			}
		}
		return bestMove;
		
	}
	
	private double getR(NextState ns, NextState nns, int action) {
		if(nns.hasLost())
			return P * LOST_REWARD;
		else 
			return P * (nns.getRowsCleared()-ns.getRowsCleared());
	}
	
	public double[] learn() {
	
		NextState s = new NextState();
		double[] prevWeight = Arrays.copyOf(weights, weights.length);
		NextState ns = new NextState();
		NextState nns = new NextState();
		
		int count = 0;
		
		//stop when weights converge or count reduce to 0
		//don't need to compare for the first iteration
		while ((diff(weights,prevWeight)>EPS || count == 0) && count < 20) {			
			prevWeight = Arrays.copyOf(weights, weights.length);
			weights = updateWeights(s, weights, ns, nns);
			count++;
			System.out.println(count+PROCESS); //print out current stage
		}
		System.out.println(Arrays.toString(weights));
		return weights;
	}
	
	private double[] updateWeights(NextState s, double[] w, NextState ns, NextState nns) {
		double reward = 0;
		double[][] A = new double[K][K];
		
		for(int j = 0; j < K; j++) {
			A[j][j]=0.00001; //an small number
		}
		double[][] B = new double[K][1];
		
		for(int i = 0; i < limit; i++) {

			generateRandomState(s);
			
			//to get summation of all the possible action and nextStates
			for(int action = 0; action < s.legalMoves().length; action++) {

				ns.copyState(s);
				ns.makeMove(action);
				
				if(!ns.hasLost()) {
					reward = 0;
					double[][] phi1 = new double[K][1];
					double[][] phi2 = new double[K][1];
					double[][] phiSum = new double[K][1];
					phi1 = Matrix.convertToColumnVector(ff.computeFeatureVector(ns));

					//calculate summation of all the possibilities
					for(int piece = 0; piece < N_PIECES; piece++) {
						ns.setNextPiece(piece);
						nns.copyState(ns);
						nns.makeMove(pickMove(nns, w));
						
						phi2 = Matrix.convertToColumnVector(ff.computeFeatureVector(nns));
						phiSum = Matrix.matrixAdd(phiSum, phi2);
						reward += getR(ns, nns, action);
					}
					
					//find numerator
					//As both GAMMA and P is constant
					double[][] tempSum = Matrix.multiplyByConstant(phiSum, GAMMA*P);					
					double[][] transposed = Matrix.transpose(Matrix.matrixSub(phi1, tempSum));
					double[][] numerator = Matrix.matrixMulti(A, phi1);					
					numerator = Matrix.matrixMulti(numerator, transposed);
					numerator = Matrix.matrixMulti(numerator, A);
					
					//find denominator
					double[][] temp = Matrix.matrixMulti(transposed, A);
					temp = Matrix.matrixMulti(temp, phi1);
					//temp is a 1*1 array
					double denominator = 1.0 + temp[0][0];
					
					A = Matrix.matrixSub(A, Matrix.multiplyByConstant(numerator, 1.0 / denominator));
					B = Matrix.matrixAdd(B, Matrix.multiplyByConstant(phi1, reward));
				}
			}	
		}
		
		w = Matrix.convertToArray(Matrix.matrixMulti(A, B));
		return w;
	}

	private double diff(double[] w, double[] prevWeight) {
		int diff = 0;
		for (int i = 0; i < w.length; i++) {
			diff += (w[i] - prevWeight[i])*(w[i] - prevWeight[i]);
		}
		
		return diff;
	}

	private void generateRandomState(NextState s) {
		int [][] fields = new int[ROWS][COLS];
		int [] tops = new int[COLS];
		for(int j = 0; j < ROWS-1; j++)
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



//Author:https://github.com/ngthnhan/Tetris/blob/final/src/PlayerSkeleton.java
class Matrix {

	/**
	 * Performs matrix multiplication.
	 * @param A input matrix
	 * @param B input matrix
	 * @return the new matrix resultMatrix
	 */
	public static double [][] matrixMulti(double [][] A, double [][]B){
		int aRows = A.length;
		int aCols = A[0].length;
		int bRows = B.length;
		int bCols = B[0].length;
		if(aCols != bRows){
			throw new IllegalArgumentException("The first matrix's rows is not equal to the second matrix's columns, cannot perform matrix multiplication");
		}
		else{
			double [][] resultMatrix = new double [aRows][bCols];
			for (int i = 0; i < aRows; i++) {
				for (int j = 0; j < bCols; j++) {
					resultMatrix[i][j] = 0.00000;
				}
			}
			for (int i = 0; i < aRows; i++) {
				for (int j = 0; j < bCols; j++) {
					for (int k = 0; k < aCols; k++) {
						resultMatrix[i][j] += A[i][k] * B[k][j];
					}
				}
			}
			return resultMatrix;
		}
	}

	/**
	 * returns the transpose of the input matrix M
	 */
	public static double [][] transpose(double [][] M){
		int mRows = M.length;
		int mCols = M[0].length;
		double [][] resultMatrix = new double [mCols][mRows];
		for(int i = 0; i < mRows; i++){
			for(int j = 0; j < mCols; j++){
				resultMatrix[j][i] = M[i][j];
			}
		}
		return resultMatrix;
	}

	/**
	 * creates positiv or negativ matrix addition of matrix A relative matrix B based of character c
	 * @param A input matrix
	 * @param B additon matrix
	 * @param c either '-' or '+'
	 * @return output matrix of this addition
	 */
	public static double[][] matrixAddition(double[][] A, double[][] B, char c) {
		int aRows = A.length;
		int aCols = A[0].length;
		int bRows = B.length;
		int bCols = B[0].length;
		if (aRows != bRows || aCols !=bCols ) {
			throw new IllegalArgumentException("both input matrix needs to be in the same format");
		}
		double [][] resultmatrix = new double [aRows][aCols];
		for ( int i = 0 ; i < aRows ; i++ ) {
			for (int j = 0; j < aCols; j++) {
				if(c== '+'){
					resultmatrix[i][j] = A[i][j] +  B[i][j];
				}
				else if (c=='-'){
					resultmatrix[i][j] = A[i][j] -  B[i][j];
				}
				else{
					throw new IllegalArgumentException("character input can only be '-' or '+'");
				}
			}
		}
		return resultmatrix;
	}

	//Matrix addition. A add B
	public static double[][] matrixAdd(double[][] A, double[][] B) {
		return matrixAddition(A,B,'+');
	}

	//Matrix substitution. A minus B
	public static double[][] matrixSub(double[][] A, double[][] B) {
		return matrixAddition(A,B,'-');
	}


	/**
	 * Creates the submatrix of a given position of the input matrix M
	 * @param M input matrix
	 * @param exclude_row excluding row
	 * @param exclude_col excluding column
	 * @return the new matrix resultMatrix
	 */
	public static double [][] createSubMatrix(double [][] M, int exclude_row, int exclude_col) {
		int mRows = M.length;
		int mCols = M[0].length;
		double[][] resultMatrix = new double[mRows - 1][mCols - 1];
		int resultMatrixRow = 0;

		for (int i = 0; i < mRows; i++) {
			//excludes the aaa row
			if (i == exclude_row) {
				continue;
			}
			int resultMatrixCol = 0;
			for (int j = 0; j < mCols; j++) {
				//excludes the aaa column
				if (j == exclude_col){
					continue;
				}
				resultMatrix[resultMatrixRow][resultMatrixCol] = M[i][j];
				resultMatrixCol+=1;
			}
			resultMatrixRow+=1;
		}
		return resultMatrix;
	}

	/**
	 * Calculate the determinant of the input matrix
	 * @param M input matrix
	 * @return the determinant
	 * @throws IllegalArgumentException
	 */
	public static double determinant(double [][] M) throws IllegalArgumentException {
		int aRows = M.length;
		int aCols = M[0].length;
		double sum = 0.0;

		if (aRows!=aCols) {
			throw new IllegalArgumentException("matrix need to be square.");
		}
		else if(aRows ==1){
			return M[0][0];
		}
		if (aRows==2) {
			return (M[0][0] * M[1][1]) - ( M[0][1] * M[1][0]);
		}
		// breaks down larger matrix into smaller Submatrix
		// calculates their determinant by recursion
		for (int j=0; j<aCols; j++) {
			sum += placeSign(0,j) * M[0][j] * determinant(createSubMatrix(M, 0, j));
		}
		return sum;
	}

	/**
	 * Checks if the place sign is positive or negative
	 */
	private static double placeSign(int i, int j) {
		if((i+j)%2 ==0 ){
			return 1.0;
		}
		return -1.0;
	}

	/**
	 * function creating the Adjugate of a matrix
	 * @param M input matrix
	 * @return the Adjugate matrix called resultMatrix
	 * @throws IllegalArgumentException
	 */
	public static double [][] matrixAdjugate(double[][] M) throws IllegalArgumentException{
		int mRows = M.length;
		int mCols = M[0].length;
		double [][] resultMatrix = new double [mRows][mCols];

		for (int i=0;i<mRows;i++) {
			for (int j=0; j<mCols;j++) {
				// i j is reversed to get the transpose of the cofactor matrix
				resultMatrix[j][i] = placeSign(i,j)* determinant(createSubMatrix(M, i, j));
			}
		}
		return resultMatrix;
	}


	/**
	 * Add constant c to every element in the matrix M
	 */
	public static double[][] multiplyByConstant(double[][] M, double c) {
		int mRows = M.length;
		int mCols = M[0].length;
		double [][] resultMatrix = new double [mRows][mCols];

		for(int i = 0; i < mRows; i++){
			for(int j = 0; j < mCols; j++){
				resultMatrix[i][j] = c*M[i][j];
			}
		}
		return resultMatrix;
	}

	/**
	 * Return the Inverse of the matrix
	 */
	public static double [][] matrixInverse(double [][] M) throws IllegalArgumentException {
		double det = determinant(M);
		if(det==0){
			throw new IllegalArgumentException("The determinant is Zero, the matrix doesn't have an inverse");
		}
		return (multiplyByConstant(matrixAdjugate(M), 1.0/det));
	}

	public static double [][] convertToRowVector(double[] singleArray){
		double[][] rowVector = new double[1][singleArray.length];
		for(int i=0;i<singleArray.length;i++) {
			rowVector[0][i]=singleArray[i];
		}
		return rowVector;
	}

	public static double [][] convertToColumnVector(double[] singleArray){
		double[][] columnVector = new double[singleArray.length][1];
		for(int i=0;i<singleArray.length;i++) {
			columnVector[i][0]=singleArray[i];
		}
		return columnVector;
	}

	public static double[] convertToArray(double[][] myMatrix){
		double[] myArray = new double[myMatrix.length];
		for(int i=0;i<myMatrix.length;i++) {
			myArray[i]=myMatrix[i][0];
		}
		return myArray;
	}

}


class Genetic {

    // Set these values if you want to automatically generate chromosomes
    // According to some research, the optimal setting is 100 chromosomes with
    // 200 generations
    // I suggest you set MAX_ALLOWED_EVOLUTIONS to 1, and run the program
    // periodically rather than
    // setting MAX_ALLOWED_EVOLUTIONS to a large number,
    // since each generation can take a significant amount of time.
    // For the first run, have chromosomes auto generated randomly.
    // The chromosomes will be saved into log.txt
    // For future runs, have chromosomes manually retrieved from log.txt.
    //
    public static final int NUM_THREADS = 4;
    public static final int MAX_ALLOWED_EVOLUTIONS = 5000000;
    public static final int POPULATION_SIZE = 100;

    public void start() throws Exception {

        // Transfer all readings from log.txt to log_old.txt
        // log_old.txt is like a backup
        backupLog();

        Configuration conf = new DefaultConfiguration();
        FitnessFunction fitnessFunction = new PlayerFitnessFunction();
        conf.setFitnessFunction(fitnessFunction);

        Gene[] genes = new Gene[8];
        genes[0] = new DoubleGene(conf, -1000000, 1000000);
        genes[1] = new DoubleGene(conf, -1000000, 1000000);
        genes[2] = new DoubleGene(conf, -1000000, 1000000);
        genes[3] = new DoubleGene(conf, -1000000, 1000000);
        genes[4] = new DoubleGene(conf, -1000000, 1000000);
        genes[5] = new DoubleGene(conf, -1000000, 1000000);
        genes[6] = new DoubleGene(conf, -1000000, 1000000);
        genes[7] = new DoubleGene(conf, -1000000, 1000000);
        Chromosome chromosome = new Chromosome(conf, genes);
        conf.setSampleChromosome(chromosome);

        // AUTO RANDOM CHROMOSOME GENERATION --------------------------
        // Uncomment this section to auto generate chromosomes
        // Make sure manual generation section is commented out

//        conf.setPopulationSize(POPULATION_SIZE);
//        Genotype population = Genotype.randomInitialGenotype(conf);
        //
        // ------------------------------------------------------------

        // MANUAL GENERATION OF CHROMOSOMES ---------------------------
        // Uncomment this section to manually generate chromosomes from log.txt
        // Population size will set to number of chromosomes read from log.txt
         BufferedReader in = new BufferedReader(new FileReader("log.txt"));
         Population p = new Population(conf);
         String line = null;
         int count = 0;
         while ((line = in.readLine()) != null){
         String[] arr = line.split(" "); genes = new Gene[8]; 
         genes[0] = new DoubleGene(conf,-1000000, 1000000);
         genes[0].setAllele(Double.parseDouble(arr[0]));
         genes[1] = new DoubleGene(conf, -1000000, 1000000);
         genes[1].setAllele(Double.parseDouble(arr[1]));
         genes[2] = new DoubleGene(conf, -1000000, 1000000);
         genes[2].setAllele(Double.parseDouble(arr[2]));
         genes[3] = new DoubleGene(conf, -1000000, 1000000);
         genes[3].setAllele(Double.parseDouble(arr[3]));
         genes[4] = new DoubleGene(conf, -1000000, 1000000);
         genes[4].setAllele(Double.parseDouble(arr[4]));
         genes[5] = new DoubleGene(conf, -1000000, 1000000);
         genes[5].setAllele(Double.parseDouble(arr[5]));
         genes[6] = new DoubleGene(conf, -1000000, 1000000);
         genes[6].setAllele(Double.parseDouble(arr[6]));
         genes[7] = new DoubleGene(conf, -1000000, 1000000);
         genes[7].setAllele(Double.parseDouble(arr[7]));
         Chromosome c = new Chromosome(conf, genes);
         p.addChromosome(c);
         count++;
         }
         in.close();
         conf.setPopulationSize(count);
         Genotype population = new Genotype(conf, p);

        // ------------------------------------------------------------

     // WRITE RANDOM POPULATION TO LOG.TXT
        // WARNING: This will delete your current log.txt contents
//        IChromosome[] chromos = population.getPopulation().toChromosomes();
//        updateLog(chromos);
        
        for (int i = 0; i < MAX_ALLOWED_EVOLUTIONS; i++) {
            System.out.println("EVOLUTION CYCLE NO. " + i);
            population.evolve();
            IChromosome[] chromosomes = population.getPopulation().toChromosomes();
            IChromosome fittest = population.getFittestChromosome();
            Gene[] gene_array = fittest.getGenes();
            String s = "Fittest chromosome weights and value: ";
            for (int g = 0; g < gene_array.length; g++){
             s += gene_array[g].getAllele() + " ";
            }
            s += fittest.getFitnessValue() + " ";
            System.out.println(s);
            updateLog(chromosomes, i);
        }

        IChromosome bestSolutionSoFar = population.getFittestChromosome();

//        Arrays.stream(bestSolutionSoFar.getGenes()).mapToDouble(gene -> (double) gene.getAllele())
//            .forEach(System.out::println);
    }

    public static void updateLog(IChromosome[] chromosomes, int index) throws IOException {
    //    backupLog();
    //    clearLog();
    	String fileName = "log" + index + ".txt";
        PrintWriter weightsOut = new PrintWriter(new BufferedWriter(new FileWriter(fileName, true)));
        double r = 0;
        for (int j = 0; j < chromosomes.length; j++) {
            String s = "";
            IChromosome c = chromosomes[j];
            Gene[] gene_array = c.getGenes();
            for (int k = 0; k < gene_array.length; k++) {
                Gene g = gene_array[k];
                s += (double) g.getAllele() + " ";
            }
            r += c.getFitnessValue();
            weightsOut.println(s);
        }
        weightsOut.close();
        double avg = r/chromosomes.length;
        System.out.println("Average rows is: " + avg);
    }

    public static void clearLog() throws FileNotFoundException {
        PrintWriter out = new PrintWriter("log.txt");
        out.print("");
        out.close();
    }

    public static void backupLog() throws IOException {
        FileInputStream srcStream = new FileInputStream("log.txt");
        FileOutputStream destStream = new FileOutputStream("log_old.txt");
        FileChannel src = srcStream.getChannel();
        FileChannel dest = destStream.getChannel();
        dest.transferFrom(src, 0, src.size());
        srcStream.close();
        destStream.close();
    }

}