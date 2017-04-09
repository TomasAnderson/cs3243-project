import java.util.Scanner;

public class PlayerSkeleton implements Runnable{
	private FeatureFunction featureFunction;
	private static double[] weights = new double[] {
			-34904.28880829364,
			-32902.55791710992,
			-64531.26922394128,
			-78371.96104970135,
			-60635.57439678378,
			-24966.539028577772,
			-97514.01014137706,
			-36032.72102494168
			//8 feature functions
	};
	private static int NUMBEROFLEARNER = 4;
	
	private NextState nextstate;

	public PlayerSkeleton() {
		featureFunction = new FeatureFunction();
		nextstate = new NextState();
	}
	
	public PlayerSkeleton(double[] weights) {
		featureFunction = new FeatureFunction();
		nextstate = new NextState();
		this.weights = weights;
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
		
		System.out.println("Choose: 1. Play; 2. Learn");
		Scanner sc = new Scanner(System.in);

		if(Integer.valueOf(sc.nextLine()) == 1) {
			Thread[] threads = new Thread[NUMBEROFLEARNER ];
			for (int i = 0; i < NUMBEROFLEARNER; i++) {
				threads[i] = new Thread(new PlayerSkeleton());
				threads[i].start();
			}

			try {
				for (Thread t: threads) {
					t.join();
				} 
			}catch (InterruptedException e) {
			}
//			PlayerSkeleton p = new PlayerSkeleton();
//			p.run2();

		}
		
		else {
			System.out.println("Key in number of threads and file name of the orginal weights");
			System.out.println("eg.4 weights.txt");
			String input = sc.nextLine();
			Learner learner = new Learner(input.split(" "));
			learner.learn();
		}
	}

	public void run() {
		State s = new State();
		//			new TFrame(s);
		PlayerSkeleton p = new PlayerSkeleton();
		while(!s.hasLost()) {
			s.makeMove(p.pickMove(s,s.legalMoves()));
			//				s.draw();
			//				s.drawNext(0,0);
			try {
				Thread.sleep(0);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		System.out.println("You have completed "+s.getRowsCleared()+" rows.");
	}
	
	public int run2() {
		State s = new State();
//					new TFrame(s);
		PlayerSkeleton p = new PlayerSkeleton();
		while(!s.hasLost()) {
			s.makeMove(p.pickMove(s,s.legalMoves()));
//							s.draw();
//							s.drawNext(0,0);
			try {
				Thread.sleep(0);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		System.out.println("You have completed "+s.getRowsCleared()+" rows.");
		return s.getRowsCleared();
	}
	
}
