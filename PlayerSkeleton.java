
public class PlayerSkeleton {
	private FeatureFunction featureFunction;
	private static double[] weights = new double[] {
		-18632.774652174616, 6448.762504425676, -29076.013395444257,
			-36689.271441668505, -16894.091937650956, -49926.16836221889,
            -8720.173920864327, -47198.39106032252
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

			if (currentValue > bestValue) {
				bestMove = currentMove;
				bestValue = currentValue;
			}
		}
		return bestMove;
	}
	
	public static void main(String[] args) {
		State s = new State();
		new TFrame(s);
		PlayerSkeleton p = new PlayerSkeleton();
		while(!s.hasLost()) {
			s.makeMove(p.pickMove(s,s.legalMoves()));
			s.draw();
			s.drawNext(0,0);
			try {
				Thread.sleep(300);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		System.out.println("You have completed "+s.getRowsCleared()+" rows.");
	}
	
}
