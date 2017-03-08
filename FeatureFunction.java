/**
 * Created by zhouyou on 7/3/17.
 */
public class FeatureFunction {
    private static final int NUM_OF_FEATURE = 8;
    public static final int F1 	= 0; // Landing height
    public static final int F2 	= 1; // Rows clear
    public static final int F3 	= 2; // Row transition
    public static final int F4 	= 3; // Col transition
    public static final int F5 	= 4; // Num of holes
    public static final int F6	= 6; // Empty cells below some filled cell in the same column
    public static final int F7 	= 5; // Well sum
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

        int[] ff456 = features456(s);
        features[F4] = ff456[0];
        features[F5] = ff456[1];
        features[F6] = ff456[2];

        double[] ff78 = features78(s);
        features[F7] = ff78[0];
        features[F8] = ff78[1];
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
        int piece = s.getNextPiece();

        double height = -1;
        for (int i=0, col=slot; i<s.getpWidth()[piece][orient];i++,col++) {
            height = Math.max(height, s.getTop()[col] - s.getpBottom()[piece][orient][i]);
        }
        return height + s.getpHeight()[piece][orient] / 2.0;
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

    public int[] features456(State s) {
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

    public double[] features78(State s) {
        int[] top = s.getTop();
        double cumulativeWells = 0, total=0;

        for (int i = 0; i < State.COLS; i++){
            total += top[i];
            // Feature 7:
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
