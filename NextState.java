import java.util.Arrays;

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