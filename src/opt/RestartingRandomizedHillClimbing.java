package opt;

import shared.Instance;

/**
 * A randomized hill climbing algorithm
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class RestartingRandomizedHillClimbing extends OptimizationAlgorithm {
    
    /**
     * The current optimization data
     */
    private Instance cur;
    
    /**
     * The current value of the data
     */
    private double curVal;
    
    /**
     * The best current optimization data seen across all restarts
     */
    private Instance bestCur;
    
    /**
     * The best current value of the data seen across all restarts
     */
    private double bestCurVal;
    
    /**
     * The number of iterations without improvement to train before restarting in a random state (0 to disable restarting)
     */
    private int restartThreshold;
    
    /**
     * The current number of iterations without improvement
     */
    private int restartCounter;
    
    /**
     * Make a new randomized hill climbing
     */
    public RestartingRandomizedHillClimbing(int resThreshold, HillClimbingProblem hcp) {
        super(hcp);
        cur = hcp.random();
        curVal = hcp.value(cur);
        bestCur = cur;
        bestCurVal = curVal;
        restartCounter = 0;
        restartThreshold = resThreshold;
    }

    /**
     * @see shared.Trainer#train()
     */
    public double train() {
        HillClimbingProblem hcp = (HillClimbingProblem) getOptimizationProblem();
        Instance neigh = hcp.neighbor(cur);
        double neighVal = hcp.value(neigh);
        if (neighVal > curVal) {
            curVal = neighVal;
            cur = neigh;
            restartCounter = 0; // there was improvement, so reset the counter
        } else {
        	restartCounter++; // no improvement, so increment counter
	        // Check if above the threshold for lack of improvement and reset to random state, if necessary
        	if ((restartThreshold > 0) && (restartCounter >= restartThreshold)) {
	        	//System.out.println("RESTARTING");
	        	cur = hcp.random();
	        	curVal = hcp.value(cur);
	        	restartCounter = 0;
	        }
        }
        // Check if the new state has a better value than ever seen and, if so, update "best" vars
        if (curVal > bestCurVal) {
        	bestCur = cur;
        	bestCurVal = curVal;
        }
        return curVal;
    }

    /**
     * @see opt.OptimizationAlgorithm#getOptimalData()
     */
    public Instance getOptimal() {
        return bestCur;
    }

}