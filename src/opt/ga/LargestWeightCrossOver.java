package opt.ga;

import shared.Instance;

import dist.Distribution;

/**
 * Implementation of a "largest weight" crossover function for genetic algorithms.
 *
 * @author Brandon Potocki
 * @version 1.0
 */
public class LargestWeightCrossOver implements CrossoverFunction {

    /**
     * Mates two candidate solutions by using whichever parent has the largest weight (absolute value).
     *
     * @param a the first solution
     * @param b the second solution
     * @return the mated solution.
     */
    public Instance mate(Instance a, Instance b) {
        // Create space for the mated solution
        double[] newData = new double[a.size()];

        // Assign bits to the mated solution
        for (int i = 0; i < newData.length; i++) {
        	if (Math.abs(a.getContinuous(i)) > Math.abs(b.getContinuous(i))) {
        		newData[i] = a.getContinuous(i);
        	} else {
        		newData[i] = b.getContinuous(i);
        	}
        }

        // Return the mated solution
        return new Instance(newData);
    }

}