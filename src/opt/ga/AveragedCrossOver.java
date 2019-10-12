package opt.ga;

import shared.Instance;

import dist.Distribution;

/**
 * Implementation of an "averaging" crossover function for genetic algorithms.
 *
 * @author Brandon Potocki
 * @version 1.0
 */
public class AveragedCrossOver implements CrossoverFunction {

    /**
     * Mates two candidate solutions by averaging their values.
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
        	newData[i] = (a.getContinuous(i) + b.getContinuous(i)) / 2.0;
        }

        // Return the mated solution
        return new Instance(newData);
    }

}