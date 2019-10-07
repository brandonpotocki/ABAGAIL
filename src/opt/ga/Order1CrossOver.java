package opt.ga;

import java.util.Arrays;

import dist.Distribution;

import shared.Instance;

/**
 * Implementation of the Order 1 crossover function for genetic algorithms.
 * Used for NQueens.
 *
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class Order1CrossOver implements CrossoverFunction {

    /**
     * Mates two candidate solutions using single point crossover by choosing a point in the bit string, and creating
     * a crossover mask of 0s up to that point, then 1s after. The mated solution takes the first bits from the second
     * solution, and the remaining bits from the first.
     *
     * @param a the first solution
     * @param b the second solution
     * @return the mated solution
     */
    public Instance mate(Instance a, Instance b) {
        // Create space for the mated solution
        double[] newData = new double[a.size()];
        Arrays.fill(newData, -1);

        // Randomly assign the dividing point.. must be length - 2 to prevent being identical to parent
        int cutoffPoint = Distribution.random.nextInt(newData.length-2);
        int bi = 0;
        
        // Get permutation from parent A between startPoint and endPoint
        for (int i = 0; i < newData.length; i++) {
        	if (i <= cutoffPoint) {
        		newData[i] = a.getContinuous(i);
        	} else {
        		while (valueInArray(b.getContinuous(bi),newData)) {
        			bi++;
        		}
        		newData[i] = b.getContinuous(bi);
        		bi++;
        	}
        	
        }

        // Return the mated solution
        return new Instance(newData);
    }
    
    private boolean valueInArray(double val, double[] arr) {
    	boolean inArray = false; 
        for (double e : arr) { 
            if (e == val) { 
                inArray = true; 
                break; 
            } 
        }
        return inArray;
    }

}