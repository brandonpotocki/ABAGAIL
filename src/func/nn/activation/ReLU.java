

package func.nn.activation;
/**
 * A ReLU activation function
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 * 
 * Borrowed from JonTay
 * 
 */
public class ReLU extends DifferentiableActivationFunction {

    /**
     * @see nn.function.ActivationFunction#activation(double)
     */
	public double value(double value) {
        if (value < 0) {
        	return 0;
        } else {
        	return value;
        }
	}

	/**
	 * @see nn.function.DifferentiableActivationFunction#derivative(double)
	 */
    public double derivative(double value) {
        if (value < 0) {
        	return 0;
        } else {
        	return 1;
        }
	}

}
