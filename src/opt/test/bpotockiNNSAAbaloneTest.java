package opt.test;

import dist.*;
import opt.*;
import opt.example.*;
import opt.ga.*;
import shared.*;
import func.nn.backprop.*;

import java.util.*;
import java.io.*;
import java.net.URL;
import java.text.*;

/**
 * Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to
 * find optimal weights to a neural network that is classifying abalone as having either fewer 
 * or more than 15 rings. 
 *
 * @author Hannah Lau
 * @version 1.0
 */
public class bpotockiNNSAAbaloneTest {
    private static Instance[] instances = initializeInstances(); // Data loading and pre-processing

    private static int inputLayer = 10;
    private static int hiddenLayer1 = 10;
    private static int hiddenLayer2 = 10;
    private static int hiddenLayer3 = 10;
    private static int outputLayer = 1;
    
    private static double testSize = 0.2;
    
    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();
    
    private static GradientErrorMeasure measure = new SumOfSquaresError();

    private static DataSet set = new DataSet(instances);

    private static BackPropagationNetwork network = factory.createClassificationNetwork(
            new int[] {inputLayer, hiddenLayer1, hiddenLayer2, hiddenLayer3, outputLayer}); // defaults to ReLU activation
    
    private static NeuralNetworkOptimizationProblem nnop = new NeuralNetworkOptimizationProblem(set, network, measure);

    private static OptimizationAlgorithm oa = new SimulatedAnnealing(1000,0.5,nnop);
    private static String oaName = "SA";
    private static String results = "";

    private static DecimalFormat df = new DecimalFormat("0.000");

    public static void main(String[] args) throws Exception {
    	
    	String filenameStart = "NN-Abalone-";
        String filenameEnd = "-" + new SimpleDateFormat("yyyyMMdd-HHmmss'.csv'").format(new Date());
    	
    	List<Instance> instanceList = new ArrayList<Instance>();
    	for (int i = 0; i < instances.length; i++) {
    		instanceList.add(instances[i]);
    	}

    	boolean goodSplit = false;
    	
    	Instance[] trainingInstances = new Instance[(int)Math.ceil(instances.length * (1-testSize))];
    	Instance[] testingInstances = new Instance[instances.length - (int)Math.ceil(instances.length * (1-testSize))];
    	
    	while(!goodSplit) {
    	
	    	Collections.shuffle(instanceList);
	    	
	    	trainingInstances = new Instance[(int)Math.ceil(instances.length * (1-testSize))];
	    	testingInstances = new Instance[instances.length - (int)Math.ceil(instances.length * (1-testSize))];
	    	
	    	double trainCount0 = 0;
	    	double trainCount1 = 0;
	    	double testCount0 = 0;
	    	double testCount1 = 0;
	    	
	    	for (int i = 0; i < trainingInstances.length; i++) {
	    		trainingInstances[i] = instanceList.get(i);
	    		//System.out.println(trainingInstances[i].getLabel().getData());
	    		if (trainingInstances[i].getLabel().getData().get(0) == 0) {
	    			trainCount0++;
	    		} else {
	    			trainCount1++;
	    		}
	    	}
	    	for (int i = 0; i < testingInstances.length; i++) {
	    		testingInstances[i] = instanceList.get(i + trainingInstances.length);
	    		if (testingInstances[i].getLabel().getData().get(0) == 0) {
	    			testCount0++;
	    		} else {
	    			testCount1++;
	    		}
	    	}
	    	
	    	double trainRatio = (trainCount0)/(trainCount0+trainCount1);
	    	double testRatio = (testCount0)/(testCount0+testCount1);
	    	
	    	if (Math.abs(trainRatio - testRatio) < 0.001) {
	    		goodSplit = true;
	    	}
	    	
	    	//System.out.println("Training 0:1 ratio = " + String.valueOf(((trainCount0)/(trainCount0+trainCount1))));
	    	//System.out.println("Testing 0:1 ratio = " + String.valueOf(((testCount0)/(testCount0+testCount1))));
	    	//System.out.println("Difference = " + String.valueOf((trainRatio - testRatio)));
    	
    	}
    	
    	DataSet trainingSet = new DataSet(trainingInstances);
    	
    	String nnCsvFile = filenameStart + oaName + filenameEnd;
        FileWriter nnWriter = new FileWriter(nnCsvFile);
        CSVUtils.writeLine(nnWriter, Arrays.asList("Algorithm", "Iteration", "Init Temp", "Decay",
        		"Test Size", "Train Time", "Train MSE",
        		"Train Test Time", "Train Test Accuracy", "Train Test Balanced Accuracy",
        		"Train Test Correct", "Train Test Incorrect", "Train Test True Positive Count", "Train Test True Negative Count",
        		"Train Test False Positive Count", "Train Test False Negative Count",
        		"Train Test Precision", "Train Test Recall", "Train Test F1 Score",
        		"Test Test Time", "Test Test Accuracy", "Test Test Balanced Accuracy",
        		"Test Test Correct", "Test Test Incorrect", "Test Test True Positive Count", "Test Test True Negative Count",
        		"Test Test False Positive Count", "Test Test False Negative Count",
        		"Test Test Precision", "Test Test Recall", "Test Test F1 Score",
        		"Weights"));
    	
    	int[] nnIterations = new int[] {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384};
    	//nnIterations = new int[] {1, 2, 4};
        double[] saInitTemps = {0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0};
        double[] saDecays = {0.9, 0.99, 0.999, 0.9999, 0.99999};
        
    	for (int iter : nnIterations) {
    		for (double initTemp : saInitTemps) {
        		for (double decay : saDecays) {
		    		network = factory.createClassificationNetwork(
		    	            new int[] {inputLayer, hiddenLayer1, hiddenLayer2, hiddenLayer3, outputLayer});
		    	
		    		nnop = new NeuralNetworkOptimizationProblem(trainingSet, network, measure);
			    	oa = new SimulatedAnnealing(initTemp, decay, nnop);
		        
			    	double trainTrainStart, trainTrainEnd, trainTestStart, trainTestEnd, testTestStart, testTestEnd;
			    	double trainTrainTime, trainTestTime, testTestTime;
			    	double trainTestCorrect = 0, trainTestIncorrect = 0, testTestCorrect = 0, testTestIncorrect = 0;
			    	
			    	double trainError;
			    	
			    	trainTrainStart = System.nanoTime();
			    	trainError = train(oa, network, oaName, iter, trainingInstances);
			        trainTrainEnd = System.nanoTime();
			        trainTrainTime = trainTrainEnd - trainTrainStart;
			        trainTrainTime /= Math.pow(10,9);
			        
			        double[] networkWeights = network.getWeights();
			        //System.out.println(networkWeights.length);
			        
			        double trainTestPredicted, trainTestActual;
			        double trainTestTruePositive = 0, trainTestFalsePositive = 0;
			        double trainTestFalseNegative = 0, trainTestTrueNegative = 0;
		
			        Instance optimalInstance = oa.getOptimal();
			        network.setWeights(optimalInstance.getData());
		
			        trainTestStart = System.nanoTime();
			        for(int j = 0; j < trainingInstances.length; j++) {
			        	network.setInputValues(trainingInstances[j].getData());
			        	network.run();
			        	
			        	trainTestActual = Double.parseDouble(trainingInstances[j].getLabel().toString());
			        	trainTestPredicted = Double.parseDouble(network.getOutputValues().toString());
			        	
			        	boolean correctPrediction = (Math.abs(trainTestPredicted - trainTestActual) < 0.5);
			        	if (correctPrediction) {
			    			trainTestCorrect++;
			        		if (trainTestActual == 0) {
			        			trainTestTrueNegative++;
			        		} else {
			        			trainTestTruePositive++;
			        		}
			        	} else {
			        		trainTestIncorrect++;
			        		if (trainTestActual == 0) {
			        			trainTestFalsePositive++;
			        		} else {
			        			trainTestFalseNegative++;
			        		}
			        	}
			        }
			        trainTestEnd = System.nanoTime();
			        trainTestTime = trainTestEnd - trainTestStart;
			        trainTestTime /= Math.pow(10, 9);
			        
			        double trainTestAccuracy = (trainTestTruePositive + trainTestTrueNegative) / (trainTestTruePositive + trainTestTrueNegative + trainTestFalsePositive + trainTestFalseNegative);
			        double trainTestTruePositiveRate = (trainTestTruePositive) / (trainTestTruePositive + trainTestFalseNegative);
			        double trainTestTrueNegativeRate = (trainTestTrueNegative) / (trainTestTrueNegative + trainTestFalsePositive);
			        double trainTestBalancedAccuracy = (trainTestTruePositiveRate + trainTestTrueNegativeRate) / 2.0;
			        double trainTestPrecision = trainTestTruePositive / (trainTestTruePositive + trainTestFalsePositive);
			        double trainTestRecall = trainTestTruePositive / (trainTestTruePositive + trainTestFalseNegative);
			        double trainTestF1Score = (2 * trainTestPrecision * trainTestRecall) / (trainTestPrecision + trainTestRecall);
			        if (trainTestTruePositive == 0) {
			        	trainTestPrecision = 0;
			        	trainTestRecall = 0;
			        	trainTestF1Score = 0;
			        }
			        
			        
			        double testTestPredicted, testTestActual;
			        double testTestTruePositive = 0, testTestFalsePositive = 0;
			        double testTestFalseNegative = 0, testTestTrueNegative = 0;
			        
			        testTestStart = System.nanoTime();
			        for(int j = 0; j < testingInstances.length; j++) {
			        	network.setInputValues(testingInstances[j].getData());
			        	network.run();
			        	
			        	testTestActual = Double.parseDouble(testingInstances[j].getLabel().toString());
			        	testTestPredicted = Double.parseDouble(network.getOutputValues().toString());
			        	
			        	boolean correctPrediction = (Math.abs(testTestPredicted - testTestActual) < 0.5);
			        	if (correctPrediction) {
			    			testTestCorrect++;
			        		if (testTestActual == 0) {
			        			testTestTrueNegative++;
			        		} else {
			        			testTestTruePositive++;
			        		}
			        	} else {
			        		testTestIncorrect++;
			        		if (testTestActual == 0) {
			        			testTestFalsePositive++;
			        		} else {
			        			testTestFalseNegative++;
			        		}
			        	}
			        }
			        testTestEnd = System.nanoTime();
			        testTestTime = testTestEnd - testTestStart;
			        testTestTime /= Math.pow(10, 9);
			        
			        double testTestAccuracy = (testTestTruePositive + testTestTrueNegative) / (testTestTruePositive + testTestTrueNegative + testTestFalsePositive + testTestFalseNegative);
			        double testTestTruePositiveRate = (testTestTruePositive) / (testTestTruePositive + testTestFalseNegative);
			        double testTestTrueNegativeRate = (testTestTrueNegative) / (testTestTrueNegative + testTestFalsePositive);
			        double testTestBalancedAccuracy = (testTestTruePositiveRate + testTestTrueNegativeRate) / 2.0;
			        double testTestPrecision = testTestTruePositive / (testTestTruePositive + testTestFalsePositive);
			        double testTestRecall = testTestTruePositive / (testTestTruePositive + testTestFalseNegative);
			        double testTestF1Score = (2 * testTestPrecision * testTestRecall) / (testTestPrecision + testTestRecall);
			        if (testTestTruePositive == 0) {
			        	testTestPrecision = 0;
			        	testTestRecall = 0;
			        	testTestF1Score = 0;
			        }
		
			        results = "\nResults for " + oaName + " [Iteration " + String.valueOf(iter) + "]"
			        		+ "[Init Temp " + String.valueOf(initTemp) + "]"
			        		+ "[Decay " + String.valueOf(decay) + "]:"
			        		+ "\nTraining Size: " + Double.toString((1 - testSize)*100.0) + "%"
			        		+ "\nTraining Time: " + df.format(trainTrainTime)+ " seconds"
			        		+ "\nTraining Mean Squared Error: " + df.format(trainError/(double)trainingInstances.length)
			        		+ "\n"
			        		+ "\n[Training Set] Correct: " + trainTestCorrect
			        		+ " (True Positives: " + Double.toString(trainTestTruePositive) + "; True Negatives: " + Double.toString(trainTestTrueNegative) + ")"
			        		+ "\n[Training Set] Incorrect: " + trainTestIncorrect
			        		+ " (False Positives: " + Double.toString(trainTestFalsePositive) + "; False Negatives: " + Double.toString(trainTestFalseNegative) + ")"
			        		+ "\n[Training Set] Accuracy: " + df.format(trainTestAccuracy*100.0) + "%"
			        		+ "\n[Training Set] Balanced Accuracy: " + df.format(trainTestBalancedAccuracy*100.0) + "%"
			        		+ "\n[Training Set] F1 Score: " + df.format(trainTestF1Score)
			        		+ "\n[Training Set] Testing Time: " + df.format(trainTestTime) + " seconds"
			        		+ "\n"
			        		+ "\n[Testing Set] Correct: " + testTestCorrect
			        		+ " (True Positives: " + Double.toString(testTestTruePositive) + "; True Negatives: " + Double.toString(testTestTrueNegative) + ")"
			        		+ "\n[Testing Set] Incorrect: " + testTestIncorrect
			        		+ " (False Positives: " + Double.toString(testTestFalsePositive) + "; False Negatives: " + Double.toString(testTestFalseNegative) + ")"
			        		+ "\n[Testing Set] Accuracy: " + df.format(testTestAccuracy*100.0) + "%"
			                + "\n[Testing Set] Balanced Accuracy: " + df.format(testTestBalancedAccuracy*100.0) + "%"
			                + "\n[Testing Set] F1 Score: " + df.format(testTestF1Score)
			        		+ "\n[Testing Set] Testing Time: " + df.format(testTestTime) + " seconds\n";
			        
			        System.out.println(results);
			        
			        CSVUtils.writeLine(nnWriter, Arrays.asList(oaName, String.valueOf(iter), String.valueOf(initTemp), String.valueOf(decay),
			        		String.valueOf(testSize), String.valueOf(trainTrainTime),String.valueOf(trainError/(double)trainingInstances.length),
			        		String.valueOf(trainTestTime), String.valueOf(trainTestAccuracy), String.valueOf(trainTestBalancedAccuracy),
			        		String.valueOf(trainTestCorrect), String.valueOf(trainTestIncorrect), String.valueOf(trainTestTruePositive), String.valueOf(trainTestTrueNegative),
			        		String.valueOf(trainTestFalsePositive), String.valueOf(trainTestFalseNegative),
			        		String.valueOf(trainTestPrecision), String.valueOf(trainTestRecall), String.valueOf(trainTestF1Score),
			        		String.valueOf(testTestTime), String.valueOf(testTestAccuracy), String.valueOf(testTestBalancedAccuracy),
			        		String.valueOf(testTestCorrect), String.valueOf(testTestIncorrect), String.valueOf(testTestTruePositive), String.valueOf(testTestTrueNegative),
			        		String.valueOf(testTestFalsePositive), String.valueOf(testTestFalseNegative),
			        		String.valueOf(testTestPrecision), String.valueOf(testTestRecall), String.valueOf(testTestF1Score),
			        		Arrays.toString(networkWeights)));
        		}
    		}
    	}
        
    	nnWriter.flush();
        nnWriter.close();
        
    }

    private static double train(OptimizationAlgorithm oa, BackPropagationNetwork network, String oaName, int numIters, Instance[] trainInstances) {
        System.out.println("\nError results for " + oaName + "\n---------------------------");
        double error = 0;
        for(int i = 0; i < numIters; i++) {
            oa.train();

            error = 0;
            for(int j = 0; j < trainInstances.length; j++) {
                network.setInputValues(trainInstances[j].getData());
                network.run();

                Instance output = trainInstances[j].getLabel(), example = new Instance(network.getOutputValues());
                example.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
                error += measure.value(output, example);
            }

            //System.out.println(df.format(error));
            if (i % 1000 == 0) {
            	System.out.println("Iteration " + Integer.toString(i));
            }
        }
        return error;
    }

    private static Instance[] initializeInstances() {

    	// Data loading and pre-processing
        int numInstances = 4177;
        int numAttributesBeforeEncoding = 8;
        int numAttributesAfterEncoding = 10;
        String[][][] baseAttributes = new String[numInstances][][];

        try {
        	URL abaloneUrl = bpotockiNNSAAbaloneTest.class.getResource("abalone.data");
        	File abaloneFile = new File(abaloneUrl.getPath());
            BufferedReader br = new BufferedReader(new FileReader(abaloneFile));

            for(int i = 0; i < baseAttributes.length; i++) {
                Scanner scan = new Scanner(br.readLine());
                scan.useDelimiter(",");

                baseAttributes[i] = new String[2][];
                baseAttributes[i][0] = new String[numAttributesBeforeEncoding]; // 8 attributes
                baseAttributes[i][1] = new String[1]; // classifier (0/1)

                for(int j = 0; j < numAttributesBeforeEncoding; j++)
                	baseAttributes[i][0][j] = scan.next();

                baseAttributes[i][1][0] = scan.next();
            }
        }
        catch(Exception e) {
            e.printStackTrace();
        }
        
        double[][][] cleanedAttributes = new double[numInstances][][];
        for (int i = 0; i < baseAttributes.length; i++) {
        	cleanedAttributes[i] = new double[2][];
        	cleanedAttributes[i][0] = new double[numAttributesAfterEncoding];
        	cleanedAttributes[i][1] = new double[1];
        	
        	for (int j = 0; j < numAttributesBeforeEncoding; j++) {
        		if (j != 0) {
        			cleanedAttributes[i][0][j-1] = Double.parseDouble(baseAttributes[i][0][j]);
        		} else { // One-Hot encoding of this column
        			if (baseAttributes[i][0][j].equals("M")) {
        				cleanedAttributes[i][0][7] = 1.0;
        				cleanedAttributes[i][0][8] = 0.0;
        				cleanedAttributes[i][0][9] = 0.0;
        			} else if (baseAttributes[i][0][j].equals("F")) {
        				cleanedAttributes[i][0][7] = 0.0;
        				cleanedAttributes[i][0][8] = 1.0;
        				cleanedAttributes[i][0][9] = 0.0;
        			} else {
        				cleanedAttributes[i][0][7] = 0.0;
        				cleanedAttributes[i][0][8] = 0.0;
        				cleanedAttributes[i][0][9] = 1.0;        				
        			}
        		}
        	}
        	
        	cleanedAttributes[i][1][0] = Double.parseDouble(baseAttributes[i][1][0]);
        }
        
        //System.out.println(baseAttributes[1]);

        
        Instance[] instances = new Instance[cleanedAttributes.length];
        
        for(int i = 0; i < instances.length; i++) {
            instances[i] = new Instance(cleanedAttributes[i][0]);
            // classifications range from 0 to 30; split into 0 - 9 and 10 - 30
            instances[i].setLabel(new Instance(cleanedAttributes[i][1][0] <= 9 ? 0 : 1));
        }

        return instances;
    }
}
