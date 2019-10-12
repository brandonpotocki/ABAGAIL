package opt.test;

import java.util.Arrays;
import java.util.Date;
import java.util.Random;
import java.lang.Math;

import dist.DiscreteDependencyTree;
import dist.DiscretePermutationDistribution;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.RestartingRandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.SwapNeighbor;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.SingleCrossOver;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.NQueensFitnessFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.ga.SwapMutation;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;
import shared.ConvergenceTrainer;

import java.io.FileWriter;
import java.text.SimpleDateFormat;

/**
 * Copied from ContinuousPeaksTest
 * @version 1.0
 */
public class bpotockiTravelingSalesman10TestWithReallyRHC {
    /** The n value */
    private static final int N = 10;
    
    public static void main(String[] args) throws Exception {
        Random random = new Random();
        // create the random points
        double[][] points = new double[N][2];
        for (int i = 0; i < points.length; i++) {
            points[i][0] = random.nextDouble();
            points[i][1] = random.nextDouble();   
        }
        // for rhc, sa, and ga we use a permutation based encoding
        TravelingSalesmanEvaluationFunction ef = new TravelingSalesmanRouteEvaluationFunction(points);
        Distribution odd = new DiscretePermutationDistribution(N);
        NeighborFunction nf = new SwapNeighbor();
        MutationFunction mf = new SwapMutation();
        CrossoverFunction cf = new TravelingSalesmanCrossOver(ef);
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        
        // for mimic we use a sort encoding
        TravelingSalesmanEvaluationFunction mimic_ef = new TravelingSalesmanSortEvaluationFunction(points);
        int[] mimic_ranges = new int[N];
        Arrays.fill(mimic_ranges, N);
        Distribution mimic_odd = new  DiscreteUniformDistribution(mimic_ranges);
        Distribution mimic_df = new DiscreteDependencyTree(.1, mimic_ranges); 
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(mimic_ef, mimic_odd, mimic_df);
        
        String filenameStart = "TSP10-";
        String filenameEnd = "-" + new SimpleDateFormat("yyyyMMdd-HHmmss'.csv'").format(new Date());
        
        long startTime;
        long endTime;
        long duration;
        
        int avgHowManyRuns = 5;

        int[] rhcIterations = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384};
        //rhcIterations = new int[] {};
        int[] rhcRestartThresholds = {0, 10, 50, 100, 500, 1000};
        //rhcRestartThresholds = new int[] {100};
        
        int[] saIterations = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384};
        //saIterations = new int[] {};
        double[] saInitTemps = {0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0};
        double[] saDecays = {0.09, 0.9, 0.999, 0.9999};
        
        int[] gaIterations = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};
        //gaIterations = new int[] {2};
        int[] gaPopSizes = {1000, 3000, 5000};
        //gaPopSizes = new int[] {5000};
        double[] gaMatePercents = {0.50, 0.75, 0.95};
        //gaMatePercents = new double[] {0.95};
        double[] gaMutatePercents = {0.1, 0.2};
        //gaMutatePercents = new double[] {0.1};
        
        int[] mmIterations = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};
        //mmIterations = new int[] {};
        int[] mmSamples = {1000, 3000, 5000};
        //mmSamples = new int[] {3000};
        double[] mmKeepPercents = {0.1, 0.2, 0.3};
        //mmKeepPercents = new double[] {0.2};
        
        
        
        // RHC
        if (rhcIterations.length > 0) {
	        String rhcCsvFile = filenameStart + "RHC" + filenameEnd;
	        FileWriter rhcWriter = new FileWriter(rhcCsvFile);
	        CSVUtils.writeLine(rhcWriter, Arrays.asList("Averaged Runs", "Iteration", "Algorithm", "Fitness", "Time (ns)", "Restart Threshold"));
	        
	        for (int iter : rhcIterations) {
	        	for (int res : rhcRestartThresholds) {
		        	double[] avgRHCOptimals = new double[avgHowManyRuns];
		        	double[] avgRHCTimes = new double[avgHowManyRuns];
		        	for (int r = 0; r < avgHowManyRuns; r++) {
				        startTime = System.nanoTime();
				        RestartingRandomizedHillClimbing rhc = new RestartingRandomizedHillClimbing(res, hcp);
				        FixedIterationTrainer fit = new FixedIterationTrainer(rhc, iter);
				        fit.train();
				        endTime = System.nanoTime();
				        duration = (endTime - startTime);
				        avgRHCOptimals[r] = ef.value(rhc.getOptimal());
				        avgRHCTimes[r] = duration;
				        //System.out.println("RHC: " + restartOptimal + " [global max]");
		        	}
		        	
		        	double avgRHCOptimalSum = 0;
		            for ( int i = 0; i < avgHowManyRuns; i++){
		            	avgRHCOptimalSum += avgRHCOptimals[i];
		               }
		            double avgRHCOptimal = 1.0d * avgRHCOptimalSum/avgHowManyRuns;
		        	
		        	double avgRHCTimeSum = 0;
		            for ( int i = 0; i < avgHowManyRuns; i++){
		            	avgRHCTimeSum += avgRHCTimes[i];
		               }
		            double avgRHCTime = 1.0d * avgRHCTimeSum/avgHowManyRuns;
			        
		            System.out.println("[RHC] [Iteration " + String.valueOf(iter) + "] [Restart Threshold " + String.valueOf(res) + 
		            		"] ~~~ Fitness: " + String.valueOf(avgRHCOptimal) + " ~~~ " + String.valueOf(1.0d * avgRHCTime/1000000000) + " seconds");
		        	CSVUtils.writeLine(rhcWriter, Arrays.asList(String.valueOf(avgHowManyRuns), String.valueOf(iter), "RHC", String.valueOf(avgRHCOptimal),String.valueOf(avgRHCTime),String.valueOf(res)));
	        	}
	        }
	        
	        rhcWriter.flush();
	        rhcWriter.close();
        }
        
        // SA
        if (saIterations.length > 0) {
	        String saCsvFile = filenameStart + "SA" + filenameEnd;
	        FileWriter saWriter = new FileWriter(saCsvFile);
	        CSVUtils.writeLine(saWriter, Arrays.asList("Averaged Runs", "Iteration", "Algorithm", "Fitness", "Time (ns)", "Initial Temp", "Decay"));
	        
	        for (int iter : saIterations) {
	        	for (double initTemp : saInitTemps) {
	        		for (double decay : saDecays) {
	        			double[] avgSAOptimals = new double[avgHowManyRuns];
	                	double[] avgSATimes = new double[avgHowManyRuns];
	                	for (int r = 0; r < avgHowManyRuns; r++) {
	        		        startTime = System.nanoTime();
	        		        SimulatedAnnealing sa = new SimulatedAnnealing(initTemp, decay, hcp);
	        		        FixedIterationTrainer fit = new FixedIterationTrainer(sa, iter);
	        		        fit.train();
	        		        endTime = System.nanoTime();
	        		        duration = (endTime - startTime);
	        		        avgSAOptimals[r] = ef.value(sa.getOptimal());
	        		        avgSATimes[r] = duration;
	                	}
	                	
	                	double avgSAOptimalSum = 0;
	                    for ( int i = 0; i < avgHowManyRuns; i++){
	                    	avgSAOptimalSum += avgSAOptimals[i];
	                       }
	                    double avgSAOptimal = 1.0d * avgSAOptimalSum/avgHowManyRuns;
	                	
	                	double avgSATimeSum = 0;
	                    for ( int i = 0; i < avgHowManyRuns; i++){
	                    	avgSATimeSum += avgSATimes[i];
	                       }
	                    double avgSATime = 1.0d * avgSATimeSum/avgHowManyRuns;
	        	        
	                    System.out.println("[SA] [Iteration " + String.valueOf(iter) + "] [Init_Temp " + String.valueOf(initTemp) + 
	                    		"] [Decay " + String.valueOf(decay) + "] ~~~ Fitness: " + 
	                    		String.valueOf(avgSAOptimal) + " ~~~ " + String.valueOf(1.0d * avgSATime/1000000000) + " seconds");
	                	CSVUtils.writeLine(saWriter, Arrays.asList(String.valueOf(avgHowManyRuns), String.valueOf(iter), "SA", String.valueOf(avgSAOptimal),String.valueOf(avgSATime),String.valueOf(initTemp),String.valueOf(decay)));
	
	        		}
	        	}
	        }
	        
	    	saWriter.flush();
	    	saWriter.close();
        }
        
        // GA
        if (gaIterations.length > 0) {
	        String gaCsvFile = filenameStart + "GA" + filenameEnd;
	        FileWriter gaWriter = new FileWriter(gaCsvFile);
	        CSVUtils.writeLine(gaWriter, Arrays.asList("Averaged Runs", "Iteration", "Algorithm", "Fitness", "Time (ns)", "Pop Size", "Mate Pct", "Mutate Pct"));
	        
	        for (int iter : gaIterations) {
	        	for (int popSize : gaPopSizes) {
	        		for (double matePct : gaMatePercents) {
	        			for (double mutPct : gaMutatePercents) {
		        			double[] avgGAOptimals = new double[avgHowManyRuns];
		                	double[] avgGATimes = new double[avgHowManyRuns];
		                	for (int r = 0; r < avgHowManyRuns; r++) {
		        		        startTime = System.nanoTime();
		        		        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(popSize, (int) Math.ceil(matePct * popSize), (int) Math.ceil(mutPct * popSize), gap);
		        		        FixedIterationTrainer fit = new FixedIterationTrainer(ga, iter);
		        		        fit.train();
		        		        endTime = System.nanoTime();
		        		        duration = (endTime - startTime);
		        		        avgGAOptimals[r] = ef.value(ga.getOptimal());
		        		        avgGATimes[r] = duration;
		                	}
		                	
		                	double avgGAOptimalSum = 0;
		                    for ( int i = 0; i < avgHowManyRuns; i++){
		                    	avgGAOptimalSum += avgGAOptimals[i];
		                       }
		                    double avgGAOptimal = 1.0d * avgGAOptimalSum/avgHowManyRuns;
		                	
		                	double avgGATimeSum = 0;
		                    for ( int i = 0; i < avgHowManyRuns; i++){
		                    	avgGATimeSum += avgGATimes[i];
		                       }
		                    double avgGATime = 1.0d * avgGATimeSum/avgHowManyRuns;
		        	        
		                    System.out.println("[GA] [Iteration " + String.valueOf(iter) + "] [Pop Size " + String.valueOf(popSize) + 
		                    		"] [Mate Pct " + String.valueOf(matePct) + 
		                    		"] [Mutate Pct " + String.valueOf(mutPct) +
		                    		"] ~~~ Fitness: " + 
		                    		String.valueOf(avgGAOptimal) + " ~~~ " + String.valueOf(1.0d * avgGATime/1000000000) + " seconds");
		                	CSVUtils.writeLine(gaWriter, Arrays.asList(String.valueOf(avgHowManyRuns), String.valueOf(iter), "GA", String.valueOf(avgGAOptimal),String.valueOf(avgGATime),String.valueOf(popSize),String.valueOf(matePct),String.valueOf(mutPct)));
	        			}
	        		}
	        	}
	        }
	        
	    	gaWriter.flush();
	    	gaWriter.close();
        }
        
        // MIMIC
        if (mmIterations.length > 0) {
	        String mmCsvFile = filenameStart + "MIMIC" + filenameEnd;
	        FileWriter mmWriter = new FileWriter(mmCsvFile);
	        CSVUtils.writeLine(mmWriter, Arrays.asList("Averaged Runs", "Iteration", "Algorithm", "Fitness", "Time (ns)", "Samples", "Keep Pct"));
	        
	        for (int iter : mmIterations) {
	        	for (int sample : mmSamples) {
	        		for (double keepPct : mmKeepPercents) {
	        			double[] avgMMOptimals = new double[avgHowManyRuns];
	                	double[] avgMMTimes = new double[avgHowManyRuns];
	                	for (int r = 0; r < avgHowManyRuns; r++) {
	        		        startTime = System.nanoTime();
	        		        MIMIC mimic = new MIMIC(sample, (int) Math.ceil(keepPct * sample), pop);
	        		        FixedIterationTrainer fit = new FixedIterationTrainer(mimic, iter);
	        		        fit.train();
	        		        endTime = System.nanoTime();
	        		        duration = (endTime - startTime);
	        		        avgMMOptimals[r] = mimic_ef.value(mimic.getOptimal());
	        		        avgMMTimes[r] = duration;
	                	}
	                	
	                	double avgMMOptimalSum = 0;
	                    for ( int i = 0; i < avgHowManyRuns; i++){
	                    	avgMMOptimalSum += avgMMOptimals[i];
	                       }
	                    double avgMMOptimal = 1.0d * avgMMOptimalSum/avgHowManyRuns;
	                	
	                	double avgMMTimeSum = 0;
	                    for ( int i = 0; i < avgHowManyRuns; i++){
	                    	avgMMTimeSum += avgMMTimes[i];
	                       }
	                    double avgMMTime = 1.0d * avgMMTimeSum/avgHowManyRuns;
	        	        
	                    System.out.println("[MIMIC] [Iteration " + String.valueOf(iter) + "] [Samples " + String.valueOf(sample) + 
	                    		"] [Keep Pct " + String.valueOf(keepPct) + "] ~~~ Fitness: " + 
	                    		String.valueOf(avgMMOptimal) + " ~~~ " + String.valueOf(1.0d * avgMMTime/1000000000) + " seconds");
	                	CSVUtils.writeLine(mmWriter, Arrays.asList(String.valueOf(avgHowManyRuns), String.valueOf(iter), "MIMIC", String.valueOf(avgMMOptimal),String.valueOf(avgMMTime),String.valueOf(sample),String.valueOf(keepPct)));
	
	        		}
	        	}
	        }
	        
	    	mmWriter.flush();
	    	mmWriter.close();
        }
        
    }
}
