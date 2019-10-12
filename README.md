ABAGAIL
=======

Changes in this Fork
--------------------

* (2019-10-11) Added several new test cases, images, and results
* (2019-10-11) Added `LargestWeightCrossOver` crossover function
	* Used with `GeneticAlgorithmProblem` with continuous data
	* Creates a child sample by looking at the absolute value of each parent's respective bits and using the one with largest magnitude.
* (2019-10-11) Added `AveragedCrossOver` crossover function
	* Used with `GeneticAlgorithmProblem` with continuous data
	* Creates a child sample based on averaging the respective bits of data between the two parents
* (2019-10-07) Bugfix to `Neuron` connections
	* Added a check to prevent nodes in one layer of a neural network from being able to connect to the bias node in the next layer.
* (2019-10-07) Added `ReLU` activation function
	* NOTE: Defaulted `BackPropagationNetworkFactory.createClassificationNetwork()` to utilize ReLU activation.
* (2019-10-07) Added `Order1CrossOver` crossover function
	* Used with `GeneticAlgorithmProblem`.
	* Maintains permutation integrity when mating two parents.
	* Uses a random length of data from parent 1 (in order), then fills in the remainder with the unused values in the order they appear in parent 2.
* (2019-10-03) Added `RestartingRandomizedHillClimbing` algorithm
	* Similar to `RandomizedHillClimbing`, but takes an additional `int restartThreshold` parameter.
	* If more than `restartThreshold` iterations pass without any fitness improvement, a new random state is selected.
	* [Not guaranteed not to break other things depending on what you are doing... so use with caution!]

[![Build Status](https://travis-ci.org/pushkar/ABAGAIL.svg?branch=master)](https://travis-ci.org/pushkar/ABAGAIL)

The library contains a number of interconnected Java packages that implement machine learning and artificial intelligence algorithms. These are artificial intelligence algorithms implemented for the kind of people that like to implement algorithms themselves.

Usage
------

* See [FAQ](https://github.com/pushkar/ABAGAIL/blob/master/faq.md)
* See [Wiki](https://github.com/pushkar/ABAGAIL/wiki)

Issues
-------

See [Issues page](https://github.com/pushkar/ABAGAIL/issues?state=open).

Contributing
------------

1. Fork it.
2. Create a branch (`git checkout -b my_branch`)
3. Commit your changes (`git commit -am "Awesome feature"`)
4. Push to the branch (`git push origin my_branch`)
5. Open a [Pull Request][1]
6. Enjoy a refreshing Diet Coke and wait 

Features
========

### Hidden Markov Models

* Baum-Welch reestimation algorithm, scaled forward-backward algorithm, Viterbi algorithm
* Support for Input-Output Hidden Markov Models
* Write your own output or transition probability distribution or use the provided distributions, including neural network based conditional probability distributions
* Neural Networks

### Feed-forward backpropagation neural networks of arbitrary topology
* Configurable error functions with sum of squares, weighted sum of squares
* Multiple activation functions with logistic sigmoid, linear, tanh, and soft max
* Choose your weight update rule with standard update rule, standard update rule with momentum, Quickprop, RPROP
* Online and batch training
* Support Vector Machines

### Fast training with the sequential minimal optimization algorithm
* Support for linear, polynomial, tanh, radial basis function kernels
* Decision Trees

### Information gain or GINI index split criteria
* Binary or all attribute value splitting
* Chi-square signifigance test pruning with configurable confidence levels
* Boosted decision stumps with AdaBoost
* K Nearest Neighbors

### Fast kd-tree implementation for instance based algorithms of all kinds
* KNN Classifier with weighted or non-weighted classification, customizable distance function
* Linear Algebra Algorithms

### Basic matrix and vector math, a variety of matrix decompositions based on the standard algorithms
* Solve square systems, upper triangular systems, lower triangular systems, least squares
* Singular Value Decomposition, QR Decomposition, LU Decomposition, Schur Decomposition, Symmetric Eigenvalue Decomposition, Cholesky Factorization
* Make your own matrix decomposition with the easy to use Householder Reflection and Givens Rotation classes
* Optimization Algorithms

### Randomized hill climbing, simulated annealing, genetic algorithms, and discrete dependency tree MIMIC
* Make your own crossover functions, mutation functions, neighbor functions, probability distributions, or use the provided ones.
* Optimize the weights of neural networks and solve travelling salesman problems
* Graph Algorithms

### Kruskals MST and DFS
* Clustering Algorithms

### EM with gaussian mixtures, K-means
* Data Preprocessing

### PCA, ICA, LDA, Randomized Projections
* Convert from continuous to discrete, discrete to binary
* Reinforcement Learning

### Value and policy iteration for Markov decision processes

[1]: https://help.github.com/articles/using-pull-requests
