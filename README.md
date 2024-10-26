# Project Overview

This repository implements logistic regression for binary classification tasks using Python and NumPy. It includes two scripts that analyze different datasets related to admissions and microchip tests.

To run this project you are first required to run these commands:
```
pip install numpy
pip install matplotlib
```

## Basic Logistic Regression
This script implements a simple logistic regression model for classifying admission results based on exam scores.

### Key Features:
- Data Visualization: Loads and visualizes training data from ex2data1.txt, illustrating the distinction between admitted and not admitted candidates.
- Sigmoid Function: Applies the sigmoid function to generate probability scores.
- Cost Function Calculation: Evaluates the average loss across training examples.
- Gradient Calculation: Computes gradients for weights and bias to facilitate optimization.
- Gradient Descent Optimization: Performs parameter updates iteratively, monitoring the cost.
- Decision Boundary Visualization: Visualizes the linear decision boundary derived from model parameters.
- Model Evaluation: Calculates and prints training accuracy.

## Logistic Regression with feature mapping

This script utilizes logistic regression to classify microchip test results, employing feature mapping to handle non-linear boundaries.

### Key Features:

- Data Visualization: Loads and visualizes training data from ex2data2.txt, distinguishing between accepted and rejected microchips.
- Feature Mapping: Transforms the input features into a higher-dimensional space to improve classification accuracy.
#### Regularized Logistic Regression:
- Implements the sigmoid function for probability calculations.

- Computes the regularized cost to enhance model robustness.
- Derives gradients for efficient parameter updates.
- Gradient Descent Optimization: Adjusts model parameters through iterative gradient descent while tracking cost convergence.
- Decision Boundary Visualization: Plots the decision boundary to illustrate classification effectiveness.
- Model Evaluation: Computes and displays training accuracy.


