import numpy as np
import matplotlib.pyplot as plt
from utils import *

# Load and visualize data
X_train, y_train = load_data("data/ex2data2.txt")
print(X_train.shape)
print(y_train.shape)
print(X_train[:5])

# Plot examples
plot_data(X_train, y_train[:], pos_label="Accepted", neg_label="Rejected")
plt.ylabel('Microchip Test 2')
plt.xlabel('Microchip Test 1')
plt.legend(loc="upper right")
plt.show()

# Feature Mapping
print("Original shape of data:", X_train.shape)
mapped_X = map_feature(X_train[:, 0], X_train[:, 1])
print("Shape after feature mapping:", mapped_X.shape)
print("X_train[0]:", X_train[0])
print("mapped X_train[0]:", mapped_X[0])

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Regularized cost function
def compute_cost_reg(X, y, w, b, lambda_=1):
    m = X.shape[0]
    cost = 0

    # Compute loss for each example
    for i in range(m):
        f_wb = np.dot(X[i], w) + b
        loss = -y[i] * np.log(sigmoid(f_wb)) - (1 - y[i]) * np.log(1 - sigmoid(f_wb))
        cost += loss

    # Average over m examples
    cost /= m

    # Add regularization term (excluding the bias term)
    reg = (lambda_ / (2 * m)) * np.sum(np.square(w))
    cost_reg = cost + reg
    return cost_reg

# Gradient for regularized logistic regression
def compute_gradient_reg(X, y, w, b, lambda_=1):
    m, n = X.shape
    dj_dw = np.zeros(w.shape)
    dj_db = 0

    # Loop through each example
    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i], w) + b)
        error = f_wb_i - y[i]

        # Update gradients
        dj_db += error  # Sum the error for the bias gradient
        dj_dw += error * X[i]  # Sum the error * input for each feature

    # Average over the batch
    dj_db /= m
    dj_dw /= m

    # Regularize the weights gradient (but not the bias)
    dj_dw += (lambda_ / m) * w

    return dj_db, dj_dw

# Gradient descent function
def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_):
    m = len(X)
    J_history = []
    w_history = []

    for i in range(num_iters):
        # Calculate the gradient and update the parameters
        dj_db, dj_dw = gradient_function(X, y, w_in, b_in, lambda_)
        w_in = w_in - alpha * dj_dw
        b_in = b_in - alpha * dj_db

        # Save cost J at each iteration
        if i < 100000:
            cost = cost_function(X, y, w_in, b_in, lambda_)
            J_history.append(cost)

        # Print cost every 10 iterations
        if i % max(1, num_iters // 10) == 0 or i == (num_iters - 1):
            w_history.append(w_in)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}")

    return w_in, b_in, J_history, w_history

# Feature-mapped data
X_mapped = map_feature(X_train[:, 0], X_train[:, 1])

# Initialize parameters and compute cost
np.random.seed(1)
initial_w = np.random.rand(X_mapped.shape[1]) - 0.5
initial_b = 1.0
lambda_ = 0.01
iterations = 10000
alpha = 0.01

# Run gradient descent
w, b, J_history, _ = gradient_descent(X_mapped, y_train, initial_w, initial_b,
                                      compute_cost_reg, compute_gradient_reg,
                                      alpha, iterations, lambda_)

# Print final cost to check if gradient descent worked well
print("Final cost after gradient descent:", J_history[-1])


def plot_decision_boundary(w, b, X, y):
    # Plot data points
    plot_data(X[:, 0:2], y, pos_label="Accepted", neg_label="Rejected")

    # Check if decision boundary is linear or non-linear
    if X.shape[1] <= 2:
        # For linear decision boundary
        plot_x = np.array([min(X[:, 0]), max(X[:, 0])])
        plot_y = (-1. / w[1]) * (w[0] * plot_x + b)

        plt.plot(plot_x, plot_y, c="b", label="Decision Boundary")
    else:
        # For non-linear decision boundary
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)

        z = np.zeros((len(u), len(v)))

        # Evaluate z = theta*x over the grid
        for i in range(len(u)):
            for j in range(len(v)):
                # Compute z values for the contour plot using map_feature
                z[i, j] = sigmoid(np.dot(map_feature(u[i], v[j]), w) + b).item()

        # Transpose z before calling contour
        z = z.T

        # Plot z = 0.5 contour line as decision boundary
        plt.contour(u, v, z, levels=[0.5], colors="g")

    plt.xlabel("Microchip Test 1")
    plt.ylabel("Microchip Test 2")
    plt.legend(loc="upper right")
    plt.show()

# Call the function to plot the decision boundary
plot_decision_boundary(w, b, X_mapped, y_train)


def predict(X, w, b):
    m, n = X.shape
    p = np.zeros(m)
    for i in range(m):
        z_wb = np.dot(X[i], w) + b  # Calculate z = w*x + b directly
        f_wb = sigmoid(z_wb)
        p[i] = 1 if f_wb > 0.5 else 0  # Threshold at 0.5
    return p

p = predict(X_mapped, w, b)
print("Train Accuracy : %f"%(np.mean(p == y_train)*100))
