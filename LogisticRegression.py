import numpy as np
import matplotlib.pyplot as plt
from utils import *
import copy
import math

# Seeing and Understanding the data
X_train, Y_train = load_data("data/ex2data1.txt")
print(X_train.shape, Y_train.shape)
print(X_train[:5])
print(Y_train[:5])
plot_data(X_train, Y_train, pos_label="Admitted", neg_label="Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend(loc="upper right")
plt.show()

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

print(sigmoid(0))

# Computing Function for logistic regression
def compute_cost(w, b, X, Y):
    m, n = X.shape
    cost = 0
    for i in range(m):
        Z = np.dot(w, X[i]) + b
        f_wb = sigmoid(Z)
        loss = (-Y[i] * np.log(f_wb) - (1 - Y[i]) * np.log(1 - f_wb))
        cost += loss
    total_cost = cost / m
    return total_cost

# Initialize weights as a 1D array
np.random.seed(1)
initial_w = 0.01 * (np.random.rand(2) - 0.5)  # Shape is now (2,)
initial_b = -8

# Update the compute_gradient function
def compute_gradient(w, b, X, Y, lambda_=None):
    m, n = X.shape
    dj_dw = np.zeros(w.shape)  # Shape will be (2,)
    dj_db = 0

    for i in range(m):
        f_wb_i = sigmoid(np.dot(w, X[i]) + b)  # np.dot(w, X[i]) should now work
        err_i = f_wb_i - Y[i]
        dj_dw += err_i * X[i]  # Shape is compatible now
        dj_db += err_i

    dj_db /= m
    dj_dw /= m
    return dj_dw, dj_db

# Gradient descent function
def gradient_descent(X, Y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_):
    m = len(X)
    J_history = []
    w_history = []

    for i in range(num_iters):
        # Calculate the gradient and update the parameters
        dj_dw, dj_db = gradient_function(w_in, b_in, X, Y, lambda_)
        w_in = w_in - alpha * dj_dw
        b_in = b_in - alpha * dj_db

        # Save cost J at each iteration
        if i < 100000:
            cost = cost_function(w_in, b_in, X, Y)
            J_history.append(cost)

        # Print cost every 10 iterations
        if i % max(1, num_iters // 10) == 0 or i == (num_iters - 1):
            w_history.append(w_in)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}")

    return w_in, b_in, J_history, w_history

# Some gradient descent settings
iterations = 10000
alpha = 0.001

# Running gradient descent
w, b, J_history, _ = gradient_descent(X_train, Y_train, initial_w, initial_b,
                                      compute_cost, compute_gradient, alpha, iterations, 0)

# Plotting data and decision boundary
plot_data(X_train, Y_train, pos_label="Admitted", neg_label="Not Admitted")

# Calculate decision boundary line
x_values = np.array([np.min(X_train[:, 0]), np.max(X_train[:, 0])])
y_values = -(w[0] * x_values + b) / w[1]  # Rearranged line equation for x2: w0*x1 + w1*x2 + b = 0

# Plot the decision boundary
plt.plot(x_values, y_values, color="blue", label="Decision Boundary")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend(loc="upper right")
plt.show()


def predict(X, w, b):
    m, n = X.shape
    p = np.zeros(m)
    for i in range(m):
        z_wb = np.dot(X[i], w) + b  # Calculate z = w*x + b directly
        f_wb = sigmoid(z_wb)
        p[i] = 1 if f_wb > 0.5 else 0  # Threshold at 0.5
    return p


p = predict(X_train, w, b)
print('Train Accuracy : %f'%(np.mean(p == Y_train)*100))