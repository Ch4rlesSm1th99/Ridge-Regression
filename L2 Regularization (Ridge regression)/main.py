import numpy as np
import matplotlib.pyplot as plt


def generate_data(n_samples=100):
    np.random.seed(0)
    X = 2 - 3 * np.random.normal(0, 1, n_samples)
    y = X - 2 * (X ** 2) + 0.5 * (X ** 3) + np.random.normal(-3, 3, n_samples)
    X = X[:, np.newaxis]
    y = y[:, np.newaxis]
    return X, y


def normalize_features(X):
    mean = np.mean(X, axis=0)
    std_dev = np.std(X, axis=0)
    X_normalized = (X - mean) / std_dev
    return X_normalized


# New function: Compute the L2 norm (Euclidean norm)
def l2_norm(weights):
    return np.sqrt(np.sum(weights ** 2))  # Calculate L2 norm of weights


def polynomial_regression(X, y, degree, l2_penalty, lr, n_iterations):
    X_poly = X
    for d in range(2, degree + 1):
        X_poly = np.hstack((X_poly, np.power(X, d)))

    X_poly = normalize_features(X_poly)

    X_poly = np.hstack((np.ones((X_poly.shape[0], 1)), X_poly))

    weights = np.random.randn(X_poly.shape[1], 1)

    for i in range(n_iterations):
        predictions = np.dot(X_poly, weights)
        residuals = predictions - y
        # Modification for L2: The gradient calculation now includes the L2 penalty term
        gradients = 2 / X.shape[0] * np.dot(X_poly.T, residuals) + l2_penalty * weights  # L2 regularization term added
        weights -= lr * gradients

    return weights, X_poly


def plot_predictions(X, y, weights, X_poly):
    plt.scatter(X, y, color='blue', s=30, marker='o', label="Input data")

    X_continuous = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    X_continuous_poly = X_continuous
    for d in range(2, X_poly.shape[1]):
        X_continuous_poly = np.hstack((X_continuous_poly, np.power(X_continuous, d)))
    X_continuous_poly = normalize_features(X_continuous_poly)
    X_continuous_poly = np.hstack((np.ones((X_continuous_poly.shape[0], 1)), X_continuous_poly))

    predictions = np.dot(X_continuous_poly, weights)
    plt.plot(X_continuous, predictions, color='red', label="Fitted polynomial regression")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("Polynomial Regression with L2 regularization")  # Updated title to reflect L2 regularization
    plt.legend(loc='upper right')
    plt.show()


X, y = generate_data(100)

degree = 2
l2_penalty = 0.1  # This is now the L2 regularization penalty (lambda)
lr = 0.01
n_iterations = 10000

weights, X_poly = polynomial_regression(X, y, degree, l2_penalty, lr, n_iterations)

plot_predictions(X, y, weights, X_poly)

# Print L2 norm of the weights
print(f"L2 norm of weights: {l2_norm(weights)}")  # Updated to calculate and print the L2 norm

# Print final weights
print("Final weights are:", weights)
