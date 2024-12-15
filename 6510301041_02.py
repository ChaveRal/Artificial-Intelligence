"""
Stud ID: 6510301041
Name   : Cheewapron Sutus
"""
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.datasets import make_blobs

# Generate adjusted data for two clusters
X1_adjusted, y1_adjusted = make_blobs(n_samples=100,
                                      n_features=2,
                                      centers=[(-1, -1)],  # Center for Class 1 (purple)
                                      cluster_std=0.4,
                                      random_state=69)

X2_adjusted, y2_adjusted = make_blobs(n_samples=100,
                                      n_features=2,
                                      centers=[(1, 1)],  # Center for Class 2 (yellow)
                                      cluster_std=0.4,
                                      random_state=69)

# Combine data and labels
X = np.vstack([X1_adjusted, X2_adjusted])
y = np.hstack([np.zeros(len(X1_adjusted)), np.ones(len(X2_adjusted))])  # Labels: 0 and 1

# Perceptron Class
class Perceptron:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr  # Learning rate
        self.epochs = epochs  # Number of training epochs
        self.weights = None  # Weight vector
        self.bias = None  # Bias term

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)  # Initialize weights
        self.bias = 0  # Initialize bias
        y_ = np.where(y == 1, 1, -1)  # Convert labels to +1 and -1

        # Training loop
        for _ in range(self.epochs):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                if y_[idx] * linear_output <= 0:  # Misclassified point
                    self.weights += self.lr * y_[idx] * x_i
                    self.bias += self.lr * y_[idx]

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.sign(linear_output)

# Train perceptron
perceptron = Perceptron(lr=0.1, epochs=1000)
perceptron.fit(X, y)

# Plot decision boundary learned by the perceptron
def plot_decision_boundary(X, y, model):
    # Generate grid of points
    x1_range = np.linspace(-3, 3, 500)
    x2_range = np.linspace(-3, 3, 500)
    x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)

    # Calculate decision values
    grid = np.c_[x1_grid.ravel(), x2_grid.ravel()]
    predictions = model.predict(grid).reshape(x1_grid.shape)

    # Plot decision regions
    plt.contourf(x1_grid, x2_grid, predictions, levels=[-np.inf, 0, np.inf], colors=['red', 'blue'], alpha=0.5)
    plt.contour(x1_grid, x2_grid, predictions, levels=[0], colors='black', linewidths=2)  # Decision boundary

    # Scatter plot of data points
    plt.scatter(X1_adjusted[:, 0], X1_adjusted[:, 1], c='purple', s=20, label="Class 1")
    plt.scatter(X2_adjusted[:, 0], X2_adjusted[:, 1], c='yellow', s=20, label="Class 2")

    # Configure plot
    plt.xlabel('Feature x1', fontsize=10)
    plt.ylabel('Feature x2', fontsize=10)
    plt.title('Decision Plane', fontsize=14)
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.axhline(0, color='gray', linestyle='--', linewidth=1.5)  # Horizontal axis
    plt.axvline(0, color='gray', linestyle='--', linewidth=1.5)  # Vertical axis
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

# Plot the decision boundary
plot_decision_boundary(X, y, perceptron)



