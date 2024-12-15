import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from keras.api.models import Sequential
from keras.api.layers import Dense
from keras.api.optimizers import Adam

# Step 1: Generate two datasets (A and B) using make_blobs
X1_adjusted, y1_adjusted = make_blobs(n_samples=100,
                                      n_features=2,
                                      centers=[(2.0, 2.0)],  # Corrected center for Class 1
                                      cluster_std=0.75,
                                      random_state=69)

X2_adjusted, y2_adjusted = make_blobs(n_samples=100,
                                      n_features=2,
                                      centers=[(3.0, 3.0)],  # Corrected center for Class 2
                                      cluster_std=0.75,
                                      random_state=69)

# Combine the two datasets into one
X = np.vstack([X1_adjusted, X2_adjusted])
y = np.hstack([np.zeros(len(X1_adjusted)), np.ones(len(X2_adjusted))])  # Labels: 0 and 1

# Step 2: Neural Network Model
model_nn = Sequential()
model_nn.add(Dense(16, input_dim=2, activation='relu'))  # 2 features
model_nn.add(Dense(1, activation="sigmoid"))  # Single output for binary classification
optimizer = Adam(0.0001)
model_nn.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Train the neural network
model_nn.fit(X, y, epochs=300, batch_size=200, verbose=0)

# Step 3: Perceptron Model
class Perceptron:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        y_ = np.where(y == 1, 1, -1)

        for _ in range(self.epochs):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                if y_[idx] * linear_output <= 0:  # Misclassified point
                    self.weights += self.lr * y_[idx] * x_i
                    self.bias += self.lr * y_[idx]

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.sign(linear_output)

# Train the perceptron model
perceptron = Perceptron(lr=0.1, epochs=1000)
perceptron.fit(X, y)

# Function to rotate the decision boundary by a given angle
def rotate_angle(weights, bias, angle_deg):
    angle_rad = np.radians(angle_deg)
    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                [np.sin(angle_rad), np.cos(angle_rad)]])
    rotated_weights = np.dot(rotation_matrix, weights)
    return rotated_weights, bias

# Function to plot the decision boundary for perceptron with rotation
def plot_decision_boundary_angle(X, y, model, angle_deg, model_type='Perceptron'):
    x1_range = np.linspace(-3, 3, 500)
    x2_range = np.linspace(-3, 3, 500)
    x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)

    if model_type == 'Perceptron':
        rotated_weights, rotated_bias = rotate_angle(model.weights, model.bias, angle_deg)
        grid = np.c_[x1_grid.ravel(), x2_grid.ravel()]
        decision_values = np.dot(grid, rotated_weights) + rotated_bias
    else:  # Neural Network
        grid = np.c_[x1_grid.ravel(), x2_grid.ravel()]
        decision_values = model.predict(grid)
    
    predictions = np.sign(decision_values).reshape(x1_grid.shape)
    predictions = -predictions  # Flip regions

    plt.figure(figsize=(8, 6))
    plt.contourf(x1_grid, x2_grid, predictions, levels=[-np.inf, 0, np.inf], colors=['red', 'blue'], alpha=0.5)
    plt.contour(x1_grid, x2_grid, predictions, levels=[0], colors='black', linewidths=2)  # Decision boundary

    plt.scatter(X1_adjusted[:, 0], X1_adjusted[:, 1], c='red', s=20, label="Class 1")
    plt.scatter(X2_adjusted[:, 0], X2_adjusted[:, 1], c='blue', s=20, label="Class 2")

    plt.xlabel('Feature x1')
    plt.ylabel('Feature x2')
    plt.title(f'Decision Boundary ({model_type} Model, Rotated by {angle_deg}°)', fontsize=14)

    # Set the x and y limits to be from -3 to 3
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)

    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

# Plot decision boundaries for both models
angle_deg = 270
plot_decision_boundary_angle(X, y, perceptron, angle_deg, model_type='Perceptron')
plot_decision_boundary_angle(X, y, model_nn, angle_deg, model_type='Neural Network')
