import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.api.models import Sequential
from keras.api.layers import Dense
from keras.api.optimizers import Adam
import pandas as pd

# -----------------------------------------
# STEP 1: LOAD TITANIC DATA
# -----------------------------------------
url = "titanic.csv"
data = pd.read_csv(url)

# Selecting features and target variable (Age, Fare, Survived)
X_titanic = data[['Age', 'Fare']].values
y_titanic = data['Survived'].values

# Handle missing values by replacing them with minimum values
X_titanic[np.isnan(X_titanic)] = np.nanmin(X_titanic)

# Standardize Titanic data
scaler = StandardScaler()
X_titanic = scaler.fit_transform(X_titanic)

# -----------------------------------------
# STEP 2: GENERATE BLOB DATA
# -----------------------------------------
X_blob1, _ = make_blobs(n_samples=100, centers=[(2.0, 2.0)], cluster_std=0.75, random_state=69)
X_blob2, _ = make_blobs(n_samples=100, centers=[(3.0, 3.0)], cluster_std=0.75, random_state=69)

# Combine blob datasets
X_blob = np.vstack([X_blob1, X_blob2])
y_blob = np.hstack([np.zeros(len(X_blob1)), np.ones(len(X_blob2))])

# Standardize blob data
X_blob = scaler.fit_transform(X_blob)

# -----------------------------------------
# STEP 3: NEURAL NETWORK MODEL
# -----------------------------------------
# Build and train a neural network on blob data
model_nn = Sequential([
    Dense(10, activation='relu', input_shape=(2,)),
    Dense(1, activation='sigmoid')
])
model_nn.compile(optimizer=Adam(learning_rate=0.01), loss='binary_crossentropy', metrics=['accuracy'])
model_nn.fit(X_blob, y_blob, epochs=20, batch_size=8, verbose=0)

# -----------------------------------------
# STEP 4: PLOT DECISION BOUNDARIES
# -----------------------------------------
def plot_decision_boundary(X, y, model, model_type='Neural Network'):
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    x1_range, x2_range = np.linspace(x1_min, x1_max, 500), np.linspace(x2_min, x2_max, 500)
    x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
    
    grid = np.c_[x1_grid.ravel(), x2_grid.ravel()]
    if model_type == 'Neural Network':
        predictions = (model.predict(grid) > 0.5).astype(int).reshape(x1_grid.shape)
    
    # Plot decision boundaries
    plt.figure(figsize=(8, 6))
    plt.contourf(x1_grid, x2_grid, predictions, levels=[-1, 0, 1], colors=['red', 'blue'], alpha=0.5)
    plt.scatter(X[y == 1][:, 0], X[y == 0][:, 1], c='red', edgecolor='k', label='Class 0')
    plt.scatter(X[y == 0][:, 0], X[y == 1][:, 1], c='blue', edgecolor='k', label='Class 1')
    plt.title(f'{model_type} Decision Boundary')
    plt.xlabel('Feature x1')
    plt.ylabel('Feature x2')
    plt.legend()
    plt.show()

# Plot decision boundaries for Neural Network
print("Neural Network Decision Boundary (Blob Data):")
plot_decision_boundary(X_blob, y_blob, model_nn, model_type='Neural Network')

# -----------------------------------------
# STEP 5: TITANIC TABLE PREVIEW
# -----------------------------------------
print("\nTitanic Dataset Preview (Age and Fare):")
print(pd.DataFrame(X_titanic, columns=['Age', 'Fare']).head())