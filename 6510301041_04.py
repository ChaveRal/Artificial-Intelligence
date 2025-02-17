import keras.api.models as mod
import keras.api.layers as lay
import numpy as np
import matplotlib.pyplot as plt

# model = mod.Sequential()
# model.add(lay.SimpleRNN(units = 1,
#                         input_shape = (1,1),
#                         activation = "relu"))

# model.summary()
# model.save("RNN.h5")

# pitch = 20
# step = 1
# N = 100
# n_train = int(N*0.7) # 70% for Training test

# def gen_data(x):
#     return (x%pitch)/pitch

# t = np.arange(1, N+1)
# y = [gen_data(i) for i in t]
# y = np.array(y)
# y = np.sin(0.05*t*10) + 0.8 * np.random.rand(N)

# def convertToMatrix(data, step=1):
#     X, Y = [], []
#     for i in range(len(data)-step):
#         d = i + step
#         X.append(data[i:d,])
#         Y.append(data[d,])
#     return np.array(X), np.array(Y)

# train, test = y[0:n_train], y[n_train:N]

# x_train, y_train = convertToMatrix(train, step)
# x_test, y_test = convertToMatrix(test, step)

# print("Dimension (Before): ", train.shape, test.shape)
# print("Dimenion (After): ", x_train.shape, x_test.shape)


# Model = mod.Sequential()
# Model.add(lay.SimpleRNN(units=32,
#                         input_shape=(step,1),
#                         activation="relu"))
# Model.add(lay.Dense(units=1))

# Model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])
# hist = Model.fit(x_train, y_train, epochs=30, batch_size=1, verbose=1)

# plt.figure()
# plt.plot(y)

# # plt.plot(hist.history['loss'])
# plt.show()

#............................................................................

# Define the model
model = mod.Sequential()
model.add(lay.SimpleRNN(units=1, input_shape=(1, 1), activation="relu"))
model.summary()

# Save the model
model.save("RNN.h5")

# Generate data (Adjust the signal characteristics here as per the problem statement)
pitch = 20
step = 1
N = 100
n_train = int(N * 0.7)  # 70% for Training test

def gen_data(x):
    return (x % pitch) / pitch

t = np.arange(1, N + 1)
y_original = [gen_data(i) for i in t]
y_original = np.array(y_original)

# Modify signal characteristics (change the function here)
y = np.sin(0.05 * t * 10) + 0.8 * np.random.rand(N)

def convertToMatrix(data, step=1):
    X, Y = [], []
    for i in range(len(data) - step):
        d = i + step
        X.append(data[i:d, ])
        Y.append(data[d, ])
    return np.array(X), np.array(Y)

# Split data
train, test = y[0:n_train], y[n_train:N]
x_train, y_train = convertToMatrix(train.reshape(-1, 1), step)
x_test, y_test = convertToMatrix(test.reshape(-1, 1), step)

print("Dimension (Before):", train.shape, test.shape)
print("Dimension (After):", x_train.shape, x_test.shape)

# Define and train the model
Model = mod.Sequential()
Model.add(lay.SimpleRNN(units=32, input_shape=(step, 1), activation="relu"))
Model.add(lay.Dense(units=1))

Model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])
hist = Model.fit(x_train, y_train, epochs=30, batch_size=1, verbose=1)

# Make predictions
train_predict = Model.predict(x_train)
test_predict = Model.predict(x_test)

# Flatten predictions
train_predict = train_predict.flatten()
test_predict = test_predict.flatten()

# Combine train and test predictions
predicted = np.concatenate([train_predict, test_predict])

# Define a function to plot the comparison
def plot_comparison(t, y, train_predict, test_predict, n_train, step):
    plt.figure(figsize=(10, 6))
    plt.plot(t, y, label="Original", color="blue", linestyle="-")
    plt.plot(t[step:n_train], train_predict, label="Train Predict", color="red", linestyle="--")
    plt.plot(t[n_train + step:], test_predict, label="Test Predict", color="red", linestyle="--")

    # Add a vertical line to separate train and test
    plt.axvline(x=n_train, color="purple", linestyle="-", linewidth=2)

    # Add labels and legend
    plt.title("Comparison of Original vs Predicted")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

# Call the function to plot the comparison
plot_comparison(t, y, train_predict, test_predict, n_train, step)

# Additional experiment: Change signal input characteristics
y_new = np.cos(0.1 * t * 5) + 0.5 * np.random.rand(N)  # Experiment with new signal characteristics

# Repeat training and prediction steps if needed with new signal
# Update train/test split and retrain model
