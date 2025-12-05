import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

# 1. SETUP NETWORK
# We use 'tanh' because it fits waves naturally.
Net = Sequential([
    Input(shape=(1,)),
    Dense(50, activation="tanh"), 
    Dense(50, activation="tanh"),
    Dense(1, activation="linear")
])

# CRITICAL FIX 1: Boost the Learning Rate
# Default Adam is 0.001. We use 0.01 to learn faster.
Net.compile(loss='mean_squared_error', 
            optimizer=Adam(learning_rate=0.01))

def my_target(y):
    return np.sin(y)*np.sin(y/2)

# 2. TRAINING
# We train for more batches to ensure convergence
training_batches = 2000 
batchsize = 64
costs = np.zeros(training_batches)

# CRITICAL FIX 2: Safe Range
# We train on -6.0 to +6.0. This is roughly -2pi to +2pi.
# This avoids the "saturation" problem of huge numbers like 20.
range_width = 6.0 

print(f"Training on range [-{range_width}, +{range_width}]...")

for j in range(training_batches):
    # Sample uniformly from the safe range
    y_in = np.random.uniform(low=-range_width, high=+range_width, size=[batchsize,1])
    y_target = my_target(y_in)
    
    costs[j] = Net.train_on_batch(y_in, y_target)
    
    if j % 200 == 0:
        print(f"Batch {j}, Cost: {costs[j]:.5f}", end="\r")

# 3. PLOTTING
N = 400
y_in = np.zeros([N,1])

# Plot exactly the range we trained on
y_in[:,0] = np.linspace(-range_width, range_width, N) 
y_out = Net.predict_on_batch(y_in)

plt.figure(figsize=(10,6))
plt.plot(y_in, y_out, label="NN Prediction", linewidth=3, color="blue")
plt.plot(y_in, my_target(y_in), color="orange", label="True Function", linestyle='--', linewidth=2)
plt.title(f"Sine Wave Approximation (Range: -{range_width} to +{range_width})")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()