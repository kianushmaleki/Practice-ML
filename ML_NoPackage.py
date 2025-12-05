import numpy as np
import matplotlib.pyplot as plt
import scipy as sp


### This is a code to generate a simple machine learning code without using any ML packages.
# I has 1 hidden latyers
# it has two inputs and opne output
# the goal is to approximate an arbitrary function


# Assigin some initial values to the layers
def my_target(y):
    return np.sin(y)*np.sin(y/2)

# Let's define our cost function
def cost_function(y_sample, y_true):
    return 0.5 * np.sum((y_sample - y_true)**2)

# activation function
def f_act(x):
    return np.tanh(x), np.multiply(1.0 - np.tanh(x)**2,1.0) # return value and derivative

# learning rate
lr = 0.01

# Network architecture
N_in = 2 # input layer size
N_1 = 5 # hidden layer size
N_out =1 # output layer size

# making the arrays
y_in=np.zeros([N_in]) # input array
y_n1=np.zeros([N_1]) # hidden layer 1 array
y_out=np.zeros([N_out]) # output array

# Weights matrecis with some arbitrtary initial values
w_01=np.random.uniform(low=-1,high=+1,size=(N_1,N_in)) # weights from input to hidden layer 1
b_1=np.random.uniform(low=-0.5,high=+0.5,size=N_1) # biases for hidden layer 1
w_1out=np.random.uniform(low=-1,high=+1,size=(N_out,N_1)) # weights from hidden layer 2 to output
b_out=np.random.uniform(low=-0.5,high=+0.5,size=N_out) # biases for output layer


# calculating the initial value of each node
y_n1 = f_act(np.dot(w_01,y_in) + b_1)[0]
print("Initial hidden layer 1:", y_n1)
y_out = f_act(np.dot(w_1out,y_n1) + b_out)[0]
print("Initial output:", y_out)

# Training loop
training_batches = 1
batchsize = 32  
costs = np.zeros(training_batches)
for j in range(training_batches):
    # Sample uniformly from -10 to +10
    y_in = np.random.uniform(low=-10.0, high=+10.0, size=(N_in))
    y_target = my_target(y_in[0])  # Target function based on the first input

    # For these input parameters can our initial and abitrary weights approximate the target function?
    # Forward pass
    y_n1, dy_n1 = f_act(np.dot(w_01,y_in) + b_1) # Calculate the nodes in hidden layer 1
    y_out, dy_out = f_act(np.dot(w_1out,y_n1) + b_out) # Calculate the output node
    print("Input:", y_in)
    print("Target:", y_target)
    print("Output:", y_out) # it is obvious that thr output is way off

    # Compute cost : This steps quantifies "how off" we are
    costs[j] = cost_function(y_out, y_target)
    print("Cost:", costs[j]) 
    # the cost does not need to be an array here since we have only one output. 
    # But we keep it as an array for checking the cost over multiple batches

    # based on "how off" we are, we update the weights using backpropagation
    # Backpropagation


    ### continue from here
    error_out = (y_out - y_target) * dy_out
    print("Output layer error:", error_out)
    
    error_n1 = np.dot(w_1out.T, error_out) * dy_n1
   

    # Update weights and biases
    w_1out -= lr * np.outer(error_out, y_n1)
    b_out -= lr * error_out
    w_01 -= lr * np.outer(error_n1, y_in)
    b_1 -= lr * error_n1

    if j % 100 == 0:
        print(f"Batch {j}, Cost: {costs[j]:.5f}", end="\r")

