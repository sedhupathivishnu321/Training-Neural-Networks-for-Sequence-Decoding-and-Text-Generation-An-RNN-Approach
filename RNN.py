import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
timesteps = 10  # number of timesteps in sequence
hidden_size = 5  # number of units in the RNN layer
learning_rate = 0.01

# Initialize random inputs and weights
x = np.random.randn(timesteps)  # sequential input data
W_h = np.random.randn(hidden_size, hidden_size)  # hidden state weights
W_x = np.random.randn(hidden_size)  # input to hidden weights
b = np.zeros(hidden_size)  # bias term

# Initialize hidden state
h = np.zeros((timesteps, hidden_size))
h_grad = np.zeros((timesteps, hidden_size))  # gradient of hidden state for visualization

# Forward pass: Compute hidden states
for t in range(timesteps):
    h[t] = np.tanh(np.dot(W_h, h[t-1]) + W_x * x[t] + b)

# Backward pass: Calculate gradients
h_grad[-1] = np.random.randn(hidden_size)  # initial gradient for final timestep
for t in reversed(range(timesteps - 1)):
    h_grad[t] = h_grad[t + 1] * (1 - h[t] ** 2)  # backpropagation through time (simplified)

# Visualization of hidden states and gradients
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

# Plot forward hidden states
ax1.plot(range(timesteps), h, marker='o')
ax1.set_title("Hidden States (Forward Pass)")
ax1.set_xlabel("Timesteps")
ax1.set_ylabel("Hidden State Value")
ax1.legend([f"Unit {i}" for i in range(hidden_size)])

# Plot gradients of hidden states in backward pass
ax2.plot(range(timesteps), h_grad, marker='x')
ax2.set_title("Gradients of Hidden States (Backward Pass)")
ax2.set_xlabel("Timesteps")
ax2.set_ylabel("Gradient Value")
ax2.legend([f"Grad Unit {i}" for i in range(hidden_size)])

plt.tight_layout()
plt.show()
