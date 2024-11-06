import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
timesteps = 10  # Number of timesteps in sequence
hidden_size = 5  # Number of units in the LSTM cell
input_size = 1  # Size of input at each timestep

# Initialize random inputs
x = np.random.randn(timesteps, input_size)  # Sequential input data

# LSTM weights initialization (simplified, random values for demonstration)
Wf, Wi, Wo, Wc = [np.random.randn(hidden_size, hidden_size + input_size) for _ in range(4)]
bf, bi, bo, bc = [np.zeros(hidden_size) for _ in range(4)]

# Initialize cell state and hidden state
cell_state = np.zeros((timesteps, hidden_size))
hidden_state = np.zeros((timesteps, hidden_size))

# LSTM cell forward pass
for t in range(timesteps):
    combined = np.concatenate((x[t], hidden_state[t-1])) if t > 0 else np.concatenate((x[t], np.zeros(hidden_size)))
    
    # Forget gate
    ft = 1 / (1 + np.exp(-(np.dot(Wf, combined) + bf)))
    # Input gate
    it = 1 / (1 + np.exp(-(np.dot(Wi, combined) + bi)))
    # Candidate cell state
    ct_hat = np.tanh(np.dot(Wc, combined) + bc)
    # Update cell state
    cell_state[t] = ft * cell_state[t-1] + it * ct_hat
    
    # Output gate
    ot = 1 / (1 + np.exp(-(np.dot(Wo, combined) + bo)))
    # Update hidden state
    hidden_state[t] = ot * np.tanh(cell_state[t])

# Visualization of cell state and hidden state
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot cell states over time
ax1.plot(range(timesteps), cell_state, marker='o')
ax1.set_title("Cell States (Memory) Over Time")
ax1.set_xlabel("Timesteps")
ax1.set_ylabel("Cell State Value")
ax1.legend([f"Unit {i}" for i in range(hidden_size)])

# Plot hidden states over time
ax2.plot(range(timesteps), hidden_state, marker='x')
ax2.set_title("Hidden States Over Time")
ax2.set_xlabel("Timesteps")
ax2.set_ylabel("Hidden State Value")
ax2.legend([f"Unit {i}" for i in range(hidden_size)])

plt.tight_layout()
plt.show()
