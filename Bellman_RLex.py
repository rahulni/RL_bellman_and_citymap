import numpy as np
# NumPy is used for numerical operations and easy reshaping of arrays

# -------------------------------
# 1. Grid and Environment Setup
# -------------------------------

N = 4                          # Grid size: 4x4
num_states = N * N             # Total number of states = 16

# -------------------------------
# 2. Hyperparameters
# -------------------------------

gamma = 1.0                    # Discount factor (no discounting)
theta = 1e-4                   # Convergence threshold

# -------------------------------
# 3. Initialize Value Function
# -------------------------------

# V[s] represents the value of state s
# Initialize all state values to 0 as per the problem statement
V = np.zeros(num_states)

# -------------------------------
# 4. Terminal State Definition
# -------------------------------

# Bottom-right corner of the grid is the terminal state
# State indexing is row-major, so last index = 15
terminal_state = num_states - 1

# -------------------------------
# 5. Action Space
# -------------------------------

# Actions represented as (row change, column change)
# Up, Down, Left, Right
actions = [
    (-1, 0),   # Up
    (1, 0),    # Down
    (0, -1),   # Left
    (0, 1)     # Right
]

# Each action is taken with equal probability (0.25)

# -------------------------------
# 6. Helper Functions
# -------------------------------

def state_to_pos(s):
    """
    Convert a 1D state index to a 2D grid position (row, col)
    Example: state 6 -> (1, 2)
    """
    return divmod(s, N)

def pos_to_state(r, c):
    """
    Convert a 2D grid position (row, col) back to a 1D state index
    Example: (1, 2) -> 6
    """
    return r * N + c

# -------------------------------
# 7. Value Iteration Loop
# -------------------------------

while True:
    delta = 0                  # Tracks maximum change in value function
    V_new = V.copy()           # Copy old values to compute updates safely

    # Iterate over all states
    for s in range(num_states):

        # Skip terminal state (value remains 0)
        if s == terminal_state:
            continue

        # Convert state index to grid coordinates
        r, c = state_to_pos(s)

        value = 0              # Accumulate expected value for state s

        # -------------------------------
        # 8. Bellman Expectation Update
        # -------------------------------

        # For each possible action
        for dr, dc in actions:

            # Compute tentative next position
            nr, nc = r + dr, c + dc

            # Boundary handling:
            # If action takes agent outside the grid,
            # agent stays in the same state
            if nr < 0 or nr >= N or nc < 0 or nc >= N:
                s_next = s
            else:
                s_next = pos_to_state(nr, nc)

            # Reward is -1 for every move
            reward = -1

            # Bellman expectation equation:
            # V(s) += 0.25 * [reward + gamma * V(s')]
            value += 0.25 * (reward + gamma * V[s_next])

        # Store updated value for state s
        V_new[s] = value

        # Track the maximum change across states
        delta = max(delta, abs(V_new[s] - V[s]))

    # Update value function after full sweep
    V = V_new

    # -------------------------------
    # 9. Convergence Check
    # -------------------------------

    # Stop when maximum change is below threshold
    if delta < theta:
        break

# -------------------------------
# 10. Display Final Value Function
# -------------------------------

# Reshape the 1D value array into a 4x4 grid for readability
V_grid = V.reshape(N, N)
print(V_grid)
