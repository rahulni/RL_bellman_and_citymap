# GridWorld Value Iteration (Bellman Expectation Equation)

This project contains a minimal and educational implementation of **Value Iteration** for a classic **4×4 GridWorld** environment using the **Bellman Expectation Equation**.

It is designed for learning, interviews, and reinforcement learning fundamentals, with no external RL libraries used.

---

## Problem Description

- Environment: **4×4 GridWorld** (16 states)
- Terminal state: **Bottom-right corner**
- Actions: Up, Down, Left, Right
- Policy: **Uniform random** (each action with probability 0.25)
- Reward: **-1 for every move**
- Boundary condition: If an action moves the agent outside the grid, it stays in the same state
- Discount factor (γ): **1.0**

---

## Algorithm

The value function is computed using the **Bellman Expectation Equation**:

V(s) = Σₐ π(a|s) [ r + γ V(s′) ]

The algorithm iteratively updates state values until the maximum change across all states falls below a small threshold.

---

## Hyperparameters

| Parameter | Value | Description |
|---------|------|-------------|
| N | 4 | Grid size |
| gamma | 1.0 | Discount factor |
| theta | 1e-4 | Convergence threshold |

---

## Code Overview

- Environment and grid setup
- Helper functions for state ↔ position conversion
- Value Iteration loop using Bellman expectation updates
- Convergence check
- Final value function reshaped into a grid

---

## Final Output

After convergence, the value function is printed as a 4×4 grid:

```
[[-59.42367735 -57.42387125 -54.2813141  -51.71012579]
 [-57.42387125 -54.56699476 -49.71029394 -45.13926711]
 [-54.2813141  -49.71029394 -40.85391609 -29.99766609]
 [-51.71012579 -45.13926711 -29.99766609   0.        ]]
```

---

## Interpretation

- Values become less negative as states get closer to the terminal state
- The terminal state has value 0
- Negative values represent accumulated step penalties
- The value function approximates the expected number of steps to reach the goal

---

## How to Run

### Jupyter Notebook
Open and run:
```
Bellman_RLex.ipynb
```

### Python Script
```
python Bellman_RLex.py
```

---

## Possible Extensions

- Policy extraction
- Policy Iteration
- Discounted rewards (γ < 1)
- Obstacles or multiple terminal states
- Value function visualization

---

## License

This project is intended for educational use. You are free to use, modify, and extend it.
