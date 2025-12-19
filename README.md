# Autonomous City Navigation with Deep Q-Network (Multi‑Target RL)

This project extends the earlier **GridWorld / Bellman-based RL work** into a **realistic city-scale navigation problem** using **Deep Q-Learning (DQN)**, physics-based motion, and a visual simulation.

An autonomous car (agent) learns to navigate a **custom city map**, sequentially visiting **three targets (A1 → A2 → A3)** while avoiding obstacles and learning optimal control policies.

---

## Key Objectives Addressed

This implementation explicitly answers the following assignment requirements:

### 1. New City Map
- A **custom city map** (different from GridWorld) is used.
- Roads and obstacles are represented via image brightness.
- Any city layout (e.g., circular roads, intersections, blocks) can be loaded via PNG/JPG.

### 2. Deep Neural Network with Additional FC Layer
- The DQN uses a **deep fully-connected architecture** with multiple hidden layers.
- Compared to a minimal DQN, **an extra FC layer is added** to improve representational capacity.
- This allows learning from:
  - 7 distance sensors
  - Angle to target
  - Distance to target

### 3. Sequential Multi‑Target Navigation (A1 → A2 → A3)
- The agent must **reach targets in strict order**:
  - Target 1 (A1)
  - Target 2 (A2)
  - Target 3 (A3)
- Only the **current active target** provides reward.
- Upon reaching a target, the agent automatically switches to the next one.
- Visual indicators clearly show:
  - Active target
  - Completed targets
  - Target index

---

## Environment Overview

- **Agent**: Autonomous car with continuous position and orientation
- **Sensors**: 7 forward-facing ray sensors (collision and road detection)
- **Actions**:
  - Left
  - Straight
  - Right
  - Sharp Left
  - Sharp Right
- **Rewards**:
  - +100 for reaching a target
  - −100 for crashing
  - Dense shaping reward for moving toward target
  - Penalty for moving away from target

---

## Neural Network Architecture (DQN)

```
Input (9 features)
 → FC(128) → ReLU
 → FC(256) → ReLU
 → FC(512) → ReLU
 → FC(512) → ReLU   ← extra FC layer (added)
 → FC(256) → ReLU
 → FC(128) → ReLU
 → Output (5 actions)
```

This deeper network improves:
- Stability
- Long-horizon planning
- Multi-target generalization

---

## Reinforcement Learning Details

- Algorithm: **Deep Q-Network (DQN)**
- Experience Replay:
  - Regular memory
  - Priority memory for successful episodes
- Target Network:
  - Soft updates using Polyak averaging (τ)
- Exploration:
  - Epsilon-greedy with slow decay
- Crash handling:
  - Automatic reset after consecutive failures

---

## User Interaction Flow

1. Load or auto-generate a city map
2. Click once to place the **car**
3. Click three times to place **targets A1, A2, A3**
4. Press **START**
5. Observe learning, rewards, and target switching in real time

---

## Visual Features

- Live reward chart (raw + moving average)
- Animated sensors
- Pulsing targets with numbering
- Real-time logs (target reached, crash, learning stats)

---

## Sample Demonstration Video

You can embed or reference the following sample video to demonstrate similar multi-target RL navigation:

🔗 **YouTube (Sample Demo)**  
[https://www.youtube.com/watch?v=2pWv7GOvuf0](https://youtu.be/uH80_QW2yHs)

*(This is a representative RL navigation demo; replace with your own run if desired.)*

### Embedded (for GitHub / Markdown viewers)

<iframe width="560" height="315"
src="https://youtu.be/uH80_QW2yHs"
frameborder="0" allowfullscreen></iframe>


## How to Run

```bash
python citymap_assignment_RL.py
```

Requirements:
- Python 3.9+
- PyTorch
- PyQt6
- NumPy

---

## Relation to Earlier Work (Reference)

This project builds directly on the concepts introduced in the earlier repository section:

**GridWorld Value Iteration (Bellman Expectation Equation)**

Shared concepts:
- State-value estimation
- Discounting (γ)
- Convergence vs stability
- Reward shaping intuition

Key differences:
| GridWorld | City Navigation |
|---------|----------------|
| Tabular V(s) | Deep Neural Network |
| Discrete grid | Continuous space |
| Single terminal | Sequential targets |
| Deterministic | Physics + noise |

Refer to the earlier README in this repository for foundational theory and equations.

---

## Educational Value

This project demonstrates:
- Transition from classical RL → Deep RL
- Multi-goal task decomposition
- Reward shaping in continuous environments
- Practical DQN engineering (replay, target nets, epsilon decay)

---

## License

This project is intended for **educational and instructional use**.
You are free to modify, extend, and experiment.

---
