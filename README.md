# Distributed-MultiAgent-RL-With-Ray

This project demonstrates a multi-agent reinforcement learning environment in which two agents collaborate to achieve a shared goal. It uses Ray's RLlib, to show advanced distributed training for multi-agent systems in a 7×7 grid-based environment.

The *Key Features* include a custom multi-agent environment, a shared reward system, distributed RL training, a modular codebase, and live visualization.

## Project Structure

```plaintext
MultiAgent_RLlib_Project/
├── custom_env.py        # Multi-agent environment implementation
├── train.py             # Training script using RLlib
├── visualize.py         # Dash-based visualization of agent behavior
├── requirements.txt     # Dependencies for the project
└── README.md            # Project documentation
```

## Getting Started
1. Setup Environment (assuming the virtual environment is set up and Python 3.8+ is installed on the system):
   - ```bash
     pip install -r requirements.txt
2. Train the Agents:
   - ```bash
     python train.py
3. Visualize Agent Behavior:
   - ```bash
     python visualize.py
4. After the above steps are completed, agents' coordinates can be monitored in the browser at: `http://127.0.0.1:8050` or `localhost:8050`

## Customization
- Grid Size: Modify the `grid_size` parameter in custom_env.py.
- Reward Function: Adjust the cooperative reward logic in `custom_env.py` to experiment with agent strategies.
- Multi-Agent Policies: Define separate policies for each agent or share a single policy.

## Further Enhancements
- Implement curriculum learning for dynamic grid sizes.
- Experiment with other RL algorithms like APEX or IMPALA.
- Add trajectory replay or advanced visualization features.
