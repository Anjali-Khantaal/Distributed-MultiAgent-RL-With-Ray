import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv

class CoopGridWorld(MultiAgentEnv):
    """Cooperative Multi-Agent GridWorld environment."""

    def __init__(self, config=None):
        super().__init__()
        config = config or {}
        self.grid_size = config.get("grid_size", 7)
        self.max_steps = config.get("max_steps", 50)
        self.arrival_window = config.get("arrival_window", 3)
        self.goal_position = (self.grid_size - 1, self.grid_size - 1)

        # Define action and observation spaces per agent
        self.possible_agents = ["agent_1", "agent_2"]
        self.action_spaces = {agent: Discrete(5) for agent in self.possible_agents}
        self.observation_spaces = {
            agent: Box(
                low=np.array([0, 0, 0, 0, 0], dtype=np.float32),
                high=np.array([self.grid_size - 1, self.grid_size - 1,
                               self.grid_size - 1, self.grid_size - 1,
                               self.max_steps], dtype=np.float32),
            )
            for agent in self.possible_agents
        }

        # Initialize other attributes
        self.agents = self.possible_agents[:]
        self.agent_positions = {}
        self.steps_taken = 0
        self.agent_arrival_steps = {}

    def reset(self, *, seed=None, options=None):
        """Reset the environment."""
        self.steps_taken = 0
        self.agents = self.possible_agents[:]  # Reset active agents
        self.agent_arrival_steps = {agent_id: None for agent_id in self.agents}
        self.agent_positions = {
            agent_id: (np.random.randint(0, self.grid_size),
                       np.random.randint(0, self.grid_size))
            for agent_id in self.agents
        }

        # Return observations and an empty info dict
        observations = {agent_id: self._get_obs(agent_id) for agent_id in self.agents}
        info = {agent_id: {} for agent_id in self.agents}
        return observations, info

    def step(self, action_dict):
        """Take a step in the environment."""
        self.steps_taken += 1
        rewards = {agent_id: 0.0 for agent_id in self.agents}
        terminateds = {agent_id: False for agent_id in self.agents}
        truncateds = {agent_id: False for agent_id in self.agents}
        done = False

        # Update agent positions based on actions
        for agent_id, action in action_dict.items():
            if self.agent_arrival_steps[agent_id] is None:
                self.agent_positions[agent_id] = self._apply_action(
                    self.agent_positions[agent_id], action
                )
                # Check if agent arrived at the goal
                if self.agent_positions[agent_id] == self.goal_position:
                    self.agent_arrival_steps[agent_id] = self.steps_taken

        # Check if both agents have arrived at the goal
        all_arrived = all(
            step is not None for step in self.agent_arrival_steps.values()
        )
        if all_arrived:
            arrival_times = list(self.agent_arrival_steps.values())
            if abs(arrival_times[0] - arrival_times[1]) <= self.arrival_window:
                for agent_id in self.agents:
                    rewards[agent_id] = 10.0
            done = True

        # Check if max steps are reached
        if self.steps_taken >= self.max_steps:
            done = True

        # Update termination and truncation flags
        if done:
            for agent_id in self.agents:
                terminateds[agent_id] = True
                truncateds[agent_id] = self.steps_taken >= self.max_steps

        # Provide last observations even on truncation
        observations = {agent_id: self._get_obs(agent_id) for agent_id in self.agents}

        # Clear agents if episode is done
        if done:
            self.agents = []

        # Prepare termination and truncation dicts
        terminateds["__all__"] = done
        truncateds["__all__"] = self.steps_taken >= self.max_steps

        info = {agent_id: {} for agent_id in self.possible_agents}
        return observations, rewards, terminateds, truncateds, info

    def _apply_action(self, position, action):
        """Apply an action to an agent's position."""
        row, col = position
        if action == 0:  # Up
            row = max(0, row - 1)
        elif action == 1:  # Down
            row = min(self.grid_size - 1, row + 1)
        elif action == 2:  # Left
            col = max(0, col - 1)
        elif action == 3:  # Right
            col = min(self.grid_size - 1, col + 1)
        # Stay (4) does nothing
        return row, col

    def _get_obs(self, agent_id):
        """Get observation for an agent."""
        other_agent = "agent_2" if agent_id == "agent_1" else "agent_1"
        row, col = self.agent_positions.get(agent_id, (0, 0))
        other_row, other_col = self.agent_positions.get(other_agent, (0, 0))
        steps_left = self.max_steps - self.steps_taken
        return np.array([row, col, other_row, other_col, steps_left], dtype=np.float32)
