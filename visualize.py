# ------------------------------------------------
# visualize.py
# ------------------------------------------------
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import numpy as np
import os

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from custom_env import CoopGridWorld

# ------------------------------------------------
# GLOBALS for demonstration only:
# ------------------------------------------------
# Insert the path to your checkpoint here:
CHECKPOINT_PATH = os.path.abspath("./rllib_results/PPO_2025-01-26_21-13-51/PPO_CoopGridWorld_e9d5a_00000_0_2025-01-26_21-13-51/checkpoint_000000")

# RLlib objects:
ALGO = None
ENV = None
OBS = None
DONE = False

def setup_rllib_and_env():
    """Initialize RLlib, load the policy, and create the environment."""
    global ALGO, ENV, OBS, DONE

    ray.init(ignore_reinit_error=True)

    # Rebuild same config from training (adjusted to match train.py)
    temp_env = CoopGridWorld(config={"grid_size": 7, "max_steps": 50, "arrival_window": 3})

    config = (
        PPOConfig()
        .environment(env=CoopGridWorld, env_config={"grid_size": 7, "max_steps": 50, "arrival_window": 3})
        .framework("torch")  # Use torch as in train.py
        .multi_agent(
            policies={
                "shared_policy": (
                    None,
                    temp_env.observation_spaces["agent_1"],
                    temp_env.action_spaces["agent_1"],
                    {}
                )
            },
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy"
        )
        .training(
            model={
                "fcnet_hiddens": [128, 128],
                "fcnet_activation": "relu",
            },
            gamma=0.99,
            lr=1e-3,
            train_batch_size=2048,
            num_epochs=10  # Align with train.py
        )
    )

    # Build PPO from config and restore checkpoint
    algo = config.build()
    algo.restore(CHECKPOINT_PATH)

    # Create environment and reset
    env = CoopGridWorld({"grid_size": 7})
    obs, _ = env.reset()  # Return obs and info for compatibility

    ALGO = algo
    ENV = env
    OBS = obs
    DONE = False

def plot_grid(agent_positions, goal_pos, grid_size=7):
    """
    Create a Plotly figure showing agent positions and the goal on a grid.
    agent_positions: dict like {"agent_1": (r, c), "agent_2": (r, c)}
    goal_pos: (r, c)
    """
    # We'll produce a scatter plot with a background grid.
    # The x-axis is columns (c), y-axis is rows (r), but we flip so that row=0 is at the top.
    # For a 7x7, coordinates go 0..6 in both row & col.

    agent1_pos = agent_positions["agent_1"]
    agent2_pos = agent_positions["agent_2"]

    # For convenience, create a small x,y set for the entire grid to visually display squares:
    x_coords = np.arange(grid_size)
    y_coords = np.arange(grid_size)

    # We'll create a background "mesh" as faint squares via Heatmap or Scatter.
    # Here, we do an invisible Heatmap to show grid lines easily:
    heatmap = go.Heatmap(
        z=[[0]*grid_size]*grid_size, 
        x=x_coords, 
        y=y_coords,
        showscale=False,
        hoverinfo='skip',
        colorscale='Greys',
        opacity=0.2
    )

    # Mark agent 1
    agent1_scatter = go.Scatter(
        x=[agent1_pos[1]],  # col -> x
        y=[agent1_pos[0]],  # row -> y
        mode='markers',
        marker=dict(size=20, color='blue'),
        name='Agent 1'
    )

    # Mark agent 2
    agent2_scatter = go.Scatter(
        x=[agent2_pos[1]],
        y=[agent2_pos[0]],
        mode='markers',
        marker=dict(size=20, color='red'),
        name='Agent 2'
    )

    # Mark goal
    goal_scatter = go.Scatter(
        x=[goal_pos[1]],
        y=[goal_pos[0]],
        mode='markers',
        marker=dict(size=20, color='green', symbol='star'),
        name='Goal'
    )

    fig = go.Figure(data=[heatmap, agent1_scatter, agent2_scatter, goal_scatter])
    fig.update_layout(
        title="CoopGridWorld Live Visualisation",
        xaxis=dict(
            scaleanchor='y',  # Ensure x/y scales match
            showgrid=True,
            tickmode='linear',
            dtick=1,
            range=[-0.5, grid_size-0.5]
        ),
        yaxis=dict(
            autorange='reversed',  # row=0 at top
            showgrid=True,
            tickmode='linear',
            dtick=1,
            range=[-0.5, grid_size-0.5]
        )
    )
    return fig

# ------------------------------------------------
# Dash App
# ------------------------------------------------
app = dash.Dash(__name__)
app.title = "Multi-Agent RL Visualisation"

app.layout = html.Div([
    html.H1("Two-Agent Cooperative GridWorld (PPO)"),
    dcc.Graph(id='live-graph', style={'height': '70vh'}),
    dcc.Interval(
        id='interval-component',
        interval=500,  # Update every 500ms
        n_intervals=0
    ),
    html.Div("The environment will reset after each episode ends.")
])

@app.callback(
    Output('live-graph', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_graph_live(n):
    """
    Called periodically by the Interval component to step the environment 
    and update the figure.
    """
    global ENV, ALGO, OBS, DONE

    if DONE:
        # Reset if episode is done
        OBS, _ = ENV.reset()
        DONE = False

    print(f"OBS before actions: {OBS}")
    # Use RLlib policy to get actions for both agents
    # compute_actions() can handle multi-agent dict input.
    actions_dict = ALGO.compute_actions(OBS)
    print(f"Actions: {actions_dict}")  
    next_obs, rewards, dones, infos = ENV.step(actions_dict)
    print(f"Next OBS: {next_obs}, Rewards: {rewards}, Dones: {dones}")
    OBS = next_obs

    # If all done, set DONE
    if dones.get("__all__", False):
        DONE = True

    # Build figure
    agent_positions = {
        "agent_1": ENV.agent_positions["agent_1"],
        "agent_2": ENV.agent_positions["agent_2"]
    }
    print(f"Agent Positions: {agent_positions}")
    goal_pos = ENV.goal_position
    fig = plot_grid(agent_positions, goal_pos, grid_size=ENV.grid_size)
    return fig

# ------------------------------------------------
# Main entry point
# ------------------------------------------------
if __name__ == "__main__":
    # 1. Setup the environment & RLlib policy
    setup_rllib_and_env()

    # 2. Run Dash app
    #    By default, it runs on http://127.0.0.1:8050
    app.run_server(debug=True)
