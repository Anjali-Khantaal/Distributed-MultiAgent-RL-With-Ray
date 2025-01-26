# ------------------------------------------------
# train.py
# ------------------------------------------------
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from custom_env import CoopGridWorld  # Your custom environment
import os

def main():
    # 1. Shut down any previous Ray session and init a fresh one
    ray.shutdown()
    ray.init()

    # Temporary environment to fetch observation and action spaces
    temp_env = CoopGridWorld(config={"grid_size": 7, "max_steps": 50, "arrival_window": 3})

    # 2. Build the RLlib config
    config = (
        PPOConfig()
        # Environment configuration
        .environment(
            env=CoopGridWorld,
            env_config={
                "grid_size": 7,
                "max_steps": 50,
                "arrival_window": 3
            }
        )
        # Multi-agent setup
        .multi_agent(
            policies={
                "shared_policy": (
                    None,  # Default PPO policy
                    temp_env.observation_space,
                    temp_env.action_space,
                    {}
                )
            },
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy"
        )
        # Framework selection
        .framework("torch")
        # Training parameters
        .training(
            model={
                "fcnet_hiddens": [128, 128],
                "fcnet_activation": "relu",
            },
            gamma=0.99,
            lr=1e-3,
            train_batch_size=2048,
            num_epochs=10 # Replaces deprecated num_sgd_iter
        )
        # Resource allocation
        .resources(
            num_gpus=0  # Set to 1 if GPU is available
        )
    )

    # 3. Run the training via Tune
    stop_criteria = {"training_iteration": 30}
    tune.run(
        "PPO",
        config=config.to_dict(),
        stop=stop_criteria,
        storage_path=os.path.abspath("./rllib_results"),
        checkpoint_at_end=True
    )

if __name__ == "__main__":
    main()
