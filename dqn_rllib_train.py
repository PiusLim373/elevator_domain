import os
from ray import air, tune
from ray.tune.registry import register_env
from ray.rllib.algorithms.dqn import DQN, DQNConfig
from custom_elevator_gym import *

# Register custom environment
env_name = "elevator_domain"
register_env(env_name, lambda config: Elevator(instance=5))

# Path to save models
cwd = os.getcwd()
checkpoint_path = os.path.join(cwd, "models")

# Define DQN config (vanilla)
config = (
    DQNConfig()
    .environment(env=env_name)
    .framework("torch")
    .rollouts(num_rollout_workers=8)  # Adjust based on your CPU
    .training(
        gamma=0.99,
        lr=1e-3,
        train_batch_size=32,
        double_q=False
    )
)

# Run training using Ray Tune
tune.Tuner(
    DQN,
    param_space=config.to_dict(),
    run_config=air.RunConfig(
        stop={"episodes_total": 3001},
        storage_path=checkpoint_path,
        checkpoint_config=air.CheckpointConfig(
            checkpoint_frequency=100,
            checkpoint_at_end=True,
        ),
    ),
).fit()