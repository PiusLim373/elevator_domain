import os
from ray import air, tune
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.ppo import PPO, PPOConfig  # Use PPO class directly
from custom_elevator_gym import *

# Register custom environment
env_name = "elevator_domain"
register_env(env_name, lambda config: Elevator(instance=5))

# Path to save models
cwd = os.getcwd()
checkpoint_path = os.path.join(cwd, "models")

# Define PPO config with action-masked custom model
config = (
    PPOConfig()
    .environment(env=env_name)
    .framework("torch")
    .rollouts(num_rollout_workers=8)  # Adjust based on your CPU
    .training(
        model={
            "fcnet_hiddens": [128, 128, 128, 128],
            "fcnet_activation": "relu",
        },
        lr=5e-4,
        train_batch_size=1600,
        sgd_minibatch_size=200,
        num_sgd_iter=10,
        clip_param=0.2,
        gamma=0.99,
        lambda_=0.95,
        use_gae=True,
        vf_loss_coeff=1.0,
        entropy_coeff=0.01,
    )
)

# Run training using Ray Tune 
tune.Tuner(
    PPO,  
    param_space=config.to_dict(),
    run_config=air.RunConfig(
        stop={"episodes_total": 3001},
        storage_path=checkpoint_path,  # Specify the custom save directory
        checkpoint_config=air.CheckpointConfig(
            checkpoint_frequency=100,
            checkpoint_at_end=True, # Also save at the end of training
        ),
    ),
).fit()
