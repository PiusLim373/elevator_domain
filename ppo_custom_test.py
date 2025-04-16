#!/usr/bin/python3
import os
import numpy as np
from pyRDDLGym.Elevator import Elevator
from ppo.ppo_agent import Agent
from ppo.hyperparams import *
# from ppo_custom_train import INPUT_DIMS, ACTIONS_DIMS, LEARNING_RATE, DISCOUNT, GAE_LAMBDA, CRITIC_LOSS_COEFF, ENTROPY_COEFF, PPO_CLIP, BATCH_SIZE, N_EPOCH
# Configuration
RENDER = True
CHECKPOINT_DIR = "model/"
USE_AUTOSAVE = True  # Set to True if you want to load autosave

ACTOR_MODEL = "sample_actor_torch_ppo.pth"  # Change this if you need to load different model
CRITIC_MODEL = "sample_critic_torch_ppo.pth"  # Change this if you need to load different model
# actor_model_path = input("Input Actor Model Path: \n")
# critic_model_path = input("Input Critic Model Path: \n")
# # Determine which checkpoint files to load
# if actor_model_path and critic_model_path:
#     ACTOR_CHECKPOINT = actor_model_path
#     CRITIC_CHECKPOINT = critic_model_path
# else: 
#     if USE_AUTOSAVE:
#         ACTOR_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "autosave_actor_torch_ppo.pth")
#         CRITIC_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "autosave_critic_torch_ppo.pth")
#     else:
#         ACTOR_CHECKPOINT = os.path.join(CHECKPOINT_DIR, ACTOR_MODEL)
#         CRITIC_CHECKPOINT = os.path.join(CHECKPOINT_DIR, CRITIC_MODEL)

ACTOR_CHECKPOINT = os.path.join(CHECKPOINT_DIR, ACTOR_MODEL)
CRITIC_CHECKPOINT = os.path.join(CHECKPOINT_DIR, CRITIC_MODEL)
# Initialize environment and agent
env = Elevator(is_render=RENDER, instance=5)
env_features = list(env.observation_space.keys())

def convert_state_to_list(state, env_features):
    out = []
    for i in env_features:
        out.append(state[i])
    return out

agent = Agent(
    input_dims=INPUT_DIMS,
    action_dims=ACTIONS_DIMS,
    learning_rate=LEARNING_RATE,
    discount=DISCOUNT,
    gae_lambda=GAE_LAMBDA,
    critic_loss_coeff=CRITIC_LOSS_COEFF, 
    entropy_coeff=0,
    ppo_clip=PPO_CLIP,
    batch_size=BATCH_SIZE,
    n_epoch=N_EPOCH,
    checkpoint_dir=CHECKPOINT_DIR,
)

if os.path.isfile(ACTOR_CHECKPOINT) and os.path.isfile(CRITIC_CHECKPOINT):
    agent.actor.checkpoint_file = ACTOR_CHECKPOINT
    agent.critic.checkpoint_file = CRITIC_CHECKPOINT
    agent.load_models()  # Use the load_models method
    agent.actor.eval()  # Set the actor model to evaluation mode
    agent.critic.eval()  # Set the actor model to evaluation mode
    
else:
    print("Starting fresh, no models loaded.")

# Testing the agent
n_episodes = 10  # Number of episodes to run
scores = []  # Initialize a list to store episode scores

for episode in range(n_episodes):
    raw_state = env.reset()  # Reset the environment for a new episode
    state_desc = env.disc2state(raw_state)
    state_list = convert_state_to_list(state_desc, env_features)
    observation = np.array(state_list)
    done = False
    score = 0

    while not done:
        action, prob, val = agent.choose_action(observation)  # Choose action
        new_raw_state, reward, done, info = env.step(action)
        new_state_desc = env.disc2state(new_raw_state)
        new_state_list = convert_state_to_list(new_state_desc, env_features)
        new_observation = np.array(new_state_list)
        score += reward  # Accumulate score
        observation = new_observation  # Update observation for the next step

    scores.append(score)  # Add episode score to the scores list
    print(f"Episode {episode + 1}: Reward= {score}")

# Calculate and print average reward
average_reward = np.mean(scores)
print(f"Average Reward over {n_episodes} episodes: {average_reward}")

# End of the testing
env.close()  # Close the environment window if applicable
