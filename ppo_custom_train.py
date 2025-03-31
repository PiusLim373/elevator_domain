#!/usr/bin/python3

################################################### Team information
'''
    Pius Lim Zhen Ye
    E1264220
    A0285094W

    Cheah Tze Ang
    E1326018 
    A0290565X

    Connection to gdrive....
'''

################################################### Installation and setup
'''
    requirement.txt blah blah blah
    pip install -r requirements.txt
'''
from ppo.ppo_agent import *
import matplotlib.pyplot as plt
import tqdm
from pyRDDLGym.Elevator import Elevator
from IPython.display import Image, display, clear_output

################################################### Environment Initialization
## IMPORTANT: Do not change the instance of the environment.
env = Elevator(instance=5)

print("Discrete environment actions:")
print(env.disc_actions)
print("Continuous environment actions:")
print(env.base_env.action_space)
print(f"Observation space size for the discrete Elevator Environment: {len(env.disc_states)}")

################################################### Hyperparameters

INPUT_DIMS = 13
ACTIONS_DIMS = 6
LEARNING_RATE = 0.0003
DISCOUNT = 0.99
GAE_LAMBDA = 0.95
CRITIC_LOSS_COEFF = 0.5
ENTROPY_COEFF = 0.01
PPO_CLIP = 0.2
BATCH_SIZE = 256
TRAIN_EVERY_N_STEPS = 2048
N_EPOCH = 20
N_EPISODES = 3000
CHECKPOINT_DIR = "saves/"

################################################### Model Definition
# copy paste all ppo.ppo_network.py stuff here

################################################### Feature Extraction
env_features = list(env.observation_space.keys())


def convert_state_to_list(state, env_features):
    out = []
    for i in env_features:
        out.append(state[i])
    return out


################################################### Neural Net Initialization
agent = Agent(
    input_dims=INPUT_DIMS,
    action_dims=ACTIONS_DIMS,
    learning_rate=LEARNING_RATE,
    discount=DISCOUNT,
    gae_lambda=GAE_LAMBDA,
    critic_loss_coeff=CRITIC_LOSS_COEFF,
    entropy_coeff=ENTROPY_COEFF,
    ppo_clip=PPO_CLIP,
    batch_size=BATCH_SIZE,
    n_epoch=N_EPOCH,
    checkpoint_dir=CHECKPOINT_DIR,
)

################################################### Live Plotting Setup
# Create a figure for plotting
plt.style.use("ggplot")
fig, ax = plt.subplots(figsize=(10, 6))
plt.ion()

# Lists to store rewards and episode numbers
rewards_list = []
episodes = []


def exponential_smoothing(data, alpha=0.1):
    """Compute exponential smoothing."""
    smoothed = [data[0]]  # Initialize with the first data point
    for i in range(1, len(data)):
        st = alpha * data[i] + (1 - alpha) * smoothed[-1]
        smoothed.append(st)
    return smoothed


def live_plot(data_dict, figure, ylabel="Total Rewards"):
    """Plot the live graph."""
    clear_output(wait=True)
    ax.clear()
    for label, data in data_dict.items():
        if label == "Total Reward":
            ax.plot(data, label=label, color="yellow", linestyle="--")

            # Compute and plot moving average for total reward
            ma = exponential_smoothing(data)
            ma_idx_start = len(data) - len(ma)
            ax.plot(
                range(ma_idx_start, len(data)), ma, label="Smoothed Value", linestyle="-", color="purple", linewidth=2
            )
        else:
            ax.plot(data, label=label)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left")
    display(figure)

################################################### RL Algorithm
# copy paste all ppo.ppo_agent.py stuff here

################################################### Training loop with live plotting
learning_counter = 0
step_count = 0
episode_loss = 0
best_reward = -int(1e6)

progress_bar = tqdm.tqdm(range(N_EPISODES), postfix={"Total Reward": 0, "Loss": 0})
for episode in progress_bar:
    total_reward = 0
    raw_state = env.reset()
    state_desc = env.disc2state(raw_state)
    state_list = convert_state_to_list(state_desc, env_features)
    observation = np.array(state_list)

    while True:
        action, prob, val = agent.choose_action(observation)
        # reward, done, observation_new = gym_env.step(ACTIONS[action])
        new_raw_state, reward, done, _ = env.step(action)
        new_state_desc = env.disc2state(new_raw_state)
        new_state_list = convert_state_to_list(new_state_desc, env_features)
        new_observation = np.array(new_state_list)
        step_count += 1
        agent.remember(observation, action, prob, val, reward, done)

        # update the networks every TRAIN_EVERY_N_STEPS collected data, aka train
        if step_count % TRAIN_EVERY_N_STEPS == 0:
            critic_loss, actor_loss, total_loss = agent.learn()
            episode_loss = total_loss
            learning_counter += 1

        observation = new_observation
        total_reward += reward
        if done:
            break
    rewards_list.append(total_reward)
    episodes.append(episode)

    live_plot({"Total Reward": rewards_list}, fig)


    # Save the model if new episode_reward_mean is greater than the best_reward
    average_reward = np.mean(rewards_list)
    # print(
    #     f"\nEposide {episode} | Reward this episode: {total_reward:.4f} | Mean reward of all episodes: {average_reward:.4f} | Learning Counter: {learning_counter} | Data collected: {step_count}"
    # )
    if average_reward > best_reward:
        best_reward = average_reward
        agent.save_models(autosave=True)
        print(f"\nNew best average reward: {best_reward}, autosaving model\n")
        
    progress_bar.set_postfix({"Total Reward": total_reward, "Average reward": average_reward, "Loss": episode_loss})

# End of the training, saving the model
agent.save_models()

################################################### Compute the mean rewards
print(f"\nMean Rewards: {np.mean(rewards_list)}")
# close the environment
env.close()
