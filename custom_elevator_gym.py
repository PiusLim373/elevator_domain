import numpy as np
from pyRDDLGym import RDDLEnv
from pyRDDLGym import ExampleManager
import itertools
import pickle
from gymnasium import spaces
import gymnasium as gym
from pyRDDLGym.Visualizer.MovieGenerator import MovieGenerator
from pathlib import Path
import random
import time


class Elevator(gym.Env):
    def __init__(self, config=None, is_render=False, render_path="temp_vis", instance=4):
        """
        Discrete version of the Elevator example. Please do not modify this
        """

        # build RDDL example including elevator environment
        ExampleManager.RebuildExamples()

        # Select Elevator domain
        ENV = "Elevators"
        instance = instance  #
        EnvInfo = ExampleManager.GetEnvInfo(ENV)

        self.base_env = RDDLEnv.RDDLEnv(domain=EnvInfo.get_domain(), instance=EnvInfo.get_instance(instance))

        # Extracting global variables
        self.num_waiting_threshold = int(self.base_env.sampler.subs["MAX-PER-FLOOR"])
        self.num_elevators = int(len(self.base_env.sampler.subs["num-person-in-elevator"]))
        self.num_floors = int(len(self.base_env.sampler.subs["ARRIVE-PARAM"]))
        self.max_in_elevator = int(self.base_env.sampler.subs["MAX-PER-ELEVATOR"])
        print(
            f"The building has {self.num_floors} floors and {self.num_elevators} elevators. Each floor has maximum {self.num_waiting_threshold} people waiting. Each elevator can carry maximum of {self.max_in_elevator} people."
        )

        # Load Transition information
        if instance == 5:
            P1 = np.load(f"{EnvInfo.path_to_env}/instance_{instance}_next_states.npy")
            P2 = np.load(f"{EnvInfo.path_to_env}/instance_{instance}_rewards.npy")

            self.Prob = self.convert_to_Prob_matrix(P1, P2)
        else:
            self.Prob = pickle.load(open(f"{EnvInfo.path_to_env}/instance_{instance}.pkl", "rb"))

        self.numConcurrentActions = self.base_env.numConcurrentActions
        self.horizon = self.base_env.horizon
        self.disc_states = self.init_states()
        self.disc_actions = self.init_actions()

        self.action_space = spaces.Discrete(6)
        self.default_observation_space = self.base_env.observation_space
        print(self.default_observation_space)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64),
            high=np.array(
                [10, 10, 10, 10, 10, 10, 1, 1, 1, 1, 1, 1, 1],
                dtype=np.int64,
            ),
            shape=(13,),
            dtype=np.int64,
        )
        self.env_features = list(self.default_observation_space.keys())
        self.is_render = is_render
        self.render_path = render_path
        if is_render:
            self.render_path = render_path
            self.MovieGen = MovieGenerator(render_path, ENV, 1000, save_as_mp4=True)
            self.base_env.set_visualizer(EnvInfo.get_visualizer(), movie_gen=self.MovieGen, movie_per_episode=True)
            Path(self.render_path).mkdir(parents=True, exist_ok=True)

    def convert_to_Prob_matrix(self, P1, P2):
        Prob = {}
        for s in range(len(P1)):
            Prob[s] = {}
            for a in range(6):
                Prob[s][a] = [[1.0, P1[s, a], P2[s, a], 0]]

        return Prob

    def step(self, action):
        cont_action = self.disc2action(action)
        next_state, reward, done, info = self.base_env.step(cont_action)
        truncated = False
        if self.is_render:
            self.render()
        return np.array(self.convert_state_to_list(next_state, self.env_features)), reward, done, truncated, info

    def convert_state_to_list(self, state, env_features):
        out = []
        for i in env_features:
            if type(state[i]) == np.bool_:
                out.append(int(state[i]))
            else:
                out.append(state[i])
        return out

    def reset(self, seed=None, options=None):
        state = self.base_env.reset(seed=seed)
        return np.array(self.convert_state_to_list(state, self.env_features)), {}

    def render(self):
        self.base_env.render(to_display=True)

    def save_render(self):
        if self.is_render:
            self.MovieGen.save_animation("elevator")
            return open(f"{self.render_path}/elevator.gif", "rb").read()

    def init_states(self):
        """
        Initializes discrete states
        """

        # each combination of num_person_waiting is a state
        num_people_combinations = list(
            itertools.product(np.arange(self.num_waiting_threshold + 1), repeat=self.num_floors)
        )
        temp_list = []
        for num_people in num_people_combinations:
            txt = f""
            for i in range(self.num_floors):
                txt += f"f{i}_{num_people[i]}|"
            temp_list.append(txt)

        # elevator door close/open
        num_combinations = list(itertools.product(np.arange(2), repeat=self.num_elevators))
        temp_list_new = []
        for num in num_combinations:
            txt = ""
            for i in range(self.num_elevators):
                txt += f"elevdoor{i}_{num[i]}|"
            for _v in temp_list:
                temp_text = txt + _v
                temp_list_new.append(temp_text)
        temp_list = temp_list_new

        # elevator direction down/up
        num_combinations = list(itertools.product(np.arange(2), repeat=self.num_elevators))
        temp_list_new = []
        for num in num_combinations:
            txt = ""
            for i in range(self.num_elevators):
                txt += f"elevdir{i}_{num[i]}|"
            for _v in temp_list:
                temp_text = txt + _v
                temp_list_new.append(temp_text)
        temp_list = temp_list_new

        # elevator location
        num_combinations = list(itertools.product(np.arange(self.num_floors), repeat=self.num_elevators))
        temp_list_new = []
        for num in num_combinations:
            txt = ""
            for i in range(self.num_elevators):
                txt += f"elevfloor{i}_{num[i]}|"
            for _v in temp_list:
                temp_text = txt + _v
                temp_list_new.append(temp_text)
        temp_list = temp_list_new

        # people inside elevator
        num_combinations = list(itertools.product(np.arange(self.max_in_elevator + 1), repeat=self.num_elevators))
        temp_list_new = []
        for num in num_combinations:
            txt = ""
            for i in range(self.num_elevators):
                txt += f"elevpeople{i}_{num[i]}|"
            for _v in temp_list:
                temp_text = txt + _v
                temp_list_new.append(temp_text)
        temp_list = temp_list_new

        disc_states = {}
        for i, _v in enumerate(temp_list):
            disc_states[i] = _v

        return disc_states

    def init_actions(self):
        """
        Initializes discrete actions
        """

        actions = ["movcurdir_0", "movcurdir_1", "close_0", "close_1", "open_0", "open_1"]

        # uncomment this to have one action per elevator
        temp_actions_dict = {}
        for k in range(self.num_elevators):
            temp_actions = []
            for a in actions:
                temp_actions.append(f"e{k}_{a}")
            temp_actions_dict[k] = temp_actions

        disc_actions_list = list(itertools.product(*temp_actions_dict.values()))

        disc_actions = {}
        for i, a_def in enumerate(disc_actions_list):
            disc_actions[i] = a_def

        return disc_actions

    def find_state(self, s, context):
        a = s.split("|")
        for _s in a:
            if f"{context}" in _s:
                try:
                    res = int(_s[-1])
                except:
                    res = 1
                return res
        return None

    def action2disc(self, original_action):

        action_list = []
        for val in original_action.keys():
            if "move-current-dir" in val:
                action_list.append(f"e{int(val[-1])}_movcurdir_{original_action[val]}")

            if "close-door" in val:
                action_list.append(f"e{int(val[-1])}_close_{original_action[val]}")

            if "open-door" in val:
                action_list.append(f"e{int(val[-1])}_open_{original_action[val]}")

        # find this action
        for val in self.disc_actions:
            count = 0
            for i in range(len(action_list)):
                if self.disc_actions[val][i] == action_list[i]:
                    count += 1
            if count == len(action_list):
                return int(val)

        return None

    def disc2action(self, a):
        """
        Converts discrete action into Elevator environment
        Input:
            - a (int): action
        Return:
            - a (definition): action that is compatible with Elevator environment
        """
        a_def = self.disc_actions[a]
        action = {}
        for i in range(self.num_elevators):
            if self.find_state(a_def[i], f"e{i}_movcurdir") is not None:
                action[f"move-current-dir___e{i}"] = self.find_state(a_def[i], f"e{i}_movcurdir")

            if self.find_state(a_def[i], f"e{i}_open") is not None:
                action[f"open-door___e{i}"] = self.find_state(a_def[i], f"e{i}_open")

            if self.find_state(a_def[i], f"e{i}_close") is not None:
                action[f"close-door___e{i}"] = self.find_state(a_def[i], f"e{i}_close")

        return action

    def disc2state(self, s):
        """
        Converts discrete state into Elevator environment state
        Input:
            - s (int): action
        Return:
            - s (definition): state that is compatible with Elevator environment
        """
        s_def = self.disc_states[s]
        # number people in elevator
        state = {}
        for i in range(self.num_elevators):
            state[f"num-person-in-elevator___e{i}"] = self.find_state(s_def, f"elevpeople{i}")

            if self.find_state(s_def, f"elevdir{i}") == 1:
                state[f"elevator-dir-up___e{i}"] = True
            if self.find_state(s_def, f"elevdir{i}") == 0:
                state[f"elevator-dir-up___e{i}"] = False

            if self.find_state(s_def, f"elevdoor{i}") == 0:
                state[f"elevator-closed___e{i}"] = True
            if self.find_state(s_def, f"elevdoor{i}") == 1:
                state[f"elevator-closed___e{i}"] = False

            for j in range(self.num_floors):
                state[f"elevator-at-floor___e{i}__f{j}"] = False

            state[f"elevator-at-floor___e{i}__f{self.find_state(s_def, f'elevfloor{i}')}"] = True

        for j in range(self.num_floors):
            state[f"num-person-waiting___f{j}"] = self.find_state(s_def, f"f{j}")

        return state

    def state2disc(self, original_state):
        txt = ""
        for i in range(self.num_elevators):
            txt += f"elevpeople{i}_{original_state[f'num-person-in-elevator___e{i}']}|"

            for j in range(self.num_floors):
                if original_state[f"elevator-at-floor___e{i}__f{j}"]:
                    txt += f"elevfloor{i}_{j}|"

            if original_state[f"elevator-dir-up___e{i}"]:
                txt += f"elevdir{i}_1|"
            else:
                txt += f"elevdir{i}_0|"

            if original_state[f"elevator-closed___e{i}"]:
                txt += f"elevdoor{i}_0|"
            else:
                txt += f"elevdoor{i}_1|"

        for j in range(self.num_floors):
            txt += f"f{j}_{original_state[f'num-person-waiting___f{j}']}|"

        # find this state
        for val in self.disc_states:
            if self.disc_states[val] == txt:
                return val

        return None


def train():
    env = Elevator(instance=5)
    print(env.observation_space.sample())
    obs, _ = env.reset()
    print(obs)

    for i in range(255):
        obs, reward, done, truncated, info = env.step(random.randint(0, 5))
        print(obs)
        print(reward)


def test():
    env = Elevator(is_render=False, instance=5)
    print(env.observation_space.sample())
    obs, _ = env.reset()
    done = False
    while not done:
        obs, reward, done, truncated, info = env.step(random.randint(0, 5))
        print(reward, done)
        done = done or truncated


if __name__ == "__main__":
    train()
