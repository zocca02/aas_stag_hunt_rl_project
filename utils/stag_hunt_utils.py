import numpy as np
import matplotlib.pyplot as plt
import time
import copy
import torch
import pickle

from utils.utils import StagHuntEnvsWrapper, StagHuntEnvsLooper, arr_avg_last_n



class StagHuntLog:
    def __init__(self, n_envs):
        self.episode_rewards = [[] for _ in range(n_envs)]
        self.episode_foragings = [[] for _ in range(n_envs)]
        self.episode_stags = [[] for _ in range(n_envs)]
        self.episode_maulings = [[] for _ in range(n_envs)]
        self.hyperparameters = {}

    def save(self, file_name):
        with open(file_name, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file_name):
        with open(file_name, "rb") as f:
            log = pickle.load(f)
        return log

    def get_plot_data(self, episodes_to_mean=25):

        tot_rewards = np.array(self.episode_rewards).mean(axis=0)
        foragings = np.array(self.episode_foragings).mean(axis=0)
        stags = np.array(self.episode_stags).mean(axis=0)
        maulings = np.array(self.episode_maulings).mean(axis=0)

        # In the form of (title, data, trend, x_label, y_label)
        data = [
            ("Rewards", tot_rewards, arr_avg_last_n(tot_rewards, episodes_to_mean), "Episode", "Tot. Reward"),
            ("Foragings", foragings, arr_avg_last_n(foragings, episodes_to_mean), "Episode", "N. Foragins"),
            ("Stags", stags, arr_avg_last_n(stags, episodes_to_mean), "Episode", "N. Stags"),
            ("Maulings", maulings, arr_avg_last_n(maulings, episodes_to_mean), "Episode", "N. maulings")
        ]

        return data

    def plot_stats(self, file_name, fig_n=1, plot_layout=(2, 2), episodes_to_mean=100):

        plt.figure(fig_n, figsize=(20, 10))

        plt.subplot(*plot_layout, 1)
        plt.title("Agent 1 (blue) Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Tot. Reward")
        plt.plot(self.episode_rewards[0])
        plt.plot(arr_avg_last_n(self.episode_rewards[0], episodes_to_mean), color="red")

        plt.subplot(*plot_layout, 2)
        plt.title("Agent 1 (blue) Foragings")
        plt.xlabel("Episode")
        plt.ylabel("N. Foragings")
        plt.plot(self.episode_foragings[0])
        plt.plot(arr_avg_last_n(self.episode_foragings[0], episodes_to_mean), color="red")

        plt.subplot(*plot_layout, 3)
        plt.title("Agent 1 (blue) Stags")
        plt.xlabel("Episode")
        plt.ylabel("N. Stags")
        plt.plot(self.episode_stags[0])
        plt.plot(arr_avg_last_n(self.episode_stags[0], episodes_to_mean), color="red")

        plt.subplot(*plot_layout, 4)
        plt.title("Agent 1 (blue) Maulings")
        plt.xlabel("Episode")
        plt.ylabel("N. Maulings")
        plt.plot(self.episode_maulings[0])
        plt.plot(arr_avg_last_n(self.episode_maulings[0], episodes_to_mean), color="red")

        plt.savefig(file_name)


class StagHuntWrapper(StagHuntEnvsWrapper):
    def __init__(self, env, print_rewards=False, test=False):
        super().__init__(env, print_rewards=print_rewards)
        
        self.total_foragings = np.array([0, 0])
        self.total_stags = np.array([0, 0])
        self.total_maulings = np.array([0, 0])

    def step(self, action):
        new_observation, rewards, termination, truncation, info = super().step(action)
        for i in range(2):
            if self.unwrapped.forage_reward == info["rewards"][i]:
                self.total_foragings[i]+=1
            if self.unwrapped.stag_reward == info["rewards"][i]:
                self.total_stags[i]+=1
            if self.unwrapped.mauling_punishment == info["rewards"][i]:
                self.total_maulings[i]+=1

        if termination or truncation:
            info["total_foragings"] = self.total_foragings
            info["total_stags"] = self.total_stags
            info["total_maulings"] = self.total_maulings

            self.total_foragings = np.array([0, 0])
            self.total_stags = np.array([0, 0])
            self.total_maulings = np.array([0, 0])

        return new_observation, rewards, termination, truncation, info

class StagHuntLooper(StagHuntEnvsLooper):
    def __init__(self, n_envs, env_params, parallelization="sync", print_rewards=False, test=False, evaluate_during_training=True, evaluate_every_n_steps=200, n_evaluations=1, seed=42):
        super().__init__(parallelization, print_rewards, test, evaluate_during_training, evaluate_every_n_steps, n_evaluations, seed)

        self.load_default_env_creation_fns("StagHunt-Hunt-v0", n_envs, env_params, StagHuntWrapper)

        self.log_agents = [StagHuntLog(n_envs), StagHuntLog(n_envs)]
        self.evaluation_log_agents = [StagHuntLog(n_evaluations), StagHuntLog(n_evaluations)]

        self.n_plants = env_params["forage_quantity"]
        self.width, self.height = env_params["grid_size"][0], env_params["grid_size"][1]

    def update_logs(self, logs_arr, mask, info):
        for i in np.where(mask)[0]:
            for j in range(2):
                logs_arr[j].episode_rewards[i].append(info["final_info"]["total_rewards"][i, j])
                logs_arr[j].episode_foragings[i].append(info["final_info"]["total_foragings"][i, j])
                logs_arr[j].episode_maulings[i].append(info["final_info"]["total_maulings"][i, j])
                logs_arr[j].episode_stags[i].append(info["final_info"]["total_stags"][i, j])

    def on_episode_end(self, info, terminations, truncations):
        endeds = truncations | terminations
        if endeds.any():
            self.update_logs(self.log_agents, endeds, info)
    
    def on_evaluation_episode_end(self, info, terminations, truncations, already_ended):
        endeds = truncations | terminations
        endeds[already_ended] = False
        if endeds.any():
            self.update_logs(self.evaluation_log_agents, endeds, info)

    def convert_state(self, states, n_envs, info):
        new_states = torch.zeros((2, n_envs, 4+2*int(self.n_plants)), dtype=torch.float32)
        for env_idx in range(n_envs):
            for actor_idx in range(2):
                new_states[actor_idx, env_idx, 0] = (int(states[env_idx, actor_idx, 2])-int(states[env_idx, actor_idx, 0]))/self.width
                new_states[actor_idx, env_idx, 1] = (int(states[env_idx, actor_idx, 3])-int(states[env_idx, actor_idx, 1]))/self.height
                new_states[actor_idx, env_idx, 2] = (int(states[env_idx, actor_idx, 4])-int(states[env_idx, actor_idx, 0]))/self.width
                new_states[actor_idx, env_idx, 3] = (int(states[env_idx, actor_idx, 5])-int(states[env_idx, actor_idx, 1]))/self.height
                for i in range(self.n_plants):
                    new_states[actor_idx, env_idx, 4+2*i] = (int(states[env_idx, actor_idx, 6+2*i]) - int(states[env_idx, actor_idx, 0]))/self.width
                    new_states[actor_idx, env_idx, 4+2*i+1] = (int(states[env_idx, actor_idx, 6+2*i+1]) - int(states[env_idx, actor_idx, 1]))/self.height
        return new_states
