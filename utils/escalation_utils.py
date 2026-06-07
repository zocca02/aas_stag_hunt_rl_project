import numpy as np
import matplotlib.pyplot as plt
import time
import copy
import torch
import pickle

from utils.utils import StagHuntEnvsWrapper, StagHuntEnvsLooper, arr_avg_last_n
from rl.agents import RLAgent



class EscalationLog:
    def __init__(self, n_envs):
        self.episode_rewards = [[] for _ in range(n_envs)]
        self.episode_collaborations = [[] for _ in range(n_envs)]
        self.episode_defections = [[] for _ in range(n_envs)]
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
        collaborations = np.array(self.episode_collaborations).mean(axis=0)
        defections = np.array(self.episode_defections).mean(axis=0)

        # In the form of (title, data, trend, x_label, y_label)
        data = [
            ("Rewards", tot_rewards, arr_avg_last_n(tot_rewards, episodes_to_mean), "Episode", "Tot. Reward"),
            ("Collaborations", collaborations, arr_avg_last_n(collaborations, episodes_to_mean), "Episode", "N. Collaborations"),
            ("Defections", defections, arr_avg_last_n(defections, episodes_to_mean), "Episode", "N. Defections")
        ]

        return data

    def plot_stats(self, file_name, fig_n=1, plot_layout=(2, 2), episodes_to_mean=100):

        plt.figure(fig_n, figsize=(20, 10))

        plt.subplot(*plot_layout, 1)
        plt.title("Agent Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Tot. Reward")
        plt.plot(self.episode_rewards[0])
        plt.plot(arr_avg_last_n(self.episode_rewards[0], episodes_to_mean), color="red")

        plt.subplot(*plot_layout, 2)
        plt.title("Agent Collaborations")
        plt.xlabel("Episode")
        plt.ylabel("N. Collaborations")
        plt.plot(self.episode_collaborations[0])
        plt.plot(arr_avg_last_n(self.episode_collaborations[0], episodes_to_mean), color="red")

        plt.subplot(*plot_layout, 3)
        plt.title("Agent Defections")
        plt.xlabel("Episode")
        plt.ylabel("N. Defections")
        plt.plot(self.episode_defections[0])
        plt.plot(arr_avg_last_n(self.episode_defections[0], episodes_to_mean), color="red")

        plt.savefig(file_name)


class EscalationWrapper(StagHuntEnvsWrapper):
    def __init__(self, env, print_rewards=False, test=False):
        super().__init__(env, print_rewards, test)

        self.total_collaborations = np.array([0, 0])
        self.total_defections = np.array([0, 0])

        self.current_streak = 0
        self.current_step = 0

        self.total_rewards_correction = np.array([0, 0])

    def reset(self, **kwargs):
        observation, info = super().reset(**kwargs)

        self.total_collaborations = np.array([0, 0])
        self.total_defections = np.array([0, 0])
        self.current_step = 0
        self.current_streak = 0
        self.total_rewards_correction = np.array([0, 0])
        
        if info is None:
            info = {}
        info["streak_len"] = 0
        info["current_step"] = 0
        
        return observation, info

    def step(self, action):
        new_observation, rewards, termination, truncation, info = super().step(action)

        # Patch for a possible bug of Escalation rewards when switching episode (probably when AutoresetMode.SAME_STEP is used in vectorized environments)
        # It happens if the last action of the previous episode was collaboration and in the new episode the first action of the agent is going into the stag
        # the agent which goes over the stag in the new episode, that theoretically should get a 0.0 reward, receive a punishment of -multiplier*length of the last streak of the previous episode instead
        if self.current_step==0 and (info["rewards"][0]<0 or info["rewards"][1]<0):
            self.total_rewards_correction = -info["rewards"]
            info["rewards"][:] = 0
  
        self.current_step+=1

        collaboration=True
        for i in range(2):
            if info["rewards"][i]>0:
                self.total_collaborations[i]+=1
            else:
                collaboration = False
            if info["rewards"][i]<0:
                self.total_defections[i]+=1
        
        if collaboration:
            self.current_streak+=1
        else:
            self.current_streak=0

        info["streak_len"] = self.current_streak
        info["current_step"] = self.current_step

        if termination or truncation:
            info["total_collaborations"] = self.total_collaborations
            info["total_defections"] = self.total_defections
            # Correction of the total reward which is computed by the superclass, unaware of the problem
            info["total_rewards"] += self.total_rewards_correction

            self.total_collaborations = np.array([0, 0])
            self.total_defections = np.array([0, 0])
            self.current_step = 0
            self.current_streak = 0
            self.total_rewards_correction = np.array([0, 0])

        return new_observation, rewards, termination, truncation, info

class EscalationLooper(StagHuntEnvsLooper):
    def __init__(self, n_envs, env_params, parallelization="sync", print_rewards=False, test=False, evaluate_during_training=True, evaluate_every_n_steps=200, n_evaluations=1,
                 add_streak_len=False, max_streak_len=200, seed=42):
        super().__init__(parallelization, print_rewards, test, evaluate_during_training, evaluate_every_n_steps, n_evaluations, seed)

        self.load_default_env_creation_fns("StagHunt-Escalation-v0", n_envs, env_params, EscalationWrapper)#lambda env, print_rewards=False: EscalationWrapper(env, print_rewards, scale_rewards, episode_time_steps))

        self.log_agents = [EscalationLog(n_envs), EscalationLog(n_envs)]
        self.evaluation_log_agents = [EscalationLog(n_evaluations), EscalationLog(n_evaluations)]

        self.width, self.height = env_params["grid_size"][0], env_params["grid_size"][1]
        self.add_streak_len = add_streak_len
        self.max_streak_len = max_streak_len

    def update_logs(self, logs_arr, mask, info):
        for i in np.where(mask)[0]:
            for j in range(2):
                logs_arr[j].episode_rewards[i].append(info["final_info"]["total_rewards"][i, j])
                logs_arr[j].episode_collaborations[i].append(info["final_info"]["total_collaborations"][i, j])
                logs_arr[j].episode_defections[i].append(info["final_info"]["total_defections"][i, j])

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
        new_states = torch.zeros((2, n_envs, 4+int(self.add_streak_len)), dtype=torch.float32)
        for env_idx in range(n_envs):
            for actor_idx in range(2):
                new_states[actor_idx, env_idx, 0] = (int(states[env_idx, actor_idx, 2])-int(states[env_idx, actor_idx, 0]))/self.width
                new_states[actor_idx, env_idx, 1] = (int(states[env_idx, actor_idx, 3])-int(states[env_idx, actor_idx, 1]))/self.height
                new_states[actor_idx, env_idx, 2] = (int(states[env_idx, actor_idx, 4])-int(states[env_idx, actor_idx, 0]))/self.width
                new_states[actor_idx, env_idx, 3] = (int(states[env_idx, actor_idx, 5])-int(states[env_idx, actor_idx, 1]))/self.height
                if self.add_streak_len and "streak_len" in info:
                    new_states[actor_idx, env_idx, 4] = info["streak_len"][env_idx]/self.max_streak_len
        return new_states
