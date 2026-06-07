import gymnasium as gym
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv, AutoresetMode

import numpy as np
import torch
import time
import copy
from abc import ABC, abstractmethod

from rl.agents import RLAgent

def to_tensor(x):
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    return x

def state_to_tensor(x):
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    return x

def state_to_np(state):
    return np.array(state)

def arr_avg_last_n(arr, n=25, ignore_incompletes=False):
    arr = np.array(arr)
    arr_avgs = np.ndarray((len(arr),))
    for i in range(len(arr)):
        if ignore_incompletes and i<n-1:
            arr_avgs[i] = np.nan
        else:
            arr_avgs[i] = np.mean(arr[np.max([0, i-n+1]):i+1])
    
    return arr_avgs


class SeededResetWrapper(gym.Wrapper):
    def __init__(self, env, seed):
        super().__init__(env)
        self.local_rng = np.random.default_rng(seed)

    def reset(self, **kwargs):
        if kwargs.get("seed") is None:
            kwargs["seed"] = int(self.local_rng.integers(0, 2**32 - 1))
        return self.env.reset(**kwargs)

'''
Problem: gym expects reward to be a scalar in vectorized environments
I create the wrapper which returns a mock 0.0 reward and store the real reward in the info object
'''
class StagHuntEnvsWrapper(gym.Wrapper):
    def __init__(self, env, print_rewards=False, test=False):
        super().__init__(env)

        self.print_rewards = print_rewards
        self.total_reward = np.array([0.0, 0.0])
    
    def step(self, action):
        new_observation, rewards, termination, truncation, info = self.env.step(action)


        info["rewards"] = np.array(rewards)
        if self.print_rewards:
            print(info["rewards"])

        for i in range(2):
            self.total_reward[i] += rewards[i]
        
        if termination or truncation:
            info["total_rewards"] = self.total_reward
            if self.print_rewards:
                print(f"Episode cumulative reward: {self.total_reward}")

            self.total_reward = np.array([0.0, 0.0])

        return new_observation, 0.0, termination, truncation, info


class StagHuntEnvsLooper(ABC):

    def __init__(self, parallelization="sync", print_rewards=False, test=False, evaluate_during_training=True, evaluate_every_n_steps=200, n_evaluations=1, seed=42, reward_normalization_factor=1.0):
        assert parallelization in ["sync", "async"]
        self.print_rewards = print_rewards
        self.test=test
        self.parallelization = parallelization
        self.evaluate_during_training = evaluate_during_training
        self.evaluate_every_n_steps = evaluate_every_n_steps
        self.evaluation_envs = None
        self.n_evaluations = n_evaluations
        self.reward_normalization_factor = reward_normalization_factor
        
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def load_default_env_creation_fns(self, env_id, n_envs, env_params, wrapper):
        self.n_envs = n_envs
        self.envs_creation_fns = [lambda: wrapper(gym.make(env_id, **env_params), self.print_rewards, self.test) for _ in range(n_envs)]
        if self.evaluate_during_training:
            self.evaluation_envs_fns = [lambda: wrapper(gym.make(env_id, **env_params), False, True) for _ in range(self.n_evaluations)]
        self.create_envs()

    def load_custom_env_creation_fns(self, envs_creation_fns, evaluation_envs_fns=None):
        self.envs_creation_fns = envs_creation_fns
        self.n_envs = len(envs_creation_fns)
        if self.evaluate_during_training:
            assert evaluation_envs_fns is not None
            self.evaluation_envs_fns = evaluation_envs_fns
            self.n_evaluations = len(evaluation_envs_fns)
        self.create_envs()


    def create_envs(self):
        seeded_envs_fns = [lambda f=fn, s=int(self.rng.integers(0, 2**32 - 1)): SeededResetWrapper(f(), s) for fn in self.envs_creation_fns]
        
        if self.evaluate_during_training:
            seeded_eval_envs_fns = [lambda f=fn, s=int(self.rng.integers(0, 2**32 - 1)): SeededResetWrapper(f(), s) for fn in self.evaluation_envs_fns]

        if self.parallelization=="sync":
            self.envs = SyncVectorEnv(seeded_envs_fns, autoreset_mode=AutoresetMode.SAME_STEP)
            if self.evaluate_during_training:
                self.evaluation_envs = SyncVectorEnv(seeded_eval_envs_fns, autoreset_mode=AutoresetMode.SAME_STEP)
        else:
            self.envs = AsyncVectorEnv(seeded_envs_fns, autoreset_mode=AutoresetMode.SAME_STEP)
            if self.evaluate_during_training:
                self.evaluation_envs = AsyncVectorEnv(seeded_eval_envs_fns, autoreset_mode=AutoresetMode.SAME_STEP)

    def get_action_space(self):
        return self.envs.action_space

    @abstractmethod
    def on_episode_end(self, info, terminations, truncations):
        return

    @abstractmethod
    def on_evaluation_episode_end(self, info, terminations, truncations, already_ended):
        return
    
    @abstractmethod
    def convert_state(self, observation, n_envs, info):
        raise NotImplemented

    def extract_rewards_from_info(self, info, new_observations, n_envs):
        if "_final_obs" in info:
            
            next_iter_observations = copy.deepcopy(new_observations)
            for i in np.where(info["_final_obs"])[0]:
                new_observations[i] = info["final_obs"][i]
            new_observations = self.convert_state(new_observations, n_envs, info)
            next_iter_observations = self.convert_state(next_iter_observations, n_envs, info)

            assert "_final_info" in info
            rewards = np.zeros((n_envs, 2), dtype=np.float32)
            rewards[info["final_info"]["_rewards"]] = info["final_info"]["rewards"]

            if "_rewards" in info:
                rewards[info["_rewards"]] = info["rewards"]
            
            rewards = rewards.T

        else:
            new_observations = self.convert_state(new_observations, n_envs, info)
            next_iter_observations = new_observations
            rewards = info["rewards"].T
        
        return rewards, new_observations, next_iter_observations

    def training_loop(self, agent1: RLAgent, agent2: RLAgent, max_steps, common_reward=False, enable_gui=False, printing_delay=1, ts_to_print=1000):

        assert self.n_envs==1 or (self.n_envs>1 and not enable_gui)

        start_time = time.perf_counter()
        steps = 0

        observations, info = self.envs.reset()
        observations = self.convert_state(observations, self.n_envs, info)

        while True:

            ### Policy Section
            actions_agent1, actions_agent2 = agent1.policy(observations[0]), agent2.policy(observations[1])

            ### Step Section
            if enable_gui:
                time.sleep(printing_delay)
                self.envs.render()

            new_observations, rewards, terminations, truncations, info = self.envs.step(np.stack([actions_agent1, actions_agent2]).T)

            rewards, new_observations, next_iter_observations = self.extract_rewards_from_info(info, new_observations, self.n_envs)

            
            reward_agent1 = (rewards[0] + rewards[1] if common_reward else rewards[0])*self.reward_normalization_factor
            reward_agent2 = (rewards[0] + rewards[1] if common_reward else rewards[1])*self.reward_normalization_factor

            ### Give the step to the agents
            agent1.on_action_performed(observations[0], actions_agent1, reward_agent1, new_observations[0], terminations, truncations)
            agent2.on_action_performed(observations[1], actions_agent2, reward_agent2, new_observations[1], terminations, truncations)

            ### Step End section
            if steps !=0 and steps%ts_to_print == 0:
                print(f"Step {steps}/{max_steps}")

            ### Episode End section
            if (terminations | truncations).any():
                self.on_episode_end(info, terminations, truncations)

            if not self.test and self.evaluate_during_training and steps%self.evaluate_every_n_steps==0:
                self.evaluate_one_episode(agent1, agent2)

            steps+=1

            if steps>=max_steps:
                break

            observations = next_iter_observations

        end_time = time.perf_counter()
        print(f"Training lasted for {end_time - start_time:.2f}s")
    
    def evaluate_one_episode(self, agent1: RLAgent, agent2: RLAgent):
        assert not agent1.test and not agent2.test

        agent1.test, agent2.test = True, True

        observations, info = self.evaluation_envs.reset()
        observations = self.convert_state(observations, self.n_evaluations, info)
        endeds = np.zeros(observations.shape[1], dtype=np.bool_)
        while not endeds.all():

            ### Policy Section
            actions_agent1, actions_agent2 = agent1.policy(observations[0]), agent2.policy(observations[1])

            new_observations, rewards, terminations, truncations, info = self.evaluation_envs.step(np.stack([actions_agent1, actions_agent2]).T)
            

            rewards, new_observations, next_iter_observations = self.extract_rewards_from_info(info, new_observations, self.n_evaluations)

            observations = next_iter_observations
        
            ### Episode End section
            if (terminations | truncations).any():
                self.on_evaluation_episode_end(info, terminations, truncations, endeds)
                endeds = endeds | terminations | truncations


        agent1.test, agent2.test = False, False
