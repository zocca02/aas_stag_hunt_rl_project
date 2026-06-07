import torch
import numpy as np
import torch.nn.functional as F
import copy
import torch.nn as nn

from rl.agents import RLAgent
from utils.experience_replay import ExperienceReplayBuffer, PrioritizedExperienceReplayBuffer
from utils.utils import to_tensor


class QLearningAgent(RLAgent):
    def __init__(self, q_estimations, action_space, alpha=1e-4, start_epsilon=1.0, 
                 epsilon_decay=0.99, min_epsilon=0.05, gamma=0.99, test=False, linear_decay=False):
        self.q_estimations = q_estimations
        self.alpha = alpha

        self.gamma = gamma
        self.action_space = action_space

        self.start_epsilon = start_epsilon
        self.epsilon = start_epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.linear_decay = linear_decay
        self.test = test
    
    def select_best_action(self, state):
        return np.argmax(self.q_estimations[*state, :])
    
    def policy(self, state) -> int:

        if np.random.random() < self.epsilon and not self.test:
            action = self.action_space.sample()
        else:
            action = self.select_best_action(state)
        return action
    
    def on_action_performed(self, state, action, reward, new_state, terminated, truncated, env_idx=None) -> None:
        if self.test:
            return

        estimation = self.q_estimations[*state, action]
        new_estimation = np.max(self.q_estimations[*new_state, :])
        self.q_estimations[*state, action] = estimation + self.alpha*(reward + self.gamma*new_estimation - estimation)

        self.update_epsilon()

    def on_episode_end(self) -> None:
        return

    def update_epsilon(self) -> None:
        if not self.test:
            if self.linear_decay:
                self.epsilon = max(self.min_epsilon, self.epsilon - self.epsilon_decay)
            else:
                self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

class DQNAgent(RLAgent):
    def __init__(self, dqn, er_buffer: ExperienceReplayBuffer, action_space, lr=1e-4, batch_size=32, start_epsilon=1.0, 
                 epsilon_decay=0.99, min_epsilon=0.05, gamma=0.99, device="cpu", test=False, linear_decay=False, update_every_n_steps=1, n_envs=1,
                 use_lr_scheduler=False, end_lr=1e-7, lr_decay_steps=100000):
        self.dqn = dqn
        self.lr = lr
        self.optim = torch.optim.Adam(dqn.parameters(), lr=lr)

        self.gamma = gamma
        self.action_space = action_space

        self.er_buffer = er_buffer
        self.batch_size=batch_size

        self.start_epsilon = start_epsilon
        self.epsilon = start_epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.device = device
        self.dqn = self.dqn.to(device)
        self.test = test
        self.linear_decay = linear_decay
        self.update_every_n_steps = update_every_n_steps
        self.step_counter = 0
        self.n_envs = n_envs

        self.use_lr_scheduler = use_lr_scheduler
        self.end_lr = end_lr
        self.lr_decay_steps = lr_decay_steps
        if use_lr_scheduler:
            self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(self.optim, start_factor=1.0, end_factor=end_lr/lr, total_iters=lr_decay_steps)
    
    def policy(self, state) -> int:
        state = state.to(self.device)
        with torch.no_grad():
            actions = self.dqn(state).argmax(dim=1)
        
        if not self.test:
            epsilon_probs = torch.rand(self.n_envs, device=self.device)
            random_actions = torch.randint(0, self.action_space.nvec[0], size=(self.n_envs,), device=self.device)
            actions = torch.where(epsilon_probs < self.epsilon, random_actions, actions)


        return actions.cpu().numpy()

    def save_new_experience(self, state, action, reward, new_state, terminated, truncated):
        state, new_state = to_tensor(state), to_tensor(new_state)
        for i in range(self.n_envs):
            self.er_buffer.add_experience(
                state[i],
                action[i],
                reward[i],
                new_state[i],
                1 if terminated[i] or truncated[i] else 0
            )
  
    def update_network(self):
        self.step_counter = 0
        batch = self.er_buffer.sample_batch(self.batch_size)
        states, actions, rewards, next_states, terminals = batch
        states, next_states = states.to(self.device), next_states.to(self.device)
        actions, rewards, terminals = actions.to(self.device), rewards.to(self.device), terminals.to(self.device)

        q = torch.gather(self.dqn(states), dim=1, index=actions)

        with torch.no_grad():
            next_q = self.dqn(next_states).max(dim=1, keepdim=True)[0]
        target = rewards + self.gamma*next_q*(1-terminals)

        loss = F.mse_loss(target.squeeze(), q.squeeze())

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()  
        if self.use_lr_scheduler:
            self.lr_scheduler.step()

    def on_action_performed(self, state, action, reward, new_state, terminated, truncated) -> None:
        if self.test:
            return

        self.save_new_experience(state, action, reward, new_state, terminated, truncated)
        self.update_epsilon()

        self.step_counter+=1
        if self.step_counter<self.update_every_n_steps:
            return

        if not self.er_buffer.full:
            return

        self.update_network()

    
    def update_epsilon(self) -> None:
        if not self.test:
            if self.linear_decay:
                self.epsilon = max(self.min_epsilon, self.epsilon - self.epsilon_decay)
            else:
                self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

class DDQNAgent(RLAgent):
    def __init__(self, dqn, er_buffer: ExperienceReplayBuffer, action_space, lr=1e-4, batch_size=32, start_epsilon=1.0, 
                 epsilon_decay=0.99, min_epsilon=0.05, gamma=0.99, device="cpu", test=False, linear_decay=False, update_every_n_steps=1, n_envs=1, target_update_freq=1000,
                 use_lr_scheduler=False, end_lr=1e-7, lr_decay_steps=100000):
        self.dqn = dqn
        self.lr = lr
        self.optim = torch.optim.Adam(dqn.parameters(), lr=lr)

        self.target_dqn = copy.deepcopy(self.dqn)
        self.target_update_freq = target_update_freq
        self.total_steps = 0

        self.gamma = gamma
        self.action_space = action_space

        self.er_buffer = er_buffer
        self.batch_size=batch_size

        self.start_epsilon = start_epsilon
        self.epsilon = start_epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.device = device
        self.dqn = self.dqn.to(device)
        self.target_dqn = self.target_dqn.to(device)
        self.test = test
        self.linear_decay = linear_decay
        self.update_every_n_steps = update_every_n_steps
        self.step_counter = 0
        self.n_envs = n_envs

        self.use_lr_scheduler = use_lr_scheduler
        self.end_lr = end_lr
        self.lr_decay_steps = lr_decay_steps
        if use_lr_scheduler:
            self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(self.optim, start_factor=1.0, end_factor=end_lr/lr, total_iters=lr_decay_steps)

    def policy(self, state) -> int:
        state = state.to(self.device)
        with torch.no_grad():
            actions = self.dqn(state).argmax(dim=1)
        
        if not self.test:
            epsilon_probs = torch.rand(self.n_envs, device=self.device)
            random_actions = torch.randint(0, self.action_space.nvec[0], size=(self.n_envs,), device=self.device)
            actions = torch.where(epsilon_probs < self.epsilon, random_actions, actions)


        return actions.cpu().numpy()

    def save_new_experience(self, state, action, reward, new_state, terminated, truncated):
        state, new_state = to_tensor(state), to_tensor(new_state)
        for i in range(self.n_envs):
            self.er_buffer.add_experience(
                state[i],
                action[i],
                reward[i],
                new_state[i],
                1 if terminated[i] or truncated[i] else 0
            )
  
    def update_network(self):
        self.step_counter = 0
        batch = self.er_buffer.sample_batch(self.batch_size)
        states, actions, rewards, next_states, terminals = batch
        states, next_states = states.to(self.device), next_states.to(self.device)
        actions, rewards, terminals = actions.to(self.device), rewards.to(self.device), terminals.to(self.device)

        q = torch.gather(self.dqn(states), dim=1, index=actions)

        with torch.no_grad():
            best_next_actions = self.dqn(next_states).argmax(dim=1, keepdim=True)
            target_next_qs = self.target_dqn(next_states)
            next_q = torch.gather(target_next_qs, dim=1, index=best_next_actions)

        target = rewards + self.gamma*next_q*(1-terminals)

        loss = F.mse_loss(target.squeeze(), q.squeeze())

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()  
        if self.use_lr_scheduler:
            self.lr_scheduler.step()

    def on_action_performed(self, state, action, reward, new_state, terminated, truncated) -> None:
        if self.test:
            return

        self.save_new_experience(state, action, reward, new_state, terminated, truncated)
        self.update_epsilon()

        self.step_counter += 1
        self.total_steps += 1

        if self.total_steps % self.target_update_freq == 0:
            self.target_dqn.load_state_dict(self.dqn.state_dict())

        if self.step_counter<self.update_every_n_steps:
            return

        if not self.er_buffer.full:
            return

        self.update_network()

    
    def update_epsilon(self) -> None:
        if not self.test:
            if self.linear_decay:
                self.epsilon = max(self.min_epsilon, self.epsilon - self.epsilon_decay)
            else:
                self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    
class DuelingDDQNAgent:
    def __init__(self, dqn, er_buffer: PrioritizedExperienceReplayBuffer, action_space, lr=1e-4, batch_size=32, start_epsilon=1.0, 
                 epsilon_decay=0.99, min_epsilon=0.05, gamma=0.99, device="cpu", test=False, linear_decay=False, update_every_n_steps=1, n_envs=1, target_update_freq=1000,
                 use_lr_scheduler=False, end_lr=1e-7, lr_decay_steps=100000):
        
        self.dqn = dqn
        self.lr = lr
        self.optim = torch.optim.Adam(dqn.parameters(), lr=lr)

        self.target_dqn = copy.deepcopy(self.dqn)
        self.target_update_freq = target_update_freq
        self.total_steps = 0

        self.gamma = gamma
        self.action_space = action_space

        self.er_buffer = er_buffer
        self.batch_size = batch_size

        self.start_epsilon = start_epsilon
        self.epsilon = start_epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.device = device
        self.dqn = self.dqn.to(device)
        self.target_dqn = self.target_dqn.to(device)
        self.test = test
        self.linear_decay = linear_decay
        self.update_every_n_steps = update_every_n_steps
        self.step_counter = 0
        self.n_envs = n_envs

        self.use_lr_scheduler = use_lr_scheduler
        self.end_lr = end_lr
        self.lr_decay_steps = lr_decay_steps
        if use_lr_scheduler:
            self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(self.optim, start_factor=1.0, end_factor=end_lr/lr, total_iters=lr_decay_steps)

    def policy(self, state) -> int:
        state = state.to(self.device)
        with torch.no_grad():
            actions = self.dqn(state).argmax(dim=1)
        
        if not self.test:
            epsilon_probs = torch.rand(self.n_envs, device=self.device)
            n_actions = self.action_space.nvec[0] if hasattr(self.action_space, 'nvec') else self.action_space.n
            random_actions = torch.randint(0, n_actions, size=(self.n_envs,), device=self.device)
            actions = torch.where(epsilon_probs < self.epsilon, random_actions, actions)

        return actions.cpu().numpy()

    def save_new_experience(self, state, action, reward, new_state, terminated, truncated):
        state, new_state = to_tensor(state), to_tensor(new_state)
        for i in range(self.n_envs):
            self.er_buffer.add_experience(
                state[i],
                action[i],
                reward[i],
                new_state[i],
                1 if terminated[i] or truncated[i] else 0
            )
  
    def update_network(self):
        self.step_counter = 0
 
        batch = self.er_buffer.sample_batch(self.batch_size)

        states, actions, rewards, next_states, terminals, indices, weights = batch

        states, next_states = states.to(self.device), next_states.to(self.device)
        actions, rewards, terminals = actions.to(self.device), rewards.to(self.device), terminals.to(self.device)

        q = torch.gather(self.dqn(states), dim=1, index=actions)

       
        with torch.no_grad():
            best_next_actions = self.dqn(next_states).argmax(dim=1, keepdim=True)
            target_next_qs = self.target_dqn(next_states)
            next_q = torch.gather(target_next_qs, dim=1, index=best_next_actions)

        target = rewards + self.gamma * next_q * (1 - terminals)
        

        td_errors = target.squeeze() - q.squeeze()
        self.er_buffer.update_priorities(indices, td_errors.detach())

        loss = (weights * (td_errors ** 2)).mean()

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()  
        
        if self.use_lr_scheduler:
            self.lr_scheduler.step()

    def on_action_performed(self, state, action, reward, new_state, terminated, truncated) -> None:
        if self.test:
            return

        self.save_new_experience(state, action, reward, new_state, terminated, truncated)
        self.update_epsilon()

        self.step_counter += 1
        self.total_steps += 1

        if self.total_steps % self.target_update_freq == 0:
            self.target_dqn.load_state_dict(self.dqn.state_dict())

        if self.step_counter < self.update_every_n_steps:
            return

        if not self.er_buffer.full:
            return

        self.update_network()

    def update_epsilon(self) -> None:
        if not self.test:
            if self.linear_decay:
                self.epsilon = max(self.min_epsilon, self.epsilon - self.epsilon_decay)
            else:
                self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    