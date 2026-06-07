import numpy as np
import torch

class ExperienceReplayBuffer:
    def __init__(self, state_shape, dim, start_after=None):
        self.dim = dim
        
        self.states = torch.zeros((dim, *state_shape), dtype=torch.float32)
        self.actions = torch.zeros(dim, dtype=torch.int64)
        self.rewards = torch.zeros(dim, dtype=torch.float32)
        self.next_states = torch.zeros((dim, *state_shape), dtype=torch.float32)
        self.terminals = torch.zeros(dim, dtype=torch.float32)

        self.current_idx = 0
        self.full = False
        self.total_full = False
        if start_after==None:
            self.start_after = dim
        else:
            self.start_after = start_after
    
    def add_experience(self, state, action, reward, next_state, terminal):
        self.states[self.current_idx] = torch.as_tensor(state, dtype=torch.float32)
        self.actions[self.current_idx] = torch.as_tensor(action, dtype=torch.int64)
        self.rewards[self.current_idx] = torch.as_tensor(reward, dtype=torch.float32)
        self.next_states[self.current_idx] = torch.as_tensor(next_state, dtype=torch.float32)
        self.terminals[self.current_idx] = torch.as_tensor(terminal, dtype=torch.float32)

        self.current_idx = (self.current_idx + 1) % self.dim

        if not self.full and (self.current_idx == 0 or self.current_idx>=self.start_after):
            self.full = True
        
        if not self.total_full and self.current_idx == 0:
            self.total_full=True
    
    def sample_batch(self, batch_size, return_as_tensors=True):
        if not self.full:
            raise ValueError("Il buffer non è ancora pieno!")

        # For efficiency instead of torch.randperm (small statistical difference for a big buffer)
        batch_idxs = torch.randint(0, self.dim if self.total_full else self.current_idx, size=(batch_size,))

        b_states = self.states[batch_idxs]
        b_next_states = self.next_states[batch_idxs]
        
        b_actions = self.actions[batch_idxs, None]
        b_rewards = self.rewards[batch_idxs, None]
        b_terminals = self.terminals[batch_idxs, None]

        return b_states, b_actions, b_rewards, b_next_states, b_terminals



class PrioritizedExperienceReplayBuffer:
    def __init__(self, state_shape, dim, start_after=None, alpha=0.6, beta=0.4, beta_iters=100000, epsilon=1e-6):
        self.dim = dim
        
        self.states = torch.zeros((dim, *state_shape), dtype=torch.float32)
        self.actions = torch.zeros(dim, dtype=torch.int64)
        self.rewards = torch.zeros(dim, dtype=torch.float32)
        self.next_states = torch.zeros((dim, *state_shape), dtype=torch.float32)
        self.terminals = torch.zeros(dim, dtype=torch.float32)
        self.priorities = torch.zeros(dim, dtype=torch.float32)

        self.alpha = alpha
        self.beta = beta
        self.beta_increment = (1.0 - beta) / beta_iters
        self.max_priority = 1.0
        self.epsilon = epsilon

        self.current_idx = 0
        self.full = False
        self.total_full = False
        if start_after==None:
            self.start_after = dim
        else:
            self.start_after = start_after
    
    def add_experience(self, state, action, reward, next_state, terminal):
        self.states[self.current_idx] = torch.as_tensor(state, dtype=torch.float32)
        self.actions[self.current_idx] = torch.as_tensor(action, dtype=torch.int64)
        self.rewards[self.current_idx] = torch.as_tensor(reward, dtype=torch.float32)
        self.next_states[self.current_idx] = torch.as_tensor(next_state, dtype=torch.float32)
        self.terminals[self.current_idx] = torch.as_tensor(terminal, dtype=torch.float32)
        self.priorities[self.current_idx] = self.max_priority

        self.current_idx = (self.current_idx + 1) % self.dim

        if not self.full and (self.current_idx == 0 or self.current_idx >= self.start_after):
            self.full = True
        
        if not self.total_full and self.current_idx == 0:
            self.total_full = True
    
    def sample_batch(self, batch_size):
        if not self.full:
            raise ValueError("Il buffer non è ancora pieno!")


        dim = self.dim if self.total_full else self.current_idx
        priorities = self.priorities[:dim]

        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = torch.multinomial(probs, batch_size, replacement=True)

        weights = (dim * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        self.beta = min(1.0, self.beta + self.beta_increment)

        b_states = self.states[indices]
        b_next_states = self.next_states[indices]
        b_actions = self.actions[indices, None]
        b_rewards = self.rewards[indices, None]
        b_terminals = self.terminals[indices, None]

        return b_states, b_actions, b_rewards, b_next_states, b_terminals, indices, weights


    def update_priorities(self, indices, td_errors):
        new_priorities = (torch.abs(td_errors) + self.epsilon).cpu()
        self.priorities[indices] = new_priorities
        self.max_priority = max(self.max_priority, new_priorities.max().item())

