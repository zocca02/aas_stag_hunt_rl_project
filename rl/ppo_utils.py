from abc import ABC, abstractmethod
import torch
from torch.optim import Adam
import matplotlib.pyplot as plt
import pickle

from utils.utils import arr_avg_last_n


class PPOModelWrapper(ABC):

    @abstractmethod
    def forward(self, x):
        raise NotImplemented
    
    @abstractmethod
    def act(self, x):
        raise NotImplemented
    
    @abstractmethod
    def evaluate(self, x):
        raise NotImplemented
    
    @abstractmethod
    def optimize(self, loss_policy, loss_critic, loss_entropy):
        raise NotImplemented

    @abstractmethod
    def update_lr(self):
        raise NotImplemented
    
    @abstractmethod
    def get_n_optimizers(self):
        raise NotImplemented

    @abstractmethod
    def get_current_lr(self):
        raise NotImplemented
    
    def __call__(self, x):
        return self.forward(x)

class PPOSignleModelWrapper(PPOModelWrapper):
    def __init__(self, network, lr, lr_annealing=False, max_updates=None, device="cpu"):
        assert not lr_annealing or (max_updates is not None)

        self.device = device
        self.network = network.to(device)

        self.lr = lr

        self.lr_annealing = lr_annealing
        self.max_updates = max_updates

        self.optimizer = Adam(network.parameters(), lr=lr, eps=1e-5)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1, end_factor=0, total_iters=max_updates) if lr_annealing else None
    
    def forward(self, x):
        return self.network(x)
    
    def act(self, x):
        return self.network.act(x)

    def evaluate(self, x):
        return self.network.evaluate(x)
    
    def optimize(self, loss_policy, loss_critic, loss_entropy, clip_grad_norm=True, max_grad_norm=0.5):
        loss = -loss_policy + loss_critic - loss_entropy

        self.optimizer.zero_grad()
        loss.backward()
        
        max_grad_norm = max_grad_norm if clip_grad_norm else float('inf')
        grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_grad_norm)

        self.optimizer.step()

        return grad_norm

    def update_lr(self):
        if self.lr_annealing:
            self.scheduler.step()
    
    def get_n_optimizers(self):
        return 1

    def get_current_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def save_model(self, base_name):
        self.network = self.network.cpu()
        torch.save(self.network.state_dict(), f"{base_name}.pth")
        self.network = self.network.to(self.device)
    
    def load_model(self, base_name):
        self.network = self.network.cpu()
        self.network.load_state_dict(torch.load(f"{base_name}.pth"))
        self.network = self.network.to(self.device)

class PPOSplittedModelWrapper(PPOModelWrapper):
    def __init__(self, network_policy, network_critic, lr_policy, lr_critic, lr_annealing=False, max_updates=None, device="cpu"):
        assert not lr_annealing or (max_updates is not None)

        self.device = device
        self.network_policy = network_policy.to(device)
        self.network_critic = network_critic.to(device)

        self.lr_policy = lr_policy
        self.lr_critic = lr_critic

        self.lr_annealing = lr_annealing
        self.max_updates = max_updates

        self.optimizer_policy = Adam(network_policy.parameters(), lr=lr_policy, eps=1e-5)
        self.optimizer_critic = Adam(network_critic.parameters(), lr=lr_critic, eps=1e-5)
        self.scheduler_policy = torch.optim.lr_scheduler.LinearLR(self.optimizer_policy, start_factor=1, end_factor=0, total_iters=max_updates) if lr_annealing else None
        self.scheduler_critic = torch.optim.lr_scheduler.LinearLR(self.optimizer_critic, start_factor=1, end_factor=0, total_iters=max_updates) if lr_annealing else None
    
    def forward(self, x):
        return self.network_policy(x), self.network_critic(x)
    
    def act(self, x):
        return self.network_policy(x)

    def evaluate(self, x):
        return self.network_critic(x)
    
    def optimize(self, loss_policy, loss_critic, loss_entropy, clip_grad_norm=True, max_grad_norm=0.5):

        loss_policy = -loss_policy - loss_entropy

        self.optimizer_policy.zero_grad()
        self.optimizer_critic.zero_grad()

        loss_policy.backward()
        loss_critic.backward()

        max_grad_norm = max_grad_norm if clip_grad_norm else float('inf')
        policy_grad_norm = torch.nn.utils.clip_grad_norm_(self.network_policy.parameters(), max_grad_norm)
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.network_critic.parameters(), max_grad_norm)

        self.optimizer_policy.step()
        self.optimizer_critic.step()

        return policy_grad_norm, critic_grad_norm

    def update_lr(self):
        if self.lr_annealing:
            self.scheduler_policy.step()
            self.scheduler_critic.step()
    
    def get_n_optimizers(self):
        return 2

    def get_current_lr(self):
        return self.optimizer_policy.param_groups[0]['lr']

    def save_model(self, base_name):
        self.network_policy = self.network_policy.cpu()
        self.network_critic = self.network_critic.cpu()

        torch.save(self.network_policy.state_dict(), f"{base_name}_policy.pth")
        torch.save(self.network_critic.state_dict(), f"{base_name}_critic.pth")

        self.network_policy = self.network_policy.to(self.device)
        self.network_critic = self.network_critic.to(self.device)
    
    def load_model(self, base_name):
        self.network_policy = self.network_policy.cpu()
        self.network_critic = self.network_critic.cpu()

        self.network_policy.load_state_dict(torch.load(f"{base_name}_policy.pth"))
        self.network_critic.load_state_dict(torch.load(f"{base_name}_critic.pth"))

        self.network_policy = self.network_policy.to(self.device)
        self.network_critic = self.network_critic.to(self.device)


class RolloutBuffer:
    def __init__(self, n_envs, size, state_shape, batch_size, gamma, lambda_gae, device="cpu"):
        assert (size*n_envs) % batch_size == 0

        self.n_envs = n_envs
        self.size = size
        self.batch_size = batch_size
        self.n_batches = self.size*self.n_envs // self.batch_size
        self.state_shape = state_shape
        self.device = device

        self.gamma = gamma
        self.lambda_gae = lambda_gae

        self.reset()
    
    def reset(self):
        self.current_idx = 0
        self.advantages_computed = False
        self.buffer_flattened = False

        self.states = torch.zeros(self.size, self.n_envs, *self.state_shape, device=self.device)
        self.state_values = torch.zeros(self.size, self.n_envs, device=self.device)

        self.actions = torch.zeros(self.size, self.n_envs, device=self.device)
        self.log_probs = torch.zeros(self.size, self.n_envs, device=self.device)
        self.rewards = torch.zeros(self.size, self.n_envs, device=self.device)

        self.dones = torch.zeros(self.size, self.n_envs, device=self.device)

        self.advantages = torch.zeros(self.size, self.n_envs, device=self.device)
        self.returns = torch.zeros(self.size, self.n_envs, device=self.device)

    def add_transition(self, transition_dict):
        assert self.current_idx < self.size

        self.states[self.current_idx] = transition_dict["states"]
        self.state_values[self.current_idx] = transition_dict["state_values"]
        self.actions[self.current_idx] = transition_dict["actions"]
        self.log_probs[self.current_idx] = transition_dict["log_probs"]
        self.rewards[self.current_idx] = transition_dict["rewards"]
        self.dones[self.current_idx] = transition_dict["dones"]
        self.current_idx += 1

    def compute_advantages(self, last_state_values):
        # for sv in self.state_values:
        #     assert not sv.requires_grad
        # assert not last_state_values.requires_grad

        next_state_values = last_state_values
        next_adv = 0.0

        for t in reversed(range(self.size)):
            delta = self.rewards[t] + self.gamma*next_state_values*(1-self.dones[t]) - self.state_values[t]
            adv = delta + self.gamma*self.lambda_gae*next_adv*(1-self.dones[t])

            next_adv = adv
            next_state_values = self.state_values[t]

            self.advantages[t] = adv
            self.returns[t] = adv + self.state_values[t]

        self.advantages_computed = True

    def flatten_buffer(self):
        assert self.current_idx == self.size
        assert self.advantages_computed

        self.flattened_states = self.states.flatten(end_dim=1)
        self.flatten_state_values = self.state_values.flatten(end_dim=1)
        self.flatten_actions = self.actions.flatten(end_dim=1)
        self.flatten_log_probs = self.log_probs.flatten(end_dim=1)
        self.flatten_rewards = self.rewards.flatten(end_dim=1)
        self.flatten_dones = self.dones.flatten(end_dim=1)
        self.flatten_advantages = self.advantages.flatten(end_dim=1)
        self.flatten_returns = self.returns.flatten(end_dim=1)

        self.buffer_flattened = True

    def get_batches(self):
        assert self.current_idx == self.size
        assert self.advantages_computed
        assert self.buffer_flattened

        idxs = torch.randperm(self.flattened_states.shape[0])

        for i in range(self.n_batches):
            batch_idxs = idxs[i*self.batch_size:(i+1)*self.batch_size]

            yield (self.flattened_states[batch_idxs], self.flatten_state_values[batch_idxs], 
                    self.flatten_actions[batch_idxs], self.flatten_log_probs[batch_idxs],
                    self.flatten_rewards[batch_idxs], self.flatten_dones[batch_idxs], 
                    self.flatten_advantages[batch_idxs], self.flatten_returns[batch_idxs])

    def is_full(self):
        return self.current_idx == self.size



class PPOStats:
    def __init__(self, single_grad_norm=False, last_n_avg=50):
        self.policy_loss = []
        self.critic_loss = []
        self.entropy = []

        self.grad_norm = []
        self.policy_grad_norm = []
        self.critic_grad_norm = []

        self.approx_kl = []
        self.clip_fraction = []
        self.lr = []

        self.hyperparameters = None
        self.episodic_rewards = []

        self.last_n_avg = last_n_avg
        self.single_grad_norm = single_grad_norm
    
    def save(self, file_name):
        with open(file_name, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file_name):
        with open(file_name, "rb") as f:
            log = pickle.load(f)
        return log

    def add_stats(self, policy_loss, critic_loss, entropy, approx_kl, clip_fraction, lr, policy_grad_norm=None, critic_grad_norm=None, grad_norm=None):
        assert (policy_grad_norm is not None and critic_grad_norm is not None and not self.single_grad_norm) or (grad_norm is not None and self.single_grad_norm)

        self.policy_loss.append(policy_loss)
        self.critic_loss.append(critic_loss)
        self.entropy.append(entropy)
        self.lr.append(lr)
    
        if not self.single_grad_norm:
            self.policy_grad_norm.append(policy_grad_norm)
            self.critic_grad_norm.append(critic_grad_norm)
        else:
            self.grad_norm.append(grad_norm)

        self.approx_kl.append(approx_kl)
        self.clip_fraction.append(clip_fraction)
    
    
    def plot_stats(self, fig_num=2, filename="ppo_stats.png", show_lr=False, max_grad_norm=None, episodic_rewards=None, 
                   policy_loss_y_clipping=None, critic_loss_y_clipping=None, grad_norm_y_clipping=None, policy_grad_norm_y_clipping=None, critic_grad_norm_y_clipping=None):
        if self.single_grad_norm:
            fig_layout=(2, 4)
        else:
            fig_layout=(2, 4)

        plt.style.use('seaborn-v0_8-whitegrid')

        fig, axes = plt.subplots(*fig_layout, num=fig_num, figsize=(16, 8))
        axes = axes.flatten()

        # Raccogliamo i dati in una lista per ciclare e scrivere codice più pulito
        metrics = [
            ("Policy Loss", self.policy_loss, 'tab:blue', "Loss", [], policy_loss_y_clipping),
            ("Critic Loss", self.critic_loss, 'tab:red', "Loss", [], critic_loss_y_clipping),
            ("Entropy", self.entropy, 'tab:green', "Entropy", [], None),
            ("Approx KL", self.approx_kl, 'tab:brown', "KL Divergence", [0.05], None),
            ("Clip Fraction", self.clip_fraction, 'tab:cyan', "Percentage", [0.25], None),
        ]

        if episodic_rewards is not None:
            metrics.insert(0, ("Episodic Rewards", episodic_rewards, 'tab:blue', "Reward", [], None))

        if show_lr:
            metrics.append(("Learning Rate", self.lr, 'tab:purple', "Rate", [], None))

        grad_norm_ref = [] if max_grad_norm is None else [max_grad_norm]
        if self.single_grad_norm:
            metrics.append(("Grad Norm", self.grad_norm, 'tab:orange', "Norm", grad_norm_ref, grad_norm_y_clipping))
        else:
            metrics.append(("Policy Grad Norm", self.policy_grad_norm, 'tab:orange', "Norm", grad_norm_ref, policy_grad_norm_y_clipping))
            metrics.append(("Critic Grad Norm", self.critic_grad_norm, 'tab:pink', "Norm", grad_norm_ref, critic_grad_norm_y_clipping))

        for i, (title, data, color, y_label, ref_lines, y_clipping) in enumerate(metrics):
            ax = axes[i]
            ax.plot(data, linestyle='-', color=color, linewidth=2, alpha=0.2)

            if title!="Learning Rate":
                ax.plot(arr_avg_last_n(data, self.last_n_avg), linestyle='-', color=color, linewidth=1, alpha=1.0)

            if y_clipping is not None:
                ax.set_ylim(-1, y_clipping)

            ax.set_title(title, fontsize=12, fontweight='bold', color='#333333')
            ax.set_xlabel("Updates", fontsize=10, color='#555555')
            ax.set_ylabel(y_label, fontsize=10, color='#555555')
            ax.tick_params(axis='both', which='major', labelsize=9)

            for ref in ref_lines:
                ax.axhline(y=ref, color='red', linestyle='--', linewidth=1, alpha=0.7)
            
            ax.grid(True, linestyle='--', alpha=0.6)

        for j in range(len(metrics), len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle("PPO Training Statistics", fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
