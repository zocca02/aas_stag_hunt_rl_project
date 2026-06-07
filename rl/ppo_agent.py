import torch
from torch.distributions import Categorical
import numpy as np

from rl.ppo_utils import PPOModelWrapper, RolloutBuffer, PPOStats
from rl.agents import RLAgent


class PPOAgent(RLAgent):
    
    def __init__(self, model_wrapper: PPOModelWrapper, n_envs, rollout_steps, state_shape, batch_size, 
                 clip_coef=0.2, epochs=4, gamma=0.9, lambda_gae=0.95,
                 critic_loss_coef=0.5, entropy_temperature = 0.0,
                 normalize_advantages=False, clip_grad_norm=True, max_grad_norm=0.5,
                 device="cpu", test=False, greedy_on_test=False):
        
        ###
        self.n_envs = n_envs
        self.device = device
        self.state_shape = state_shape
        self.test = test
        self.greedy_on_test = greedy_on_test

        ###
        self.model_wrapper = model_wrapper
        self.rollout_steps = rollout_steps
        self.epochs = epochs
        self.batch_size = batch_size
        
        ###
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.clip_coef = clip_coef
        self.entropy_temperature = entropy_temperature
        self.critic_loss_coef = critic_loss_coef
        self.max_grad_norm = max_grad_norm

        ###
        self.normalize_advantages = normalize_advantages
        self.clip_grad_norm = clip_grad_norm
                
        self.rollout_buffer = RolloutBuffer(n_envs, rollout_steps, state_shape, batch_size, gamma, lambda_gae, device=device)

        self.stats = PPOStats(single_grad_norm=(model_wrapper.get_n_optimizers()==1))

        self.current_transition = {
            "states": None, "state_values": None, 
            "actions": None, "log_probs": None, 
            "rewards": None, "dones": None
        }

    def policy(self, states):
        if torch.is_tensor(states):
            states = states.to(self.device)
        else:
            states = torch.tensor(states, device=self.device)
            
        with torch.no_grad():
            logits, state_values = self.model_wrapper(states)
        
        dist = Categorical(logits=logits)
        
        if self.test and self.greedy_on_test:
            actions = torch.argmax(logits, dim=-1)
        else:
            actions = dist.sample()
            
        if not self.test:
            self.current_transition["states"] = states
            self.current_transition["state_values"] = state_values.squeeze()
            self.current_transition["actions"] = actions
            self.current_transition["log_probs"] = dist.log_prob(actions)

        return actions.cpu().numpy()
    
    def on_action_performed(self, states, actions, rewards, new_states, terminations, truncations):
        if self.test:
            return
        
        self.current_transition["rewards"] = torch.tensor(rewards, device=self.device)
        self.current_transition["dones"] = torch.tensor(terminations, dtype=torch.int32, device=self.device)

        self.rollout_buffer.add_transition(self.current_transition)

        if self.rollout_buffer.is_full():
            if torch.is_tensor(new_states):
                new_states = new_states.to(self.device)
            else:
                new_states = torch.tensor(new_states, device=self.device)
            with torch.no_grad():
                new_state_values = self.model_wrapper.evaluate(new_states).squeeze()
            self.rollout_buffer.compute_advantages(new_state_values.cpu())
            self.learn()
            self.rollout_buffer.reset()
        
        
    def learn(self):

        # Stats Init
        policy_losses, critic_losses, entropies = [], [], []
        policy_grad_norms, critic_grad_norms, grad_norms = [], [], []
        approx_kls, clip_fractions = [], []

        self.rollout_buffer.flatten_buffer()
        #dl = self.rollout_buffer.get_dl()
        for e in range(self.epochs):
            for states, state_values, actions, log_probs, rewards, dones, advatages, returns in self.rollout_buffer.get_batches():
                
                if self.normalize_advantages:
                    advatages = (advatages - advatages.mean()) / (advatages.std() + 1e-8)

                states = states.to(self.device)

                # Policy loss
                logits = self.model_wrapper.act(states)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(actions)

                log_prob_ratio = new_log_probs - log_probs
                prob_ratio = log_prob_ratio.exp()
                clipped_prob_ratio = torch.clip(prob_ratio, 1-self.clip_coef, 1+self.clip_coef)

                policy_loss = torch.min(prob_ratio*advatages, clipped_prob_ratio*advatages).mean()

                # Entropy loss
                entropy = dist.entropy().mean()
                entropy_loss = self.entropy_temperature*entropy

                # Critic loss
                values = self.model_wrapper.evaluate(states).squeeze()
                critic_loss = self.critic_loss_coef*((returns-values)**2).mean()

                grad_norm = self.model_wrapper.optimize(policy_loss, critic_loss, entropy_loss, clip_grad_norm=self.clip_grad_norm, max_grad_norm=self.max_grad_norm)




                ############### Stats update
                policy_losses.append(policy_loss.item())
                critic_losses.append(critic_loss.item())
                entropies.append(entropy.item())


                if self.model_wrapper.get_n_optimizers() == 1:
                    grad_norms.append(grad_norm.item())
                else:
                    policy_grad_norms.append(grad_norm[0].item())
                    critic_grad_norms.append(grad_norm[1].item())

                with torch.no_grad():
                    approx_kl = (log_prob_ratio).mean().item()
                    clipped = (prob_ratio > (1+self.clip_coef)) | (prob_ratio < (1-self.clip_coef))
                    clip_fraction = clipped.float().mean().item()

                approx_kls.append(approx_kl)
                clip_fractions.append(clip_fraction)
                ###############

        # Stats
        if self.model_wrapper.get_n_optimizers() == 1:
            self.stats.add_stats(np.mean(policy_losses), np.mean(critic_losses), np.mean(entropies), np.mean(approx_kls), np.mean(clip_fractions), 
                                 lr=self.model_wrapper.get_current_lr(), grad_norm=np.mean(grad_norms))
        else:
            self.stats.add_stats(np.mean(policy_losses), np.mean(critic_losses), np.mean(entropies), np.mean(approx_kls), np.mean(clip_fractions), 
                                 lr=self.model_wrapper.get_current_lr(), policy_grad_norm=np.mean(policy_grad_norms), critic_grad_norm=np.mean(critic_grad_norms))

        self.model_wrapper.update_lr()

    def on_end(self):
        pass
