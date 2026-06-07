import gymnasium as gym
from gymnasium import spaces
import gymnasium_stag_hunt

import random
import numpy as np

import torch
import torch.nn as nn
import argparse
import os

from utils.escalation_utils import EscalationLooper
from rl.q_agents import DuelingDDQNAgent
from utils.experience_replay import PrioritizedExperienceReplayBuffer

SEED = 42

os.environ['PYTHONHASHSEED'] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

arg_parser = argparse.ArgumentParser()

# Generic
arg_parser.add_argument("-x", "--use-code-params", action="store_true", help="Use params embedded into the code")
arg_parser.add_argument("-t", "--test", action="store_true", help="Enables test mode")
arg_parser.add_argument("-s", "--save", action="store_true", help="Save weights or estimations")
arg_parser.add_argument("-c", "--checkpoint", action="store_true", help="Start from checkpoint")

arg_parser.add_argument("-bf", "--base-folder", type=str, default="savings/escalation/", help="Stats file name")
arg_parser.add_argument("-n", "--name", type=str, default="escalation", help="Generic name")

# Environment
arg_parser.add_argument("-E", "--n-envs", type=int, default=1, help="N of parallel environments")
arg_parser.add_argument("-W", "--width", type=int, default=4, help="Width of the grid")
arg_parser.add_argument("-H", "--height", type=int, default=4, help="Height of the grid")
#arg_parser.add_argument("-N", "--n-steps", type=int, default=200e+3, help="N of steps of the training")
arg_parser.add_argument("-S", "--time-steps", type=int, default=200, help="N of time steps in each episode")
arg_parser.add_argument("-e", "--episodes", type=int, default=1000, help="N of episodes")

arg_parser.add_argument("-A", "--async-envs", action="store_true", help="Async parallelization of the environments (otherwise sync)")
arg_parser.add_argument("-R", "--common-rewards", action="store_true", help="Enables common rewards. Each agent will recieve as reward the sum of both agents' rewards")

# Escalation
arg_parser.add_argument("-bp", "--streak-brak-punishment", type=float, default=.5, help="Streak break punishment factor")
arg_parser.add_argument("-sl", "--streak-len", action="store_true", help="Add streak length to the state")

# PPO
arg_parser.add_argument("-Dse", "--start-epsilon", type=float, default=1.0, help="Start epsilon")
arg_parser.add_argument("-Dee", "--end-epsilon", type=float, default=0.1, help="End epsilon")
arg_parser.add_argument("-Dd", "--time-to-decay", type=float, default=0.7, help="Perc. of time steps to reach epsilon end")
arg_parser.add_argument("-Derb", "--er-buffer-size", type=int, default=10000, help="Experience replay buffer size")
arg_parser.add_argument("-Dbs", "--batch-size", type=int, default=32, help="Batch size")
arg_parser.add_argument("-Dus", "--update-steps", type=int, default=1, help="Update every n steps")
arg_parser.add_argument("-Dduf", "--ddqn-update-freq", type=int, default=200, help="DDQN update frequency")

arg_parser.add_argument("-g", "--gamma", type=float, default=.9, help="Gamma discount")
arg_parser.add_argument("-lr", "--learning-rate", type=float, default=3e-4, help="Learning rate")
arg_parser.add_argument("-a", "--alpha", type=float, default=0.6, help="Alpha for prioritized experience replay")
arg_parser.add_argument("-b", "--beta", type=float, default=0.4, help="Beta for prioritized experience replay")
arg_parser.add_argument("-bi", "--beta-iters", type=int, default=100000, help="Beta for prioritized experience replay")

device = "cuda" if torch.cuda.is_available() else "cpu"
args = arg_parser.parse_args()

if args.use_code_params:
    args = arg_parser.parse_args([
        # Generic
        "-s", 
        #"-c",
        
        # Env
        "-E", "1", "-W", "6", "-H", "6", "-S", "200", "-e", "10", "-A",

        # Escalation
        "-bp", ".5", "-sl", "-cs",

        # DQN
        "-g", ".9", "-lr", "2e-3",
        "-Dse", "1.0", "-Dee", "0.05", "-Dd", "0.8", 
        "-Derb", f"{int(100e+3)}", "-Dbs", "64", "-Dus", "4"
    ])
    
########################
# CONFIG
########################


ENABLE_GUI = False
PRINTING_DELAY = .5
ENABLE_GRAPHIC = False
PRINT_REWARDS = False

if args.test:
    ENABLE_GUI = True
    PRINTING_DELAY = .5
    ENABLE_GRAPHIC = True
    PRINT_REWARDS = True

    args.time_steps = 200
    args.episodes = 20
    
PRINTING_TS = 100*args.time_steps
N_STEPS = args.episodes*args.time_steps
DECAY_STEPS = args.time_to_decay*N_STEPS
EPSILON_DECAY = (args.start_epsilon-args.end_epsilon)/(DECAY_STEPS)
    

# Env config

env_params = {
    "obs_type": "coords",
    "grid_size": (args.width, args.height),
    "screen_size": (1500, 1500),
    "load_renderer": ENABLE_GUI and ENABLE_GRAPHIC,
    "max_timesteps": args.time_steps,

    "streak_break_punishment_factor": args.streak_brak_punishment
}



looper = EscalationLooper(args.n_envs, env_params, parallelization="async" if args.async_envs else "sync",print_rewards=PRINT_REWARDS, test=args.test, seed=SEED,
                            evaluate_during_training=not args.test, evaluate_every_n_steps=args.time_steps, n_evaluations=4,
                            add_streak_len=args.streak_len, max_streak_len=args.time_steps)
action_space = looper.get_action_space()
n_actions = looper.envs.action_space.nvec[0]
state_shape = (4+int(args.streak_len),)


########################
# DQN
########################

class DuelingDQN(nn.Module):
    def __init__(self, input_size, n_actions):
        super(DuelingDQN, self).__init__()
        
        self.features_net = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU()
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.advantages_head = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, n_actions)
        )

    def forward(self, x):
        features = self.features_net(x)
        
        values = self.value_head(features)
        advantages = self.advantages_head(features)
        
        q_vals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_vals
    

########################
# AGENT1
########################

er_buffer_agent1 = PrioritizedExperienceReplayBuffer(state_shape, args.er_buffer_size, start_after=int(0.1*args.er_buffer_size), alpha=args.alpha, beta=args.beta, beta_iters=args.beta_iters)
dqn_agent1 = DuelingDQN(state_shape[0], n_actions)

if args.test or args.checkpoint:
    dqn_agent1.load_state_dict(torch.load(f"{args.base_folder}/models/{args.name}_ag1.pth"))

agent1 = DuelingDDQNAgent(dqn_agent1, er_buffer_agent1, action_space, lr=args.learning_rate, batch_size=args.batch_size, update_every_n_steps=args.update_steps, n_envs = args.n_envs, target_update_freq=args.ddqn_update_freq,
                  start_epsilon=args.start_epsilon, epsilon_decay=EPSILON_DECAY, min_epsilon=args.end_epsilon, gamma=args.gamma, device=device, test=args.test, linear_decay=True)

 
########################
# AGENT2
########################

er_buffer_agent2 = PrioritizedExperienceReplayBuffer(state_shape, args.er_buffer_size, start_after=int(0.1*args.er_buffer_size), alpha=args.alpha, beta=args.beta, beta_iters=args.beta_iters)
dqn_agent2 = DuelingDQN(state_shape[0], n_actions)

if args.test or args.checkpoint:
    dqn_agent2.load_state_dict(torch.load(f"{args.base_folder}/models/{args.name}_ag2.pth"))

agent2 = DuelingDDQNAgent(dqn_agent2, er_buffer_agent2, action_space, lr=args.learning_rate, batch_size=args.batch_size, update_every_n_steps=args.update_steps, n_envs = args.n_envs, target_update_freq=args.ddqn_update_freq,
                  start_epsilon=args.start_epsilon, epsilon_decay=EPSILON_DECAY, min_epsilon=args.end_epsilon, gamma=args.gamma, device=device, test=args.test, linear_decay=True)


########################
# Loop
########################

looper.training_loop(agent1, agent2, N_STEPS, common_reward=args.common_rewards, enable_gui=ENABLE_GUI, printing_delay=PRINTING_DELAY, ts_to_print=PRINTING_TS)

########################
# Savings
########################

if args.save:
    looper.evaluation_log_agents[0].hyperparameters = args
    looper.evaluation_log_agents[1].hyperparameters = args

    looper.evaluation_log_agents[0].save(f"{args.base_folder}/logs/{args.name}_ag1.pkl")
    looper.evaluation_log_agents[1].save(f"{args.base_folder}/logs/{args.name}_ag2.pkl")

    torch.save(agent1.dqn.state_dict(), f"{args.base_folder}/models/{args.name}_ag1.pth")
    torch.save(agent2.dqn.state_dict(), f"{args.base_folder}/models/{args.name}_ag2.pth")

looper.envs.close()