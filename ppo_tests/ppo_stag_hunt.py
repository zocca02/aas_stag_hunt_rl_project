import gymnasium as gym
from gymnasium import spaces
import gymnasium_stag_hunt

import torch
import torch.nn as nn
import argparse

from utils.stag_hunt_utils import StagHuntLooper
from rl.ppo_agent import PPOAgent
from rl.ppo_utils import PPOSplittedModelWrapper
from utils.experience_replay import ExperienceReplayBuffer


arg_parser = argparse.ArgumentParser()

# Generic
arg_parser.add_argument("-x", "--use-code-params", action="store_true", help="Use params embedded into the code")
arg_parser.add_argument("-t", "--test", action="store_true", help="Enables test mode")
arg_parser.add_argument("-s", "--save", action="store_true", help="Save weights or estimations")
arg_parser.add_argument("-c", "--checkpoint", action="store_true", help="Start from checkpoint")

arg_parser.add_argument("-bf", "--base-folder", type=str, default="savings/stag_hunt/", help="Stats file name")
arg_parser.add_argument("-n", "--name", type=str, default="ppo_stag_hunt", help="Generic name")

# Environment
arg_parser.add_argument("-E", "--n-envs", type=int, default=1, help="N of parallel environments")
arg_parser.add_argument("-W", "--width", type=int, default=4, help="Width of the grid")
arg_parser.add_argument("-H", "--height", type=int, default=4, help="Height of the grid")
#arg_parser.add_argument("-N", "--n-steps", type=int, default=200e+3, help="N of steps of the training")
arg_parser.add_argument("-S", "--time-steps", type=int, default=200, help="N of time steps in each episode")
arg_parser.add_argument("-e", "--episodes", type=int, default=1000, help="N of episodes")

arg_parser.add_argument("-A", "--async-envs", action="store_true", help="Async parallelization of the environments (otherwise sync)")
arg_parser.add_argument("-R", "--common-rewards", action="store_true", help="Enables common rewards. Each agent will recieve as reward the sum of both agents' rewards")

# Stag Hunt
arg_parser.add_argument("-sf", "--stag-follows", action="store_true", help="Stag follows flag")
arg_parser.add_argument("-sr", "--stag-reward", type=float, default=5.0, help="Stag reward")
arg_parser.add_argument("-mp", "--mauling-punishment", type=float, default=-.5, help="Mauling punishment")
arg_parser.add_argument("-fr", "--forage-reward", type=float, default=1.0, help="Forage reward")
arg_parser.add_argument("-np", "--n-plants", type=int, default=2, help="N of plants")

# PPO
arg_parser.add_argument("-Prs", "--rollout-steps", type=int, default=128, help="N of time steps rollout of PPO")
arg_parser.add_argument("-Pbs", "--batch-size", type=int, default=64, help="Size of PPO mini-batches")
arg_parser.add_argument("-Pe", "--epochs", type=int, default=4, help="N of PPO epochs")
arg_parser.add_argument("-Ppc", "--clip-coef", type=float, default=0.2, help="PPO clipping coefficient")
arg_parser.add_argument("-Pl", "--lambda-gae", type=float, default=0.95, help="PPO GAE's lambda")
arg_parser.add_argument("-Pt", "--entropy-temperature", type=float, default=0.0, help="PPO temperature for entropy")
arg_parser.add_argument("-Pcc", "--critic-loss-coef", type=float, default=0.5, help="PPO critic loss coefficient")

arg_parser.add_argument("-Pna", "--normalize-advantages", action="store_true", help="Normalize PPO advantages")
arg_parser.add_argument("-Pnr", "--normalize-rewards", action="store_true", help="Normalize PPO rewards")
arg_parser.add_argument("-Plra", "--lr-annealing", action="store_true", help="Apply learning rate annealing to PPO")
arg_parser.add_argument("-Pcg", "--clip-gradient", action="store_true", help="Apply gradient clipping to PPO")
arg_parser.add_argument("-Pcgc", "--clipping-gradient-coef", type=float, default=0.5, help="PPO clipping gradient coefficient")

arg_parser.add_argument("-g", "--gamma", type=float, default=.9, help="Gamma discount")
arg_parser.add_argument("-lrp", "--lr-policy", type=float, default=3e-4, help="Policy learning rate")
arg_parser.add_argument("-lrc", "--lr-critic", type=float, default=1e-3, help="Critic learning rate")


device = "cuda" if torch.cuda.is_available() else "cpu"
args = arg_parser.parse_args()

if args.use_code_params:
    args = arg_parser.parse_args([
        # Generic
        "-s", 
        #"-c",
        
        # Env
        "-E", "1", "-W", "6", "-H", "6", "-S", "200", "-e", "10", "-A",

        # Stag Hunt
        "-sf", 
        "-sr", "5.0", "-fr", "1.0", "-mp", "-0.5",

        # PPO
        "-g", ".9", "-lrp", "5e-4", "-lrc", "1e-3",
        "--rollout-steps", "128", "--batch-size", "32", "--epochs", "6", "--clip-coef", "0.3",
        "--lambda-gae", "0.9", "--entropy-temperature", "0.1", "--critic-loss-coef", "0.5",
        #"--normalize-rewards",
        #"--lr-annealing",
        #"--ppo-clip-gradient", "--ppo-clipping-gradient-coef", "0.5"
    ])

PRINTING_TS = 100*args.time_steps
N_STEPS = args.episodes*args.time_steps
MAX_UPDATES = N_STEPS//args.rollout_steps

########################
# CONFIG
########################


ENABLE_GUI = False
PRINTING_DELAY = 1
ENABLE_GRAPHIC = False
PRINT_REWARDS = False

if args.test:
    ENABLE_GUI = True
    PRINTING_DELAY = .2
    ENABLE_GRAPHIC = True
    PRINT_REWARDS = True

    args.time_steps = 200
    args.episodes = 20
    
    

# Env config
RUN_AWAY_AFTER_MAUL = False


env_params = {
    "obs_type": "coords",           # obs_type= image or coords
    "grid_size": (args.width, args.height),
    "screen_size": (1500, 1500),
    "load_renderer": ENABLE_GUI and ENABLE_GRAPHIC,
    "max_timesteps": args.time_steps,
    "forage_quantity": args.n_plants,
    "forage_reward": args.forage_reward,
    "stag_reward": args.stag_reward,
    "mauling_punishment": args.mauling_punishment,
    "stag_follows": args.stag_follows,
    "run_away_after_maul": RUN_AWAY_AFTER_MAUL
}

state_shape = (4+2*int(args.n_plants),)
looper = StagHuntLooper(args.n_envs, env_params, parallelization="async" if args.async_envs else "sync", print_rewards=PRINT_REWARDS,
                        evaluate_during_training=False)
action_space = looper.get_action_space()
n_actions = looper.envs.action_space.nvec[0]


########################
# DQN
########################

class CriticNet(nn.Module):
    def __init__(self, input_size, width, height):
        super(CriticNet, self).__init__()
        self.layers = nn.Sequential(nn.Linear(input_size, 32), nn.ReLU(),
                                    nn.Linear(32, 32), nn.ReLU(),
                                    nn.Linear(32, 1))

    def forward(self, x):
        return self.layers(x)

class PolicyNet(nn.Module):
    def __init__(self, input_size, n_actions, width, height):
        super(PolicyNet, self).__init__()
        self.layers = nn.Sequential(nn.Linear(input_size, 32), nn.ReLU(),
                                    nn.Linear(32, 32), nn.ReLU(),
                                    nn.Linear(32, n_actions))

    def forward(self, x):
        return self.layers(x)
    

########################
# AGENT1
########################

critic_net1 = CriticNet(state_shape[0], args.width, args.height)
policy_net1 = PolicyNet(state_shape[0], n_actions, args.width, args.height)
model_wrapper1 = PPOSplittedModelWrapper(policy_net1, critic_net1, args.lr_policy, args.lr_critic, max_updates=MAX_UPDATES, lr_annealing=args.lr_annealing, device=device)

if args.test or args.checkpoint:
    model_wrapper1.load_model(f"{args.base_folder}/models/{args.name}_ag1")

agent1 = PPOAgent(model_wrapper1, args.n_envs, args.rollout_steps, state_shape, args.batch_size, 
                  clip_coef=args.clip_coef, epochs=args.epochs,gamma=args.gamma, lambda_gae=args.lambda_gae, entropy_temperature=args.entropy_temperature, 
                  critic_loss_coef=args.critic_loss_coef, normalize_advantages=args.normalize_advantages, clip_grad_norm=args.clip_gradient, max_grad_norm=args.clipping_gradient_coef,
                  device=device, test=args.test)

########################
# AGENT2
########################

critic_net2 = CriticNet(state_shape[0], args.width, args.height)
policy_net2 = PolicyNet(state_shape[0], n_actions, args.width, args.height)
model_wrapper2 = PPOSplittedModelWrapper(policy_net2, critic_net2, args.lr_policy, args.lr_critic, max_updates=MAX_UPDATES, lr_annealing=args.lr_annealing, device=device)

if args.test or args.checkpoint:
    model_wrapper2.load_model(f"{args.base_folder}/models/{args.name}_ag2")

agent2 = PPOAgent(model_wrapper2, args.n_envs, args.rollout_steps, state_shape, args.batch_size, 
                  clip_coef=args.clip_coef, epochs=args.epochs,gamma=args.gamma, lambda_gae=args.lambda_gae, entropy_temperature=args.entropy_temperature, 
                  critic_loss_coef=args.critic_loss_coef, normalize_advantages=args.normalize_advantages, clip_grad_norm=args.clip_gradient, max_grad_norm=args.clipping_gradient_coef,
                  device=device, test=args.test)

########################
# Loop
########################

looper.training_loop(agent1, agent2, N_STEPS, common_reward=args.common_rewards, enable_gui=ENABLE_GUI, printing_delay=PRINTING_DELAY, ts_to_print=PRINTING_TS)

########################
# Savings
########################

if args.save:

    agent1.stats.hyperparameters = args
    agent1.stats.episodic_rewards = looper.log_agents[0].episode_rewards[0]
    agent1.stats.save(f"{args.base_folder}/logs/{args.name}_ppo_stats.pkl")
    agent1.stats.plot_stats(fig_num=10, filename=f"{args.base_folder}/plots/ppo_stats_{args.name}_{args.stats_name}.png", 
                            episodic_rewards=looper.log_agents[0].episode_rewards[0],
                            max_grad_norm=args.max_grad_norm if args.clip_gradient else None)

    looper.log_agents[0].hyperparameters = args
    looper.log_agents[1].hyperparameters = args

    looper.log_agents[0].save(f"{args.base_folder}/logs/{args.name}_ag1.pkl")
    looper.log_agents[1].save(f"{args.base_folder}/logs/{args.name}_ag2.pkl")

    # looper.evaluation_log_agents[0].plot_stats(f"{args.base_folder}/plots/{args.name}_ag1.png", fig_n=1)
    # looper.evaluation_log_agents[1].plot_stats(f"{args.base_folder}/plots/{args.name}_ag2.png", fig_n=2)
    # looper.log_agents[0].plot_stats(f"{args.base_folder}/plots/{args.name}_training_ag1.png", fig_n=3)
    # looper.log_agents[1].plot_stats(f"{args.base_folder}/plots/{args.name}_training_ag2.png", fig_n=4)

    model_wrapper1.save_model(f"{args.base_folder}/models/{args.name}_ag1")
    model_wrapper2.save_model(f"{args.base_folder}/models/{args.name}_ag2")

looper.envs.close()