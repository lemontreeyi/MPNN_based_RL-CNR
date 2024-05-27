import torch
import gym
import numpy as np
from utils import ReplayBuffer
from mpnn_A2C import A2C
from collections import deque
import random, os
from easydict import EasyDict
from torch.utils.tensorboard import SummaryWriter
import gym_environments

cfg = EasyDict()
cfg.algorithm_name = "MPNN_based_A2C"
cfg.env_name = "GraphEnv-v1"
cfg.seed = 1
cfg.save_model = True
cfg.load_model = False
cfg.train_episodes = int(2e2)
cfg.evaluate_episodes = 5
cfg.start_size = int(3e2)
cfg.max_steps = int(1e4)
cfg.evaluate_freq = 10

# listofDemands = [4, 8, 32, 48]
listofDemands = [8, 32, 64]
env = gym.make(cfg.env_name)
env.seed(cfg.seed)
env.generate_environment(1, listofDemands)

hparams = {
    "l2": 0.1,
    #'dropout_rate': 0.05,
    "link_state_dim": 20,
    "readout_units": 35,
    "lr": 0.001,
    "T": 4,
    "num_demands": len(listofDemands),
    "batch_size": 32,
    "discount_rate": 0.95,
}

agent = A2C(cfg, env, hparams)
filename = f"{cfg.algorithm_name}_{cfg.env_name}_{cfg.seed}"
agent.load(filename)

evaluate_rewards_list = [0] * 50
rest_capacity_list = [0] * 50
bw_utilization_list = [0] * 50

for ep in range(50):
    state, old_demand, old_src, old_dst = env.reset()
    done = False
    while True:
        action_dist, _ = agent.get_action_dist(
            env, state, old_demand, old_src, old_dst
        )
        action = torch.argmax(action_dist.probs)
        next_state, reward, done, new_demand, new_src, new_dst = (
            env.make_step(
                state, action.item(), old_demand, old_src, old_dst
            )
        )
        state = next_state
        old_demand = new_demand
        old_src = new_src
        old_dst = new_dst
        evaluate_rewards_list[ep] += reward
        if done:
            break
    for i in range(env.numEdges):
        rest_capacity_list[ep] += env.graph_state[i][0]
    bw_utilization_list[ep] = 1 - rest_capacity_list[ep] / (env.numEdges * 200.0)
    agent.evaluate_return_array = np.append(agent.evaluate_return_array, evaluate_rewards_list[ep])
    agent.bw_utilization_array = np.append(agent.bw_utilization_array, bw_utilization_list[ep])

np.savez('results/geant2_test.npz',
             evaluate_return_array=agent.evaluate_return_array,
             bw_utilization_array = agent.bw_utilization_array
             )
