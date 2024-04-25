import torch
import gym
import numpy as np
from utils import ReplayBuffer
from mpnn_A2C import A2C
from collections import deque
import random, os
from easydict import EasyDict
import gym_environments


class ReplayBuffer(object):
    def __init__(self, max_size=int(1e6)):
        self.max_size = max_size
        self.size = 0  # buffer size

        self.buffer = deque(maxlen=self.max_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, action_dist, next_state, reward, done):
        """传入sasr数据组"""
        experience = {
            "features_critic": state,
            "action": action,
            "action_dist": action_dist,
            "next_features_critic": next_state,
            "reward": reward,
            "not_done": 1 - done,
        }
        self.buffer.append(experience)
        self.size = len(self.buffer)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return batch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

graph_topology_id = 0
listofDemands = [8, 32, 64]
n_actions = [0, 1, 2, 3]

hparams = {
    "l2": 0.1,
    #'dropout_rate': 0.05,
    "link_state_dim": 20,
    "readout_units": 35,
    "learning_rate": 0.0001,
    "T": 4,
    "num_demands": len(listofDemands),
    "batch_size": 32,
    "discount_rate": 0.99,
}

cfg = EasyDict(
    {
        "algorithm_name": "MPNN_based_A2C",
        "env_name": "GraphEnv-v1",
        "seed": 1,
        "save_model": True,
        "load_model": True,
        "train_episodes": 5e2,
        "start_step": 1e3,
        "evaluate_freq": 10,
        "max_steps": 1e4,
    }
)

if __name__ == "__main__":
    filename = f"{cfg.algorithm_name}_{cfg.env_name}_{cfg.seed}"

    if not os.path.exists("./results"):
        os.makedirs("./results")
    if cfg.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    env = gym.make(cfg.env_name)
    env_eval = gym.make(cfg.env_name)

    # set seed for all
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    env.seed(cfg.seed)
    env_eval.seed(cfg.seed)
    env.generate_environment(graph_topology_id, listofDemands)
    env_eval.generate_environment(graph_topology_id, listofDemands)

    agent = A2C(cfg, env, hparams)
    # 为agent加载参数
    if cfg.load_model:
        agent.load(filename)
        print("---------------------------------------")
        print("loaded model config successfully!")
    else:
        print("---------------------------------------")
        print("no model config files...")

    replay_buffer = ReplayBuffer()

    """开始run agent并训练，注意为off-policy训练"""
    print("---------------------------------------")
    print(
        f"Policy: {cfg.MPNN_based_A2C}, Env: {cfg.env_name}, Seed: {cfg.seed}, Device: {cfg.device}, Traning start..."
    )
    print("---------------------------------------")

    for i_ep in range(int(cfg.train_episodes)):
        print("EPISODE: ", i_ep + 1)
        state, old_demand, old_src, old_dst = env.reset()
        ep_rewards = 0

        for _ in range(int(cfg.max_steps)):
            # 通过actor得到一个分布
            action_dist, k_paths_feature_tensor = agent.get_action_dist(
                env, state, old_demand, old_src, old_dst
            )

            # 通过action分布采样一个action, 仍为tensor类型
            action = action_dist.sample()

            # 与env交互
            next_state, reward, done, new_demand, new_src, new_dst = env.make_step(
                state, action.item(), old_demand, old_src, old_dst
            )

            # 将经验数据存入replay buffer
            features_critic = agent.get_graph_features_critic(env, state)
            next_features_critic = agent.get_graph_features_critic(env, next_state)
            replay_buffer.add(
                features_critic, action, action_dist, next_features_critic, reward, done
            )

            # 更新变量
            state = next_state
            old_demand = new_demand
            old_src = new_src
            old_dst = new_dst
            ep_rewards += reward

            """一个回合结束后，直接开启下一个回合"""
            if done:
                break
            
            """收集足够数据后开始训练"""
            if replay_buffer.size > cfg.start_steps:
            