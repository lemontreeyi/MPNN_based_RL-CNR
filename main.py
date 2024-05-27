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


class ReplayBuffer(object):
    def __init__(self, max_size=int(1e6)):
        self.max_size = max_size
        self.size = 0  # buffer size

        self.buffer = deque(maxlen=self.max_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, actor_input, state, action, next_state, reward, done):
        """传入sasr数据组"""
        experience = {
            "actor_input": actor_input,
            "features_critic": state,
            "action": action,
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
# listofDemands = [2, 4, 8, 16, 32]

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
cfg = EasyDict()
cfg.algorithm_name = "MPNN_based_A2C"
cfg.env_name = "GraphEnv-v1"
cfg.seed = 2
cfg.save_model = True
cfg.load_model = False
cfg.train_episodes = int(2e2)
cfg.evaluate_episodes = 5
cfg.start_size = int(3e2)
cfg.max_steps = int(1e4)
cfg.evaluate_freq = 10


if __name__ == "__main__":
    filename = f"{cfg.algorithm_name}_{cfg.env_name}_{cfg.seed}"

    if not os.path.exists("./results"):
        os.makedirs("./results")
    if cfg.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    print("----------------initialize env & agent begin...------------------")
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
        f"Policy: {cfg.algorithm_name}, Env: {cfg.env_name}, Seed: {cfg.seed}, Device: {device}, Traning start..."
    )
    print("---------------------------------------")

    for i_ep in range(cfg.train_episodes):
        state, old_demand, old_src, old_dst = env.reset()
        ep_rewards = 0

        for _ in range(cfg.max_steps):
            # 通过actor得到一个分布
            action_dist, actor_input = agent.get_action_dist(
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
                actor_input, features_critic, action, next_features_critic, reward, done
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
            if replay_buffer.size > cfg.start_size:
                agent.train(env, replay_buffer)

        # if replay_buffer.size > cfg.start_size and (i_ep + 1) % cfg.evaluate_freq == 0:
        if replay_buffer.size >= cfg.start_size:
            evaluate_rewards_list = [0] * cfg.evaluate_episodes
            rest_capacity_list = [0] * cfg.evaluate_episodes
            # 对目前的actor进行评估
            for ep in range(cfg.evaluate_episodes):
                state, old_demand, old_src, old_dst = env_eval.reset()
                done = False
                while True:
                    action_dist, _ = agent.get_action_dist(
                        env_eval, state, old_demand, old_src, old_dst
                    )
                    #  取概率最大的动作而不是按照概率分布采样
                    action = torch.argmax(action_dist.probs)
                    next_state, reward, done, new_demand, new_src, new_dst = (
                        env_eval.make_step(
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
                rest_capacity_list[ep] = 0
                for i in range(env_eval.numEdges):
                    rest_capacity_list[ep] += env_eval.graph_state[i][0]
            evaluate_return = sum(evaluate_rewards_list) / len(evaluate_rewards_list)
            rest_capacity = sum(rest_capacity_list) / len(rest_capacity_list)
            bw_utilization = 1 - rest_capacity / (env_eval.numEdges * 200.0)
            # 存储过程数据
            agent.evaluate_return_array = np.append(agent.evaluate_return_array, evaluate_return)
            agent.bw_utilization_array = np.append(agent.bw_utilization_array, bw_utilization)
            print(
                f"Evaluation over {cfg.evaluate_episodes}, average return: {evaluate_return:.3f}"
            )
            if cfg.save_model:
                agent.save(filename)
        print(
            f"training episode: no.{i_ep + 1}, episode_return: {ep_rewards}, buffer_size: {replay_buffer.size}"
        )

    np.savez('results/data_lr2.npz',
             evaluate_return_array=agent.evaluate_return_array,
             bw_utilization_array = agent.bw_utilization_array,
             actor_loss_array = agent.actor_loss_array,
             critic_loss_array = agent.critic_loss_array,
             )