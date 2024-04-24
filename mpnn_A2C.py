import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch_scatter import scatter_add

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_weight(m):
    if isinstance(m, (nn.Linear, nn.GRUCell)):
        nn.init.normal_(m.weight, std=0.1)


class Actor(nn.Module):
    """
    通过MPNN架构实现actor
    注意这里不默认有batch_size，或者说batch_size与edges数量绑定
    """

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.link_state_dim = self.hparams["link_state_dim"]
        self.readout_units = self.hparams["readout_units"]
        self.T = self.hparams["T"]

        """MPNN结构网络"""
        # message function m(·)，函数输入为两个edges的pair
        # 且M_k = sum_{i} m(h_k, h_i)为边k的Message
        # 由于需要学习网络的拓扑邻接关系，输入包含mainEdge和neighbours的link_state
        self.Message = nn.Sequential(
            nn.Linear(self.link_state_dim * 2, self.link_state_dim),
            nn.SELU(),
        )

        # update function u(·), 输入为 边k的h_k和M_k
        # new_h_k = u(h_k, M_k)
        self.Update = nn.GRUCell(self.link_state_dim, self.link_state_dim)

        # readout function r(·)
        self.Readout = nn.Sequential(
            nn.Linear(self.link_state_dim, self.readout_units),
            nn.SELU(),
            nn.Linear(self.readout_units, self.readout_units),
            nn.SELU(),
            nn.Linear(self.readout_units, 1),
            nn.Softmax(),
        )

        self.Message.apply(init_weight)
        self.Update.apply(init_weight)
        self.Readout.apply(init_weight)

    def forward(
        self,
        links_state,
        K,
        id_mainEdges,
        id_neighbourEdges,
        num_edges,
        is_train=False,
    ):
        """
        输出为k条可选path的分布
        """

        # 循环执行T次
        for _ in range(self.T):
            # 将mainEdge和neighbour的state结合起来
            states_mainEdges = links_state[id_mainEdges]
            states_neighbourEdges = links_state[id_neighbourEdges]

            # 让MPNN学习网络的邻接关系
            states_concatEdges = torch.cat(
                [states_mainEdges, states_neighbourEdges], dim=1
            )

            """1.a 对每条边，与其neighbours做 message passing"""
            m = self.Message(states_concatEdges)

            """1.b 计算每条边m的加和，通过neighbours中的id实现"""
            # 此时links_M.shape=(num_edges, link_state_dim)
            links_M = scatter_add(src=m, index=id_neighbourEdges, dim_size=num_edges)

            """为每条链路更新state"""
            outputs, links_state_list = self.Update(links_M, [links_state])
            links_state = links_state_list[0]

        K_paths_ids = []
        for i in range(K):
            K_paths_ids += [i] * num_edges
        K_paths_ids = torch.tensor(K_paths_ids).to(device)

        # 迭代更新完state后 根据k条paths做加和
        k_path_outputs = scatter_add(links_state, K_paths_ids, dim=0)

        # 送入readout函数，求出k条paths对应的值作为分布
        r = self.Readout(k_path_outputs)
        # 建立一个概率分布，方便后续采样action
        dist = Categorical(r)
        return dist


class Critic(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.link_state_dim = self.hparams["link_state_dim"]
        self.readout_units = self.hparams["readout_units"]
        self.T = self.hparams["T"]

        """MPNN结构网络"""
        # message function m(·)，函数输入为两个edges的pair
        # 且M_k = sum_{i} m(h_k, h_i)为边k的Message
        # 由于需要学习网络的拓扑邻接关系，输入包含mainEdge和neighbours的link_state
        self.Message = nn.Sequential(
            nn.Linear(self.link_state_dim * 2, self.link_state_dim),
            nn.SELU(),
        )

        # update function u(·), 输入为 边k的h_k和M_k
        # new_h_k = u(h_k, M_k)
        self.Update = nn.GRUCell(self.link_state_dim, self.link_state_dim)

        # readout function r(·)
        self.Readout = nn.Sequential(
            nn.Linear(self.link_state_dim, self.readout_units),
            nn.SELU(),
            nn.Linear(self.readout_units, self.readout_units),
            nn.SELU(),
            nn.Linear(self.readout_units, 1),
            nn.Softmax(),
        )

        self.Message.apply(init_weight)
        self.Update.apply(init_weight)
        self.Readout.apply(init_weight)

    def forward(
        self,
        links_state,
        id_mainEdges,
        id_neighbourEdges,
        num_edges,
        is_train=False,
    ):
        """
        输出对应状态的state value
        """

        # 循环执行T次
        for _ in range(self.T):
            # 将mainEdge和neighbour的state结合起来
            states_mainEdges = links_state[id_mainEdges]
            states_neighbourEdges = links_state[id_neighbourEdges]

            # 让MPNN学习网络的邻接关系
            states_concatEdges = torch.cat(
                [states_mainEdges, states_neighbourEdges], dim=1
            )

            """1.a 对每条边，与其neighbours做 message passing"""
            m = self.Message(states_concatEdges)

            """1.b 计算每条边m的加和，通过neighbours中的id实现"""
            # 此时links_M.shape=(num_edges, link_state_dim)
            links_M = scatter_add(src=m, index=id_neighbourEdges, dim_size=num_edges)

            """为每条链路更新state"""
            outputs, links_state_list = self.Update(links_M, [links_state])
            links_state = links_state_list[0]

        # 迭代更新完state后 将所有的边做加和
        total_state = torch.sum(links_state, dim=0)

        # 送入readout函数，求出k条paths对应的值作为分布
        r = self.Readout(total_state)

        return r


class A2C:
    def __init__(self, cfg, env, hparams):
        self.hparams = hparams
        self.link_state_dim = self.hparams["link_state_dim"]
        self.num_demands = self.hparams["num_demands"]

        self.capacity_feature = None
        self.bw_allocated_feature = np.zeros((env.numEdges, len(env.listofDemands)))

        self.actor = Actor(self.hparams)
        self.critic = Critic(self.hparams)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.hparams.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.hparams.lr)

    def get_graph_features_actor(self, env, links_state):
        self.bw_allocated_feature.fill(0.0)
        # 将capacity归一化到(-0.5, 0.5)
        self.capacity_feature = (links_state[:, 0] - 100.0) / 200.0

        # 遍历处理bandwidth_allocated
        idx = 0
        for i in links_state[:, 1]:
            # 假设为离散型带宽需求，用one-hot表示
            if i == 8:
                self.bw_allocated_feature[idx][0] = 1
            elif i == 32:
                self.bw_allocated_feature[idx][1] = 1
            elif i == 64:
                self.bw_allocated_feature[idx][2] = 1
            idx += 1

        num_edges = env.numEdges
        length = env.firstTrueSize
        # 将ndArray转为tensor，注意to(device)
        betweenness = torch.tensor(env.between_feature, dtype=torch.float32).to(device)
        bw_allocated = torch.tensor(self.bw_allocated_feature, dtype=torch.float32).to(
            device
        )
        capacities = torch.tensor(self.capacity_feature, dtype=torch.float32).to(device)
        main_edges_id = torch.tensor(env.first, dtype=torch.int32).to(device)
        neighbour_edges_id = torch.tensor(env.second, dtype=torch.int32).to(device)

        # 规范它们的shape
        capacities = torch.reshape(capacities[0:num_edges], (num_edges, 1))
        betweenness = torch.reshape(betweenness[0:num_edges], (num_edges, 1))

        # 拼接hidden_state，按列拼接
        hidden_states = torch.cat([capacities, betweenness, bw_allocated], dim=1)
        paddings = torch.zeros(num_edges, self.link_state_dim - 2 - self.num_demands)

        # 将state进行扩充，以储存MPNN中更多其他信息
        links_state = torch.cat([hidden_states, paddings], dim=1)

        features_actor = {
            "links_state": links_state,
            "main_edges_id": main_edges_id,
            "neighbour_edges_id": neighbour_edges_id,
            "num_edges": num_edges,
        }
        return features_actor

    def select_action(self, env, state, bw_demand, src, dst):
        listGraphs = []
        k_paths_features = []

        # 初始化一个action
        action = 0
        # 根据(src, dst)pair获取对应的K-paths
        pathList = env.allPaths[str(src) + ":" + str(dst)]
        path_id = 0

        # 通过k条可选paths分配需求 (src, dst, bw_demand)
        while path_id < len(pathList):
            state_copy = np.copy(state)
            currentPath = pathList[path_id]
            i = 0
            j = 0

            # 对所选path进行逐edge遍历, 并将demand的带宽进行分配
            while j < len(currentPath):
                currentEdgeId = env.edgesDict[
                    str(currentPath[i]) + ":" + str(currentPath[j])
                ]
                state_copy[currentEdgeId][1] = bw_demand
                i += 1
                j += 1

            # 将分配demand后的link_state储存
            listGraphs.append(state_copy)
            # 注意返回的features中多维数据都是tensor
            features = self.get_graph_features_actor(env, state_copy)
            # 将每条可选path作用后对应的features储存
            k_paths_features.append(features)
            path_id += 1

        # 注意：
