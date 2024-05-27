import torch
import gym
import numpy as np
import gym_environments
import random

env = gym.make("GraphEnv-v1")
env.seed(1)
env.generate_environment(0,[4, 8, 32, 48])

random_score_array = np.array([])
random_utilization_array = np.array([])

for ep in range(50):
    state, old_demand, old_src, old_dst = env.reset()
    done = False
    rewards = 0
    while True:
        action = random.choice([0,1,2,3])
        next_state, reward, done, new_demand, new_src, new_dst = (
            env.make_step(
                state, action, old_demand, old_src, old_dst
            )
        )
        state = next_state
        old_demand = new_demand
        old_src = new_src
        old_dst = new_dst
        rewards += reward
        if done:
            break
    rest = 0
    for i in range(env.numEdges):
        rest += env.graph_state[i][0]
    utilization = 1 - rest / (env.numEdges * 200)
    random_score_array = np.append(random_score_array, rewards)
    random_utilization_array = np.append(random_utilization_array, utilization)

np.savez("results/random_results_geant2_1.npz", random_score_array=random_score_array, random_utilization_array=random_utilization_array)