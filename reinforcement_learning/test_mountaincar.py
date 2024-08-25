import os
from datetime import datetime
import gym
import json
import torch
from agent.dqn_agent import DQNAgent
from train_mountaincar import run_episode
from agent.networks import *
import numpy as np
from kan import *

np.random.seed(0)

if __name__ == "__main__":

    env = gym.make("MountainCar-v0").unwrapped
    # env.length = 1.0
    env.gravity = 0.01
    env.reset()

    # TODO: load DQN agent
    # ...

    state_dim = 2
    num_actions = 3

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # Q_network = MLP(state_dim=2, action_dim=3).to(device)
    # Q_target = MLP(state_dim=2, action_dim=3).to(device)
    # agent = DQNAgent(Q_network, Q_target, num_actions=3, device=device)
    # agent.load("./models_mountaincar/mlp.pt")

    Q_network = KAN(width=[state_dim, 10, num_actions], grid=5, k=3, seed=0).to(device)
    Q_target = KAN(width=[state_dim, 10, num_actions], grid=5, k=3, seed=0).to(device)
    agent = DQNAgent(Q_network, Q_target, num_actions=4, device=device)
    agent.load("./models_mountaincar/kan10.pt")

    # Q_network = KAN(width=[state_dim, 25, num_actions], grid=5, k=3, seed=0).to(device)
    # Q_target = KAN(width=[state_dim, 25, num_actions], grid=5, k=3, seed=0).to(device)
    # agent = DQNAgent(Q_network, Q_target, num_actions=4, device=device)
    # agent.load("./models_mountaincar/kan25.pt")

    n_test_episodes = 15

    episode_rewards = []
    for i in range(n_test_episodes):
        stats = run_episode(
            env, agent, deterministic=True, do_training=False, rendering=True
        )
        episode_rewards.append(stats.episode_reward)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()

    if not os.path.exists("./results_mountaincar"):
        os.mkdir("./results_mountaincar")

    fname = "./results_mountaincar/mountaincar_results_dqn-%s.json" % datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )
    fh = open(fname, "w")
    json.dump(results, fh)

    env.close()
    print("... finished")
