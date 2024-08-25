import numpy as np
import gym
import matplotlib.pyplot as plt
import torch
from agent.dqn_agent import ADRQN, ADRQNKAN
from agent.utils import run_episode

np.random.seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
env = gym.make('CartPole-v0').unwrapped
state_size = env.observation_space.shape[0]
n_actions = env.action_space.n
embedding_size = 8
distribution_ranges = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
n_test_episodes = 30

plt.figure(figsize=(10, 6))

for agent_type in ['adrqnkan', 'adrqn']:
    if agent_type == 'adrqn':
        agent = ADRQN(n_actions, state_size, embedding_size).to(device)
        agent.load_state_dict(torch.load('./models_adrqn/adrqn_cartpole_best.pth'))
    else:
        agent = ADRQNKAN(n_actions, state_size, embedding_size).to(device)
        agent.load_state_dict(torch.load('./models_adrqn/adrqnkan_cartpole_best.pth'))

    last_action = 0
    observation = env.reset()

    means = []
    stds = []

    for distribution_range in distribution_ranges:
        episode_rewards = []
        for i in range(n_test_episodes):
            stats = run_episode(
                env, agent, observation, last_action, 0, rendering=False, max_timesteps=200, 
                in_distribution_range=distribution_range
            )
            episode_rewards.append(stats.episode_reward)

        # save results in a dictionary and write them into a .json file
        results = dict()
        results["episode_rewards"] = episode_rewards
        results["mean"] = np.array(episode_rewards).mean()
        results["std"] = np.array(episode_rewards).std()

        means.append(results["mean"])
        stds.append(results["std"])

        env.close()
        print(f"... finished. Mean reward: {results['mean']}, std: {results['std']}")

    plt.plot(distribution_ranges, means, label=agent_type)
    plt.fill_between(distribution_ranges, np.array(means) - np.array(stds), np.array(means) + np.array(stds), alpha=0.2)

plt.xlabel("Distribution Range")
plt.xticks(distribution_ranges)
plt.ylabel("Mean Reward")
plt.grid(True)
plt.legend()
plt.savefig("cartpole_adrqn_adrqnkan.png")