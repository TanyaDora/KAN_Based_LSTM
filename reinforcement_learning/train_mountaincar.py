import sys

sys.path.append("../")

import os
import numpy as np
import gym
import torch
import itertools as it
from agent.dqn_agent import DQNAgent
from agent.networks import MLP
from agent.utils import EpisodeStats
from tensorboard_evaluation import Evaluation
from kan import *
import random

def initialize_env_settings(env):
    # Default values
    default_min_position = -1.2
    default_max_position = 0.6
    default_max_speed = 0.07
    default_goal_position = 0.5
    default_force = 0.001
    default_gravity = 0.0025

    # Sample range ±10% around default values
    env.min_position = default_min_position * random.uniform(0.8, 1.2)
    env.max_position = default_max_position * random.uniform(0.8, 1.2)
    env.max_speed = default_max_speed * random.uniform(0.8, 1.2)
    env.goal_position = default_goal_position * random.uniform(0.8, 1.2)
    env.force = default_force * random.uniform(0.8, 1.2)
    env.gravity = default_gravity * random.uniform(0.8, 1.2)

def run_episode(
    env, agent, deterministic, do_training=True, rendering=False, max_timesteps=200
):
    """
    This methods runs one episode for a gym environment.
    deterministic == True => agent executes only greedy actions according the Q function approximator (no random actions).
    do_training == True => train agent
    """

    stats = EpisodeStats()  # save statistics like episode reward or action usage
    state = env.reset()

    step = 0
    while True:

        action_id = agent.act(state=state, deterministic=deterministic)
        next_state, reward, terminal, info = env.step(action_id)

        if do_training:
            agent.train(state, action_id, next_state, reward, terminal)

        stats.step(reward, action_id)

        state = next_state

        if rendering:
            env.render()

        if terminal or step > max_timesteps:
            break

        step += 1

    return stats

def train_online(
    env,
    agent,
    name,
    num_episodes=1000,
    model_dir="./models_mountaincar",
    tensorboard_dir="./tensorboard",
):
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    print("... train agent")

    tensorboard = Evaluation(
        os.path.join(tensorboard_dir, "train"),
        "Reinforcement Learning",
        ["reward", "left", "no acceleration", "right", "avg_reward"],
    )

    # training
    best_eval_reward = -np.inf
    for i in range(num_episodes):
        print("episode: ", i)
        stats = run_episode(
            env, agent, deterministic=False, do_training=True, rendering=False
        )
        tensorboard.write_episode_data(
            i,
            eval_dict={
                "reward": stats.episode_reward,
                "left": stats.get_action_usage(0),
                "no acceleration": stats.get_action_usage(1),
                "right": stats.get_action_usage(2)          
            },
        )

        # TODO: evaluate your agent every 'eval_cycle' episodes using run_episode(env, agent, deterministic=True, do_training=False) to
        # check its performance with greedy actions only. You can also use tensorboard to plot the mean episode reward.
        # ...
        # if i % eval_cycle == 0:
        #    for j in range(num_eval_episodes):
        #       ...

        if i % eval_cycle == 0:
            reward = 0
            avg_eval_reward = 0.0
            for j in range(num_eval_episodes):
                stats = run_episode(
                    env, agent, deterministic=True, do_training=False, rendering=False
                )
                reward += stats.episode_reward
            avg_eval_reward = reward / num_eval_episodes
            tensorboard.write_episode_data(
                i,
                eval_dict={
                    "avg_reward": avg_eval_reward,
                },
            )
            if avg_eval_reward > best_eval_reward:
                best_eval_reward = avg_eval_reward
                agent.save(os.path.join(model_dir, f"{name}.pt"))

    if avg_eval_reward > best_eval_reward:
        best_eval_reward = avg_eval_reward
        agent.save(os.path.join(model_dir, f"{name}.pt"))

    tensorboard.close_session()

if __name__ == "__main__":
    
    num_eval_episodes = 5  # evaluate on 5 episodes
    eval_cycle = 20  # evaluate every 10 episodes

    # You find information about cartpole in
    # https://github.com/openai/gym/wiki/CartPole-v0
    # Hint: CartPole is considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.

    env = gym.make("MountainCar-v0").unwrapped
    name = sys.argv[1]

    state_dim = 2
    num_actions = 3

    # TODO:
    # 1. init Q network and target network (see dqn/networks.py)
    # 2. init DQNAgent (see dqn/dqn_agent.py)
    # 3. train DQN agent with train_online(...)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # Q_network = MLP(state_dim, num_actions).to(device)
    # Q_target = MLP(state_dim, num_actions).to(device)

    # Q_network = KAN(width=[state_dim, 25, num_actions], grid=5, k=3, seed=0).to(device)
    # Q_target = KAN(width=[state_dim, 25, num_actions], grid=5, k=3, seed=0).to(device)

    Q_network = KAN(width=[state_dim, 10, num_actions], grid=5, k=3, seed=0).to(device)
    Q_target = KAN(width=[state_dim, 10, num_actions], grid=5, k=3, seed=0).to(device)

    agent = DQNAgent(
        Q_network, Q_target, num_actions=3, device=device, capacity=500
    )
    train_online(env=env, agent=agent, name=name)

