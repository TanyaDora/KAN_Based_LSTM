from fastkan import FastKAN as KAN
from agent.utils import EpisodeStats, initialize_env_settings, Evaluation
from agent.networks import MLP
from agent.dqn_agent import DQNAgent
import itertools as it
import torch
import gym
import numpy as np
import os
import sys
import json
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


def run_episode(
    env, agent, deterministic, do_training=True, rendering=False, max_timesteps=200, in_distribution_range=0.2, out_distribution_range=0
):
    """
    This methods runs one episode for a gym environment.
    deterministic == True => agent executes only greedy actions according the Q function approximator (no random actions).
    do_training == True => train agent
    """

    stats = EpisodeStats()
    initialize_env_settings(env, percentage_range=in_distribution_range, out_of_distribution_percentage_range=out_distribution_range)
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
    num_episodes=1500,
    local_evaluation_interval=150,
    eval_cycle=20,
    num_eval_episodes=5,
    model_dir="./models_cartpole_changingenv",
    model_config=None,
    in_distribution_range=0.2,
    out_distribution_range=0,
):
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    model_dir = os.path.join(model_dir, name)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    # save config to folder as json
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(model_config, f)

    print("... train agent")

    tensorboard = Evaluation(
        os.path.join(model_dir, "tensorboard"),
        "Reinforcement Learning",
        ["reward", "a_0", "a_1", "avg_reward"],
    )

    # training
    best_eval_reward = 0
    best_eval_reward_local = 0
    local_model_name = f'best_model_local_0.pt'
    for i in range(num_episodes):
        print("episode: ", i)
        stats = run_episode(env, agent, deterministic=False, do_training=True, rendering=False, in_distribution_range=in_distribution_range, out_distribution_range=out_distribution_range)
        tensorboard.write_episode_data(
            i,
            eval_dict={
                "reward": stats.episode_reward,
                "a_0": stats.get_action_usage(0),
                "a_1": stats.get_action_usage(1),
            },
        )

        if i % local_evaluation_interval == 0:
            local_model_name = f'best_model_local_{i}.pt'
            best_eval_reward_local = 0

        if i % eval_cycle == 0:
            reward = 0
            avg_eval_reward = 0.0
            for j in range(num_eval_episodes):
                stats = run_episode(env, agent, deterministic=True, do_training=False, rendering=False)
                reward += stats.episode_reward
            avg_eval_reward = reward / num_eval_episodes
            tensorboard.write_episode_data(i, eval_dict={"avg_reward": avg_eval_reward})
            if avg_eval_reward >= best_eval_reward:
                best_eval_reward = avg_eval_reward
                agent.save(os.path.join(model_dir, f"best_model.pt"))
            if avg_eval_reward > best_eval_reward_local:
                best_eval_reward_local = avg_eval_reward
                agent.save(os.path.join(model_dir, local_model_name))

    if avg_eval_reward >= best_eval_reward:
        best_eval_reward = avg_eval_reward
        agent.save(os.path.join(model_dir, f"best_model.pt"))

    tensorboard.close_session()


if __name__ == "__main__":

    num_eval_episodes = 5 
    eval_cycle = 20 

    # You find information about cartpole in
    # https://github.com/openai/gym/wiki/CartPole-v0
    # Hint: CartPole is considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.

    env = gym.make("CartPole-v0").unwrapped

    state_dim = 4
    num_actions = 2

    model_configs = [
                    # ['mlp', [40], 'mlp_40', 0.2, 0],
                    # ['kan', [state_dim, 9, num_actions], 'kan_9', 0.2, 0],
                    # ['mlp', [40, 40], 'mlp_40_40', 0.2, 0], 
                    # ['kan', [state_dim, 16, 16, num_actions], 'kan_16_16', 0.2, 0],
                    ['mlp', [40, 40, 40], 'mlp_40_40_40', 0.2, 0],
                    # ['kan', [state_dim, 17, 17, 17, num_actions], 'kan_17_17_17', 0.2, 0]
                    ]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    for model_config in model_configs:
        if model_config[0] == 'kan':
            Q_network = KAN(model_config[1], num_grids=5).to(device)
            Q_target = KAN(model_config[1], num_grids=5).to(device)
        elif model_config[0] == 'mlp':
            Q_network = MLP(state_dim=state_dim, action_dim=num_actions, hidden_dims=model_config[1]).to(device)
            Q_target = MLP(state_dim=state_dim, action_dim=num_actions, hidden_dims=model_config[1]).to(device)
        else:
            raise ValueError("Model not found")

        dqn = DQNAgent(Q_network, Q_target, num_actions, device=device, capacity=500)
        train_online(env=env, 
                     agent=dqn, 
                     name=model_config[2], 
                     num_episodes=10000, 
                     num_eval_episodes=5, 
                     eval_cycle=20, 
                     local_evaluation_interval=500,
                     model_config=model_config,
                     in_distribution_range=model_config[3],
                     out_distribution_range=model_config[4])

