import numpy as np
import random
import matplotlib.pyplot as plt
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn.functional as F


LEFT = 1
RIGHT = 2
STRAIGHT = 0
ACCELERATE = 3
BRAKE = 4


class Evaluation:

    def __init__(self, store_dir, name, stats=[]):
        """
        Creates placeholders for the statistics listed in stats to generate tensorboard summaries.
        e.g. stats = ["loss"]
        """
        self.folder_id = "%s-%s" % (name, datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.summary_writer = SummaryWriter(os.path.join(store_dir, self.folder_id))
        self.stats = stats

    def write_episode_data(self, episode, eval_dict):
        """
        Write episode statistics in eval_dict to tensorboard, make sure that the entries in eval_dict are specified in stats.
        e.g. eval_dict = {"loss" : 1e-4}
        """

        for k in eval_dict:
            assert k in self.stats
            self.summary_writer.add_scalar(k, eval_dict[k], global_step=episode)

        self.summary_writer.flush()

    def close_session(self):
        self.summary_writer.close()


def sort_models(models):
    def model_sort_key(model_name):
        if model_name == 'best_model.pt':
            return float('inf')  # Ensure 'best_model.pt' is at the end
        else:
            return int(model_name.split('_')[-1].split('.')[0])

    return sorted(models, key=model_sort_key)


def plot_checkpoint_performance(data, dist_type, out_folder, model_types=None, name=""):
    plt.figure(figsize=(10, 6))
    
    if model_types is None:
        model_types = list(data.keys())

    for model_type in model_types:
        models = data[model_type]
        model_names = sort_models(list(models.keys()))
        means = [models[model][dist_type]["mean"] for model in model_names]
        stds = [models[model][dist_type]["std"] for model in model_names]
        # filter name_{idx}.pt -> idx for all model names
        model_names = [model.split('_')[-1].split('.')[0] for model in model_names]
        model_names[-1] = 'best'
        temp = model_names[:-1]
        temp = [str(int(model)+500) for model in temp]
        model_names = temp + [model_names[-1]]
        plt.errorbar(model_names, means, yerr=stds, marker='.', label=model_type, capsize=2)
    
    plt.xlabel(f'Best Models for each {model_names[1]} epochs')
    plt.ylabel('Mean Episode Reward')
    plt.title(f'Mean Episode Reward ({dist_type.replace("_", " ").title()})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if name != "":
        name = f"_{name}"

    plt.savefig(f"{out_folder}/mean_episode_reward_{dist_type}{name}.png")


def plot_model_performance(data, out_folder, checkpoint=None, model_types=None, name=""):
    plt.figure(figsize=(10, 6))
    
    if checkpoint is None:
        checkpoint = 'best_model.pt'

    if model_types is None:
        model_types = list(data.keys())

    for model_type in model_types:
        dist_types = data[model_type][checkpoint].keys()
        dist_types = sorted([int(dist_type) for dist_type in dist_types])
        dist_types_x = np.array(dist_types) / 10
        dist_types = [str(dist_type) for dist_type in dist_types]

        means = [data[model_type][checkpoint][dist_type]["mean"] for dist_type in dist_types]
        stds = [data[model_type][checkpoint][dist_type]["std"] for dist_type in dist_types]

        plt.plot(dist_types_x, means, label=model_type)
        plt.fill_between(dist_types_x, np.array(means) - np.array(stds), np.array(means) + np.array(stds), alpha=0.2)
    
    plt.xlabel(f'Distribution Range')
    plt.ylabel('Mean Reward')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if name != "":
        name = f"_{name}"

    plt.savefig(f"{out_folder}/model_performance{name}.png")


def initialize_env_settings(env, percentage_range=0.2, out_of_distribution_percentage_range=0):
    default_gravity = 9.8
    default_masscart = 1.0
    default_masspole = 0.1
    default_total_mass = default_masspole + default_masscart
    default_length = 0.5  
    default_polemass_length = default_masspole * default_length
    default_force_mag = 10.0
    default_tau = 0.02 

    uniform_range = (1-percentage_range, 1+percentage_range)

    if out_of_distribution_percentage_range != 0:
        coin_toss = random.randint(0, 1)
        if percentage_range + out_of_distribution_percentage_range > 1:
            # only allow for a maximum of 100% change and change upwards when its more
            coin_toss = 0
        if coin_toss == 1:
            uniform_range = (1-percentage_range-out_of_distribution_percentage_range, 1-percentage_range)
        else:
            uniform_range = (1+percentage_range, 1+percentage_range+out_of_distribution_percentage_range)

    env.gravity = default_gravity * random.uniform(*uniform_range)
    env.masscart = default_masscart * random.uniform(*uniform_range)
    env.masspole = default_masspole * random.uniform(*uniform_range)
    env.total_mass = default_total_mass * random.uniform(*uniform_range)
    env.length = default_length * random.uniform(*uniform_range)
    env.pole_mass_length = default_polemass_length * random.uniform(*uniform_range)
    env.force_mag = default_force_mag * random.uniform(*uniform_range)
    env.tau = default_tau * random.uniform(*uniform_range)


class EpisodeStats:
    """
    This class tracks statistics like episode reward or action usage.
    """

    def __init__(self):
        self.episode_reward = 0
        self.actions_ids = []

    def step(self, reward, action_id):
        self.episode_reward += reward
        self.actions_ids.append(action_id)

    def get_action_usage(self, action_id):
        ids = np.array(self.actions_ids)
        return len(ids[ids == action_id]) / len(ids)
    

def run_episode(env, 
                agent, 
                observation, 
                last_action, 
                epsilon, 
                rendering=False, 
                max_timesteps=200, 
                in_distribution_range=0, 
                out_distribution_range=0, 
                device=None):
    """
    This methods runs one episode for a gym environment.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stats = EpisodeStats()
    if out_distribution_range != 0 or in_distribution_range != 0:
        initialize_env_settings(env, percentage_range=in_distribution_range, out_of_distribution_percentage_range=out_distribution_range)
    observation = env.reset()

    step = 0
    while True:
        action_id, hidden = agent.act(torch.tensor(observation).float().view(1,1,-1).to(device),F.one_hot(torch.tensor(last_action), env.action_space.n).view(1,1,-1).float().to(device), hidden = None, epsilon = epsilon)
        next_state, reward, terminal, info = env.step(action_id)
        stats.step(reward, action_id)

        observation = next_state
        last_action = action_id

        if rendering:
            env.render()

        if terminal or step > max_timesteps:
            break

        step += 1

    return stats