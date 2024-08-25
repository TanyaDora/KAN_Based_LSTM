from collections import namedtuple, deque
import numpy as np
import os
import gzip
import pickle
import torch


class ReplayBuffer:

    # TODO: implement a capacity for the replay buffer (FIFO, capacity: 1e5 - 1e6)

    # Replay buffer for experience replay. Stores transitions.
    def __init__(self, capacity=100000):
        self.capacity = capacity

        self._data = namedtuple(
            "ReplayBuffer", ["states", "actions", "next_states", "rewards", "dones"]
        )
        self._data = self._data(
            states=[], actions=[], next_states=[], rewards=[], dones=[]
        )

    def add_transition(self, state, action, next_state, reward, done):
        """
        This method adds a transition to the replay buffer.
        """
        if len(self._data.states) >= self.capacity:
            self._data = self._data._replace(
                states=self._data.states[1:],
                actions=self._data.actions[1:],
                next_states=self._data.next_states[1:],
                rewards=self._data.rewards[1:],
                dones=self._data.dones[1:],
            )
        self._data.states.append(state)
        self._data.actions.append(action)
        self._data.next_states.append(next_state)
        self._data.rewards.append(reward)
        self._data.dones.append(done)

    def next_batch(self, batch_size):
        """
        This method samples a batch of transitions.
        """
        batch_indices = np.random.choice(len(self._data.states), batch_size)
        batch_states = np.array([self._data.states[i] for i in batch_indices])
        batch_actions = np.array([self._data.actions[i] for i in batch_indices])
        batch_next_states = np.array([self._data.next_states[i] for i in batch_indices])
        batch_rewards = np.array([self._data.rewards[i] for i in batch_indices])
        batch_dones = np.array([self._data.dones[i] for i in batch_indices])
        return (
            batch_states,
            batch_actions,
            batch_next_states,
            batch_rewards,
            batch_dones,
        )


class ExpBuffer():
    def __init__(self, max_storage, sample_length):
        self.max_storage = max_storage
        self.sample_length = sample_length
        self.counter = -1
        self.filled = -1
        self.storage = [0 for i in range(max_storage)]

    def write_tuple(self, aoarod):
        if self.counter < self.max_storage-1:
            self.counter +=1
        if self.filled < self.max_storage:
            self.filled += 1
        else:
            self.counter = 0
        self.storage[self.counter] = aoarod
    
    def sample(self, batch_size, device = None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #Returns sizes of (batch_size, seq_len, *) depending on action/observation/return/done
        seq_len = self.sample_length
        last_actions = []
        last_observations = []
        actions = []
        rewards = []
        observations = []
        dones = []

        for i in range(batch_size):
            if self.filled - seq_len < 0 :
                raise Exception("Reduce seq_len or increase exploration at start.")
            start_idx = np.random.randint(self.filled-seq_len)
            #print(self.filled)
            #print(start_idx)
            last_act, last_obs, act, rew, obs, done = zip(*self.storage[start_idx:start_idx+seq_len])
            last_actions.append(list(last_act))
            last_observations.append(last_obs)
            actions.append(list(act))
            rewards.append(list(rew))
            observations.append(list(obs))
            dones.append(list(done))
           
        return torch.tensor(last_actions).to(device), torch.tensor(last_observations, dtype = torch.float32).to(device), torch.tensor(actions).to(device), torch.tensor(rewards).float().to(device) , torch.tensor(observations, dtype = torch.float32).to(device), torch.tensor(dones).to(device)