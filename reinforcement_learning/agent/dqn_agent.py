import numpy as np
import torch
import torch.optim as optim
from agent.replay_buffer import ReplayBuffer
from agent.lstm import KanLSTM
from fastkan import FastKAN as KAN
import torch.nn.functional as F
import torch.nn as nn


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


class DQNAgent:

    def __init__(
        self,
        Q,
        Q_target,
        num_actions,
        device,
        gamma=0.95,
        batch_size=64,
        epsilon=0.1,
        tau=0.01,
        lr=1e-4,
        capacity=100000,
        env_name=None,
    ):
        """
        Q-Learning agent for off-policy TD control using Function Approximation.
        Finds the optimal greedy policy while following an epsilon-greedy policy.

        Args:
           Q: Action-Value function estimator (Neural Network)
           Q_target: Slowly updated target network to calculate the targets.
           num_actions: Number of actions of the environment.
           gamma: discount factor of future rewards.
           batch_size: Number of samples per batch.
           tau: indicates the speed of adjustment of the slowly updated target network.
           epsilon: Chance to sample a random action. Float betwen 0 and 1.
           lr: learning rate of the optimizer
        """
        # setup networks
        self.device = device
        self.Q = Q.to(self.device)
        self.Q_target = Q_target.to(self.device)
        self.Q_target.load_state_dict(self.Q.state_dict())

        # define replay buffer
        self.replay_buffer = ReplayBuffer(capacity=capacity)

        # parameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon

        self.loss_function = torch.nn.MSELoss().to(self.device)
        self.optimizer = optim.Adam(self.Q.parameters(), lr=lr)

        self.num_actions = num_actions

        self.env_name = env_name

    def train(self, state, action, next_state, reward, terminal):
        """
        This method stores a transition to the replay buffer and updates the Q networks.
        """
        self.replay_buffer.add_transition(state, action, next_state, reward, terminal)

        states, actions, next_states, rewards, terminal_flags = (
            self.replay_buffer.next_batch(self.batch_size)
        )
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        terminal_flags = torch.tensor(terminal_flags, dtype=torch.float32).to(
            self.device
        )

        with torch.no_grad():
            target = (
                rewards
                + (1 - terminal_flags)
                * self.gamma
                * self.Q_target(next_states).max(dim=1)[0]
            )
        prediction = (
            self.Q(states).gather(dim=1, index=actions.unsqueeze(1).long()).squeeze(1)
        )

        # Update Q-network
        loss = self.loss_function(prediction, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        soft_update(self.Q_target, self.Q, self.tau)

    def act(self, state, deterministic):
        """
        This method creates an epsilon-greedy policy based on the Q-function approximator and epsilon (probability to select a random action)
        Args:
            state: current state input
            deterministic:  if True, the agent should execute the argmax action (False in training, True in evaluation)
        Returns:
            action id
        """
        r = np.random.uniform()
        if deterministic or r > self.epsilon:
            state = (
                torch.tensor(np.array([state], dtype=np.float32))
                .float()
                .to(self.device)
            )
            action_id = self.Q(state).argmax().item()
        else:
            action_id = np.random.randint(self.num_actions)          
        return action_id

    def save(self, file_name):
        torch.save(self.Q.state_dict(), file_name)

    def load(self, file_name):
        self.Q.load_state_dict(torch.load(file_name, map_location=self.device))
        self.Q_target.load_state_dict(torch.load(file_name, map_location=self.device))


class ADRQNKAN(nn.Module):
    def __init__(self, n_actions, state_size, embedding_size):
        super(ADRQNKAN, self).__init__()
        self.n_actions = n_actions
        self.embedding_size = embedding_size
        self.embedder = KAN([n_actions, embedding_size], num_grids=5)
        # [state_size, 3, 32] for matching parameters
        self.obs_layer = KAN([state_size, 3, 32], num_grids=5)
        self.lstm = KanLSTM(input_size = 32+embedding_size, hidden_size = 128, num_outputs=128)
        self.out_layer = KAN([128, n_actions], num_grids=5)
    
    def forward(self, observation, action, hidden = None):
        #Takes observations with shape (batch_size, seq_len, obs_dim)
        #Takes one_hot actions with shape (batch_size, seq_len, n_actions)
        action_embedded = self.embedder(action)
        observation = self.obs_layer(observation)
        lstm_input = torch.cat([observation, action_embedded], dim = -1)
        if hidden is not None:
            lstm_out, hidden_out = self.lstm(lstm_input, hidden)
        else:
            lstm_out, hidden_out = self.lstm(lstm_input)

        q_values = self.out_layer(lstm_out)
        return q_values, hidden_out
    
    def act(self, observation, last_action, epsilon, hidden = None):
        q_values, hidden_out = self.forward(observation, last_action, hidden)
        if np.random.uniform() > epsilon:
            action = torch.argmax(q_values).item()
        else:
            action = np.random.randint(self.n_actions)
        return action, hidden_out
    

class ADRQN(nn.Module):
    def __init__(self, n_actions, state_size, embedding_size):
        super(ADRQN, self).__init__()
        self.n_actions = n_actions
        self.embedding_size = embedding_size
        self.embedder = nn.Linear(n_actions, embedding_size)
        self.obs_layer = nn.Linear(state_size, 16)
        self.obs_layer2 = nn.Linear(16,32)
        self.lstm = nn.LSTM(input_size = 32+embedding_size, hidden_size = 150, batch_first = True)
        self.out_layer = nn.Linear(150, n_actions)
    
    def forward(self, observation, action, hidden = None):
        #Takes observations with shape (batch_size, seq_len, obs_dim)
        #Takes one_hot actions with shape (batch_size, seq_len, n_actions)
        action_embedded = self.embedder(action)
        observation = F.relu(self.obs_layer(observation))
        observation = F.relu(self.obs_layer2(observation))
        lstm_input = torch.cat([observation, action_embedded], dim = -1)
        if hidden is not None:
            lstm_out, hidden_out = self.lstm(lstm_input, hidden)
        else:
            lstm_out, hidden_out = self.lstm(lstm_input)

        q_values = self.out_layer(lstm_out)
        return q_values, hidden_out
    
    def act(self, observation, last_action, epsilon, hidden = None):
        q_values, hidden_out = self.forward(observation, last_action, hidden)
        if np.random.uniform() > epsilon:
            action = torch.argmax(q_values).item()
        else:
            action = np.random.randint(self.n_actions)
        return action, hidden_out

