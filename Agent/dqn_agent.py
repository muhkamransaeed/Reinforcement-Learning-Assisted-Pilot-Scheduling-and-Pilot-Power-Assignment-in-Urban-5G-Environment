import random
import numpy as np
import torch
import torch.optim as optim
from collections import deque
from Deep_Q_Network.model import DQNNetwork

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.vstack(states), 
                np.array(actions), 
                np.array(rewards, dtype=np.float32), 
                np.vstack(next_states), 
                np.array(dones, dtype=np.uint8))

    def __len__(self):
        return len(self.memory)

class DQNAgent:
    def __init__(self, state_dim, action_dim, device='cpu',
                 lr=1e-3, gamma=0.99, epsilon_start=1.0,
                 epsilon_end=0.01, epsilon_decay=0.995,
                 buffer_size=10000, batch_size=64, target_update=10):
        
        self.device = torch.device(device)
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.policy_net = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_net = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criteria = torch.nn.MSELoss()

        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.replay_buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.target_update = target_update
        self.learn_step_counter = 0

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state)
            return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Compute Q(s,a)
        q_values = self.policy_net(states).gather(1, actions)

        # Compute target Q-values
        with torch.no_grad():
            max_next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        loss = self.criteria(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # Update target network periodically
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

