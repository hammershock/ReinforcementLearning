import os
import random
import pickle
from collections import deque

import gym

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

from grid_world import GridWorld, visualize_grid_world


def make_net(input_dim, hidden_dims, output_dim, dropout_rate=0.1):
    """
    Creates a neural network with variable depth, using ReLU activations,
    Batch Normalization, and Dropout.

    Parameters:
    - input_dim (int): Dimensionality of the input layer.
    - hidden_dims (list of int): A list containing the number of units in each hidden layer.
    - output_dim (int): Dimensionality of the output layer.
    - dropout_rate (float): Dropout rate to use after each hidden layer except the last one.

    Returns:
    - net (torch.nn.Sequential): The constructed neural network.
    """
    layers = []
    for i, hidden_dim in enumerate(hidden_dims):
        if i == 0:
            layers.append(nn.Linear(input_dim, hidden_dim))
        else:
            layers.append(nn.Linear(hidden_dims[i - 1], hidden_dim))
        # layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        if i < len(hidden_dims) - 1:  # No dropout after the last hidden layer
            layers.append(nn.Dropout(dropout_rate))
    
    # Adding the output layer
    layers.append(nn.Linear(hidden_dims[-1], output_dim))
    
    net = nn.Sequential(*layers)
    return net


class QNet(nn.Module):
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=1e-3, update_target_every=5):
        super(QNet, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.current_net = make_net(state_dim, [64, 32, 16], action_dim)
        self.target_net = make_net(state_dim, [64, 32, 16], action_dim)
        self.update_target_every = update_target_every
        self.optimizer = optim.Adam(self.current_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.update_count = 0
        
        self.action_counts = np.zeros((action_dim, ))  # 记录每个动作被选择的次数
        self.total_counts = 0  # 记录总的动作选择次数
        
        # 初始化目标网络参数
        self.target_net.load_state_dict(self.current_net.state_dict())
        self.target_net.eval()
        
    def load_model(self, path):
        self.load_state_dict(torch.load(path))
            
    def save(self, save_path):
        torch.save(self.state_dict(), save_path)
        
    def _calc_loss(self, state, action, reward, next_state, done, gamma):
        q_values = self.current_net(torch.from_numpy(np.array(state)))
        next_q_values = self.target_net(torch.from_numpy(np.array(next_state)))
        max_next_q_values = torch.max(next_q_values, dim=1)[0].detach()
        td_target = torch.tensor(reward, dtype=torch.float32) + gamma * max_next_q_values * (torch.tensor(1 - np.array(done)))
        loss = self.loss_fn(q_values[:, action], td_target)
        return loss
    
    def fit(self, state, action, next_state, reward, done, gamma=0.99) -> float:
        self.optimizer.zero_grad()
        loss = self._calc_loss(state, action, reward, next_state, done, gamma)
        loss.backward()
        self.optimizer.step()
        
        self.update_count += 1
        if self.update_count % self.update_target_every == 0:
            self.target_net.load_state_dict(self.current_net.state_dict())
        return loss.item()
    
    def make_decision_epsilon_greedy(self, state, epsilon=0.1):
        if random.random() > epsilon:
            # 使用当前网络进行决策
            with torch.no_grad():
                q_values = self.current_net(torch.from_numpy(np.array(state)).unsqueeze(0))
                # print(q_values)
                action = q_values.max(1)[1].item()  # 选择价值最高的动作
        else:
            # 随机选择动作
            action = random.choice([0, 1])
        return action
    
    def fit_buffer(self, buffer: 'ReplayBuffer', epoch, batch_size=16):
        current_loss = 0.0
        p_bar = tqdm(range(epoch))
        buffer.shuffle()
        for i in p_bar:
            state, action, next_state, reward, done = buffer.sample(batch_size)
            current_loss += self.fit(state, action, next_state, reward, done, gamma=self.gamma)
            p_bar.set_postfix(current_loss=current_loss / (i + 1))
            
    def evaluate_buffer(self, buffer: 'ReplayBuffer', verbose=False):
        mean_average_loss = 0.
        with torch.no_grad():
            p_bar = tqdm(buffer.buffer, total=len(buffer)) if verbose else buffer.buffer
            for i, (state, action, next_state, reward, done) in enumerate(p_bar):
                loss = self._calc_loss([state], [action], [reward], [next_state], [done], gamma=self.gamma)
                mean_average_loss += loss.item()
                if verbose:
                    p_bar.set_postfix(current_loss=mean_average_loss / (i + 1))
        mean_average_loss /= len(buffer)
        if verbose:
            print(f'Mean average loss: {mean_average_loss:.3f}')
        return mean_average_loss
    

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.position = 0
        
    def shuffle(self):
        random.shuffle(self.buffer)
        
    def save(self, save_path):
        with open(save_path, 'wb') as f:
            pickle.dump(self.buffer, f)
            
    def load(self, load_path):
        with open(load_path, 'rb') as f:
            self.buffer = pickle.load(f)
    
    def push(self, state, action, next_state, reward, done):
        self.buffer.append((state, action, next_state, reward, done))
    
    def choice(self):
        state, action, next_state, reward, done = random.choice(self.buffer)
        return state, action, next_state, reward, done
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, next_state, reward, done = zip(*batch)
        return np.array(state), np.array(action), np.array(next_state), np.array(reward), np.array(done)
    
    def __len__(self):
        return len(self.buffer)


if __name__ == '__main__':
    buffer = ReplayBuffer(capacity=100000)
    q_net = QNet(4, 2, update_target_every=30, lr=0.01, gamma=0.6)
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    env._max_episode_steps = 500
    
    q_net.load_model('./output/q_net.pth')
    buffer.load('./output/q_buffer.pkl')

    # q_net.fit_buffer(buffer, 1000, batch_size=256)
    # q_net.evaluate_buffer(buffer, verbose=True)
    # q_net.save('./output/q_net.pth')

    finished = False
    for episode in tqdm(range(100)):
        state, _ = env.reset()
        done, truncated = False, False
        length = 0
        while not done and not truncated:
            length += 1
            action = q_net.make_decision_epsilon_greedy(state)
            # action = env.action_space.sample()
            next_state, reward, done, truncated, info = env.step(action)
            buffer.push(state, action, next_state, reward, done)
            state = next_state

            cv2.imshow("", env.render())
            if cv2.waitKey(20) & 0xFF == ord('q'):
                finished = True
                break
        if finished:
            break
    # buffer.save('./output/q_buffer.pkl')
