import sys, os, math, random
CURRENT_TEST_DIR = os.getcwd()
sys.path.append(CURRENT_TEST_DIR + "/../../src")

from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import slayerSNN as snn
from learningStats import learningStats

import gym
from collections import namedtuple

netParams = snn.params('network.yaml')

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

env = gym.make('CartPole-v0').unwrapped


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# Network definition
class Network(torch.nn.Module):
    def __init__(self, netParams):
        super(Network, self).__init__()
        # Initialize slayer
        slayer = snn.layer(netParams['neuron'], netParams['simulation'])
        self.slayer = slayer
        # Define network functions
        # The commented line below should be used if the input spikes were not reshaped
        self.fc1 = slayer.dense(env.observation_space.shape[0], 512)
        self.fc2 = slayer.dense(512, env.action_space.n)

    def forward(self, spikeInput):
        spikeOut1 = self.slayer.spike(self.slayer.psp(self.fc1(spikeInput)))
        spikeOut2 = self.slayer.spike(self.slayer.psp(self.fc2(spikeOut1)))

        pspOut = self.slayer.psp(spikeOut2)
        
        return value


if __name__ == '__main__':
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 200

    batch_size = 8
    steps_done = 0

    memory = ReplayMemory(10000)

    # Define the cuda device to run the code on.
    device = torch.device('cuda')

    # Create network instance.
    policy_net = Network(netParams).to(device)
    target_net = Network(netParams).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # Create snn loss instance.
    error = snn.loss(netParams).to(device)

    # Define optimizer module.
    optimizer = optim.Adam(policy_net.parameters(), lr=0.01, amsgrad=True)

    # Learning stats instance.
    stats = learningStats()

    def select_action(state, steps_done):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        math.exp(-1. * steps_done / EPS_DECAY)

        state = state.unsqueeze(0).repeat(batch_size, 1, 1)

        inputSpikes = []

        for _ in range(netParams['simulation']['tSample']):
            inputSpikes.append(torch.rand_like(state).le(state))

        inputSpikes = torch.stack(inputSpikes, 0).permute(1, 3, 2, 0).unsqueeze(2).float()
        print(inputSpikes.shape)
        
        '''
        if sample > eps_threshold:
            with torch.no_grad():
                output = policy_net.forward(inputSpikes)
                return output
        else:
            return torch.tensor([[random.randrange(2)]], device=device, dtype=torch.long)
        '''

        with torch.no_grad():
            value = policy_net.forward(inputSpikes)
            return value

    state = torch.rand([1, 4], device=device)

    action = select_action(state, steps_done)