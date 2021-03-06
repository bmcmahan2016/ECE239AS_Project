# -*- coding: utf-8 -*-
"""
Created on Sun May 17 17:14:47 2020

@author: bmcma

DESCRIPTION:
    Train an agent to perform the center out reach task in the FetchReach-v1
    open AI gym environmnet
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import matplotlib.pyplot as plt
import scipy.io as sio
import pdb
import time

# comment line below if you are running without cuda
#device = torch.device("cuda")
device = torch.device('cpu')

class valueNet(nn.Module):
    
    def __init__(self):
        super(valueNet, self).__init__()
        self.fc1 = nn.Linear(16, 8)
        self.fc2 = nn.Linear(8, 4)
        self.fc3 = nn.Linear(4, 1)
        # variance of gaussian from which we draw actions; treated as hyper-parameter
        self.var = 1
        
    def forward(self, x):
        # flatten the state into a vector (16,)
        #x = np.hstack((state['observation'], state['achieved_goal'], state['desired_goal']))
        # place state vector into a torch tensor
        x = torch.from_numpy(x).to(device)
        x = x.float()
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        # shape into a torch column vector
        x = x.view(-1, self.num_flat_features(x))
        # forward pass 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        #value = x.cpu().detach().numpy().reshape(1)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class policyNet(nn.Module):
    
    def __init__(self):
        super(policyNet, self).__init__()
        self.fc1 = nn.Linear(16, 8)
        self.fc2 = nn.Linear(8, 8)
        self.fc3 = nn.Linear(8, 4)
        # variance of gaussian from which we draw actions; treated as hyper-parameter
        self.var = 1
        
    def forward(self, x):
        # flatten the state into a vector (16,)
        #x = np.hstack((state['observation'], state['achieved_goal'], state['desired_goal']))
        # place state vector into a torch tensor
        x = torch.from_numpy(x).to(device)
        x = x.float()
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        # shape into a torch column vector
        x = x.view(-1, self.num_flat_features(x))
        # forward pass 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        #action = x.cpu().detach().numpy().reshape(4)
        actionMean = x #+ torch.randn(4, device=device)*self.var
        return actionMean #+ self.var*np.random.randn(4)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def  setTargetPosition(env, targetPos=None):
    '''
    DESCRIPTION
    reset the environment to an initial state where the goal target is at one of
    8 target locations. 8 target locations are distributed uniformly around a 
    circle centered on the gripper/hand position. The target position for next 
    episode is drawn uniformly from the 8 allowed target positions. These target 
    positions all exist in a 2D plane to emulate a 2 dimensional human/moneky 
    center out reaching task. 
    
    the origin is located on the bottom left grey corner of the render.
    The table is located at roughly z=0.5
    x goes from roughly x=1 to x=1.5 
    y goes from roughly y = 0.4 to y=1.1
    '''
    # get the starting position of the hand (always the same)
    centerPosition = env.env.initial_gripper_xpos
    
    if targetPos is None:
        theta = np.pi*np.random.randint(0, 8)/4
        reachDirection = np.array([np.cos(theta), np.sin(theta), 0])

        # place the target at the random reach position
        scaleFactor = 0.3       
        reachTargetPos = centerPosition + scaleFactor * reachDirection
    else:
        reachTargetPos = centerPosition + scaleFactor * targetPos
    # manual place target at desired location
    env.env.goal = reachTargetPos
    return reachTargetPos

def getFeatureVector(observation):
    '''
    DESCRIPTION: takes an observation dictionary from FetchReach-v1 environment
                and converts it into a feature vector to be used in Policy and 
                value networks. 
    Parameters
    ----------
    observation : dictionary obejct generated by calling env.step(action)
    '''
    stateFeatures = np.hstack((observation['observation'], \
                    observation['achieved_goal'], observation['desired_goal']))
    return stateFeatures

def updateValueEstimate(lmbda=0.95):
    '''
    DESCRIPTION: updates the value network parameters using computed TD targets.

    Returns
    -------
    None.

    '''
    pass

def computeTD(sampleTuples, lmbd=0.95, tupleIX=None):
    '''
    DESCRIPTION: computes the TD error for a sampled tuple

    Returns
    -------
    Gt: TD target for a randomly sampled tuple or inputed tuple
    state: feature vector of sampled tuple

    '''
    numTimeSteps = len(sampleTuples)
    # if tupleIX not given, randomly sample a tuple
    if tupleIX is None:
        tupleIX = int( numTimeSteps*np.random.rand() )

    Gt = torch.zeros((1, 1))
    discountedRewards = 0
    for futureTimeStep in range(tupleIX, numTimeSteps):
        n = futureTimeStep - (tupleIX)
        futureReward = sampleTuples[futureTimeStep]['reward']
        discountedRewards += (discountFactor**n) * futureReward
        if ( (futureTimeStep + 1) != numTimeSteps):
            nextState = sampleTuples[futureTimeStep]['next_state']
            valueEstimate = value(nextState)
            #valueEstimte = valueEstimate.cpu().detach().numpy().reshape(1)
            Gt += lmbd**n * (discountedRewards + (discountFactor**n) * valueEstimate)
        else:
            Gt += (lmbd**n) * discountedRewards
    Gt = (1-lmbd)*Gt
    state = sampleTuples[tupleIX]['state']
    #print(tupleIX, Gt)
    return torch.tensor(Gt), state, sampleTuples[tupleIX]['action']
    
def computeTDBatch(sampleTuples, lmbd=0.95, tupleIX=None):
    if tupleIX is None:
        tupleIX = np.arange(len(sampleTuples))
    L = len(tupleIX)
    Gts = torch.zeros((L, 1))
    states = np.zeros((L, 16))
    actions = np.zeros((L, 4))
    for l in range(L):
        Gt, state, action = computeTD(sampleTuples, lmbd, tupleIX[l])
        Gts[l] = Gt
        states[l] = state
        actions[l] = action
    return Gts, states, actions
        

# define some values
NUM_EPISODES = 1001        # number of episodes we will train on
lr = 1e-3               # learning rate for Q-learning
discountFactor = 0.95    # discount factor
delta = 0.1    # single-dimension standard deviation of additive action noise

# create the environment
env = gym.make('FetchReach-v1')
# create policy and value networks
policy = policyNet().to(device)
value = valueNet().to(device)


# MSE loss is used for value network
valueLoss = nn.MSELoss()
# SGD is used for optimization of value network
valueOptimizer = optim.SGD(value.parameters(), lr=0.001)
policyOptimizer = optim.SGD(policy.parameters(), lr=0.001)

#####
#used for debugging
TD_targs = []
valHist = []
#####

currEpisodeNum = 0                           # initialize episode counter
MAX_TIME_STEPS = env._max_episode_steps        # how many data tuples to store
SIZE_OF_DATA = MAX_TIME_STEPS # same variable, different name, no good reason why
D = np.empty((SIZE_OF_DATA), dtype=list)     # holds the episode tuples
currTimeStep = 0

finalDist = []
testvec = np.random.randn(1, 16)
testvechist = []

t0 = time.time()
for currEpisodeNum in range(NUM_EPISODES):
    obs = env.reset()
    env.env.distance_threshold = 0.1
    targetPos = setTargetPosition(env)
    done = False
    
    D = np.empty((SIZE_OF_DATA), dtype=list)
    
    X = np.zeros((MAX_TIME_STEPS, 3))
    Vel = np.zeros((MAX_TIME_STEPS, 3))
    for currTimeStep in range(MAX_TIME_STEPS):
        old_obs = obs
        X[currTimeStep] = obs['observation'][0:3]
        action = policy(getFeatureVector(obs))
        action = action + delta*torch.randn(4, device=device)
        action = action.cpu().detach().numpy().reshape(4)
        obs, reward, done, info = env.step(action)
        Vel[currTimeStep] = obs['observation'][0:3] - X[currTimeStep]
        ### direction reward
        vel = Vel[currTimeStep]
        direction = targetPos - obs['achieved_goal']
        direction /= np.linalg.norm(direction)
        reward = np.dot(vel, direction)/np.linalg.norm(vel)
        ###
        D[currTimeStep] = {'state' : getFeatureVector(old_obs), 'next_state': getFeatureVector(obs), 'action' : action, 'reward' : reward}
        if done:
            D = D[0:currTimeStep]
            print(currEpisodeNum, currTimeStep)
            break
    #
    done = False
    
    TD_target, state, actionPlayed = computeTDBatch(D)
    valueEstimate = value(state)
    #print(TD_target.shape, valueEstimate.shape)
    loss = valueLoss(TD_target, valueEstimate)
    
    valueOptimizer.zero_grad()
    loss.backward()
    valueOptimizer.step()
    
    TD_targs.append(TD_target.detach().numpy())
    valHist.append(valueEstimate.detach().numpy())
    
    # update the policy network
    policyOptimizer.zero_grad()
    advantage = TD_target - valueEstimate
    actionMean = policy(state)
    actionPlayed = torch.FloatTensor(actionPlayed).to(device)
    v = -(actionPlayed-actionMean)*advantage
    actionMean.backward(v)
    policyOptimizer.step()
    finalDist.append(np.sum((obs['achieved_goal'] - obs['desired_goal'])**2))
    
    testvecpol = torch.norm(policy(testvec)).item()
    #print(currEpisodeNum, np.round(testvecpol, 3), np.round(finalDist[-1], 3))
    testvechist.append(testvecpol)
    pass

datadict = {}
datadict['valHist'] = valHist
datadict['TD_targs'] = TD_targs
datadict['finalDist'] = finalDist
datadict['train_time'] = time.time() - t0
datadict['testvechist'] = testvechist
sio.savemat('data_' + str(NUM_EPISODES) + 'iter.mat', datadict)

