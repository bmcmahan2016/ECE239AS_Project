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

# comment line below if you are running without cuda
device = torch.device("cuda")


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
        
    def forward(self, state):
        # flatten the state into a vector (16,)
        x = np.hstack((state['observation'], state['achieved_goal'], state['desired_goal']))
        # place state vector into a torch tensor
        x = torch.from_numpy(x).to(device)
        x = x.float()
        x = x.unsqueeze(0)
        # shape into a torch column vector
        x = x.view(-1, self.num_flat_features(x))
        # forward pass 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        action = x.cpu().detach().numpy().reshape(4)
        return action + self.var*np.random.randn(4)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def  setTargetPosition(env):
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
    # reset the environment to a random initial state
    env.reset()
    
    # get the starting position of the hand (always the same)
    centerPosition = env.env.initial_gripper_xpos
    
    # cartesian coordinates for reach angles on unit circle
    unitCirclePositions = []
    unitCirclePositions.append( np.array((1,0,0)) )
    unitCirclePositions.append( np.array((np.sqrt(2)/2,np.sqrt(2)/2,0)) )
    unitCirclePositions.append( np.array((0,1,0)) )
    unitCirclePositions.append(  np.array((-np.sqrt(2)/2,np.sqrt(2)/2,0)) )
    unitCirclePositions.append( np.array((-1,0,0)) )
    unitCirclePositions.append(  np.array((-np.sqrt(2)/2,-np.sqrt(2)/2,0)) )
    unitCirclePositions.append( np.array((0,-1,0)) )
    unitCirclePositions.append(  np.array((np.sqrt(2)/2,-np.sqrt(2)/2,0)) )
    
    # randomly choose a reach direction with uniform probability
    reachDirectionIX = int(8*np.random.rand())
    reachDirection = unitCirclePositions[reachDirectionIX]
    
    # place the target at the random reach position
    scaleFactor = 0.3       
    reachTargetPos = centerPosition + scaleFactor * reachDirection
    
    # manual place target at desired location
    env.env.goal = reachTargetPos

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

def computeTD(sampleTuples, lmbd=0.95):
    '''
    DESCRIPTION: computes the TD error for a sampled tuple

    Returns
    -------
    Gt: TD target for a randomly sampled tuple
    state: feature vector of sampled tuple

    '''
    numTimeSteps = len(sampleTuples)
    # randomly sample a tuple
    tupleIX = int( numTimeSteps*np.random.rand() )
    print("randomly selected tuple #", tupleIX)

    Gt = 0
    discountedRewards = 0
    for futureTimeStep in range(tupleIX+1, numTimeSteps):
        n = futureTimeStep - (tupleIX+1)
        futureReward = sampleTuples[futureTimeStep]['reward']
        discountedRewards += (discountFactor**n) * futureReward
        if ( (futureTimeStep + 1) != numTimeSteps):
            nextState = sampleTuples[futureTimeStep+1]['state']
            valueEstimate = value(nextState)
            #valueEstimte = valueEstimate.cpu().detach().numpy().reshape(1)
            Gt += lmbd**n * (discountedRewards + (discountFactor**n) * valueEstimate)
        else:
            Gt += (lmbd**n) * discountedRewards
    Gt = (1-lmbd)*Gt
    state = sampleTuples[tupleIX]['state']
    return torch.tensor(Gt), state
        

# define some values
NUM_EPISODES = 200        # number of episodes we will train on
lr = 1e-3               # learning rate for Q-learning
discountFactor = 0.95    # discount factor 

# create the environment
env = gym.make('FetchReach-v1')
# create policy and value networks
policy = policyNet().to(device)
value = valueNet().to(device)


# MSE loss is used for value network
valueLoss = nn.MSELoss()
# SGD is used for optimization of value network
optimizer = optim.SGD(value.parameters(), lr=0.01)


# manual over-ride of target position
obs = env.reset()
setTargetPosition(env)
done = False

#####
#used for debugging
TD_targs = []
valHist = []
#####

currEpisodeNum = 0                           # initialize episode counter
SIZE_OF_DATA = env._max_episode_steps        # how many data tuples to store
D = np.empty((SIZE_OF_DATA), dtype=list)     # holds the episode tuples
currTimeStep = 0
while ( currEpisodeNum < NUM_EPISODES ):
    
    action = policy(obs)  
    obs, reward, done, info = env.step(action)
    D[currTimeStep] = {'state' : getFeatureVector(obs), 'action' : action, 'reward' : reward}
    currTimeStep += 1
    #env.render()
    if done:
        currEpisodeNum += 1
        print('\ncurrent episode #', currEpisodeNum)
        done = False
        env.reset()
        setTargetPosition(env)
        currTimeStep=0
        TD_target, state = computeTD(D)
        valueEstimate = value(state)
        print("TD target:", TD_target.item())
        print("value function approximation:", valueEstimate.item())
        loss = valueLoss(TD_target,  valueEstimate)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    # Does the update
        TD_targs.append(TD_target.item())
        valHist.append(valueEstimate.item())

        
    

   
    # If we want, we can substitute a goal here and re-compute
    # the reward. For instance, we can just pretend that the desired
    # goal was what we achieved all along.
    #substitute_goal = obs['achieved_goal'].copy()
    #substitute_reward = env.compute_reward(
    #    obs['achieved_goal'], substitute_goal, info)
    #print('reward is {}, substitute_reward is {}'.format(
    #    reward, substitute_reward))
    #print('theta', theta)


plt.plot(np.array(TD_targs))
plt.plot(np.array(valHist))
plt.legend(["TD Target", "Value Approximator"])
#criterion = nn.MSELoss()

#loss = criterion(output, target)
#print(loss)