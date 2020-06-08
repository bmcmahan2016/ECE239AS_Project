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
import copy
import pickle

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

class multiLayer(nn.Module):
    def __init__(self, inputdim, hdims = [], outputdim=1, useTanh=False):
        super(multiLayer, self).__init__()
        self.inputdim = inputdim
        hdims = list(hdims)
        self.hdims = hdims
        self.outputdim = outputdim
        dims = [inputdim]
        if len(hdims) > 0:
            dims = dims + hdims
        dims.append(outputdim)
        
        self.fcs = nn.ModuleList([nn.Linear(dims[k], dims[k+1]) for k in range(len(dims) - 1)])
        self.useTanh = useTanh
    def forward(self, x):
        out = torch.FloatTensor(x)
        for k in range(len(self.fcs) - 1):
            layer = self.fcs[k]
            out = layer(out)
            out = F.relu(out)
        out = self.fcs[-1](out)
        if self.useTanh:
            out = torch.tanh(out)
        return out
    
def  setTargetPosition(env, targetPos=None, scaleFactor=0.15):
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
        reachTargetPos = centerPosition + scaleFactor * reachDirection
    else:
        reachTargetPos = centerPosition + scaleFactor * targetPos
    # manual place target at desired location
    env.env.goal = reachTargetPos
    return reachTargetPos

def getFeatureVector(observation, vel=None):
    '''
    DESCRIPTION: takes an observation dictionary from FetchReach-v1 environment
                and converts it into a feature vector to be used in Policy and 
                value networks. 
    Parameters
    ----------
    observation : dictionary obejct generated by calling env.step(action)
    '''
    #stateFeatures = np.hstack((observation['observation'], \
    #                observation['achieved_goal'], observation['desired_goal']))
    stateFeatures = np.hstack((observation['achieved_goal'] - env.env.initial_gripper_xpos, observation['desired_goal'] - env.env.initial_gripper_xpos))
    if vel is None:
        return stateFeatures
    else:
        return np.hstack((stateFeatures, vel))

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
    return Gt, state, sampleTuples[tupleIX]['action']
def computeTD2(sampleTuples, lmbd=0.95, t = None):
    L = len(sampleTuples)
    if t is None:
        t = 0
    #lmbdPower = lmbd**np.arange(L)
    Rt_n = np.zeros(L - t)
    Gt = torch.zeros((1, 1))
    for n in range(L - t):
        for el in range(0, n):
            Rt_n[n] += (discountFactor**el)*sampleTuples[t+el]['reward']
        Rt_n[n] += (discountFactor**n)*value(sampleTuples[t+n]['next_state'])
    for n in range(1, L - t):
        Gt += (lmbd**(n - 1))*Rt_n[n]
    Gt = (1 - lmbd)*Gt
    #Gt += (lmbd**(L - t - 1))*Rt_n[L - t]
    return Gt, sampleTuples[t]['state'], sampleTuples[t]['action']
def computeTDBatch(sampleTuples, lmbd=0.95, tupleIX=None):
    if tupleIX is None:
        tupleIX = np.arange(len(sampleTuples))
    L = len(tupleIX)
    Gts = torch.zeros((L, 1))
    states = np.zeros((L, observationDim))
    actions = np.zeros((L, 4))
    for l in range(L):
        Gt, state, action = computeTD(sampleTuples, lmbd, tupleIX[l])
        Gts[l] = Gt
        states[l] = state
        actions[l] = action
    return Gts, states, actions

def computeTDBatch2(sampleTuples, lmbd=0.95, tupleIX=None):
    if tupleIX is None:
        tupleIX = np.arange(len(sampleTuples))
    L = len(tupleIX)
    Gts = torch.zeros((L, 1))
    states = np.zeros((L, observationDim))
    actions = np.zeros((L, 4))
    for l in range(L):
        Gt, state, action = computeTD2(sampleTuples, lmbd, tupleIX[l])
        Gts[l] = Gt
        states[l] = state
        actions[l] = action
    return Gts, states, actions

def clipLoss(policy, oldpolicy, state, action, advantage, delta=0.1, eps=0.1):
    amu = action - policy(state)
    siginv = 1/delta*torch.eye(4)
    prob = torch.exp(-0.5*torch.diag(amu@siginv@amu.T))
    
    oldamu = (action - oldpolicy(state)).detach()
    oldprob = torch.exp(-0.5*torch.diag(oldamu@siginv@oldamu.T))
    wi = prob/oldprob
    output = -torch.mean(torch.min(wi*advantage, torch.clamp(wi, 1-eps, 1+eps)*advantage))
    return output

def probOfAction(policy, state, action, delta=0.1):
    amu = action - policy(state)
    siginv = 1/delta*torch.eye(4)
    prob = torch.exp(-0.5*torch.diag(amu@siginv@amu.T))
    return prob

# define some values
NUM_EPISODES = 301        # number of episodes we will train on
lr = 1e-3               # learning rate for Q-learning
discountFactor = 0.9    # discount factor
delta = 0.1   # single-dimension standard deviation of additive action noise
epsilon = 0.1
actionScale = 0.2
scaleFactor = 0.15

# create the environment
env = gym.make('FetchReach-v1')
# create policy and value networks
#policy = policyNet().to(device)
#value = valueNet().to(device)

refReaches = sio.loadmat('refReaches.mat')
cursorPos = (scaleFactor*refReaches['cursorPos']) + env.env.initial_gripper_xpos
refTargets = refReaches['targets']
refVel = scaleFactor*refReaches['cursorVel']

observationDim = 9
actionDim = 4
policy = multiLayer(observationDim, [9, 9], actionDim, useTanh=False)
value = multiLayer(observationDim, [9, 9], 1)

# MSE loss is used for value network
valueLoss = nn.MSELoss()
# SGD is used for optimization of value network
valueOptimizer = optim.SGD(value.parameters(), lr=0.01, weight_decay=0.0)
policyOptimizer = optim.SGD(policy.parameters(), lr=0.005, weight_decay=0.0)

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
testvec = np.random.randn(1, observationDim)
testvechist = []
Xhist = []
targetHist = []
valueHist = []
actionHistAll = []
velHist = []
rewardHistAll = []
valueLossHist = []

t0 = time.time()
for currEpisodeNum in range(NUM_EPISODES):
    obs = env.reset()
    env.env.distance_threshold = 0.1
    #targetPos = setTargetPosition(env, np.array([1, 0, 0]))
    refInd = np.random.randint(0, refTargets.shape[0])
    #refInd = 0
    targetPos = setTargetPosition(env, refTargets[refInd], scaleFactor=scaleFactor)
    obs['desired_goal'] = targetPos
    done = False
    
    D = np.empty((SIZE_OF_DATA), dtype=list)
    
    X = np.zeros((MAX_TIME_STEPS, 3)) + env.env.initial_gripper_xpos
    Vel = np.zeros((MAX_TIME_STEPS, 3))
    stateHist = np.zeros((MAX_TIME_STEPS, observationDim))
    actionHist = np.zeros((MAX_TIME_STEPS, 4))
    rewardHist = np.zeros(MAX_TIME_STEPS)
    for currTimeStep in range(MAX_TIME_STEPS):
        old_obs = obs
        x = obs['achieved_goal']
        X[currTimeStep] = x # obs['observation'][0:3]
        if currTimeStep > 0:
            oldvel = Vel[currTimeStep - 1]
        else:
            oldvel = np.zeros(3)
        action = policy(getFeatureVector(obs, oldvel))
        action = action + delta*torch.randn(4, device=device)
        action = action.cpu().detach().numpy().reshape(4)
        
        actionHist[currTimeStep] = action
        obs, reward, done, info = env.step(actionScale*action)
        dist = np.linalg.norm(obs['achieved_goal'] - X[0])
        
        Vel[currTimeStep] = obs['achieved_goal'][0:3] - X[currTimeStep]
        ### direction reward
        vel = Vel[currTimeStep]
        direction = targetPos - obs['achieved_goal']
        direction /= np.linalg.norm(direction)
        #reward = np.dot(vel, direction)/np.linalg.norm(vel)
        taskreward = np.exp(-5*(1 - np.dot(vel, direction)/np.linalg.norm(vel))) + np.exp(-10*(np.linalg.norm(targetPos - obs['achieved_goal']))**2)
        imitationreward = np.exp(-25*np.linalg.norm(vel - refVel[refInd, currTimeStep])**2) + np.exp(-5*np.linalg.norm(x - cursorPos[refInd, currTimeStep])**2)
        reward = 0.25*taskreward + 0.75*imitationreward
        
        if dist > 0.25 or np.abs(obs['achieved_goal'][2] - targetPos[2]) > 0.1:
            #reward = -1
            #done = True
            pass
        #reward = taskreward
        rewardHist[currTimeStep] = reward
        ###
        D[currTimeStep] = {'state' : getFeatureVector(old_obs, oldvel), 'next_state': getFeatureVector(obs, vel), 
                           'action' : action, 'reward' : reward}
        if dist > 0.3:
            #done = True
            pass
        if done:
            D = D[0:currTimeStep+1]
            #print(currEpisodeNum, currTimeStep)
            break
    #
    done = False
    
    TD_target, state, actionPlayed = computeTDBatch(D)
    #TD_target2, state2, actionPlayed2 = computeTDBatch2(D)
    #pdb.set_trace()
    valueEstimate = value(state)
    valueHist.append(valueEstimate.detach().numpy())
    #print(TD_target.shape, valueEstimate.shape)
    loss = valueLoss(TD_target, valueEstimate)
    valueLossHist.append(loss.item())
    
    valueOptimizer.zero_grad()
    loss.backward()
    valueOptimizer.step()
    
    TD_targs.append(TD_target.detach().numpy())
    valHist.append(valueEstimate.detach().numpy())
    Xhist.append(X)
    targetHist.append(targetPos)
    actionHistAll.append(actionHist)
    velHist.append(Vel)
    rewardHistAll.append(rewardHist)
    
    # update the policy network
    oldPolicyNetwork = copy.deepcopy(policy)
    policyOptimizer.zero_grad()
    advantage = (TD_target - valueEstimate).detach()
    
    #if currEpisodeNum % 10 == 0:
    #    pdb.set_trace()

    for k in range(10):
        #pdb.set_trace()
        batchInds = np.random.choice(np.arange(currTimeStep+1), size=np.minimum(20, currTimeStep+1), replace=False)
        policyClipLoss = clipLoss(policy, oldPolicyNetwork, stateHist[batchInds], torch.FloatTensor(actionHist[batchInds]), advantage[batchInds], delta=delta)
        policyClipLoss += torch.mean(F.relu(torch.abs(policy(torch.FloatTensor(stateHist[batchInds])))*actionScale - 1)**2)
        policyClipLoss.backward()
        policyOptimizer.step()
    
    #policyLoss = -(torch.FloatTensor(rewardHist))*torch.log(probOfAction(policy, stateHist, torch.FloatTensor(actionHist), delta=delta))
    #policyLoss = -(TD_target.detach() - 1)*torch.log(probOfAction(policy, stateHist, torch.FloatTensor(actionHist)))
    #policyLoss = torch.mean(policyLoss)
    #policyLoss.backward()
    #policyOptimizer.step()
    #pdb.set_trace()
    
    finalDist.append(np.sqrt(np.sum((obs['achieved_goal'] - obs['desired_goal'])**2)))
    
    testvecpol = torch.norm(policy(testvec)).item()
    #print(currEpisodeNum, np.round(testvecpol, 3), np.round(finalDist[-1], 3))
    if currEpisodeNum % 10 == 0:
        print(currEpisodeNum, np.round(np.arctan2(refTargets[refInd][1], refTargets[refInd][0]), 3),
              np.round(finalDist[-1], 3), np.round(np.mean(np.linalg.norm(actionHist, axis=1)), 3), np.round(policyClipLoss.item(), 3), 
             np.round(value(testvec).item(), 3), currTimeStep+1)
    testvechist.append(testvecpol)
    pass

datadict = {}
datadict['valHist'] = valHist
datadict['valueLossHist'] = valueLossHist
datadict['Xhist'] = Xhist
datadict['targetHist'] = targetHist
datadict['actionHist'] = actionHistAll
datadict['velHist'] = velHist
datadict['rewardHist'] = rewardHistAll
datadict['TD_targs'] = TD_targs
datadict['finalDist'] = finalDist
datadict['train_time'] = time.time() - t0
datadict['testvechist'] = testvechist
datadict['valueHist'] = valueHist
sio.savemat('data_' + str(NUM_EPISODES) + 'iter.mat', datadict)

pickle.dump(value, open('value_' + str(NUM_EPISODES) + '.bin', 'wb'))
pickle.dump(policy, open('policy_' + str(NUM_EPISODES) + '.bin', 'wb'))