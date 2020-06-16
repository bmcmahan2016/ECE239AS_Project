# -*- coding: utf-8 -*-
"""
Created on Sun May 17 17:14:47 2020

@author: bmcma

DESCRIPTION:
    Train an agent to perform the center out reach task in the FetchReach-v1
    open AI gym environment
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
import customEnv

# comment line below if you are running without cuda
#device = torch.device("cuda")
device = torch.device('cpu')

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
            #out = F.relu(out)
            out = F.elu(out)
        out = self.fcs[-1](out)
        if self.useTanh:
            out = torch.tanh(out)
        return out

class LSTM(nn.Module):
    def __init__(self, inputdim, hiddendim, outputdim=1):
        super(LSTM, self).__init__()
        self.inputdim = inputdim
        self.hiddendim = hiddendim
        self.outputdim = outputdim
        self.lstm = nn.LSTM(inputdim, hiddendim)
        self.fc = nn.Linear(hiddendim, outputdim)
        
        self.h = torch.zeros((1, 1, hiddendim))
        self.c = torch.zeros((1, 1, hiddendim))
        return
    
    def forward(self, x, h0=None, c0=None):
        x = torch.FloatTensor(x.reshape((-1, 1, self.inputdim)))
        if h0 is None or c0 is None:
            out, (hc, cn) = self.lstm(x, (self.h, self.c))
        else:
            h0 = h0.reshape((1, 1, self.hiddendim))
            c0 = c0.reshape((1, 1, self.hiddendim))
            out, (hc, cn) = self.lstm(x, (h0, c0))
        self.h = hc
        self.c = cn
        out = self.fc(out)
        return out.reshape((-1, self.outputdim))
    
    def resetState(self):
        self.h = torch.zeros((1, 1, self.hiddendim))
        self.c = torch.zeros((1, 1, self.hiddendim))
        return

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

def computeTDBatch(sampleTuples, lmbd=0.95, tupleIX=None):
    if tupleIX is None:
        tupleIX = np.arange(len(sampleTuples))
    L = len(tupleIX)
    Gts = torch.zeros((L, 1))
    states = np.zeros((L, observationDim))
    actions = np.zeros((L, actionDim))
    for l in range(L):
        Gt, state, action = computeTD(sampleTuples, lmbd, tupleIX[l])
        Gts[l] = Gt
        states[l] = state
        actions[l] = action
    return Gts, states, actions

def clipLoss(policy, oldpolicy, state, action, advantage, delta=0.1, eps=0.1):
    amu = action - policy(state)
    pdb.set_trace()
    siginv = 1/delta*torch.eye(actionDim)
    prob = torch.exp(-0.5*torch.diag(amu@siginv@amu.T))
    
    oldamu = (action - oldpolicy(state)).detach()
    oldprob = torch.exp(-0.5*torch.diag(oldamu@siginv@oldamu.T))
    wi = prob/oldprob
    output = -torch.mean(torch.min(wi*advantage, torch.clamp(wi, 1-eps, 1+eps)*advantage))
    return output

def clipLossProbs(newprobs, oldprobs, advantage, eps=0.1):
    wi = newprobs/oldprobs
    output = -torch.mean(torch.min(wi*advantage, torch.clamp(wi, 1-eps, 1+eps)*advantage))
    return output

def probOfAction(policy, state, action, delta=0.1):
    amu = action - policy(state)
    siginv = 1/delta*torch.eye(actionDim)
    prob = torch.exp(-0.5*torch.diag(amu@siginv@amu.T))
    return prob

# define some values
NUM_EPISODES = 16001        # number of episodes we will train on
lr = 1e-3               # learning rate for Q-learning
discountFactor = 0.95    # discount factor
delta = 3   # single-dimension standard deviation of additive action noise
epsilon = 0.1
actionScale = 1
scaleFactor = 1
dt = 0.022

# create the environment
env = customEnv.env(m=1, dt=dt)

refReaches = sio.loadmat('refReaches.mat')
cursorPos = (scaleFactor*refReaches['cursorPos'][:, :, 0:2])
refTargets = refReaches['targets'][:, 0:2]
refVel = scaleFactor*refReaches['cursorVel'][:, :, 0:2]

observationDim = 6
actionDim = 2
#policy = multiLayer(observationDim, [10, 10], actionDim, useTanh=False)
policy = LSTM(observationDim, 10, actionDim)
value = multiLayer(observationDim, [10, 10], 1)

# MSE loss is used for value network
valueLoss = nn.MSELoss()
# SGD is used for optimization of value network
#valueOptimizer = optim.SGD(value.parameters(), lr=0.01, weight_decay=0.01)
valueOptimizer = optim.Adam(value.parameters())
#policyOptimizer = optim.SGD(policy.parameters(), lr=0.001, weight_decay=0)
policyOptimizer = optim.Adam(policy.parameters())

#####
#used for debugging
TD_targs = []
valHist = []
#####

currEpisodeNum = 0                           # initialize episode counter
MAX_TIME_STEPS = 50        # how many data tuples to store
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
stateHist = []
startPos = []

t0 = time.time()
for currEpisodeNum in range(NUM_EPISODES):
    refInd = np.random.randint(0, refTargets.shape[0])
    #refInd = 0
    
    #obs['desired_goal'] = targetPos
    
    D = np.empty((SIZE_OF_DATA), dtype=list)
    
    startTimeStep = np.random.randint(0, MAX_TIME_STEPS - 1)
    if NUM_EPISODES - currEpisodeNum <= 100:
        startTimeStep = 0
    targetPos = refTargets[refInd]
    obs = env.reset(cursorPos[refInd, startTimeStep], refVel[refInd, startTimeStep] / dt , targetPos)
    
    policy.resetState()
    
    X = np.zeros((MAX_TIME_STEPS, 2))
    Vel = np.zeros((MAX_TIME_STEPS, 2))
    #stateHist = np.zeros((MAX_TIME_STEPS, observationDim))
    actionHist = np.zeros((MAX_TIME_STEPS, actionDim))
    rewardHist = np.zeros(MAX_TIME_STEPS)
    for currTimeStep in range(startTimeStep, MAX_TIME_STEPS):
        old_obs = obs
        x = obs[0:2]
        X[currTimeStep] = x # obs['observation'][0:3]
        oldvel = obs[2:4]*dt
        
        #pdb.set_trace()
        action = policy(obs)
        action = action + delta*torch.randn(actionDim, device=device)
        action = action.cpu().detach().numpy().reshape(actionDim)
        
        actionHist[currTimeStep] = action
        obs = env.step(actionScale*action)
        
        ### direction reward
        vel = env.v*0.022
        #print(np.linalg.norm(vel))
        Vel[currTimeStep] = oldvel
        direction = targetPos - x
        if np.linalg.norm(direction) == 0:
            pdb.set_trace()
        direction /= np.linalg.norm(direction) + 1e-12
        #reward = np.dot(vel, direction)/np.linalg.norm(vel)
        taskreward = np.exp(-5*(1 - np.dot(oldvel, direction)/(np.linalg.norm(oldvel)+1e-12))) + np.exp(-10*(np.linalg.norm(targetPos - x))**2)
        imitationreward = np.exp(-25*np.linalg.norm(oldvel - refVel[refInd, currTimeStep])**2) + np.exp(-5*np.linalg.norm(x - cursorPos[refInd, currTimeStep])**2)
        reward = 0.5*taskreward + 0.5*imitationreward
        #reward /= 4
        #reward = taskreward
        rewardHist[currTimeStep] = reward
        ###
        D[currTimeStep] = {'state' : old_obs, 'next_state': obs, 
                           'action' : action, 'reward' : reward}
        done = False
        if done:
            #print(currEpisodeNum, currTimeStep)
            break
    #
    #pdb.set_trace()
    #print(X[startTimeStep])
    D = D[startTimeStep:currTimeStep+1]
    done = False
    
    TD_target, state, actionPlayed = computeTDBatch(D)
    #TD_target2, state2, actionPlayed2 = computeTDBatch2(D)
    #pdb.set_trace()
    valueEstimate = value(state)
    valueHist.append(valueEstimate.detach().numpy())
    #print(TD_target.shape, valueEstimate.shape)
    loss = valueLoss(TD_target, valueEstimate)
    valueLossHist.append(loss.item())
    stateHist.append(state)
    
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
    #oldPolicyNetwork = copy.deepcopy(policy.state_dict())
    #pdb.set_trace()
    policy.resetState()
    oldPolicyProbs = probOfAction(policy, state, torch.FloatTensor(actionPlayed), delta=delta).detach()
    #pdb.set_trace()
    
    advantage = (TD_target - valueEstimate).detach()
    
    #if currEpisodeNum % 10 == 0:
    #    pdb.set_trace()
    
    for k in range((50-startTimeStep) // 5 + 1):
        #print(k)
        policyOptimizer.zero_grad()
        batchInds = np.random.choice(np.arange(D.shape[0]), size=np.minimum(20, D.shape[0]), replace=False)
        
        oldprobs = oldPolicyProbs[batchInds]
        policy.resetState()
        newprobs = probOfAction(policy, state, torch.FloatTensor(actionPlayed), delta=delta)[batchInds]
        #policyLoss = clipLoss(policy, oldPolicyNetwork, state[batchInds], torch.FloatTensor(actionPlayed[batchInds]), advantage[batchInds], delta=delta)
        #policyLoss += torch.mean(F.relu(torch.abs(policy(torch.FloatTensor(stateHist[batchInds])))*actionScale - 1)**2)
        policyLoss = clipLossProbs(newprobs, oldprobs, advantage[batchInds], eps=epsilon)
        policyLoss.backward()
        policyOptimizer.step()
    
    #policyLoss = -(torch.FloatTensor(rewardHist))*torch.log(probOfAction(policy, stateHist, torch.FloatTensor(actionHist), delta=delta))
    ##policyLoss = -(TD_target.detach() - 1)*torch.log(probOfAction(policy, stateHist, torch.FloatTensor(actionHist)))
    #policyLoss = torch.mean(policyLoss)
    #policyLoss.backward()
    #policyOptimizer.step()
    #pdb.set_trace()
    
    #finalDist.append(np.sqrt(np.sum((obs['achieved_goal'] - obs['desired_goal'])**2)))
    finalDist.append(np.linalg.norm(env.target - env.x))
    
    testvecpol = torch.norm(policy(testvec)).item()
    #print(currEpisodeNum, np.round(testvecpol, 3), np.round(finalDist[-1], 3))
    if currEpisodeNum % 10 == 0:
        print(currEpisodeNum, np.round(np.arctan2(refTargets[refInd][1], refTargets[refInd][0]), 3), startTimeStep,
              np.round(finalDist[-1], 3), np.round(np.mean(np.linalg.norm(actionHist, axis=1)), 3), np.round(policyLoss.item(), 3), 
             np.round(testvecpol, 3), np.round(value(testvec).item(), 3))
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
datadict['stateHist'] = stateHist
sio.savemat('data_' + str(NUM_EPISODES) + 'iter.mat', datadict)

pickle.dump(value, open('value_' + str(NUM_EPISODES) + '.bin', 'wb'))
pickle.dump(policy, open('policy_' + str(NUM_EPISODES) + '.bin', 'wb'))