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

# comment line below if you are running without cuda
#device = torch.device("cuda")
device = torch.device('cpu')


class valueNet(nn.Module):
    
    def __init__(self):
        super(valueNet, self).__init__()
        self.fc1 = nn.Linear(6, 8)
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
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
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
        self.fc1 = nn.Linear(6, 8)
        self.fc2 = nn.Linear(8, 8)
        self.fc3 = nn.Linear(8, 4)
        self.fc1.weight.data = torch.randn_like(self.fc1.weight.data)
        self.fc2.weight.data = torch.randn_like(self.fc2.weight.data)
        self.fc3.weight.data = torch.randn_like(self.fc3.weight.data)
        
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
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        #action = x.cpu().detach().numpy().reshape(4)
        actionMean = x #+ torch.randn(4, device=device)*self.var
        return actionMean #+ self.var*np.random.randn(4)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def syncPolicies(oldPolicy, currPolicy, currEpisodeNum):
    '''synchronizes old policy with the new policy after 
    every 20 episodes'''
    if currEpisodeNum %20 == 0:
        oldPolicy.load_state_dict(currPolicy.state_dict())

def createIntegratedTrajectory():
    REFERENCE_DURATION = env._max_episode_steps
    trajectoryX = np.linspace(-3, 3, REFERENCE_DURATION)
    trajectoryGauss = 1/np.sqrt(2*np.pi)*np.exp(-0.5*(trajectoryX**2))
    trajectoryIntegrated = trajectoryGauss
    trajectoryIntegrated = np.zeros((REFERENCE_DURATION))
    for tStep in range(1, REFERENCE_DURATION):
        trajectoryIntegrated[tStep] = np.sum(trajectoryGauss[:tStep])      # could multiply this by dt to incorporate custom units
    trajectoryIntegrated /= trajectoryIntegrated[-1]   
    return trajectoryIntegrated            

def fetchImitationMotion(initialPos, targetPos):
    '''
    initialPos: numpy array with shape (3,). Represents the starting position of the 
    robot gripper
    
    targetPos: numpy arrray with shape (3,) that represents the desired target position
    the agent strives to reach

    Returns
    -------
    referenceMotion: a numpy array with shape (3, REFERENCE_DURATION) that contians
    the target reference motions at everytimestep

    '''
    REFERENCE_DURATION = env._max_episode_steps    
    referenceMotion = np.zeros((3, REFERENCE_DURATION))
    
    xScale = targetPos[0] - initialPos[0]
    yScale = targetPos[1] - initialPos[1]
    referenceMotion[0,:] = xScale * trajectoryIntegrated + initialPos[0]      # Gaussian velocity in x
    referenceMotion[1,:] = yScale * trajectoryIntegrated + initialPos[1]      # Gaussian velocity in y
    referenceMotion[2,:] = np.linspace(initialPos[2], targetPos[2], REFERENCE_DURATION)      # no movement in z-plane
    return referenceMotion

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
    scaleFactor = 0.2   
    
    if targetPos is None:
        #np.random.seed(41)
        targetIX = np.random.randint(0, 8)
        theta = np.pi*targetIX/4

        reachDirection = np.array([np.cos(theta), np.sin(theta), 0])

        # place the target at the random reach position    
        reachTargetPos = centerPosition + scaleFactor * reachDirection
    else:
        reachTargetPos = centerPosition + scaleFactor * targetPos
        
    # manual place target at desired location
    referenceMotion = fetchImitationMotion(centerPosition, reachTargetPos)
    env.env.goal = reachTargetPos
    return reachTargetPos, referenceMotion, targetIX

def getFeatureVector(observation):
    '''
    DESCRIPTION: takes an observation dictionary from FetchReach-v1 environment
                and converts it into a feature vector to be used in Policy and 
                value networks. 
    Parameters
    ----------
    observation : dictionary obejct generated by calling env.step(action)
    '''
    g = 5     # input feature gain
    stateFeatures = np.hstack((g*(observation['achieved_goal']-env.env.initial_gripper_xpos), g*(observation['desired_goal']-env.env.initial_gripper_xpos)))
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
    return Gt, state, sampleTuples[tupleIX]['action']
    
def computeTDBatch(sampleTuples, lmbd=0.95, tupleIX=None):
    if tupleIX is None:
        tupleIX = np.arange(len(sampleTuples))
    L = len(tupleIX)
    Gts = torch.zeros((L, 1))
    states = np.zeros((L, 6))
    actions = np.zeros((L, 4))
    for l in range(L):
        Gt, state, action = computeTD(sampleTuples, lmbd, tupleIX[l])
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

def GaussianProbability(x, mean, cov):
    '''
    computes the probability for a Gaussian distribution with mean and a diagonal
    covariance matrix with a single value, cov, along the diagonal.
    
    Parameters
    ----------
    x : TPPE torch tensor having shape [4, N], where N is the number of samples 
    or batch size that we are considering
        DESCRIPTION represents an action
    mean : TYPE torch tensor with shape [4, N], where N is the number of samples 
    or batch size that we are considering
        DESCRIPTION is the output of the policy network for a given state 
        corresponding to the  mean action.
    cov : TYPE float
        DESCRIPTION diagonal value of the covariance matrix

    Returns
    -------
    actionProbability: torch tensor with shape (1,1) representing the probability
    of action a being played under policy from the given state

    '''
    dim = x.shape[0]     # action space dimension is 4
    # exponent is off by 1/cov
    sigma = torch.eye(dim).double()
    sigma = sigma/cov
    coef = (2*np.pi)**(-dim/2.0) * (cov**dim)**(-1/2)
    mx = x-mean
    gausExp = torch.exp(-mx.t() @ sigma @ mx/2)
    actionProbability = coef * gausExp
    actionProbability = torch.diag(actionProbability)
    return actionProbability

#def probOfAction(action, state, policy):
    
def getImitationReward(obs, referenceMotion, timeStep):
    if timeStep >= 50:
        return 0
    gripperPos = obs['achieved_goal']
    referencePos = referenceMotion[:,timeStep]
    distance = np.sqrt(np.sum( (gripperPos - referencePos)**2 ))
    if distance < 0.1:
        return 1
    else:
        return 0
    
    
def updatePolicy(actionHistory, stateHistory, advantageHistory, oldPolicy, currPolicy):
    '''
    compute the gradients for the policy network using the PPO loss, must call optimizer.zero_grad()
    and optimizer.step() before and after this function, as this function does not call the optimizer.
    
    Parameters
    ----------
    actionHistory : TYPE numpy array having shape [N, 4], where N is the number of timesteps 
    in the last episode
        DESCRIPTION represents action taken at every timestep in preceeding episdoe
        
    stateHistory : TYPE numpy array with shape [N, 16], where N is the number of time 
    steps in the past episode
        DESCRIPTION contains the state of the agent+environment at everytimestep of 
        previous episode
    advantageHistory : TYPE torch tensor with shape [N, 1]
        DESCRIPTION value of advantage function at everytime step N of previous episode
    
    policyNetwork: torch nn object used to estiamte policy

    Returns
    -------
    none.

    '''
    actionHistory = torch.from_numpy(actionHist).t()
    currActionMeans = currPolicy(stateHistory).t()
    oldActionMeans = oldPolicy(stateHistory).t()
    currActionProbabilities = GaussianProbability(actionHistory, currActionMeans, 0.1)
    oldActionProbabilities = GaussianProbability(actionHistory, oldActionMeans, 0.1)
    #unclippedPPO = -actionProbabilities @ advantageHistory.double()
    
    eps = 0.2 # this is value used in DeepMimic as well as original PPO paper
    p1 = currActionProbabilities / oldActionProbabilities
    p2 = torch.clamp(p1, 1-eps, 1+eps)
    
    PPO_loss = -torch.min(p1@advantageHistory.double(), p2@advantageHistory.double()) / advantageHistory.shape[0]
    PPO_loss.backward()
    return PPO_loss
    #print("ppo loss:", PPO_loss)
    #unclippedPPO.backward()
    
    # clipped loss has been giving me buggy results so I am not using it right now
    # may need to tune epsilon hyperparameter more to resolve this
    # currently I am not using the clipped loss, to do so just uncomment next line
    #PPOloss = torch.min(actionProbabilities @ advantageHistory.double(), \
    #          torch.clamp(actionProbabilities, 1-eps, 1+eps)@advantageHistory.double())
    #PPOloss.backward()
    #unclippedPPO.backward()
    

# define some values
NUM_EPISODES = 500_000          # number of episodes we will train on
lr_pol = 1e-2                  # this was 1e-2 in poster                  # learning rate for Q-learning
lr_val = 1e-2
discountFactor = 1             # discount factor
delta = 0.1                    # standard deviation of additive action noise
eps = 0.2

# create the environment
env = gym.make('FetchReach-v1')
# create policy and value networks
currPolicy = policyNet().to(device)       # the current policy we want to optimize
oldPolicy = policyNet().to(device)        # the old policy we used to collect the data
value = valueNet().to(device)
       


valueLoss = nn.MSELoss()            # MSE loss is used for value network
# using same learning rate for both networks
valueOptimizer = optim.SGD(value.parameters(), lr=lr_val)          # SGD is used for optimization of value network
policyOptimizer = optim.SGD(currPolicy.parameters(), lr=lr_pol)

#####
#used for debugging
TD_targs = []
valHist = []
#####

env._max_episode_steps = 10
trajectoryIntegrated = createIntegratedTrajectory()
currEpisodeNum = 0                           # initialize episode counter
MAX_TIME_STEPS = env._max_episode_steps      # how many data tuples to store
SIZE_OF_DATA = MAX_TIME_STEPS                # same variable, different name, no good reason why
D = np.empty((SIZE_OF_DATA), dtype=list)     # holds the episode tuples
currTimeStep = 0

# empty lists to hold distance for each target
finalDist = np.empty((8,), dtype=list)
Xhist = np.empty((8,), dtype=list)
for i in range(8):
    finalDist[i] = []
    Xhist[i] = []
    
testvec = np.random.randn(1, 16)
testvechist = []
targetHist = []
valueHist = []
lossHist = []
valLossHist = []
advantageNormHist = []
totalRewardHist = []

t0 = time.time()
for currEpisodeNum in range(NUM_EPISODES):
    syncPolicies(oldPolicy, currPolicy, currEpisodeNum)    # synchronize the policies
    obs = env.reset()
    #env.env.distance_threshold = 0.1                       # how close agent must get to target
    targetOveride = np.array((.5, 0.5, 0.5))
    targetPos, referenceMotion, targetIX = setTargetPosition(env)
    done = False
    
    D = np.empty((SIZE_OF_DATA), dtype=list)
    
    X = np.zeros((MAX_TIME_STEPS, 3))
    Vel = np.zeros((MAX_TIME_STEPS, 3))
    stateHist = np.zeros((MAX_TIME_STEPS, 16))
    actionHist = np.zeros((MAX_TIME_STEPS, 4))
    totalReward = 0
    for currTimeStep in range(MAX_TIME_STEPS):
        #env.render()
        old_obs = obs
        X[currTimeStep] = obs['observation'][0:3]
        action = oldPolicy(getFeatureVector(obs))
        action = action + delta*torch.randn(4, device=device)
        action = action.cpu().detach().numpy().reshape(4)
        
        actionHist[currTimeStep] = action
        obs, reward, done, info = env.step(action)
        rewardImitation = getImitationReward(obs, referenceMotion, currTimeStep)
        rewardImitation = 0   # uncoment for imitation rewards
        #print("Imitation reward:", rewardImitation)
        if reward == 0:
            reward += 100  # very large reward for hitting target
            done = True    # terminates episode when you reach target
        Vel[currTimeStep] = obs['observation'][0:3] - X[currTimeStep]
        #reward += 5*rewardImitation
        #print("total reward:", reward)
        totalReward += (reward - 5*rewardImitation)    # increment total reward collected for this episode

        ### direction reward  -- Brandon commented this out just for the sake of debuging the training without any aditional rewards
        vel = Vel[currTimeStep]
        direction = targetPos - obs['achieved_goal']
        direction /= np.linalg.norm(direction)
        #reward += np.dot(vel, direction)/np.linalg.norm(vel)

        D[currTimeStep] = {'state' : getFeatureVector(old_obs), 'next_state': getFeatureVector(obs), 
                           'action' : action, 'reward' : reward}
        if done:
            D = D[0:currTimeStep+1]
            actionHist = actionHist[0:currTimeStep+1,:]
            print("Completed episode #:", currEpisodeNum, "........... in", currTimeStep, "timesteps")
            totalRewardHist.append(totalReward)
            #print("action mean:", action)
            #print("grad", currPolicy.fc1.weight.grad)
            break
    #
    done = False
    X = X[0:currTimeStep+1]         # initialized zero positions removed when target is reached early
    
    TD_target, state, actionPlayed = computeTDBatch(D)
    valueEstimate = value(state)
    #valueHist.append(valueEstimate.detach().numpy())
    loss = valueLoss(TD_target, valueEstimate)
    #valLossHist.append(loss.detach().numpy())
    
    valueOptimizer.zero_grad()
    loss.backward()
    valueOptimizer.step()
    
    Xhist[targetIX].append(X)
    
    # update the policy network
    advantage = (TD_target - valueEstimate).detach()
    policyOptimizer.zero_grad()
    updatePolicy(actionHist, state, advantage, oldPolicy, currPolicy)
    policyOptimizer.step()
    advantageNormHist.append(np.linalg.norm(advantage))
    
    
    finalDist[targetIX].append(np.sqrt(np.sum((obs['achieved_goal'] - obs['desired_goal'])**2)))

plt.figure(1)
colors = ['c', 'm', 'y', 'orange', 'k', 'r', 'b', 'g']
for i in range(8):
    plt.plot(np.array(finalDist[i]), c=colors[i])
plt.plot(np.linspace(0,1300,1300), np.linspace(0.05,0.05,1300), c='k', alpha=0.5)
# use below line to copy data into old model
#oldPolicy.load_state_dict(currPolicy.state_dict())

def plotReaches(Xhist, env):
    center = env.env.initial_gripper_xpos
    SCALE_FACTOR = 100
    colors = ['c', 'm', 'y', 'orange', 'k', 'r', 'b', 'g']
    
    # get the target positions for each reach condition
    targetPos = np.zeros((8, 2))
    for targetIX in range(8):
        theta = np.pi*targetIX/4
        reachDirection = np.array([np.cos(theta), np.sin(theta), 0])
        reachTargetPos = 0.2 * reachDirection
        targetPos[targetIX, 0] = SCALE_FACTOR*reachTargetPos[0]
        targetPos[targetIX, 1] = SCALE_FACTOR*reachTargetPos[1]
        
    plt.figure(2)
    for targetIX in range(8):
        plt.scatter(targetPos[targetIX,0], targetPos[targetIX,1], s=1000, c=colors[targetIX])
        for trialNum in range(1, len(Xhist[targetIX]), 100):
            plt.plot(SCALE_FACTOR*(Xhist[targetIX][trialNum][:,0]-center[0]), SCALE_FACTOR*(Xhist[targetIX][trialNum][:,1]-center[1]), c=colors[targetIX], alpha=0.5)
    plt.xlim([-23,23])
    plt.ylim([-23,23])
    plt.title("all reaches")
    
    # first 100 reaches
    plt.figure(3)
    for targetIX in range(8):
        plt.scatter(targetPos[targetIX,0], targetPos[targetIX,1], s=100, c=colors[targetIX])
        for trialNum in range(1):
            plt.plot(SCALE_FACTOR*(Xhist[targetIX][trialNum][:,0]-center[0]), SCALE_FACTOR*(Xhist[targetIX][trialNum][:,1]-center[1]), c=colors[targetIX], alpha=0.5)
    plt.xlim([-23,23])
    plt.ylim([-23,23])
    plt.title("first 100 reaches")
    
    # last 100 reaches
    plt.figure(4)
    for targetIX in range(8):
        plt.scatter(targetPos[targetIX,0], targetPos[targetIX,1], s=1000, c=colors[targetIX])
        for trialNum in range(1, 4):
            plt.plot(SCALE_FACTOR*(Xhist[targetIX][-trialNum][:,0]-center[0]), SCALE_FACTOR*(Xhist[targetIX][-trialNum][:,1]-center[1]), c=colors[targetIX], alpha=0.5)
    plt.xlim([-23,23])
    plt.ylim([-23,23])
    plt.title("Final 100 reaches")
plotReaches(Xhist, env)

fName = "FetchreachAgent_" + str(time.time()) +".pt"
torch.save({"policy_fc1" : oldPolicy.fc1.weight.data,  "policy_fc2":oldPolicy.fc2.weight.data, "policy_fc3" : oldPolicy.fc3.weight.data, \
            "value_fc1" : value.fc1.weight.data, "value_fc2" : value.fc2.weight.data, "value_fc3" : value.fc3.weight.data, \
                "Xhist" : Xhist, "totalReward" : totalRewardHist, "Advantage" : advantageNormHist, "finalDist" : finalDist}, fName)