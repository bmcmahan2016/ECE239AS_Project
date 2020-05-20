# -*- coding: utf-8 -*-
"""
Created on Sun May 17 17:14:47 2020

@author: bmcma

DESCRIPTION:
    Train an agent to perform the center out reach task in the FetchReach-v1
    open AI gym environmnet
"""

import numpy as np
import gym

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

NUM_EPISODES = 20        # number of episodes we will train on
lr = 1e-3               # learning rate for Q-learning
discountFactor = 0.5    # discount factor 

env = gym.make('FetchReach-v1')
obs = env.reset()
# manual over-ride of target position
setTargetPosition(env)
done = False

theta = np.random.randn((40))     # linear function approximation vector
actionTheta = np.zeros((4))
nextActionTheta = np.zeros((4))



def policy(actionTheta):
    # Here you would implement your smarter policy. In this case,
    # we just sample random actions.
    # I am first going to try to do greedy Q action selection
    randomAction = env.action_space.sample()
    T = np.random.rand()
    thetaQ = theta[-4:]
    greedyAction = np.sign(thetaQ)
    if T > 0.8:
        return randomAction
    else:
        return greedyAction

currEpisodeNum = 0       # initialize episode counter
lastState = env.reset()['observation'].reshape(-1,1)
actionTheta[0] = np.matmul(theta[:10], lastState)
actionTheta[1] = np.matmul(theta[10:20], lastState)
actionTheta[2] = np.matmul(theta[20:30], lastState)
actionTheta[3] = np.matmul(theta[-10:], lastState)

while currEpisodeNum < NUM_EPISODES:
    print('current episode #', currEpisodeNum)
    lastAction = policy(actionTheta)
    obs, reward, done, info = env.step(lastAction)
    env.render()
    if done:
        currEpisodeNum += 1
        done = False
        env.reset()
        setTargetPosition(env)
        
    # update w with Q-learning target
    lastState = lastState.reshape(-1,1)
    newState = obs['observation'].reshape(-1,1)
    
    nextActionTheta[0] = np.matmul(theta[:10], newState)
    nextActionTheta[1] = np.matmul(theta[10:20], newState)
    nextActionTheta[2] = np.matmul(theta[20:30], newState)
    nextActionTheta[3] = np.matmul(theta[-10:], newState)
    
    nextAction = policy(actionTheta).reshape(-1,1)
    
    lastAction = lastAction.reshape(-1,1)
    Q_sa_next = np.matmul(nextActionTheta, nextAction)
    Q_sa_curr = np.matmul(actionTheta,  lastAction)
    
    theta = theta - 2*lr*(reward+discountFactor*Q_sa_next-Q_sa_curr)
    lastState = newState
    actionTheta = nextActionTheta

    # If we want, we can substitute a goal here and re-compute
    # the reward. For instance, we can just pretend that the desired
    # goal was what we achieved all along.
    substitute_goal = obs['achieved_goal'].copy()
    substitute_reward = env.compute_reward(
        obs['achieved_goal'], substitute_goal, info)
    print('reward is {}, substitute_reward is {}'.format(
        reward, substitute_reward))
    print('theta', theta)