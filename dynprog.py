import gym
import quanser_robots
import numpy as np
import random as rnd

def makeEnv(whichEnv):
    if whichEnv:
       return gym.make('Qube-v0')
    else:
       return gym.make('Pendulum-v0')

def discretizeSpace(space, grain):
    shape = space.shape
    highs = space.high
    lows = space.low
    discSpace = np.ones([shape[0],grain])
    for i in range(shape[0]):
        step = (highs[i] - lows[i]) / grain
        for j in range(grain):
            discSpace[i][j] = lows[i] + j * step
    return discSpace

def exploration_policy(obs):
    return [rnd.gauss(0,1)]

def explore(env, numSamples):
    samples = []
    while len(samples) < numSamples:
        done = False
        obs = env.reset()
        while not done:
            s = [obs]
            a = exploration_policy(obs)
            s.append(a)
            obs, r, done, info = env.step(a)
            s.append(r)
            s.append(obs)
            samples.append(s)
    return samples

def main():
    # false for pendulum, true for qube
    qube = False
    env = makeEnv(qube)
    discActions = discretizeSpace(env.action_space, 10)
    discStates = discretizeSpace(env.observation_space, 10)
    print(discActions)
    print(discStates)
    samples = explore(env, 10000)
    print(len(samples))
    print(samples[0])

# discretize state and action spaces
# gather data for regression using exploration policy
# learn dynamics function
# learn reward function
# policy iteration / value iteration
# plot results
