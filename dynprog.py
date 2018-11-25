import gym
import quanser_robots
import numpy as np
import random as rnd

# dynamic programming algorithm
# steps:
# discretize state and action spaces (TODO: try better discretizations)
# gather data for regression using exploration policy (TODO: try better exploration policies)
# TODO: learn dynamics function
# TODO: learn reward function
# TODO: policy iteration / value iteration
# TODO: plot results


# returns a gym environment depending on the boolean input
# qube for true, pendulum for false
def makeEnv(whichEnv):
    if whichEnv:
       return gym.make('Qube-v0')
    else:
       return gym.make('Pendulum-v0')

# returns a uniform discrete space
# args:
# space - continuous space to be discretized
# grain - number of desired discrete classes for each dimension of the space
def discretizeSpace(space, grain):
    shape = space.shape
    highs = space.high
    lows = space.low
    discSpace = np.ones([shape[0],grain+1])
    for i in range(shape[0]):
        step = (highs[i] - lows[i]) / grain
        for j in range(grain+1):
            discSpace[i][j] = lows[i] + j * step
    return discSpace

# classifies a continuous sample for a discrete space
# args:
# sample - continuous sample to be discretized
# space - discrete space
def discretize(sample, space):
    discSample = []
    for i in range(len(sample)):
        entry = sample[i]
        for j in range(len(space[i])):
            step = space[i][j]
            if entry == step:
                discSample.append(step)
                break
            elif entry < step:
                prev = space[i][j-1]
                if np.abs(entry - prev) > np.abs(entry - step):
                    discSample.append(step)
                else:
                    discSample.append(prev)
                break
    return discSample

# gaussian exploration policy (for pendulum only)
# obs - current state
def exploration_policy(obs):
    return [min(2.0, max(-2.0, rnd.gauss(0,1)))]

# generate a number of samples for regression, using an exploration policy
# env - learning environment
# numSamples - number of samples to be generated
# dA - discrete action space
# dS - discrete state space
def explore(env, numSamples, dA, dS):
    samples = []
    while len(samples) < numSamples:
        done = False
        obs = env.reset()
        obs = discretize(obs, dS)
        while not done:
            s = [obs]
            a = discretize(exploration_policy(obs), dA)
            s.append(a)
            obs, r, done, info = env.step(a)
            obs = discretize(obs, dS)
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
    samples = explore(env, 10000, discActions, discStates)
    print(len(samples))
    print(samples[0])

# For automatic execution
main()

