import gym
import quanser_robots
import numpy as np
import random as rnd
import matplotlib.pyplot as plt

# dynamic programming algorithm
# steps:
# discretize state and action spaces (TODO: try better discretizations)
# gather data for regression using exploration policy (TODO: try better exploration policies)
# learn dynamics function
# learn reward function
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
    return np.array(discSample)

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
    return np.array(samples)

# generates an array of random phase shifts for fourier features
def getphi(I):
    phi = []
    for i in range(I):
        phi.append(rnd.random() * 2 * np.pi - np.pi)
    return np.array(phi)

# generates a matrix of random weights for fourier features
def getP(I,J):
    P = []
    for i in range(I):
        Pi = []
        for j in range(J):
            Pi.append(rnd.gauss(0,1))
        P.append(Pi)
    return np.array(P)

# computes the fourier features for a state observation
# args:
# o - state observation
# P - weight matrix
# v - wavelength
# phi - phase shifts
# numfeat - number of fourier features to be generated
def fourier(o, P, v, phi, numfeat):
    y = []
    for i in range(numfeat):
        arg = 0
        for j in range(len(o)):
            arg += P[i][j] * o[j]
        arg /= v
        arg += phi[i]
        y.append(np.sin(arg))
    return np.array(y)

# computes the feature matrix for fourier regression
def featuremat(samples, numfeat, fourierparams):
    numobs = len(samples[0][0])
    numact = len(samples[0][1])
    numsamp = len(samples)
    P = fourierparams[0]
    v = fourierparams[1]
    phi = fourierparams[2]
    mat = np.ones([numsamp,numfeat])
    for i in range(numsamp):
        curr_samp = samples[i]
        x = np.append(curr_samp[0], curr_samp[1])
        mat[i] = fourier(x, P, v, phi, numfeat)

    return mat

# executes fourier regression for the given samples
# theta[0] contains the parameters for the reward function
# theta[1] contains the parameters for the dynamics function
def regression(samples, fourierparams):
    theta = []
    numfeat = fourierparams[3]
    features = featuremat(samples, numfeat, fourierparams)
    #print(features)
    feat_inverse = np.linalg.pinv(features)
    #print(feat_inverse)
    rewards = samples[...,2]
    #print(rewards)
    next_states = np.array(list(samples[...,3]))
    #print(next_states.shape)
    theta_r = np.dot(feat_inverse,rewards)
    #print(theta_r.shape)
    #print(theta_r)
    theta.append(theta_r)
    theta_dyn = np.dot(feat_inverse,next_states)
    #print(theta_dyn.shape)
    #print(theta_dyn)
    theta.append(theta_dyn)
    #print(theta)
    return theta

def getReward(obs, act, theta, fparams):
    x = np.append(obs,act)
    fx = fourier(x, fparams[0], fparams[1], fparams[2], fparams[3])
    return np.dot(theta[0], fx)

def getNextState(obs, act, theta, fparams, discStates):
    x = np.append(obs,act)
    fx = fourier(x, fparams[0], fparams[1], fparams[2], fparams[3])
    theta_s = theta[1]
    #print(theta_s[...,0])
    newState = []
    for i in range(len(obs)):
        newState.append(np.dot(theta_s[...,i], fx))
    #print(newState)
    return discretize(np.array(newState), discStates)


# !!! WIP -- DO NOT USE !!!
# Planes the optimal policy for a MDP defined by:
# states:   3xNo(DiscSteps) array containing cos(th), sin(th), dot(th)
# action:   1xNo(DiscSteps) array containing the torque
# theta:    1xNo(Features) array containing the coefficients for the fourier-approximation
# fparams:  parameters for evaluation of fourier approximation
def train_policy(states, actions, theta, fparams):
    # Setup value-function
    # length = No(all possible states)
    state_dim_len = states.shape[1]
    valFun = np.zeros((state_dim_len, state_dim_len, state_dim_len))

    # Contains amount of torque
    # if t == np.inf torque is random
    policy = np.full(valFun.shape, np.inf)

    # iterate
    oldPolicy = np.full(valFun.shape, np.inf)
    tempValFun = np.zeros((state_dim_len, state_dim_len, state_dim_len))
    updates = 0
    # polTorque = 0
    gamma = 0.9
    transFun = np.zeros((valFun.shape + (3,)))
    imReward = np.zeros(valFun.shape)

    while updates < 1000:
        oldPolicy = policy
        # policy evaluation
        for i in range(valFun.shape[0]):
            for j in range(valFun.shape[1]):
                for k in range(valFun.shape[2]):
                    if np.isinf(policy[i][j][k]):
                        imReward[i][j][k] = 0
                        for act in range(len(actions[0])):
                            imReward[i][j][k] += getReward((states[0][i], states[1][j], states[2][k]), actions[0][act], theta, fparams)/len(actions[0])
                        transFun[i][j][k] = getNextState((states[0][i], states[1][j], states[2][k]), np.random.choice(actions[0]), theta, fparams, states)

                    else:
                        imReward[i][j][k] = getReward((states[0][i], states[1][j], states[2][k]), policy[i][j][k], theta, fparams)
                        transFun[i][j][k] = getNextState((states[0][i], states[1][j], states[2][k]), policy[i][j][k], theta, fparams, states)

                    # tempValFun = imReward[i][j][k] + gamma*valFun[transFun[i][j][k][0], transFun[i][j][k][1], transFun[i][j][k][2]]
        valFun = tempValFun
        # valFun = policy_evaluation()

        # greedy policy improvement
        # policy = policy_greedy_update()
        updates += 1;

    # return optimal policy lookup-table
    return policy

def main():
    # false for pendulum, true for qube
    qube = False
    env = makeEnv(qube)
    discActions = discretizeSpace(env.action_space, 100)
    discStates = discretizeSpace(env.observation_space, 10)
    #print(discActions)
    #print(discStates)
    samples = explore(env, 10000, discActions, discStates)
    #print(len(samples))
    #print(samples[0])
    numobs = len(samples[0][0])
    numact = len(samples[0][1])
    numfeat = 20
    P = getP(numfeat,numobs+numact)
    v = 10
    phi = getphi(numfeat)
    fourierparams = [P, v, phi, numfeat]

    theta = regression(samples, fourierparams)

    # for visualizing the regression
    x = np.ones([4,len(samples)])
    yp = np.ones([len(samples)])
    for i in range(len(samples)):
        x[0][i] = samples[i][0][0]
        x[1][i] = samples[i][0][1]
        x[2][i] = samples[i][0][2]
        x[3][i] = samples[i][1][0]
        yp[i] = np.dot(theta[0], fourier(x[...,i], P, v, phi, numfeat))
    test1 = getReward(samples[0][0], samples[0][1], theta, fourierparams)
    test2 = getNextState(samples[0][0], samples[0][1], theta, fourierparams, discStates)
    print(test1)
    print(test2)
    # Planning via dynamic programming
    policy = train_policy(discStates, discActions, theta, fourierparams)

    # plt.scatter(x[0], np.abs(samples[...,2] - yp))
    # plt.scatter(x[0], yp)
    # plt.scatter(x[0], samples[...,2] )
    # plt.show()


# For automatic execution
main()
