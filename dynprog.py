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
# policy iteration / value iteration
# TODO: plot results



# returns a gym environment
# whichEnv: Boolean --- qube for true, pendulum for false
def makeEnv(whichEnv):
    if whichEnv:
       return gym.make('Qube-v0')
    else:
       return gym.make('Pendulum-v0')

# returns a uniform discrete space
# space:    continuous space to be discretized
# grain:    number of desired discrete classes for each dimension of the space
#           TODO: length of disc. space is 1 longer then indicated by grains
def discretize_space(space, grain):
    shape = space.shape
    highs = space.high
    lows = space.low
    discSpace = np.ones([shape[0],grain])
    for i in range(shape[0]):
        step = (highs[i] - lows[i]) / (grain - 1)
        for j in range(grain):
            discSpace[i][j] = lows[i] + j * step
    return discSpace

# classifies a continuous sample for a discrete space
# sample:   continuous sample to be discretized
# space:    discrete space
def discretize_state(sample, space):
    discSample = []
    for i in range(len(sample)):
        entry = np.clip(sample[i], space[i][0], space[i][-1])
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
# obs:  current state
def exploration_policy(obs):
    return [min(2.0, max(-2.0, rnd.gauss(0,1)))]

# generate a number of samples for regression, using an exploration policy
# env:          learning environment
# numSamples:   number of samples to be generated
# actions:      discrete action space
# states:       discrete state space
def explore(env, numSamples, actions, states):
    samples = []
    while len(samples) < numSamples:
        done = False
        obs = env.reset()
        discState = discretize_state(obs, states)
        while not done:
            s = [discState]
            a = discretize_state(exploration_policy(obs), actions)
            s.append(a)
            obs, reward, done, info = env.step(a)
            discState = discretize_state(obs, states)
            s.append(reward)
            s.append(discState)
            samples.append(s)
    return np.array(samples)

# generates an array of random phase shifts for fourier features
# numFeats:     number of fourier features
def getphi(numFeats):
    phi = []
    for i in range(numFeats):
        phi.append(rnd.random() * 2 * np.pi - np.pi)
    return np.array(phi)

# generates a matrix of random weights for fourier features
# NumFeats:             number of requested fourier features
# sumNumStateActions:   sum of the number of States and number of Actions
def getP(numFeats, sumNumStateActions):
    P = []
    for i in range(numFeats):
        Pi = []
        for j in range(sumNumStateActions):
            Pi.append(rnd.gauss(0,1))
        P.append(Pi)
    return np.array(P)

# computes the fourier features for a state observation
# obs:      state observation
# P:        weight matrix
# v:        wavelength
# phi:      phase shifts
# numfeat:  number of fourier features to be generated
def fourier(obs, P, v, phi, numFeats):
    y = []
    for i in range(numFeats):
        arg = 0
        for j in range(len(obs)):
            arg += P[i][j] * obs[j]
        arg /= v
        arg += phi[i]
        y.append(np.sin(arg))
    return np.array(y)

# computes the feature matrix for fourier regression
# samples:          contains the samples from the regression
#                       [[discState, action, reward, nextDiscState]]
# numfeat:          number of fourier features to be generated
# fourierparams:    contains P, v, phi and the desired number of fourier features
def featuremat(samples, numFeats, fourierparams):
    numobs = len(samples[0][0])
    numact = len(samples[0][1])
    numsamp = len(samples)
    P = fourierparams[0]
    v = fourierparams[1]
    phi = fourierparams[2]
    mat = np.ones([numsamp, numFeats])
    for i in range(numsamp):
        curr_samp = samples[i]
        x = np.append(curr_samp[0], curr_samp[1])
        mat[i] = fourier(x, P, v, phi, numFeats)

    return mat

# executes fourier regression for the given samples and returns theta
# ---theta[0] contains the parameters for the reward function
# ---theta[1] contains the parameters for the dynamics function
# samples:          contains the samples from the regression
#                   ---[[discState, action, reward, nextDiscState]]
# fourierparams:    contains P, v, phi and the desired number of fourier features
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

# calculates the immediate reward for a state-action pair
# obs:      state observation
# act:      action taken
# theta:    coefficients for the fourier-approximation
# fparams:  misc parameters for the fourier features
def getReward(obs, act, theta, fparams):
    x = np.append(obs,act)
    fx = fourier(x, fparams[0], fparams[1], fparams[2], fparams[3])
    return np.dot(theta[0], fx)

# calculates the projected next state for a state-action pair
# state:    previous state tupel
# act:      action taken
# theta:    coefficients for the fourier-approximation
#           ---theta[0] contains the parameters for the reward function
#           ---theta[1] contains the parameters for the dynamics function
# fparams:  misc parameters for the fourier features
# states:   discretized space of states
def getNextState(state, act, theta, fparams, states):
    x = np.append(state, act)
    fx = fourier(x, fparams[0], fparams[1], fparams[2], fparams[3])
    theta_s = theta[1]
    #print(theta_s[...,0])
    newState = []
    for i in range(len(state)):
        newState.append(np.dot(theta_s[...,i], fx))
    #print(newState)
    return discretize_state(np.array(newState), states)


# Planes the optimal policy for a MDP defined by:
# states:   3xNo(DiscSteps) array containing cos(th), sin(th), dot(th)
# action:   1xNo(DiscSteps) array containing the torque
# theta:    1xNo(Features) array containing the coefficients for the fourier-approximation
# fparams:  misc parameters for evaluation of fourier approximation
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
    valFun = np.zeros((state_dim_len, state_dim_len, state_dim_len))
    updates = 0
    # polTorque = 0
    gamma = 0.9
    transFun = np.zeros((valFun.shape + (3,)))
    imReward = np.zeros(valFun.shape)
    policyStable = False
    #while udates < 10:
    while not policyStable:
        policyStable = True
        oldPolicy = policy
        # policy evaluation
        maxDelta = 1
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

        while maxDelta > 0.1:
            maxDelta = 0
            for i in range(valFun.shape[0]):
                for j in range(valFun.shape[1]):
                    for k in range(valFun.shape[2]):

                        newValFun = imReward[i][j][k] + gamma*valFun[read_index_for_state(states, 0, transFun[i][j][k][0])][read_index_for_state(states, 1, transFun[i][j][k][1])][read_index_for_state(states, 2, transFun[i][j][k][2])]
                        maxDelta = max(maxDelta, np.abs(newValFun - valFun[i][j][k]))
                        valFun[i][j][k] = newValFun
        # greedy policy improvement
        bestTorque = 0
        tempReward = -np.inf
        bestReward = -np.inf
        tempState = np.array((0., 0., 0.))
        for i in range(valFun.shape[0]):
            for j in range(valFun.shape[1]):
                for k in range(valFun.shape[2]):
                    bestReward = -np.inf
                    for act in actions[0]:
                        tempState =  getNextState((states[0][i], states[1][j], states[2][k]), act, theta, fparams, states)
                        tempReward = gamma * valFun[read_index_for_state(states, 0, tempState[0])][read_index_for_state(states, 1, tempState[1])][read_index_for_state(states, 2, tempState[2])]
                        tempReward += getReward((states[0][i], states[1][j], states[2][k]), act, theta, fparams)
                        if tempReward > bestReward:
                            bestTorque = act
                            bestReward = tempReward
                    stable = policy[i][j][k] == bestTorque 
                    if not stable:
                        policyStable = False
                    policy[i][j][k] = bestTorque
                    bestTorque = 0
        # policy = policy_greedy_update()

        updates += 1

    # return optimal policy lookup-table
    return policy


# Finds the index of a given stateValue in its discretized Space
# states:       discretized state space
# dimension:    state dimension in which the value is to be found
# stateVal:     the Value for which to find the index
def read_index_for_state(states, dimension, stateVal):
    dimLen = len(states[dimension])
    dimMax = states[dimension][dimLen-1]
    dimMin = states[dimension][0]
    step = (dimMax - dimMin)/(dimLen-1)
    index = (stateVal - dimMin) / step
    return int(round(index))

def main():
    # false for pendulum, true for qube
    qube = False
    env = makeEnv(qube)
    numActions = 20
    numStates = 10
    discActions = discretize_space(env.action_space, numActions)
    discStates = discretize_space(env.observation_space, numStates)
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
    #x = np.ones([4,len(samples)])
    #yp = np.ones([len(samples)])
    #for i in range(len(samples)):
    #    x[0][i] = samples[i][0][0]
    #    x[1][i] = samples[i][0][1]
    #    x[2][i] = samples[i][0][2]
    #    x[3][i] = samples[i][1][0]
    #    yp[i] = np.dot(theta[0], fourier(x[...,i], P, v, phi, numfeat))
    #test1 = getReward(samples[0][0], samples[0][1], theta, fourierparams)
    #test2 = getNextState(samples[0][0], samples[0][1], theta, fourierparams, discStates)
    #print(test1)
    #print(test2)
    # Planning via dynamic programming
    policy = train_policy(discStates, discActions, theta, fourierparams)

    print('Its over')
    #print(policy)
    policy_iterator = policy.reshape(1,numStates * numStates * numStates)
    plt.scatter(range(numStates * numStates * numStates), policy_iterator[0])
    # plt.scatter(x[0], np.abs(samples[...,2] - yp))
    # plt.scatter(x[0], yp)
    # plt.scatter(x[0], samples[...,2] )
    plt.show()


# For automatic execution
main()
