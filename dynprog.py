import gym
import quanser_robots
import numpy as np
import random as rnd
import matplotlib.pyplot as plt
from time import *
from math import *


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
        return gym.make('Pendulum-v2')


# returns a uniform discrete space
# space:    continuous space to be discretized
# grain:    number of desired discrete classes for each dimension of the space
#           TODO: length of disc. space is 1 longer then indicated by grains
def discretize_space(space, grain):
    shape = space.shape
    highs = space.high
    lows = space.low
    discSpace = np.ones([shape[0], grain])
    for i in range(shape[0]):
        step = (highs[i] - lows[i]) / (grain - 1)
        for j in range(grain):
            discSpace[i][j] = lows[i] + j * step
    return discSpace

# returns a uniform discrete space
# space:    continuous space to be discretized
# grain:    number of desired discrete classes for each dimension of the space
# exponent: [0,1,sonstiges]=linear;  2=squared;  3=cubed
def discretize_space_cube(space, grain, cubed):
    shape = space.shape
    highs = space.high
    lows = space.low
    discSpace = np.ones([shape[0],grain])
    for i in range(shape[0]):
        step = (highs[i] - lows[i]) / (grain - 1)
        for j in range(grain):
            discSpace[i][j] = lows[i] + j * step
        if (cubed[i]>1):
            highest=highs[i]
            if (abs(lows[i])>highest):
                highest=lows[i]
            if(cubed[i]==3):
                discSpace[i]=(discSpace[i]**3)/(highest**2)
            elif(cubed[i]==2):
                vz=np.ones(len(discSpace[i]))#vorzeichen
                for k in range(len(discSpace[i])):
                    vz[k]=np.copysign(vz[k],discSpace[i][k])
                discSpace[i]=vz*(discSpace[i]**2)/(highest)

    return discSpace

# classifies a continuous sample for a discrete space
# sample:   continuous sample to be discretized
# space:    discrete space
def discretize_state(sample, space):
    return _discretize(sample,space)[0]

def _discretize(sample,space):
    discSample = []
    positions=[]
    for i in range(len(sample)):
        entry = np.clip(sample[i], space[i][0], space[i][-1])
        for j in range(len(space[i])):
            step = space[i][j]
            if entry == step:
                discSample.append(step)
                positions.append(j)
                break
            elif entry < step:
                prev = space[i][j - 1]
                if np.abs(entry - prev) > np.abs(entry - step):
                    discSample.append(step)
                    positions.append(j)
                else:
                    discSample.append(prev)
                    positions.append(j-1)
                break
    return [np.array(discSample),np.array(positions)]


def expand_angles(samples):
    for sample in samples:
        old_angle = sample[0][0]
        new_angle = sample[3][0]
        #print(sample)
        #print("old: " + str(old_angle) + ", new: " + str(new_angle))
        if np.abs(old_angle - new_angle) > 5:
            #print("replacing")
            if new_angle > 0:
                sample[3][0] = new_angle - 2 * np.pi
            else:
                sample[3][0] = new_angle + 2 * np.pi
            #print(sample)

# gaussian exploration policy (for pendulum only)
# obs:  current state
def exploration_policy(obs):
    zufall=rnd.gauss(0,1)
    value=zufall
    vel=obs[1]
    #value=vel/8+zufall
    if(obs[0]>1.57 or obs[0]<-1.57):
    #    if(obs[0]<0.8):
        sig=copysign(1.5,vel)
        #value=copysign(2,vel)#+rnd.gauss(0,0.5)
        value=rnd.gauss(sig,0.5)
    else:
        value=rnd.gauss(0,1)
    #if (zufall<0):
    #    value=-1
    #else:
    #    value=1
    return [min(2.0, max(-2.0, value))]
#return [0]


# generate a number of samples for regression, using an exploration policy
# env:          learning environment
# numSamples:   number of samples to be generated
# actions:      discrete action space
# states:       discrete state space
def explore(env, numSamples, actions, states):
    print("start explore")
    samples = []
    renderCount=15
    render=False
    while len(samples) < numSamples:
        done = False
        discState = env.reset()
        #print("reset")
        discState = discretize_state(discState, states)
        while not done:
            s = [discState]
            a = discretize_state(exploration_policy(discState), actions)
            s.append(a)
            obs, reward, done, info = env.step(a)
            if(render):
                env.render()
            discState = discretize_state(obs, states)
            s.append(reward)
            s.append(discState)
            samples.append(s)
        if(renderCount>0):
            renderCount-=1
            if renderCount==0:
                render=False

    print("end explore")
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
            Pi.append(rnd.gauss(0, 1))
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
    print('Begin Regression')
    theta = []
    numfeat = fourierparams[3]
    features = featuremat(samples, numfeat, fourierparams)
    # print(features)
    feat_inverse = np.linalg.pinv(features)
    # print(feat_inverse)
    rewards = samples[..., 2]
    # print(rewards)
    next_states = np.array(list(samples[..., 3]))
    # print(next_states.shape)
    theta_r = np.dot(feat_inverse, rewards)
    # print(theta_r.shape)
    # print(theta_r)
    theta.append(theta_r)
    theta_dyn = np.dot(feat_inverse, next_states)
    # print(theta_dyn.shape)
    # print(theta_dyn)
    theta.append(theta_dyn)
    print('Regression Finished')
    return theta


# calculates the immediate reward for a state-action pair
# obs:      state observation
# act:      action taken
# theta:    coefficients for the fourier-approximation
# fparams:  misc parameters for the fourier features
def getReward(obs, act, theta, fparams):
    x = np.append(obs, act)
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
    # print(theta_s[...,0])
    newState = []
    for i in range(len(state)):
        newState.append(np.dot(theta_s[..., i], fx))
    # print(newState)
    return discretize_state(np.array(newState), states)


# Planes the optimal policy for a MDP defined by:
# states:   3xNo(DiscSteps) array containing cos(th), sin(th), dot(th)
# action:   1xNo(DiscSteps) array containing the torque
# theta:    1xNo(Features) array containing the coefficients for the fourier-approximation
# fparams:  misc parameters for evaluation of fourier approximation
def train_policy(states, actions, theta, fparams):
    # Setup value-function
    # length = No(all possible states)
    print('Begin training policy')
    explicitStatesShape = ()
    for dim in range(states.shape[0]):
        explicitStatesShape += (states.shape[1],)
    valFun = np.zeros(explicitStatesShape)

    # Contains amount of torque
    # if t == np.inf torque is random
    policy = np.full(explicitStatesShape, np.inf)

    # iterate
    oldPolicy = np.full(explicitStatesShape, np.inf)
    valFun = np.zeros(explicitStatesShape)
    updates = 0
    # polTorque = 0
    gamma = 0.99
    transFun = np.zeros(explicitStatesShape + (states.shape[0],))
    imReward = np.zeros(explicitStatesShape)
    policyStable = False
    # while udates < 10:
    while not policyStable:
        policyStable = True
        oldPolicy = policy
        # policy evaluation
        maxDelta = 1

        # for i in range(valFun.shape[0]):
        #     for j in range(valFun.shape[1]):
        #         for k in range(valFun.shape[2]):
        #             if np.isinf(policy[i][j][k]):
        #                 imReward[i][j][k] = 0
        #                 for act in range(len(actions[0])):
        #                     imReward[i][j][k] += getReward((states[0][i], states[1][j], states[2][k]), actions[0][act], theta, fparams)/len(actions[0])
        #                 transFun[i][j][k] = getNextState((states[0][i], states[1][j], states[2][k]), np.random.choice(actions[0]), theta, fparams, states)
        #
        #             else:
        #                 imReward[i][j][k] = getReward((states[0][i], states[1][j], states[2][k]), policy[i][j][k], theta, fparams)
        #                 transFun[i][j][k] = getNextState((states[0][i], states[1][j], states[2][k]), policy[i][j][k], theta, fparams, states)

        it = np.nditer(valFun, flags=['multi_index'])
        tempReward = 0
        tempState = ()
        mulInd = 0
        while not it.finished:
            tempReward = 0
            tempState = ()
            mulInd = it.multi_index
            for dim in range(len(mulInd)):
                tempState += (states[dim][mulInd[dim]],)

            if np.isinf(policy.item(mulInd)):
                for act in range(len(actions[0])):
                    tempReward += getReward(tempState, actions[0][act], theta, fparams) / len(actions[0])
                imReward[mulInd] = tempReward
                # TODO: Is that the right way to start out?
                transFun[mulInd] = getNextState(tempState, np.random.choice(actions[0]), theta, fparams, states)
            else:
                imReward[mulInd] = getReward(tempState, policy.item(mulInd), theta, fparams)
                transFun[mulInd] = getNextState(tempState, policy.item(mulInd), theta, fparams, states)
            it.iternext()

        # while maxDelta > 0.1:
        #     maxDelta = 0
        #     for i in range(valFun.shape[0]):
        #         for j in range(valFun.shape[1]):
        #             for k in range(valFun.shape[2]):
        #
        #                 newValFun = imReward[i][j][k] + gamma*valFun[read_index_for_state(states, 0, transFun[i][j][k][0])][read_index_for_state(states, 1, transFun[i][j][k][1])][read_index_for_state(states, 2, transFun[i][j][k][2])]
        #                 maxDelta = max(maxDelta, np.abs(newValFun - valFun[i][j][k]))
        #                 valFun[i][j][k] = newValFun

        while maxDelta > 0.01:
            maxDelta = 0
            it = np.nditer(valFun, flags=['multi_index'])
            mulInd = 0
            tempStateInd = ()
            while not it.finished:
                mulInd = it.multi_index
                for dim in range(len(mulInd)):
                    tempStateInd += (read_index_for_state(states, dim, transFun.item(mulInd + (dim,))),)

                newValfun = imReward.item(mulInd) + gamma * valFun.item(tempStateInd)
                maxDelta = max(maxDelta, np.abs(newValfun - valFun.item(mulInd)))
                valFun[mulInd] = newValfun

                tempStateInd = ()
                it.iternext()

        # greedy policy improvement
        # bestTorque = 0
        # tempReward = -np.inf
        # bestReward = -np.inf
        # tempState = np.array((0., 0., 0.))
        # for i in range(valFun.shape[0]):
        #     for j in range(valFun.shape[1]):
        #         for k in range(valFun.shape[2]):
        #             bestReward = -np.inf
        #             for act in actions[0]:
        #                 tempState = getNextState((states[0][i], states[1][j], states[2][k]), act, theta, fparams,
        #                                          states)
        #                 tempReward = gamma * valFun[read_index_for_state(states, 0, tempState[0])][
        #                     read_index_for_state(states, 1, tempState[1])][
        #                     read_index_for_state(states, 2, tempState[2])]
        #                 tempReward += getReward((states[0][i], states[1][j], states[2][k]), act, theta, fparams)
        #                 if tempReward > bestReward:
        #                     bestTorque = act
        #                     bestReward = tempReward
        #             stable = policy[i][j][k] == bestTorque
        #             if not stable:
        #                 policyStable = False
        #             policy[i][j][k] = bestTorque
        #             bestTorque = 0

        it = np.nditer(valFun, flags=['multi_index'])
        bestTorque = 0
        tempReward = -np.inf
        bestReward = -np.inf
        prevState = ()
        nextState = ()
        nextStateInd = ()
        mulInd = 0
        while not it.finished:
            mulInd = it.multi_index
            for dim in range(len(mulInd)):
                prevState += (states[dim][mulInd[dim]],)
            for act in actions[0]:
                nextState = getNextState(prevState, act, theta, fparams, states)
                for dim in range(len(mulInd)):
                    nextStateInd += (read_index_for_state(states, dim, nextState[dim]), )
                tempReward = getReward(prevState, act, theta,fparams) + gamma * valFun [nextStateInd]
                if tempReward > bestReward:
                    bestTorque = act
                    bestReward = tempReward
                tempReward = -np.inf
                nextState = ()
                nextStateInd = ()
            stable = (policy[mulInd] == bestTorque)
            if not stable:
                policyStable = False
            policy[mulInd] = bestTorque
            bestReward = -np.inf
            bestTorque = 0
            prevState = ()
            it.iternext()

        print("update:"+str(updates))
        updates += 1

    print("updates: " + str(updates))
    # return optimal policy lookup-table
    return policy


# Finds the index of a given stateValue in its discretized Space
# states:       discretized state space
# dimension:    state dimension in which the value is to be found
# stateVal:     the Value for which to find the index
def read_index_for_state(states, dimension, stateVal):
    #import pdb; pdb.set_trace()
    #TODO @Tim unser Code soll schoener werden
    if(dimension==1):
        return _discretize([0,stateVal],states)[1][dimension]
    if(dimension==2):
        return _discretize([0,0,stateVal],states)[1][dimension]
    if(dimension==3):
        return _discretize([0,0,0,stateVal],states)[1][dimension]
    return _discretize([stateVal],states)[1][dimension]
    #dimLen = len(states[dimension])
    #dimMax = states[dimension][-1]
    #dimMin = states[dimension][0]
    #step = (dimMax - dimMin) / (dimLen - 1)
    #index = (stateVal - dimMin) / step
    #return int(round(max(dimMin, min(dimMax, index))))


def main():
    t1=clock()
    # false for pendulum, true for qube
    qube = False
    env = makeEnv(qube)
    numActions = 5
    numStates = 21
    discActions = discretize_space_cube(env.action_space, numActions, [3])
    discStates = discretize_space_cube(env.observation_space, numStates, [2,2,2,2])
    discCube=discretize_space_cube(env.observation_space,numStates,[3,3,3,3])
    plot_discretisation(
                    [np.sin(discStates[0]),np.sin(discCube[0])],
                    [np.cos(discStates[0]),np.cos(discCube[0])],['o','.'])
    #plt.show()
    #print(discActions)
    #print(discStates)
    samples = explore(env, 10000, discActions, discStates)
    # print(len(samples))
    # print(samples[0])
    numobs = len(samples[0][0])
    numact = len(samples[0][1])
    numfeat = 100
    P = getP(numfeat, numobs + numact)
    v = 5
    phi = getphi(numfeat)
    fourierparams = [P, v, phi, numfeat]
    #print(samples)
    expand_angles(samples)
    theta = regression(samples, fourierparams)

    #for visualizing the regression
    x = np.ones([10,len(samples)])
    yp = np.ones([2,len(samples)])
    for i in range(len(samples)):
       x[0][i] = samples[i][0][0]
       if np.abs(x[0][i]) > 4:
           print(x[0][i])
       x[1][i] = samples[i][0][1]
       x[2][i] = samples[i][3][0]
       x[3][i] = getNextState(samples[i][0], samples[i][1], theta, fourierparams, discStates)[0]
       yp[0][i] = getReward(samples[i][0], samples[i][1], theta, fourierparams)
       yp[1][i] = np.sqrt(np.sum(np.square(samples[i][3] - getNextState(samples[i][0], samples[i][1], theta, fourierparams, discStates))))
    #plt.scatter(x[0], np.square(samples[...,2] - yp[0]))
    #plt.scatter(x[0], yp)
    plt.scatter(x[0], x[2])
    plt.show()

    # test1 = getReward(samples[0][0], samples[0][1], theta, fourierparams)
    # test2 = getNextState(samples[0][0], samples[0][1], theta, fourierparams, discStates)
    # print(test1)
    # print(test2)
    # Planning via dynamic programming
    policy = train_policy(discStates, discActions, theta, fourierparams)

    print('Its over')
    t2=clock()
    print("Rechenzeit = " + str(t2-t1))
    #print(policy)
    #policy_iterator = policy.reshape(1,numStates * numStates * numStates)
    #plt.scatter(range(numStates * numStates * numStates), policy_iterator[0])
    validation = []
    for trials in range(1000):
        obs=env.reset()
        reward = 0
        done = False
        while not done:
            sample = [obs]
            #print("iteration:"+str(i))
            dis_obs=discretize_state(obs,discStates)
            index=[]
            index+=[read_index_for_state(discStates,0,dis_obs[0])]
            index+=[read_index_for_state(discStates,1,dis_obs[1])]
            #act=policy[index[0]][index[1]][index[2]]
            act=policy[tuple(index)]
            obs, rew, done,_ = env.step([act])
            sample.append([act])
            sample.append(rew)
            sample.append(obs)
            sample.append(getReward(sample[0], act, theta, fourierparams))
            sample.append(getNextState(sample[0], act, theta, fourierparams, discStates))
            validation.append(sample)
            reward += rew
            #print(""+str(i)+" : "+str(rew)+" -- "+str(obs[0])+" "+str(obs[1]))
            #env.render()
        print(reward)

    x = np.ones([3,len(validation)])
    yp = np.ones([len(validation)])
    for i in range(len(validation)):
       x[0][i] = validation[i][0][0]
       x[1][i] = validation[i][0][1]
       x[2][i] = validation[i][1][0]
       yp[i] = np.square(validation[i][2] - validation[i][4])
    plt.scatter(x[0], yp)
    plt.show()

def plot_discretisation(x,y,opt):
    l=len(x)
    if len(y)<l:
        l=len(y)
    fig,ax=plt.subplots()
    for i in range(l):
        ax.plot(x[i],y[i],opt[i])
    ax.set_aspect('equal')
    #plt.show()

# For automatic execution
main()
