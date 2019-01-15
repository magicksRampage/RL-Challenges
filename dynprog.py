import gym
import quanser_robots
import numpy as np
import random as rnd
import matplotlib.pyplot as plt
from time import *
from math import *
from discretization import Discretization

def makeEnv(whichEnv):
    """
    Used for personal testing. Chooses an environment

    :param whichEnv: True for qube, false for pendulum
    :return:
    """
    if whichEnv:
        return gym.make('Qube-v0')
    else:
        return gym.make('Pendulum-v2')


def expand_angles(samples):
    """
    Expands the interval for the angles in s' when they loop around

    :param samples: observed state, action, observed reward, next observed state
    :return: currated samples
    """
    for sample in samples:
        curStateAngle = sample[0][0]
        newStateAngle = sample[3][0]
        if np.abs(curStateAngle - newStateAngle) > 4:
            sample[3][0] = newStateAngle - (2 * np.pi * np.sign(newStateAngle))


def exploration_policy(obs,cube):
    """
    Gaussian exploration policy (for pendulum only)

    :param obs: observed state
    :param cube: whether the environment is the cube or pendulum
    :return:
    """
    if cube:
        zufall = rnd.gauss(0, 10)
        return zufall
    zufall = rnd.gauss(0, 1)
    value = zufall
    vel = obs[1]
    if (obs[0] > 1.57 or obs[0] < -1.57):
        sig = copysign(1.5, vel)
        value = rnd.gauss(sig, 0.5)
    else:
        value = rnd.gauss(0, 1)
    return [min(2.0, max(-2.0, value))]


def explore(env, numSamples, actions, states):
    """
    Generate a number of samples for regression, using an exploration policy

    :param env: learning environment
    :param numSamples: number of samples to be generated
    :param actions: discrete action space
    :param states: discrete state space
    :return:
    """
    print("start explore")
    samples = []
    renderCount = 15
    render = False
    cube=(str(env)==str("<TimeLimit<Qube<Qube-v0>>>"))
    while len(samples) < numSamples:
        done = False
        discState = env.reset()
        # print("reset")
        discState = Discretization.getState(discState, states)
        while not done:
            s = [discState.copy()]
            a = Discretization.getState(exploration_policy(discState,cube), actions)
            s.append(a)
            obs, reward, done, info = env.step(a)
            if (render):
                env.render()
            discState = Discretization.getState(obs, states)
            s.append(reward)
            s.append(discState.copy())
            samples.append(s)
        if (renderCount > 0):
            renderCount -= 1
            if renderCount == 0:
                render = False

    print("end explore")
    return np.array(samples)


def getphi(numFeats):
    """
    Generates an array of random phase shifts for fourier features

    :param numFeats: number of Fourier features
    :return:
    """
    phi = []
    for i in range(numFeats):
        phi.append(rnd.random() * 2 * np.pi - np.pi)
    return np.array(phi)


def getP(numFeats, sumNumStateActions):
    """
    Generates a matrix of random weights for fourier features

    :param numFeats: number of requested fourier features
    :param sumNumStateActions: sum of the number of States and number of Actions
    :return:
    """
    P = []
    for i in range(numFeats):
        Pi = []
        for j in range(sumNumStateActions):
            Pi.append(rnd.gauss(0, 1))
        P.append(Pi)
    return np.array(P)


def fourier(obs, P, v, phi, numFeats):
    """
    Computes the fourier features for a state observation

    :param obs: observed state
    :param P: weight matrix
    :param v: wavelength
    :param phi: phase shifts
    :param numFeats: number of fourier features to be generated
    :return:
    """
    y = []
    for i in range(numFeats):
        arg = 0
        for j in range(len(obs)):
            arg += P[i][j] * obs[j]
        arg /= v
        arg += phi[i]
        y.append(np.sin(arg))
    return np.array(y)


def featuremat(samples, numFeats, fourierparams):
    """
    Computes the feature matrix for fourier regression

    :param samples: contains the samples from the regression
    [[discState, action, reward, nextDiscState]]
    :param numFeats: number of fourier features to be generated
    :param fourierparams: contains P, v, phi and the desired number of fourier features
    :return:
    """
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


def regression(samples, fourierparams):
    """
    Executes fourier regression for the given samples

    :param samples: contains the samples from the regression
    [[discState, action, reward, nextDiscState]]
    :param fourierparams: contains P, v, phi and the desired number of fourier features
    :return: theta
    theta[0] contains the parameters for the reward function
    theta[1] contains the parameters for the dynamics function
    """
    print('Begin Regression')
    theta = []
    numfeat = fourierparams[3]
    features = featuremat(samples, numfeat, fourierparams)
    feat_inverse = np.linalg.pinv(features)
    rewards = samples[..., 2]
    next_states = np.array(list(samples[..., 3]))
    theta_r = np.dot(feat_inverse, rewards)
    theta.append(theta_r)
    theta_dyn = np.dot(feat_inverse, next_states)
    theta.append(theta_dyn)
    print('Regression Finished')
    return theta


def getReward(obs, act, theta, fparams):
    """
    Calculates the immediate reward for a state-action pair

    :param obs: observed state
    :param act: action taken
    :param theta: coefficients for the fourier-approximation
    :param fparams: misc parameters for the fourier features
    :return:
    """
    x = np.append(obs, act)
    fx = fourier(x, fparams[0], fparams[1], fparams[2], fparams[3])
    return np.dot(theta[0], fx)


def getNextState(state, act, theta, fparams, states):
    """
    Calculates the projected next state for a state-action pair

    :param state: previous state tupel
    :param act: action taken
    :param theta: coefficients for the fourier-approximation
    :param fparams: misc parameters for the fourier features
    :param states: discretized space of states
    :return:
    """
    x = np.append(state, act)
    fx = fourier(x, fparams[0], fparams[1], fparams[2], fparams[3])
    theta_s = theta[1]
    newState = []
    for i in range(len(state)):
        newState.append(np.dot(theta_s[..., i], fx))
    return Discretization.getState(np.array(newState), states)


def train_policy(model, states, actions ):
    """
    Trains a policy for the given model and spaces

    :param model: predicts the "consequences" of a given state-action-pair
    :param states: disc. space of states
    :param actions:  disc. space of actions
    :return: policy tensor containing the appropriate actions on the states indices
    """

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
    gamma = 0.9
    transFun = np.zeros(explicitStatesShape + (states.shape[0],))
    imReward = np.zeros(explicitStatesShape)
    policyStable = False
    unstableCounter = 0
    oldUnstableCounter = 0
    while not policyStable:
        print("update:" + str(updates))
        policyStable = True
        oldPolicy = policy
        # policy evaluation
        maxDelta = 1

        it = np.nditer(valFun, flags=['multi_index'])
        tempReward = 0
        tempState = ()
        mulInd = 0
        print("precomputing model")
        while not it.finished:
            tempReward = 0
            tempState = ()
            mulInd = it.multi_index
            for dim in range(len(mulInd)):
                tempState += (states[dim][mulInd[dim]],)

            if np.isinf(policy.item(mulInd)):
                modelresult = model(tempState, np.random.choice(actions[0]))
                for act in range(len(actions[0])):
                    tempReward += model(tempState, actions[0][act])[1] / len(actions[0])
                imReward[mulInd] = tempReward
                transFun[mulInd] = modelresult[0]
            else:
                modelresult = model(tempState, policy.item(mulInd))
                transFun[mulInd] = modelresult[0]
                imReward[mulInd] = modelresult[1]
            it.iternext()
        print("policy evaluation")
        while maxDelta > 0.1:
            maxDelta = 0
            it = np.nditer(valFun, flags=['multi_index'])
            mulInd = 0
            tempStateInd = ()
            while not it.finished:
                mulInd = it.multi_index
                tempStateInd = Discretization.getIndex(tempState, states)

                newValfun = imReward.item(mulInd) + gamma * valFun.item(tempStateInd)
                maxDelta = max(maxDelta, np.abs(newValfun - valFun.item(mulInd)))
                valFun[mulInd] = newValfun

                tempStateInd = ()
                it.iternext()

        it = np.nditer(valFun, flags=['multi_index'])
        bestTorque = 0
        tempReward = -np.inf
        bestReward = -np.inf
        prevState = ()
        nextState = ()
        nextStateInd = ()
        mulInd = 0
        unstableCounter = 0
        print("policy improvement")
        while not it.finished:
            mulInd = it.multi_index
            for dim in range(len(mulInd)):
                prevState += (states[dim][mulInd[dim]],)
            for act in actions[0]:
                nextState = model(prevState, act)[0]
                nextStateInd = Discretization.getIndex(nextState, states)
                tempReward = model(prevState, act)[1] + gamma * valFun[nextStateInd]
                if tempReward > bestReward:
                    bestTorque = act
                    bestReward = tempReward
                tempReward = -np.inf
                nextState = ()
                nextStateInd = ()
            stable = (policy[mulInd] == bestTorque)
            if not stable:
                print(mulInd)
                unstableCounter += 1
            policy[mulInd] = bestTorque
            bestReward = -np.inf
            bestTorque = 0
            prevState = ()
            it.iternext()
        if not (unstableCounter >= oldUnstableCounter):
            policyStable = False
        updates += 1
        print("unstable: " + str(unstableCounter))
        oldUnstableCounter = unstableCounter
        unstableCounter = 0


    print("updates: " + str(updates))
    # return optimal policy lookup-table
    return policy


def main():
    """
    For personal testing
    """
    t1 = clock()
    # false for pendulum, true for qube
    qube = False
    env = makeEnv(qube)
    numActions = 5
    numStates = 51
    discActions = Discretization.getSpace_extended(env.action_space, numActions, 3)
    discStates = Discretization.getSpace_extended(env.observation_space, numStates, 2)
    samples = explore(env, 10000, discActions, discStates)
    numobs = len(samples[0][0])
    numact = len(samples[0][1])
    numfeat = 50
    P = getP(numfeat, numobs + numact)
    v = 5
    phi = getphi(numfeat)
    fourierparams = [P, v, phi, numfeat]
    expand_angles(samples)
    theta = regression(samples, fourierparams)

    # for visualizing the regression
    x = np.ones([10, len(samples)])
    yp = np.ones([2, len(samples)])
    for i in range(len(samples)):
        # angle
        x[0][i] = samples[i][0][0]
        x[1][i] = samples[i][0][1]
        # true next angle
        x[2][i] = samples[i][3][0]
        # estimated next angle
        x[3][i] = getNextState(samples[i][0], samples[i][1], theta, fourierparams, discStates)[0]
        # estimated Reward
        yp[0][i] = getReward(samples[i][0], samples[i][1], theta, fourierparams)
        # squared difference between estimated and actual next State
        yp[1][i] = np.sqrt(np.sum(
            np.square(samples[i][3] - getNextState(samples[i][0], samples[i][1], theta, fourierparams, discStates))))
    #plt.scatter(x[0], np.square(samples[...,2] - yp[0]))
    #plt.scatter(x[0], yp[1])
    plt.scatter(x[0], x[3])
    #plt.show()
    policy = train_policy(discStates, discActions, theta, fourierparams)

    print('Its over')
    t2 = clock()
    print("Rechenzeit = " + str(t2 - t1))
    validation = []
    fig,ax=plt.subplots()
    plt.show()#stop zum anschauen
    for trials in range(10):
        obs = env.reset()
        reward = 0
        done = False
        while not done:
            sample = [obs]
            dis_obs = Discretization.getState(obs, discStates)
            index = []
            act = policy[tuple(index)]
            obs, rew, done, _ = env.step([act])
            sample.append([act])
            sample.append(rew)
            sample.append(obs)
            sample.append(getReward(sample[0], act, theta, fourierparams))
            sample.append(getNextState(sample[0], act, theta, fourierparams, discStates))
            validation.append(sample)
            reward += rew
            env.render()
        print(reward)

    x = np.ones([3, len(validation)])
    yp = np.ones([len(validation)])
    for i in range(len(validation)):
        x[0][i] = validation[i][0][0]
        x[1][i] = validation[i][0][1]
        x[2][i] = validation[i][1][0]
        yp[i] = np.square(validation[i][2] - validation[i][4])
    plt.scatter(x[0], yp)
    plt.show()
