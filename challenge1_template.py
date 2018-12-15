from typing import Tuple

from dynprog import *

"""
Submission template for Programming Challenge 1: Dynamic Programming.
"""

info = dict(
    group_number=None,  # change if you are an existing seminar/project group
    authors="John Doe; Lorem Ipsum; Foo Bar",
    description="Explain what your code does and how. "
                "Keep this description short "
                "as it is not meant to be a replacement for docstrings "
                "but rather a quick summary to help the grader.")


def get_model(env, max_num_samples):
    """
    Sample up to max_num_samples transitions (s, a, s', r) from env
    and fit a parametric model s', r = f(s, a).

    :param env: gym.Env
    :param max_num_samples: maximum number of calls to env.step(a)
    :return: function f: s, a -> s', r
    """

    numActions = 5
    numStates = 25
    discActions = Discretization.getSpace_extended(env.action_space, numActions, 3)
    discStates = Discretization.getSpace_extended(env.observation_space, numStates, 2)
    samples = explore(env, max_num_samples, discActions, discStates)
    numobs = len(samples[0][0])
    numact = len(samples[0][1])
    numfeat = 20
    P = getP(numfeat, numobs + numact)
    v = 5
    phi = getphi(numfeat)
    fourierparams = [P, v, phi, numfeat]
    #expand_angles(samples)
    theta = regression(samples, fourierparams)

    
    return lambda obs, act: (getNextState(obs, act, theta, fourierparams, discStates ), getReward(obs, act, theta, fourierparams))


def get_policy(model, observation_space, action_space):
    """
    Perform dynamic programming and return the optimal policy.

    :param model: function f: s, a -> s', r
    :param observation_space: gym.Space
    :param action_space: gym.Space
    :return: function pi: s -> a
    """
    numActions = 5
    numStates = 25
    discActions = Discretization.getSpace_extended(action_space, numActions, 3)
    discStates = Discretization.getSpace_extended(observation_space, numStates, 2)

    policy = new_train_policy(model, discStates, discStates)

    def return_fun(obs):
        mulInd = ()
        for dim in range(len(obs)):
            mulInd += (Discretization.getIndex(discStates, dim, obs[dim]),)
        return policy(mulInd)

    return lambda obs: return_fun(obs)
