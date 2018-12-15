## Discretization class
import numpy as np


class Discretization:

    # returns a uniform discrete space
    # space:    continuous space to be discretized
    # grain:    number of desired discrete classes for each dimension of the space
    # formerly 'discretize_space'
    def getSpace(space, grain):
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
    # formerly 'discretize_space_cube'
    def getSpace_extended(space, grain, exponent):
        shape = space.shape
        highs = space.high
        lows = space.low
        discSpace = np.ones([shape[0], grain])
        for i in range(shape[0]):
            step = (highs[i] - lows[i]) / (grain - 1)
            for j in range(grain):
                discSpace[i][j] = lows[i] + j * step
        if (exponent > 1):
            highest = highs[i]
            if (abs(lows[i]) > highest):
                highest = lows[i]
            if (exponent == 3):
                discSpace[i] = (discSpace[i] ** 3) / (highest ** 2)
            elif (exponent == 2):
                vz = np.ones(len(discSpace[i]))  # vorzeichen
                for k in range(len(discSpace[i])):
                    vz[k] = np.copysign(vz[k], discSpace[i][k])
                discSpace[i] = vz * (discSpace[i] ** 2) / (highest)

        return discSpace


    # classifies a continuous sample for a discrete space
    # sample:   continuous sample to be discretized
    # space:    discrete space
    # formerly 'discretize_state'
    def getState(sample, space):
        return Discretization._discretize(sample, space)[0]

    def _discretize(sample, space):
        discSample = []
        positions = []
        angle = False
        for i in range(len(sample)):
            #if len(sample) > 1 and i == 0:
            #    angle = True
            #else:
            angle = False
            if angle:
                entry = Discretization._castAngle(sample[i], space[i][0], space[i][-1])
            else:
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
                        positions.append(j - 1)
                    break
        return [np.array(discSample), np.array(positions)]

    # Finds the index of a given stateValue in its discretized Space
    # states:       discretized state space
    # dimension:    state dimension in which the value is to be found
    # stateVal:     the Value for which to find the index
    # formerly read_index_for_state
    def getIndex(states, dimension, stateVal):
        # import pdb; pdb.set_trace()
        # TODO @Tim unser Code soll schoener werden
        if (dimension == 1):
            return Discretization._discretize([0, stateVal], states)[1][dimension]
        if (dimension == 2):
            return Discretization._discretize([0, 0, stateVal], states)[1][dimension]
        if (dimension == 3):
            return Discretization._discretize([0, 0, 0, stateVal], states)[1][dimension]
        return Discretization._discretize([stateVal], states)[1][dimension]
        # dimLen = len(states[dimension])
        # dimMax = states[dimension][-1]
        # dimMin = states[dimension][0]
        # step = (dimMax - dimMin) / (dimLen - 1)
        # index = (stateVal - dimMin) / step
        # return int(round(max(dimMin, min(dimMax, index))))

    # loops input sample into the interval defined by low and high
    def _castAngle(sample, low, high):
        distance = high - low
        while sample < low:
            sample += distance
        while sample > high:
            sample -= distance
        return sample

    def plot(x, y, opt):
        l = len(x)
        if len(y) < l:
            l = len(y)
        fig, ax = plt.subplots()
        for i in range(l):
            ax.plot(x[i], y[i], opt[i])
        ax.set_aspect('equal')
        # plt.show()
