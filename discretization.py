## Discretization class
import numpy as np


class Discretization:

    def getSpace(space, grain):
        """
        Returns a uniform discrete space

        :param space: continuous space to be discretized
        :param grain: number of desired discrete classes for each dimension of the space
        :return:
        """
        shape = space.shape
        highs = space.high
        lows = space.low
        discSpace = np.ones([shape[0], grain])
        for i in range(shape[0]):
            step = (highs[i] - lows[i]) / (grain - 1)
            for j in range(grain):
                discSpace[i][j] = lows[i] + j * step
        return discSpace

    def getSpace_extended(space, grain, exponent):
        """
        Returns a uniform discrete space

        :param space: continuous space to be discretized
        :param grain: number of desired discrete classes for each dimension of the space
        :param exponent: [0,1,sonstiges]=linear;  2=squared;  3=cubed
        :return:
        """
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

    def getState(sample, space):
        """
        Classifies a continuous sample for a discrete space

        :param sample: continuous sample to be discretized
        :param space: discrete space
        :return:
        """
        return Discretization._discretize(sample, space)[0]

    def _discretize(sample, space):
        """

        :param sample: the continous values for the sample-dimensions (e.g. dimensions of obs)
        :param space: the discret space
        :return: ((values in each dimension),(index in each dimension))
        """
        discSample = []
        positions = []
        angle = False
        for i in range(len(sample)):
            angle = False
            high = len(space[i]) -1
            low = 0
            if angle:
                entry = Discretization._castAngle(sample[i], space[i][0], space[i][-1])
            else:
                entry = np.clip(sample[i], space[i][0], space[i][-1])
                    

            while high - low > 1:
                mid = int(np.ceil((high - low) / 2) + low)
                if entry < space[i][mid]:
                    high = mid
                else:
                    low = mid
            highval = space[i][high]
            lowval = space[i][low]
            if np.abs(entry - lowval) > np.abs(entry - highval):
                discSample.append(highval)
                positions.append(high)
            else:
                discSample.append(lowval)
                positions.append(low)
        return [np.array(discSample), np.array(positions)]

    def getIndex(sample, space):
        """
        Finds Index for a sample

        :param sample: observations sample
        :param space: discrete space
        :return: multiIndex
        """
        return tuple(Discretization._discretize(sample, space)[1])

    def _castAngle(sample, low, high):
        """
        Loops input sample into the interval defined by low and high

        :param low: lowest value of the angle Space
        :param high: highest value of the angle Space
        :return:
        """
        distance = high - low
        while sample < low:
            sample += distance
        while sample > high:
            sample -= distance
        return sample

    def plot(x, y, opt):
        """

        :param y:
        :param opt:
        :return:
        """
        l = len(x)
        if len(y) < l:
            l = len(y)
        fig, ax = plt.subplots()
        for i in range(l):
            ax.plot(x[i], y[i], opt[i])
        ax.set_aspect('equal')
        # plt.show()
