import operator
from functools import reduce
import numpy as np
import itertools


def compute_dimension(shape: tuple):
    #result = reduce(operator.mul, shape)
    result = int(1)
    for dim in shape:
        result = result * int(dim)
    return result


class Cube(object):

    def __init__(self, shape: tuple, flat):
        self.flat = flat
        self.shape = shape
        assert self._validate_shape(shape, flat), "Dimensions do not match!"
        self.multipliers = [1] * len(shape)
        for i in range(len(shape) - 1, 0, -1):
            self.multipliers[i - 1] = self.multipliers[i] * shape[i]

    @staticmethod
    def _validate_shape(shape, flat):
        # Validate if only the first dimension is non-trivial
        if len(flat.shape) > 0:
            for dim_slice in flat.shape[1:]:
                if dim_slice > 1:
                    return False
        return compute_dimension(shape) == flat.shape[0]

    def _lookup_position(self, row):
        item = list(row)
        assert len(item) == len(self.shape)
        item = [element % dimension for element, dimension in zip(item, self.shape)]
        multiplied = [element * multiplier for element, multiplier in zip(item, self.multipliers)]
        return reduce(operator.add, multiplied)

    def _reverse_lookup(self, position):
        items = []
        for i in range(len(self.multipliers)):
            items.append(position // self.multipliers[i])
            position = position % self.multipliers[i]
        return items

    def shift(self, positions: tuple):
        """
        This function shifts along the axes by the positions specified in the positions parameter
        For example, if we are dealing with 2x2 array, and positions=(1,0),
        then the indexes of the first dimension will be incremented by one,
        so the result b_00 = a_10, b_10 = a_00.

        We implement this shift as a square matrix of the dimension of len(flat),
        so that in the reduce methods it may be easier to do transformation using matrix multiplication
        :param positions:
        :return: a square matrix of the same dimension as the flattened data array
        """
        assert len(positions) == len(self.shape)
        flat_dimension = len(self.flat)
        result = np.zeros((flat_dimension, flat_dimension))
        for k in range(flat_dimension):
            items = self._reverse_lookup(k)
            items = [item + position for item, position in zip(items, positions)]
            k_prime = self._lookup_position(items) % flat_dimension
            result[k, k_prime] = 1
        return result

    def reduce_sum(self, dimensions: tuple, keepdims: bool):
        assert keepdims, "Currently not supporting keepdims=False"
        ranges = [range(self.shape[shape_slice]) if shape_slice in dimensions else range(1) for shape_slice in
                  range(len(self.shape))]
        combinations = itertools.product(*ranges)
        result = 0
        for combination in combinations:
            shift_matrix = self.shift(combination)
            result = result + shift_matrix @ self.flat
        return Cube(self.shape, flat=result)
