import operator
from functools import reduce


class Cube(object):

    def __init__(self, shape: tuple, flat):
        self.flat = flat
        self.shape = shape
        self._validate_shape(shape, flat)

    @staticmethod
    def compute_dimension(shape: tuple):
        return reduce(operator.mul, shape)

    @staticmethod
    def _validate_shape(shape, flat):
        # Validate if only the first dimension is non-trivial
        if len(flat.shape) > 0:
            for dim_slice in flat.shape[1:]:
                if dim_slice > 1:
                    return False
        return Cube.compute_dimension(shape) == flat.shape[0]

    @staticmethod
    def _lookup_position(row, shape):
        item = list(row)
        assert len(item) == len(shape)
        item = [element % dimension for element, dimension in zip(item, shape)]
        multipliers = [1] * len(shape)
        for i in range(len(shape)-1,0,-1):
            multipliers[i-1] = multipliers[i] * shape[i]
        multiplied = [element * multiplier for element, multiplier in zip(item, multipliers)]
        return reduce(operator.add, multiplied)
