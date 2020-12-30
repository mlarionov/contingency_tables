import operator
from functools import reduce


class Cube(object):

    def __init__(self, shape: tuple, flat):
        self.flat = flat
        self.shape = shape
        self.validate_shape(shape, flat)

    @staticmethod
    def compute_dimension(shape: tuple):
        return reduce(operator.mul, shape)

    @staticmethod
    def validate_shape(shape, flat):
        # Validate if only the first dimension is non-trivial
        if len(flat.shape) > 0:
            for dim_slice in flat.shape[1:]:
                if dim_slice > 1:
                    return False
        return Cube.compute_dimension(shape) == flat.shape[0]

