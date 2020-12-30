import unittest
import numpy as np
from cubes import Cube


class MyTestCase(unittest.TestCase):
    def test_validate_shape(self):
        self.assertTrue(Cube.validate_shape((2, 3, 4), np.zeros(24)))
        self.assertFalse(Cube.validate_shape((2, 3, 4), np.zeros(25)))
        self.assertFalse(Cube.validate_shape((2, 3, 4), np.zeros((24,2))))


if __name__ == '__main__':
    unittest.main()
