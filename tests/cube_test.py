import unittest
import numpy as np
from cubes import Cube


class MyTestCase(unittest.TestCase):
    def test_validate_shape(self):
        self.assertTrue(Cube._validate_shape((2, 3, 4), np.zeros(24)))
        self.assertFalse(Cube._validate_shape((2, 3, 4), np.zeros(25)))
        self.assertFalse(Cube._validate_shape((2, 3, 4), np.zeros((24, 2))))

    def test_lookup_position(self):
        self.assertEqual(3, Cube._lookup_position((1, 1), (2, 2)))
        self.assertEqual(0, Cube._lookup_position((0, 0), (2, 3)))
        self.assertEqual(36, Cube._lookup_position((1, 3, 1), (2, 4, 5)))
        self.assertEqual(3, Cube._lookup_position((-1,-1), (2, 2)))


if __name__ == '__main__':
    unittest.main()
