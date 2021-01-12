import unittest
import numpy as np
from cubes import Cube, compute_dimension


class MyTestCase(unittest.TestCase):

    def test_compute_dimension(self):
        shape=(3,5)
        self.assertEqual(15, compute_dimension(shape))

    def test_validate_shape(self):
        self.assertTrue(Cube._validate_shape((2, 3, 4), np.zeros(24)))
        self.assertFalse(Cube._validate_shape((2, 3, 4), np.zeros(25)))
        self.assertFalse(Cube._validate_shape((2, 3, 4), np.zeros((24, 2))))

    def test_lookup_position(self):
        self.assertEqual(3, Cube((2, 2), np.zeros(4))._lookup_position((1, 1)))
        self.assertEqual(0, Cube((2, 3), np.zeros(6))._lookup_position((0, 0)))
        self.assertEqual(36, Cube((2, 4, 5), np.zeros(40))._lookup_position((1, 3, 1)))
        self.assertEqual(3, Cube((2, 2), np.zeros(4))._lookup_position((-1, -1)))

    def test_reverse_lookup(self):
        cube = Cube((2, 4, 5), np.zeros(40))
        for i in range(40):
            self.assertEqual(i, cube._lookup_position(cube._reverse_lookup(i)))

    def test_shift(self):
        cube = Cube((2, 2), np.array([0, 1, 2, 3]))
        np.testing.assert_equal(np.array([[0., 0., 1., 0.],
                                          [0., 0., 0., 1.],
                                          [1., 0., 0., 0.],
                                          [0., 1., 0., 0.]]), cube.shift((1, 0)))

    def test_reduce_sum(self):
        flat_data = np.arange(4).astype('float')
        cube = Cube((2, 2), flat_data)
        cube0 = cube.reduce_sum((0,), keepdims=True)
        cube1 = cube.reduce_sum((1,), keepdims=True)
        cube01 = cube.reduce_sum((0, 1), keepdims=True)
        np.testing.assert_equal(np.array([2., 4., 2., 4.]), cube0.flat)
        np.testing.assert_equal(np.array([1., 1., 5., 5.]), cube1.flat)
        np.testing.assert_equal(np.array([6., 6., 6., 6.]), cube01.flat)


if __name__ == '__main__':
    unittest.main()
