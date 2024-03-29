"""
Test cases for the pod package.

@author: Christophe Alexandre <christophe.alexandre at pm dot me>

@license:
Copyright(C) 2010 Christophe Alexandre

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/lgpl.txt>.
"""

import unittest
import math
import logging
from decimal import Decimal

from numpy import array

import pod

logging.basicConfig(level=logging.DEBUG)


def prepare_fourier(shift, n_ref=30):
    SAMPLING = 5 * n_ref
    AMPLITUTDE = 100.0
    FREQ_STEP = 0.2
    PHASE_SHIFT = shift

    def gen_x_axis(x0, n=SAMPLING):
        return float(x0) * 2 * math.pi / n

    def square_signal(x0, ampl=AMPLITUTDE, shift=PHASE_SHIFT):
        return 1.0 * ampl if math.pi + shift < x0 < 2.0 * math.pi + shift else 0.0

    def sine(ampl, freq, axis):
        output = [ampl * math.sin(float(x0) * freq) for x0 in axis]
        return output

    def cosine(ampl, freq, axis):
        output = [ampl * math.cos(float(x0) * freq) for x0 in axis]
        return output

    # x axis
    x_axis = [gen_x_axis(value) for value in range(SAMPLING)]

    # input signal
    y = [square_signal(value) for value in x_axis]

    # creates set of sine and cosine functions

    y_i = []

    freq_range = range(int(Decimal('-0.2') * n_ref), int(Decimal('0.8') * n_ref))

    for freq in freq_range:
        y_i.append(sine(AMPLITUTDE, float(FREQ_STEP * freq), x_axis))

    z_i = []
    for freq in freq_range:
        z_i.append(cosine(AMPLITUTDE, float(FREQ_STEP * freq), x_axis))

    refs = y_i + z_i

    return x_axis, y, refs


class PodTest(unittest.TestCase):
    """
    Testing the POD algorithm.

    """

    def test_decompose_001(self):
        """
        Decomposing with 2 reference points in 2 dimensions
        """
        index1 = [0.8, 2.0]
        index2 = [1.2, -0.5]
        fund = [1.0, 0.5]
        r = pod.BaseDecomposition(array([index1, index2]), max_iter=20)
        replicate = r.resolve(array(fund))

        s_expected = [1.0, 0.5]
        for i in range(len(s_expected)):
            self.assertAlmostEqual(s_expected[i], replicate[i], 3)

    def test_decompose_002(self):
        """
        Decomposing with 2 points in 3 dimensions: overdetermined system,
        will find the best approximation given the constraint.
        """
        index1 = [0.8, 2.0, -5.0]
        index2 = [1.2, -0.5, -1.0]
        fund = [1.0, 0.5, 3.0]
        r = pod.BaseDecomposition(array([index1, index2]), max_iter=40)
        replicate = r.resolve(array(fund))

        logging.debug(f'result: {r}')
        s_expected = [-0.265192, -0.961998, 2.2127700]
        for i in range(len(s_expected)):
            self.assertAlmostEqual(s_expected[i], replicate[i], 4)

    def test_decompose_003(self):
        """
        Decomposition in a 3D space, 5 references: 2 x first point
        """
        indices = [[0.8, 2.0, -1.0], [1.2, -0.5, 0.8], [-1.0, -0.9, 4.1], [2.2, 0.8, 1.0], [-0.1, -1.5, -0.1],
                   [5.0, -3.0, 0.4]]
        #
        # target: 2 X first point
        fund = [1.6, 4.0, -2.0]
        #
        r = pod.BaseDecomposition(array(indices), max_iter=20)
        result = r.resolve(array(fund))

        logging.debug(f'result: {result}')
        s_expected = [1.6, 4.0, -2.0]
        for i in range(len(s_expected)):
            self.assertAlmostEqual(s_expected[i], result[i], 6)

    def test_decompose_004(self):
        """
        Decomposition in a 3D space, 6 references: 2 x first - 1 x last
        """
        indices = [[0.8, 2.0, -1.0], [1.2, -0.5, 0.8], [-1.0, -0.9, 4.1], [2.2, 0.8, 1.0], [-0.1, -1.5, -0.1],
                   [5.0, -3.0, 0.4]]
        #
        # target: 2 X first point - 1 X last point
        fund = [-3.4, 7.0, -2.4]
        #
        r = pod.BaseDecomposition(array(indices), epsilon=1E-10, max_iter=20)
        result = r.resolve(array(fund))

        logging.debug(f'result: {r}')
        self.assertAlmostEqual(2., r.get_reference_weight(0))
        self.assertAlmostEqual(-1., r.get_reference_weight(1))
        s_expected = [-3.4, 7.0, -2.4]
        for i in range(len(s_expected)):
            self.assertAlmostEqual(result[i], s_expected[i], 6)

    def test_decompose_005(self):
        """
        Decomposition in a 3D space, 6 references: 2 x first - 1 x last + noise
        """
        indices = [[0.8, 2.0, -1.0], [1.2, -0.5, 0.8], [-1.0, -0.9, 4.1], [2.2, 0.8, 1.0], [-0.1, -1.5, -0.1],
                   [5.0, -3.0, 0.4]]
        #
        # target: 2 X first point - 1 X last point + noise
        fund = [-3.6, 6.5, -2.7]
        #
        r = pod.BaseDecomposition(array(indices), epsilon=1E-6, max_iter=30)
        replicate = r.resolve(array(fund))

        logging.debug(f'result: {r}')
        s_expected = [-3.6, 6.5, -2.7]
        for i in range(len(s_expected)):
            self.assertAlmostEqual(replicate[i], s_expected[i], 3)

    def test_decompose_006(self):
        """
        Decomposition in a 3D space, 6 references: 2 x first - 1 x last,
        limited reference points, noisy input
        """
        indices = [[0.85, 1.95, -1.05], [1.22, -0.52, 0.79], [-0.98, -0.89, 4.11], [2.16, 0.78, 1.02],
                   [-0.08, -1.48, -0.12], [5.01, -2.97, 0.42]]
        #
        # target: 2 X first point - 1 X last point
        fund = [-3.4, 7.0, -2.4]
        #
        r = pod.BaseDecomposition(array(indices), epsilon=1E-6, max_iter=30, max_factors=2)
        replicate = r.resolve(array(fund))

        logging.debug(f'result: {r}')
        s_expected = [-3.42444, 6.940277, -2.530706]
        for i in range(len(s_expected)):
            self.assertAlmostEqual(replicate[i], s_expected[i], 3)

    def test_decompose_007(self):
        """
        Decomposing 1 index with 3 assets, 2 time samples,
        minimizing L2 norm in weights space.
        """
        assets = [[-4.0, -1.0], [-2.0, -1.0], [3.0, 4.0]]
        #
        # target
        index = [-2.0, 1.5]
        #
        replication = pod.decompose(index, assets, epsilon=1E-6, max_iter=90, max_factors=3)
        replicate = replication.get_decomposition()
        s_expected = [-2.0, 1.5]
        for i in range(len(s_expected)):
            self.assertAlmostEqual(s_expected[i], replicate[i], 3)

        self.assertAlmostEqual(0.961538, replication.get_reference_weight(0), 4)
        self.assertAlmostEqual(0.6154, replication.get_reference_weight(1), 4)
        self.assertAlmostEqual(0.0, replication.get_reference_weight(2), 4)

        self.assertAlmostEqual(0.0, replication.get_error_norm(), 4)

        self.assertEqual(0, replication.get_principal_component_index(0))
        self.assertEqual(2, replication.get_principal_component_index(1))
        self.assertEqual(1, replication.get_principal_component_index(2))

    def test_decompose_008(self):
        """
        Decomposing 1 index with 3 assets, 2 time samples,
        minimizing L2 norm in weights space.
        """
        assets = [[-4.0, -1.0], [-2.0, -1.0], [3.0, 4.0]]
        #
        # target
        index = [-2.0, 1.5]
        #
        replication = pod.decompose(index, assets, epsilon=1E-6, max_iter=90, max_factors=3, max_weight=0.5)
        replicate = replication.get_decomposition()
        s_expected = [-1.036, 0.777]
        for i in range(len(s_expected)):
            self.assertAlmostEqual(s_expected[i], replicate[i], 3)

        self.assertAlmostEqual(0.3824, replication.get_reference_weight(0), 4)
        self.assertAlmostEqual(0.3652, replication.get_reference_weight(1), 4)
        self.assertAlmostEqual(0.3012, replication.get_reference_weight(2), 4)

    def test_pseudo_fourier_decomposition(self):
        x_axis, y, refs = prepare_fourier(0.0)
        # computes decomposition
        decomposition = pod.decompose(y, refs, epsilon=1E-6, max_iter=20)
        self.assertEqual(5, decomposition.get_principal_component_index(0))
        self.assertEqual(1, decomposition.get_principal_component_index(1))
        self.assertEqual(16, decomposition.get_principal_component_index(2))
        self.assertAlmostEqual(-1.0334, decomposition.get_reference_weight(0), 4)
        self.assertAlmostEqual(0.2775, decomposition.get_reference_weight(1), 4)
        self.assertAlmostEqual(0.1579, decomposition.get_reference_weight(2), 4)

    def test_pseudo_fourier_decomposition_shift(self):
        x_axis, y, refs = prepare_fourier(-0.5 * math.pi)
        # computes decomposition
        decomposition = pod.decompose(y, refs, epsilon=1E-6, max_iter=20)
        self.assertEqual(3, decomposition.get_principal_component_index(0))
        self.assertEqual(33, decomposition.get_principal_component_index(1))
        self.assertEqual(51, decomposition.get_principal_component_index(2))
        self.assertAlmostEqual(-0.9690, decomposition.get_reference_weight(0), 4)
        self.assertAlmostEqual(-0.3085, decomposition.get_reference_weight(1), 4)
        self.assertAlmostEqual(0.21770, decomposition.get_reference_weight(2), 4)

    def test_pseudo_fourier_decomposition_small_shift(self):
        x_axis, y, refs = prepare_fourier(-0.5 * math.pi, n_ref=5)
        # computes decomposition
        decomposition = pod.decompose(y, refs, epsilon=1E-6, max_iter=20)
        self.assertEqual(4, decomposition.get_principal_component_index(0))
        self.assertEqual(9, decomposition.get_principal_component_index(1))
        self.assertEqual(0, decomposition.get_principal_component_index(2))

    def test_decompose_009(self):
        """
        Decomposing with 2 points in 3 dimensions: overdetermined system,
        will find the best approximation given the constraint.
        """
        index1 = [0.8, 2.0, -5.0]
        index2 = [1.2, -0.5, -1.0]
        fund = [1.0, -1., 3.0]
        r = pod.BaseDecomposition(array([index1, index2]), max_iter=40)
        replicate = r.resolve(array(fund))

        logging.debug(f'result: {r}')
        s_expected = [0.3715, -1.7263, 2.6089]
        for i in range(len(s_expected)):
            self.assertAlmostEqual(s_expected[i], replicate[i], 4)

    def test_decompose_010(self):
        """
        Decomposing with 2 points in 3 dimensions: overdetermined system,
        will find the best approximation given the constraint.
        """
        index1 = [0.8, 2.0, 0.]
        index2 = [1.2, -0.5, 0.]
        fund = [1.0, -1., 3.0]
        r = pod.BaseDecomposition(array([index1, index2]), max_iter=40)
        replicate = r.resolve(array(fund))

        logging.debug(f'result: {r}')
        s_expected = [1., -1., 0.]
        for i in range(len(s_expected)):
            self.assertAlmostEqual(s_expected[i], replicate[i], 4)


if __name__ == '__main__':
    unittest.main()
