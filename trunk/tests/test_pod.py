"""
Test cases for the pod package.

@author: Christophe Alexandre <ch.alexandre at bluewin dot ch>

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

import pod

logging.basicConfig(level=logging.DEBUG)

class PodTest(unittest.TestCase):

  def test_decompose_001(self):
    """
    Decomposing with 2 reference points in 2 dimensions
    """
    index1 = [0.8, 2.0]
    index2 = [1.2, -0.5]
    fund = [1.0, 0.5]
    r = pod.DecompositionBasic(index1, index2, max_iter=20)
    replicate = r.resolve(fund)
        
    logging.debug('result: ' + str(r))
    s_expected = [1.0, 0.5]
    for i in range(len(s_expected)):
      self.assertAlmostEqual(replicate[i], s_expected[i], 3)
    
  def test_decompose_002(self):
    """
    Decomposing with 2 points in 3 dimensions: overdetermined system (unusual),
    will find the best approximation given the constraint.
    """
    index1 = [0.8, 2.0, -5.0]
    index2 = [1.2, -0.5, -1.0]
    fund = [1.0, 0.5, 3.0]
    r = pod.DecompositionBasic(index1, index2, max_iter=40)
    replicate = r.resolve(fund)
    
    logging.debug('result: ' + str(r))
    s_expected = [-0.265192, -0.961998, 2.2127700]
    for i in range(len(s_expected)):
      self.assertAlmostEqual(replicate[i], s_expected[i], 4)
    
  def test_decompose_003(self):
    """
    Decomposition in a 3D space, 5 references: 2 x first point
    """
    indices = []
    indices.append([0.8, 2.0, -1.0])
    indices.append([1.2, -0.5, 0.8])
    indices.append([-1.0, -0.9, 4.1])
    indices.append([2.2, 0.8, 1.0])
    indices.append([-0.1, -1.5, -0.1])
    indices.append([5.0, -3.0, 0.4])
    #
    # target: 2 X first point
    fund = [1.6, 4.0, -2.0]
    #
    r = pod.BaseDecomposition(indices, max_iter=20)
    result = r.resolve(fund)
    
    logging.debug('result: ' + str(result))
    s_expected = [1.6, 4.0, -2.0]
    for i in range(len(s_expected)):
      self.assertAlmostEqual(result[i], s_expected[i], 6)
    
  def test_decompose_004(self):
    """
    Decomposition in a 3D space, 6 references: 2 x first - 1 x last
    """
    indices = []
    indices.append([0.8, 2.0, -1.0])
    indices.append([1.2, -0.5, 0.8])
    indices.append([-1.0, -0.9, 4.1])
    indices.append([2.2, 0.8, 1.0])
    indices.append([-0.1, -1.5, -0.1])
    indices.append([5.0, -3.0, 0.4])
    #
    # target: 2 X first point - 1 X last point
    fund = [-3.4, 7.0, -2.4]
    #
    r = pod.BaseDecomposition(indices, epsilon=1E-10, max_iter=20)
    result = r.resolve(fund)
    
    logging.debug('result: ' + str(r))
    s_expected = [-3.4, 7.0, -2.4]
    for i in range(len(s_expected)):
      self.assertAlmostEqual(result[i], s_expected[i], 6)
    
  def test_decompose_005(self):
    """
    Decomposition in a 3D space, 6 references: 2 x first - 1 x last + noise
    """
    indices = []
    indices.append([0.8, 2.0, -1.0])
    indices.append([1.2, -0.5, 0.8])
    indices.append([-1.0, -0.9, 4.1])
    indices.append([2.2, 0.8, 1.0])
    indices.append([-0.1, -1.5, -0.1])
    indices.append([5.0, -3.0, 0.4])
    #
    # target: 2 X first point - 1 X last point + noise
    fund = [-3.6, 6.5, -2.7]
    #
    r = pod.BaseDecomposition(indices, epsilon=1E-6, max_iter=30)
    replicate = r.resolve(fund)
    
    logging.debug('result: ' + str(r))
    s_expected = [-3.6, 6.5, -2.7]
    for i in range(len(s_expected)):
      self.assertAlmostEqual(replicate[i], s_expected[i], 3)
    
  def test_decompose_006(self):
    """
    Decomposition in a 3D space, 6 references: 2 x first - 1 x last,
    limited reference points, noisy input
    """
    indices = []
    indices.append([0.85, 1.95, -1.05])
    indices.append([1.22, -0.52, 0.79])
    indices.append([-0.98, -0.89, 4.11])
    indices.append([2.16, 0.78, 1.02])
    indices.append([-0.08, -1.48, -0.12])
    indices.append([5.01, -2.97, 0.42])
    #
    # target: 2 X first point - 1 X last point
    fund = [-3.4, 7.0, -2.4]
    #
    r = pod.BaseDecomposition(indices, epsilon=1E-6, max_iter=30, max_factors=2)
    replicate = r.resolve(fund)
    
    logging.debug('result: ' + str(r))
    s_expected = [-3.42444, 6.940277, -2.530706]
    for i in range(len(s_expected)):
      self.assertAlmostEqual(replicate[i], s_expected[i], 3)
    
  def test_decompose_007(self):
    """
    Decomposing 1 index with 3 assets, 2 time samples, 
    minimizing L2 norm in weights space.
    """
    assets = []
    assets.append([-4.0, -1.0])
    assets.append([-2.0, -1.0])
    assets.append([3.0, 4.0])
    #
    # target
    index = [-2.0, 1.5]
    #
    replication = pod.decompose(index, assets, epsilon=1E-6, max_iter=90, max_factors=3)
    replicate = replication.get_decomposition()
    s_expected = [-2.0, 1.5]
    for i in range(len(s_expected)):
      self.assertAlmostEqual(replicate[i], s_expected[i], 3)
    self.assertAlmostEqual(replication.get_reference_weight(0), 0.961538, 4)
    self.assertAlmostEqual(replication.get_reference_weight(1), 0.0, 4)
    self.assertAlmostEqual(replication.get_reference_weight(2), 0.61538, 4)
    self.assertAlmostEqual(replication.get_error_norm(), 0.0, 4)
    
if __name__ == '__main__':
    unittest.main()
