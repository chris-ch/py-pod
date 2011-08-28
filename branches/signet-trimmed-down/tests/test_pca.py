"""
Test cases for the pca package.
"""

import unittest
import math
import logging
from decimal import Decimal

import pca

logging.basicConfig(level=logging.INFO)

def prepare_fourier(shift):
  N_REF = 30
  SAMPLING = 5 * N_REF
  AMPLITUTDE = 100.0
  FREQ_STEP = 0.2
  PHASE_SHIFT = shift
  
  def gen_x_axis(x0, n=SAMPLING):
     return float(x0) * 2 * math.pi / n
     
  def square_signal(x0, ampl=AMPLITUTDE, shift=PHASE_SHIFT):
     if x0 > math.pi + shift and x0 < 2.0 * math.pi + shift:
        return 1.0 * ampl
     else:
        return 0.0
  
  def sine(ampl, freq, axis):
     output = [ampl * math.sin(float(x0) * freq) for x0 in axis]
     return output
     
  def cosine(ampl, freq, axis):
     output = [ampl * math.cos(float(x0) * freq) for x0 in axis]
     return output
   
  # x axis
  x_axis = map(gen_x_axis, range(SAMPLING))
    
  # input signal
  y = map(square_signal, x_axis)
  
  # creates set of sine and cosine functions
  
  y_i = []
  
  freq_range = range(Decimal('-0.2') * N_REF, (Decimal('0.8') * N_REF))
  
  for freq in freq_range:
    y_i.append(sine(AMPLITUTDE, float(FREQ_STEP * freq), x_axis))
  
  z_i = []
  for freq in freq_range:
    z_i.append(cosine(AMPLITUTDE, float(FREQ_STEP * freq), x_axis))
  
  refs = y_i + z_i
  
  return x_axis, y, refs 
  
  
class TestPCA(unittest.TestCase):
  """
  Testing the PCA algorithm.
  
  """

  def test_projection_001(self):
    """
    Projecting on a 2D line
    """
    p1 = (0.01, 0.02)
    p = (0.02, 0.03)
    p_expected = (0.08/5.0, 0.16/5.0)
    projected = pca.project(p, p1)
    self.assertEqual(projected, list(p_expected))

  def test_components_001(self):
    """
    Analysing with 2 reference points in 2 dimensions
    """
    index1 = [0.8, 2.0]
    index2 = [1.2, -0.5]
    fund = [1.0, 0.5]
    r = pca.ComponentAnalysis([index1, index2], max_iter=20)
    replicate = r.solve(fund)
        
    logging.debug('result: ' + str(r))
    s_expected = [1.0, 0.5]
    for i in range(len(s_expected)):
      self.assertAlmostEqual(replicate[i], s_expected[i], 3)
    
  def test_components_002(self):
    """
    Analysing with 2 points in 3 dimensions: overdetermined system (unusual),
    finding the best approximation given the constraints.
    """
    index1 = [0.8, 2.0, -5.0]
    index2 = [1.2, -0.5, -1.0]
    fund = [1.0, 0.5, 3.0]
    r = pca.ComponentAnalysis([index1, index2], max_iter=40)
    replicate = r.solve(fund)
    
    logging.debug('result: ' + str(r))
    s_expected = [-0.265192, -0.961998, 2.2127700]
    for i in range(len(s_expected)):
      self.assertAlmostEqual(replicate[i], s_expected[i], 4)
    
  def test_components_003(self):
    """
    Component analysis in a 3D space, 5 references: 2 x first point
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
    r = pca.ComponentAnalysis(indices, max_iter=20)
    result = r.solve(fund)
    
    logging.debug('result: ' + str(result))
    s_expected = [1.6, 4.0, -2.0]
    for i in range(len(s_expected)):
      self.assertAlmostEqual(result[i], s_expected[i], 6)
    
  def test_components_004(self):
    """
    Component analysis in a 3D space, 6 references: 2 x first - 1 x last
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
    r = pca.ComponentAnalysis(indices, epsilon=1E-10, max_iter=20)
    result = r.solve(fund)
    
    logging.debug('result: ' + str(r))
    s_expected = [-3.4, 7.0, -2.4]
    for i in range(len(s_expected)):
      self.assertAlmostEqual(result[i], s_expected[i], 6)
    
  def test_components_005(self):
    """
    Component analysis in a 3D space, 6 references: 2 x first - 1 x last + noise
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
    r = pca.ComponentAnalysis(indices, epsilon=1E-6, max_iter=30)
    replicate = r.solve(fund)
    
    logging.debug('result: ' + str(r))
    s_expected = [-3.6, 6.5, -2.7]
    for i in range(len(s_expected)):
      self.assertAlmostEqual(replicate[i], s_expected[i], 3)
    
  def test_components_006(self):
    """
    Component analysis in a 3D space, 6 references: 2 x first - 1 x last,
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
    r = pca.ComponentAnalysis(indices, epsilon=1E-6, max_iter=30, max_factors=2)
    replicate = r.solve(fund)
    
    logging.debug('result: ' + str(r))
    s_expected = [-3.42444, 6.940277, -2.530706]
    for i in range(len(s_expected)):
      self.assertAlmostEqual(replicate[i], s_expected[i], 3)
    
  def test_components_007(self):
    """
    Analysing 1 index with 3 assets, 2 time samples
    """
    assets = []
    assets.append([-4.0, -1.0])
    assets.append([-2.0, -1.0])
    assets.append([3.0, 4.0])
    #
    # target
    index = [-2.0, 1.5]
    #
    replication = pca.find_components(index, assets, epsilon=1E-6, max_iter=90, max_factors=3)
    replicate = replication.get_replicate()
    s_expected = [-2.0, 1.5]
    for i in range(len(s_expected)):
      self.assertAlmostEqual(replicate[i], s_expected[i], 3)
      
    self.assertAlmostEqual(replication.get_weightings()[0], 0.961538, 4)
    self.assertAlmostEqual(replication.get_weightings()[1], 0.0, 4)
    self.assertAlmostEqual(replication.get_weightings()[2], 0.61538, 4)
    self.assertAlmostEqual(replication.get_error_norm(), 0.0, 4)
    
  def test_pseudo_fourier_decomposition(self):
    x_axis, y, refs = prepare_fourier(0.0)
    # computes decomposition
    decomposition = pca.find_components(y, refs, epsilon=1E-6, max_iter=20)
    approx = decomposition.get_replicate()
    comp_0 = decomposition.get_component(0)
    comp_1 = decomposition.get_component(1)
    comp_2 = decomposition.get_component(2)
    
    self.assertEqual(decomposition.get_component_index(0), 4)
    self.assertEqual(decomposition.get_component_index(1), 10)
    self.assertEqual(decomposition.get_component_index(2), 15)

  def test_pseudo_fourier_decomposition_shift(self):
    x_axis, y, refs = prepare_fourier(-0.5 * math.pi)
    # computes decomposition
    decomposition = pca.find_components(y, refs, epsilon=1E-6, max_iter=20)
    approx = decomposition.get_replicate()
    comp_0 = decomposition.get_component(0)
    comp_1 = decomposition.get_component(1)
    comp_2 = decomposition.get_component(2)
            
    self.assertEqual(decomposition.get_component_index(0), 8)
    self.assertEqual(decomposition.get_component_index(1), 0)
    self.assertEqual(decomposition.get_component_index(2), 50)

    
if __name__ == '__main__':
    unittest.main()
