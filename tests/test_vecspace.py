"""
Test cases for the vecspace package.
"""

import unittest
import math
import logging

from pod import vecspace

class VecSpaceTest(unittest.TestCase):

  def test_projection_001(self):
    """
    Projecting on a 2D line
    """
    space = vecspace.VectorSpace(2)
    p1 = [0.01, 0.02]
    p = [0.02, 0.03]
    p_expected = [0.08/5.0, 0.16/5.0]
    line = space.define_line(p1)
    proj = line.project(p)
    self.assertEqual(proj.projected, list(p_expected))
    
if __name__ == '__main__':
    unittest.main()
