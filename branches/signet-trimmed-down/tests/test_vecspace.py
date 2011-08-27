"""
Test cases for the vecspace package.

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

from pod import vecspace

class VecSpaceTest(unittest.TestCase):

  def test_projection_001(self):
    """
    Projecting on a 2D line
    """
    space = vecspace.VectorSpace(2)
    p1 = space.define_point(0.01, 0.02)
    p = space.define_point(0.02, 0.03)
    p_expected = space.define_point(0.08/5.0, 0.16/5.0)
    line = space.define_line(p1)
    proj = line.project(p)
    self.assertEqual(proj.projected, p_expected)
    
if __name__ == '__main__':
    unittest.main()
