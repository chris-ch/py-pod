"""
Operations on matrices and various tools.

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

import math
import logging

class NullHandler(logging.Handler):
  """
  Null logging in order to avoid warning messages in client applications.
  """
  def emit(self, record):
    pass
  
_h = NullHandler()
_logger = logging.getLogger('util')
_logger.addHandler(_h)

def numbering(v):
  """ Maps every element of to its position."""
  return zip(range(len(v)), v)

def prod_scalar(v1, v2):
  assert len(v1) == len(v2), 'input vectors must be of the same size'
  prod = map(lambda x: x[0] * x[1], zip(v1, v2))
  return sum(prod)

def norm(v):
  return math.sqrt(prod_scalar(v, v))

# Auxiliary functions contribution by Eric Atienza

def mat_inverse(M):
  """
  @return: the inverse of the matrix M
  """
  #clone the matrix and append the identity matrix
  # [int(i==j) for j in range_M] is nothing but the i(th row of the identity matrix
  m2 = [row[:]+[int(i==j) for j in range(len(M) )] for i,row in enumerate(M) ]
  # extract the appended matrix (kind of m2[m:,...]
  return [row[len(M[0]):] for row in m2] if gauss_jordan(m2) else None

def zeros(size, zero=0):
  """
  @param size: a tuple containing dimensions of the matrix
  @param zero: the value to use to fill the matrix (zero by default)
  @return: matrix of dimension size
  """
  return [zeros(s[1:] ) for i in range(s[0] ) ] if not len(s) else zero


