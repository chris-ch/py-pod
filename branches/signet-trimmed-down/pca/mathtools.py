"""
Various vector operations.
"""

import math
from itertools import izip

def scale(alpha, point):
  return ((alpha * value) for value in point)

def sum_product(point1, point2):
  result = 0.0
  for value1, value2 in izip(point1, point2):
    result += value1 * value2
    
  return result
  
def vadd(point1, point2):
  return [(v1 + v2) for v1, v2 in izip(point1, point2)]

def vsub(point1, point2):
  return [(v1 - v2) for v1, v2 in izip(point1, point2)]
  
def norm(point):
  return math.sqrt(sum_product(point, point))

def units(point, unit_vector):
  """
  How many times a point fits in the specified units (in norm).
  
  The sign has a meaning only if both vectors are colinear.
  """
  ratio = norm(point) / norm(unit_vector)
  if norm(vsub(point, unit_vector)) > norm(unit_vector):
    # vectors are in opposite directions
    ratio = -ratio
    
  return ratio
  
def project(point, def_point):
  """
  Projects a point onto the 1-dim subspace defined by def_point.
  """
  denom = sum_product(def_point, def_point)
  num = sum_product(vsub(point, def_point), def_point)    
  component = scale(num / denom, def_point)
  projected = vadd(component, def_point)
  return projected
  
