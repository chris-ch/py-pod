"""
Various functions operating on vectors.
"""

import math

def scale(alpha, point):
  return [(alpha * value) for value in point]

def sum_product(point1, point2):
  result = 0.0
  for index, value1 in enumerate(point1):
    result += value1 * point2[index]
  return result
  
def vadd(point1, point2):
  return [(v1 + v2) for v1, v2 in zip(point1, point2)]

def vsub(point1, point2):
  return [(v1 - v2) for v1, v2 in zip(point1, point2)]
  
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
  
