"""
Various functions operating on vectors.
"""

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
