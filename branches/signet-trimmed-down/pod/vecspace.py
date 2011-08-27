"""
Toolbox for projecting onto straight lines.
"""
import logging

_logger = logging.getLogger('vecspace')

from mathtools import vadd
from mathtools import vsub
from mathtools import sum_product
from mathtools import scale

class VectorSpace(object):
  """
  Essentially a factory for geometrical objects.
  
  Joining an origin to a linear variety would create a VectorSpace.
  """  
  def __init__(self, dimension, basis=None):
    self.dimension = dimension
    self.origin = [0.0] * dimension
    
  def define_point(self, *coordinates):
    return coordinates
    
  def define_line(self, p0):
    sl = StraightLine(p0)
    return sl

class StraightLine(object):
  """
  """
  
  def __init__(self, point):
    """
    """
    self.point = point
    
  def project(self, point):
    """
    Generalizing projection onto induced subspace.
    """
    subspace = VectorSubspace(self.point)
    proj = subspace.project(vsub(point, self.point))
    proj.projected = vadd(proj.projected, self.point)
    return proj
    
  def __repr__(self):
    out = str(self.points) 
    return out

class VectorSubspace(object):
  """
  Defines a set on which we can project a point.
  """
  
  def __init__(self, point):
    """
    Defining a n-subspace generated by n points.
    """
    self.def_point = point

  def project(self, point):
    """
    The solution implies (x - x*) perpendicular to each y[i]
    with x* = sum( alpha[i] * y[i] )
    and y[i]: points generating the subspace.
    """
    space_dim = len(point)
    
    _logger.debug('space_dim: %s' % space_dim)
    
    m_value = sum_product(self.def_point, self.def_point)
    
    b_value = sum_product(point, self.def_point)
    
    alpha = b_value / m_value
    
    result = [0.0] * space_dim
    _logger.debug('result spaceholder: %s' % result)
    
    component = scale(alpha, self.def_point)
    result = vadd(result, component)
    
    return Projection(result, point)
    
class Projection(object):
  """
  Simple container gathering details about projection.
  """
  def __init__(self, projected, start):
    self.projected = projected
    self.start = start
    self.projector = vsub(projected, start)
    
    
