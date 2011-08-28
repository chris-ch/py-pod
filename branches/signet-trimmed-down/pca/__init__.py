"""
Framework for performing a Proper Orthogonal ComponentAnalysis (POD).

Usage example:

>>> import pod
>>> refs = [ [-4.0, -1.0],
>>>          [-2.0, -1.0],
>>>          [3.0, 4.0] ]
>>> target = [-2.0, 1.5]
>>> decomposition = pod.find_components(target, refs, epsilon=1E-6, max_iter=90)
>>> print decomposition.get_replicate()
[-1.9999991745134178, 1.4999993808850638]
>>> print decomposition.get_weightings()
[0.96153806466991254, 0.0, 0.61538436138874408]

The example above shows the reconstruction of the target using 3 reference
signals, from which only reference 1 and reference 3 are useful (reference 2
is assigned a weight of 0).

"""
__all__ = ['find_components', 'combined_distance', 'ComponentAnalysis']

import math
import logging
import os

_logger = logging.getLogger('pca')

from mathtools import vadd
from mathtools import vsub
from mathtools import sum_product
from mathtools import scale
from mathtools import norm
from mathtools import units
from mathtools import project

def combined_distance(generator_weight):
  """
  Distance function used for ordering the projections.
  
  A weight of 0.0 defines the distance to the projected point, while a weight
  of 1.0 defines the distance relative to the point generating the line.
  
  At each iteration step the current point is projected onto the closest of
  all lines.
  
  @param generator_weight: how much weight is assigned to the generator point
  @type generator_weight: float usually in range [0.0, 1.0]
  """
  
  def func(start_point, projected_point, 
                              generator_point, w=generator_weight):
    d1 = norm(vsub(start_point, generator_point))
    d2 = norm(vsub(start_point, projected_point))
    return w * d1 + (1.0 - w) * d2
  
  return func
  
def find_components(source, references, 
      epsilon=1E-10, max_iter=20, 
      max_factors=None,
      distance=combined_distance(0.0)
      ):
    """
    Decomposing the source using the proposed reference points.
    
    @param source: input point
    @type source: list
    @param references: list of reference points
    @type references: list
    @param epsilon: limit of the error sequence for stopping iterations
    @type epsilon: float
    @param max_iter: safeguard for stopping iterations
    @type max_iter: int
    @param max_factors: limit for the number of reference points, None allowing to use all of them
    @type max_factors: int
    @param distance: function used for finding the closest line to project on
    @type distance: a function of start point, projected point, generator point
    @return: decomposition details
    @rtype: IterativeComponentAnalysis
    """
    r = ComponentAnalysis(references, epsilon, max_iter, max_factors)
    r.solve(source)
    return r

class ComponentAnalysis(object):
    """
    ComponentAnalysis interface definition.
    """
    
    def __init__(self, references, epsilon=1E-10, max_iter=20, max_factors=None,
            distance=combined_distance(0.0)):
        """
        @param distance: function of start point, projected point and generator point
        """
        
        # basic consistency checks
        assert len(references) > 0, 'at least one reference is required'
        dim = len(references[0])
        for r in references:
            assert len(r) == dim, 'all points have to be of the same dimension'
        
        self._distance = distance
        self._dimension = dim
        
        self._reference_points = []
        self._ignores = []
        for count, r in enumerate(references):
            ref = tuple(r)
            if ref in self._reference_points:
                _logger.warning('filtered out redundant reference %d' % count)
                self._ignores.append(ref)
                
            elif norm(ref) == 0.0:
                _logger.warning('filtered out reference at origin %d' % count)
                self._ignores.append(ref)
            
            self._reference_points.append(ref)
            
        self._weights = dict()
        for p in self._reference_points:
            self._weights[p] = 0.0
            
        self._epsilon = epsilon
        self._start = None
        self._max_iter = max_iter
        if max_factors is None:
            self._max_factors = len(self._reference_points)
            
        else:
            self._max_factors = max_factors
            
        self._error_norm = None
        
    def _compute_replicate(self):
        """
        Computes current decomposition result on the fly.
        
        @return: the current relicate
        @rtype: linalg.Point
        """
        decomposition = [0.0] * self._dimension
        for d in self._weights.keys():
            w_d = self._weights[d]
            decomposition = vadd(decomposition, scale(w_d, d))
        return decomposition
    
    def _project_shortest(self, point, reference_points):
        """ Returns the projector that projects onto the closest of the 1-dim 
        subspaces defined by the reference points.
        """
        origin = [0.0] * self._dimension
        if norm(vsub(point, origin)) <= self._epsilon:
            # already matched: we are done
            return point
            
        projections = dict()
        distances = dict()
        for ref in reference_points:
            projections[ref] = project(point, ref)
            distances[ref] = self._distance(point, projections[ref], ref)
            _logger.debug('distance to reference %.3f' % distances[ref])
                
        if len(reference_points) == 0:
            # no eligible point left
            return point
        
        # finds main driver (shortest distance to ref line)
        def by_dist(ref1, ref2, d=distances):
            return cmp(d[ref1], d[ref2])
            
        reference_points.sort(by_dist)
        
        closest = reference_points[0]
        
        additional_weight = units(projections[closest], closest)
        self._weights[closest] += additional_weight
        
        return vsub(point, projections[closest])
    
    def solve(self, point):
        """
        Iterates decomposition until convergence or max iteration is reached.
        
        @param point: coordinates of the point to be decompositiond
        @type point: list
        @return: coordinates of decomposition point
        @rtype: list
        """
        _logger.info(' ------------- STARTING PROCESS -------------')
        self._start = tuple(point)
        reference_points = [ref for ref in self._reference_points if ref not in self._ignores]
        projector = self._start
        step = 1
        _logger.info('iteration count limit at %d' % self._max_iter)
        while True:
            previous = self._compute_replicate()
            _logger.info(' ------------- ITERATION %d -------------' % step)
            _logger.info('starting length: %f' % norm(projector))
            projector = self._project_shortest(projector, reference_points)
            enabled_drivers = [p for p in self._weights.keys() 
                                  if abs(self._weights[p]) > 0.0]
            
            _logger.debug('number of drivers: %d' % len(enabled_drivers))
            if len(enabled_drivers) >= self._max_factors:
                # limits number of drivers
                _logger.info('count limit reached for drivers: %d' % self._max_factors)
                reference_points = enabled_drivers
                
            replicate = self._compute_replicate()
            diff = norm(vsub(replicate, previous))
            _logger.info('replication improvement after iteration: %.03f' % diff)
            step = step + 1
            if diff <= self._epsilon or step > self._max_iter:
                break
        
        _logger.info('diff: %.03f' % diff)
        _logger.info('remainder length: %.03f' % norm(vsub(replicate, self._start)))
        return replicate

    def get_weightings(self):
        """
        Returns the weights assigned to the references in order to construct the
        suggested input.
        """
        return [self._weights[self._reference_points[i]]
                  for i in xrange(len(self._reference_points))]
        
    def get_replicate(self):
        """
        Returns the result of the decomposition process.
        
        @return: decomposition result
        @rtype: list
        """
        return self._compute_replicate()
    
    def get_error_norm(self):
        """
        Returns a measure of the replication error.
        
        @return: length of the difference between the result and the initial point
        @rtype: float
        """
        return norm(vsub(self._compute_replicate(), self._start))
    
    def get_component(self, rank):
        """
        Returns the rank-th reference influencing the input variable
        (main component: rank = 0), multiplied by its assigned weight.
        
        @param rank: the rank of the reference (0 means principal component)
        @return: a reference vector
        """
        ref_weights = self.get_weightings()
        sorted_weights = [(pos - 1, weight)
                                for pos, weight in enumerate(ref_weights)]
        sorted_weights.sort(lambda w1, w2: cmp(abs(w2[1]), abs(w1[1])))
        max_abs_weight_pos = sorted_weights[rank][0]
        weight = sorted_weights[rank][1]
        main_component = scale(weight, self._reference_points[max_abs_weight_pos])
        return main_component
        
    def get_component_index(self, rank):
        """
        Returns the position in the initial reference list of the rank-th 
        reference influencing the input variable (main component: rank = 0).
        
        @param rank: the rank of the reference (0 means principal component)
        @return: position in the initial reference list
        """
        ref_weights = self.get_weightings()
        sorted_weights = [(pos - 1, weight)
                                for pos, weight in enumerate(ref_weights)]
        sorted_weights.sort(lambda w1, w2: cmp(abs(w2[1]), abs(w1[1])))
        max_abs_weight_pos = sorted_weights[rank][0]
        return max_abs_weight_pos
    
    def __repr__(self):
        out = 'reference points:' + os.linesep
        for p in self._reference_points:
            out += str(p) + os.linesep
        out += 'weightings:' + os.linesep
        out += str(self._weights)
        return out
  
