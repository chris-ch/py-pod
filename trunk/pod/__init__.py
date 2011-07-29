"""
Framework for performing a Proper Orthogonal Decomposition (POD).

Useful references:
  - http://en.wikipedia.org/wiki/Homogeneous_coordinates
  - http://en.wikipedia.org/wiki/Transformation_matrix#Affine_transformations

Usage example:

>>> import pod
>>> refs = [ [-4.0, -1.0],
>>>          [-2.0, -1.0],
>>>          [3.0, 4.0] ]
>>> target = [-2.0, 1.5]
>>> decomposition = pod.decompose(target, refs, epsilon=1E-6, max_iter=90)
>>> print decomposition.get_decomposition()
[-1.9999991745134178, 1.4999993808850638]
>>> print decomposition.get_reference_weights()
[0.96153806466991254, 0.0, 0.61538436138874408]

The example above shows the reconstruction of the target using 3 reference
signals, from which only reference 1 and reference 3 are useful (reference 2
is assigned a weight of 0).

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

import os
import logging

import vecspace
from util import NullHandler

_h = NullHandler()
_logger = logging.getLogger('pod')
_logger.addHandler(_h)

def decompose(source, references, epsilon=1E-10, max_iter=20, max_factors=None, max_weight=None):
    """
    Decomposing the source using the proposed reference points.
    
    @param source: input point
    @type source: list
    @param references: list of reference points
    @type references: list
    @param epsilon: limit of the error sequence for stopping iteration
    @type epsilon: float
    @param max_iter: safeguard for stopping iteration
    @type max_iter: int
    @param max_factors: limit for the number of reference points, None allowing to use all of them
    @type max_factors: int
    @return: decomposition details
    @rtype: IterativeDecomposition
    """
    r = L2NormMinDecomposition(references, epsilon, max_iter, max_factors, max_weight)
    r.resolve(source)
    return r

class IterativeDecomposition(object):
    """
    Decomposition interface definition.
    """
    
    def __init__(self, references, epsilon=1E-10, max_iter=20, max_factors=None):
        assert len(references) > 0, 'at least one reference is required'
        dim = len(references[0])
        for r in references:
            assert len(r) == dim
        self._vector_space = vecspace.VectorSpace(dim)
        self._reference_points = []
        for r in references:
            ref = self._vector_space.define_point(*r)
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
        
    def _compute_decomposition(self):
        """
        Computes current decomposition result on the fly.
        
        @return: the current relicate
        @rtype: linalg.Point
        """
        decomposition = self._vector_space.origin
        for d in self._weights.keys():
            w_d = self._weights[d]
            decomposition = decomposition.add(d.scale(w_d))
        return decomposition
      
    def resolve(self, point):
        """
        Iterates decomposition until convergence or max iteration is reached.
        
        @param point: coordinates of the point to be decompositiond
        @type point: list
        @return: coordinates of decompositiond point
        @rtype: list
        """
        pass
    
    def get_reference_weight(self, position):
        """
        Returns the weight assigned to the reference provided in the constructor
        at the indicated position.
        
        @param position: position of the reference in the list provided in the constructor
        @type position: int
        """
        ref = self._reference_points[position]
        return self._weights[ref]
        
    def get_reference_weights(self):
        """
        Returns the weights assigned to the references in order to construct the
        proposed input.
        """
        return [self._weights[self._reference_points[i]]
                  for i in range(len(self._reference_points))]
    
    def get_decomposition(self):
        """
        Returns the result of the decomposition process.
        
        @return: decomposition result
        @rtype: list
        """
        return self._compute_decomposition().get_data()
    
    def get_error_norm(self):
        """
        Returns a measure of the decomposition error.
        
        @return: length of the difference between the result and the initial point
        @rtype: float
        """
        return self._compute_decomposition().sub(self._start).norm()
    
    def __repr__(self):
        out = 'reference points:' + os.linesep
        for p in self._reference_points:
            out += str(p) + os.linesep
        out += 'weightings:' + os.linesep
        out += str(self._weights)
        return out
        
class L2NormMinDecomposition(IterativeDecomposition):
    """
    Replication minimizing L2 norm in the weights space: min sum(|Wi|).
    """
    
    def __init__(self, references, epsilon=1E-10, max_iter=20, max_factors=None, max_weight=None):
        IterativeDecomposition.__init__(self, references, epsilon, max_iter, max_factors)
        self._max_weight = max_weight
    
    def _project_point(self, point, reference_points):
        """ Projects onto the closest of the straight lines defined by 
        the reference points.
        """
        # computes projection of source point to each subspace defined by ref points
        origin = self._vector_space.origin
        if point.sub(origin).norm() <= self._epsilon:
            # already matched: do nothing
            return point
          
        _logger.debug('projecting ' + str(point))      
        projections = dict()
        distances = dict()
        for ref in reference_points:
            line = self._vector_space.define_line(ref, origin)
            _logger.debug('computing projection onto ' + str(line))
            ref_proj = line.project(point)
            projections[ref] = ref_proj.projected
            distances[ref] = ref_proj.projector.norm()
            _logger.debug('distance ' + str(distances[ref]))
          
        ref_points = []
        for p in reference_points:
            if self._max_weight is None:
                ref_points.append(p)
            else:
                additional_weight = projections[p].units(p)
                suggested_weight = self._weights[p] + additional_weight
                if abs(suggested_weight) < self._max_weight:
                    ref_points.append(p)
                
        if len(ref_points) == 0:
            # no eligible point left
            return point
        
        # finds main driver (shortest distance to ref line)
        def by_dist(ref1, ref2, d=distances):
            return cmp(d[ref1], d[ref2])
        ref_points.sort(by_dist)
        
        closest = ref_points[0]            
        additional_weight = projections[closest].units(closest)
        self._weights[closest] += additional_weight
        _logger.debug('closest driver: %s, weight=%f' % (str(closest), self._weights[closest]))
        
        return point.sub(projections[closest])
    
    def resolve(self, point):
        """
        Iterates decomposition until convergence or max iteration is reached.
        
        @param point: coordinates of the point to be decompositiond
        @type point: list
        @return: coordinates of decompositiond point
        @rtype: list
        """
        IterativeDecomposition.resolve(self, point)
        _logger.debug(' ------------- STARTING PROCESS -------------')
        target = self._vector_space.define_point(*point)
        self._start = target
        reference_points = self._reference_points
        projector = self._project_point(target, reference_points)
        diff = None
        _logger.debug('distance to projection: %f' % projector.norm())
        _logger.debug('drivers: ' + str(self._weights))
        i = 0
        while (diff is None) or (diff > self._epsilon and i < self._max_iter):
            i = i + 1
            previous = self._compute_decomposition()
            _logger.debug(' ------------- ITERATION %d -------------' % i)
            projector = self._project_point(projector, reference_points)
            enabled_drivers = [p for p in self._weights.keys() if abs(self._weights[p]) > 0.0]
            drivers_count = len(enabled_drivers)
            _logger.debug('number of drivers: %d' % drivers_count)
            if drivers_count >= self._max_factors:
                # limits number of drivers
                _logger.debug('count limit reached for drivers: %d' % self._max_factors)
                reference_points = enabled_drivers
            decomposition = self._compute_decomposition()
            diff = decomposition.sub(previous).norm()
            _logger.debug('improvement: %f' % diff)
            _logger.debug('distance to projection: %f' % projector.norm())
            _logger.debug('drivers: ' + str(self._weights))
            _logger.debug('decomposition: ' + str(decomposition))
        
        _logger.debug('start:' + str(target))
        _logger.debug('diff:' + str(decomposition.sub(target)))
        return decomposition.get_data()
    
class DecompositionBasic(L2NormMinDecomposition):
  
    def __init__(self, ref1, ref2, max_iter):
        L2NormMinDecomposition.__init__(self, [ref1, ref2], max_iter=max_iter)
    
