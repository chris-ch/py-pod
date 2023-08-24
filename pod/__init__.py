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
>>> print(decomposition.get_decomposition())
[-1.9999991745134178, 1.4999993808850638]
>>> print(decomposition.reference_weights)
[0.96153806466991254, 0.0, 0.61538436138874408]

The example above shows the reconstruction of the target using 3 reference
signals, from which only reference 1 and reference 3 are useful (reference 2
is assigned a weight of 0).

@author: Christophe Alexandre <christophe.alexandre at pm dot me>

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
from __future__ import annotations

__all__ = ['decompose']

import os
import logging
from typing import Iterable, List, Optional

from numpy import ndarray, array, zeros, matmul, subtract, multiply, dot, argmin, zeros_like
from numpy.linalg import linalg


def decompose(source: List[float], references: List[List[float]],
              epsilon: float = 1E-10, max_iter: int = 20,
              max_factors: Optional[int] = None, max_weight: Optional[float] = None
              ) -> BaseDecomposition:
    """
    Decomposing the source using the proposed reference points.
    
    @param max_weight:
    @param source: input point
    @param references: list of reference points
    @param epsilon: limit of the error sequence for stopping iteration
    @param max_iter: safeguard for stopping iteration
    @param max_factors: limit for the number of reference points, None allowing to use all of them
    @return: decomposition details
    @rtype: IterativeDecomposition
    """
    r = BaseDecomposition(array(references), epsilon, max_iter, max_factors, max_weight)
    r.resolve(array(source))
    return r


class BaseDecomposition(object):
    """
    Decomposition interface definition.
    """

    def __init__(self, references: ndarray, epsilon: float = 1E-10,
                 max_iter: int = 20, max_factors: Optional[int] = None,
                 max_weight: Optional[float] = None):
        self._max_weight = max_weight
        if references.shape == (0,):
            raise ValueError('at least one reference is required')

        reference_points: List[ndarray] = []
        self._ignores = []
        for count, row in enumerate(references):
            if row.tolist() in (item.tolist() for item in reference_points):
                logging.warning(f'filtered out redundant vector {count:d}')
                self._ignores.append(count)

            elif linalg.norm(row) == 0.0:
                logging.warning(f'filtered out vector at origin {count:d}')
                self._ignores.append(count)

            reference_points.append(row)

        self._reference_points = array(reference_points)
        self._weights = zeros(len(self._reference_points))
        self._epsilon = epsilon
        self._start = None
        self._max_iter = max_iter
        if max_factors is None:
            self._max_factors = len(self._reference_points)

        else:
            self._max_factors = max_factors

        self._error_norm = None

    def _compute_decomposition(self) -> ndarray:
        """
        Computes current decomposition result on the fly.
        """
        return matmul(self._weights, self._reference_points)

    def get_reference_weight(self, position: int) -> float:
        """
        Returns the weight assigned to the reference provided in the constructor
        at the indicated position.
        
        @param position: position of the reference in the list provided in the constructor
        """
        return self._weights[self.get_principal_component_index(position)]

    @property
    def reference_weights(self) -> ndarray:
        """
        Returns the weights assigned to the references in order to construct the
        proposed input.
        """
        return self._weights

    def get_decomposition(self) -> ndarray:
        """
        Returns the result of the decomposition process.
        
        @return: decomposition result
        """
        return self._compute_decomposition()

    def get_error_norm(self) -> float:
        """
        Returns a measure of the decomposition error.
        
        @return: length of the difference between the result and the initial point
        """
        return linalg.norm(subtract(self._compute_decomposition(), self._start))

    def get_principal_component(self, rank: int):
        """
        Returns the rank-th reference influencing the input variable
        (main component: rank = 0), multiplied by its assigned weight.
        
        @param rank: the rank of the reference (0 means principal component)
        @return: a reference vector
        """
        ref_weights = self.reference_weights
        sorted_weights = [(pos - 1, weight) for pos, weight in enumerate(ref_weights)]
        sorted_weights.sort(key=lambda x: abs(x[1]), reverse=True)
        max_abs_weight_pos = sorted_weights[rank][0]
        weight = sorted_weights[rank][1]
        main_component = multiply(self._reference_points[max_abs_weight_pos], weight)
        return main_component.to_list()

    def get_principal_component_index(self, rank: int) -> int:
        """
        Returns the position in the initial reference list of the rank-th 
        reference influencing the input variable (main component: rank = 0).
        
        @param rank: the rank of the reference (0 means principal component)
        @return: position in the initial reference list
        """
        ref_weights = self.reference_weights
        sorted_weights = list(enumerate(ref_weights))
        sorted_weights.sort(key=lambda x: abs(x[1]), reverse=True)
        return sorted_weights[rank][0]

    def __repr__(self):
        out = f'reference points:{os.linesep}'
        for p in self._reference_points:
            out += str(p) + os.linesep
        out += f'weightings:{os.linesep}'
        out += str(self._weights)
        return out

    def _project_point(self, point: ndarray, reference_points_indices: Iterable[int]) -> ndarray:
        """ Projects onto the closest of the straight lines defined by 
        the reference points.
        """
        # computes projection of source point to each subspace defined by ref points
        if linalg.norm(point) <= self._epsilon:
            # already matched: do nothing
            return point

        def project(vector: ndarray, support: ndarray) -> ndarray:
            return multiply(support, dot(vector, support)) / dot(support, support)

        def units(vector: ndarray, unit_vector: ndarray) -> float:
            """
            How many times the vector fits in the specified units (in norm).

            The sign is meaningful only if both vectors are colinear.
            """
            ratio = linalg.norm(vector) / linalg.norm(unit_vector)
            if dot(vector, unit_vector) < 0:
                # vectors are in opposite directions
                ratio = -ratio
            return ratio

        ref_points_indices = []
        for count in reference_points_indices:
            if self._max_weight is None:
                ref_points_indices.append(count)

            else:
                additional_weight = units(project(point, self._reference_points[count]), self._reference_points[count])
                suggested_weight = self._weights[count] + additional_weight
                if abs(suggested_weight) < self._max_weight:
                    ref_points_indices.append(count)

        if not ref_points_indices:
            # no eligible point left
            return point

        # finds main driver (shortest distance to ref line)
        distances = array(
            [linalg.norm(subtract(point, project(point, self._reference_points[ref_point_index]))) for
             ref_point_index in ref_points_indices])

        closest_position = argmin(distances)
        closest_point = self._reference_points[ref_points_indices[closest_position]]
        projected = project(point, closest_point)
        self._weights[ref_points_indices[closest_position]] += units(projected, closest_point)
        logging.debug(
            f'closest driver: {str(closest_point)}, weight={str(self._weights[ref_points_indices[closest_position]])}')

        return subtract(point, projected)

    def resolve(self, point: ndarray) -> ndarray:
        """
        Iterates decomposition until convergence or max iteration is reached.
        
        @param point: coordinates of the point to decompose
        @return: coordinates of decomposed point
        """
        logging.debug(' ------------- STARTING PROCESS -------------')
        self._start = point
        reference_points_indices = [count for count in range(len(self._reference_points)) if count not in self._ignores]
        projector = self._project_point(array(point), reference_points_indices)
        diff = None
        logging.debug(f'distance to projection: {linalg.norm(projector):f}')
        i = 0
        decomposition = None
        while (diff is None) or (diff > self._epsilon and i < self._max_iter):
            i += 1
            previous = self._compute_decomposition()
            logging.debug(f' ------------- ITERATION {i:d} -------------')
            projector = self._project_point(projector, reference_points_indices)

            enabled_drivers = [count for count in reference_points_indices if abs(self._weights[count]) > 0.0]
            drivers_count = len(enabled_drivers)
            logging.debug(f'number of drivers: {drivers_count:d}')
            if drivers_count >= self._max_factors:
                # limits number of drivers
                logging.debug('count limit reached for drivers: %d' % self._max_factors)
                reference_points_indices = enabled_drivers

            decomposition = self._compute_decomposition()
            diff = linalg.norm(subtract(decomposition, previous))
            logging.debug('improvement: %f' % diff)
            logging.debug(f'distance to projection: {linalg.norm(projector):f}')
            logging.debug(f'decomposition: {str(decomposition)}')

        logging.debug(f'start:{array(point)}')
        if decomposition.any():
            logging.debug(f'diff:{subtract(decomposition, array(point))}')
        return decomposition
