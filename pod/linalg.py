"""
Basic linear algebra components and functions.

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

import math
import os
import logging
from typing import Optional, List

from pod.util import NullHandler
from pod.util import prod_scalar

_h = NullHandler()
_logger = logging.getLogger('linalg')
_logger.addHandler(_h)


class Vector(object):
    """
    """

    def __init__(self, dim: int):
        self._dim = dim
        self._values = {}

    @property
    def dimension(self) -> int:
        return self._dim

    def set_values(self, *values):
        """ Using specified values to initialize the vector. """
        for n, v in enumerate(values):
            self.set_component(n, v)
        return self

    def set_data(self, data):
        """ Using raw data (python list) to initialize the vector. """
        self.set_values(*data)

    def get_component(self, i: int) -> float:
        assert i < self.dimension, 'index %d exceeding dimension %d' % (i, self.dimension)
        assert i >= 0, 'non positive index %d' % i
        return 0.0 if i not in self._values else self._values[i]

    def set_component(self, i: int, v: float) -> None:
        assert i < self.dimension, 'index %d exceeding dimension %d' % (i, self.dimension)
        assert i >= 0, 'non positive index %d' % i
        self._values[i] = v

    def get_data(self) -> List[float]:
        """ Raw data as built-in python list."""
        return [self.get_component(i) for i in range(self.dimension)]

    def sub(self, vector: Vector) -> Vector:
        result = Vector(self.dimension)
        for i in range(self.dimension):
            result.set_component(i, self.get_component(i) - vector.get_component(i))
        return result

    def add(self, vector: Vector) -> Vector:
        result = Vector(self.dimension)
        for i in range(self.dimension):
            result.set_component(i, self.get_component(i) + vector.get_component(i))
        return result

    def scale(self, a: float) -> Vector:
        result = Vector(self.dimension)
        for i in range(self.dimension):
            result.set_component(i, a * self.get_component(i))
        return result

    def product(self, vector: Vector) -> float:
        return prod_scalar(vector.get_data(), self.get_data())

    def norm(self) -> float:
        return math.sqrt(self.product(self))

    def units(self, unit_vector: Vector) -> float:
        """
        How many times the current vector fits in the specified units (in norm).

        The sign has a meaning only if both vectors are colinear.
        """
        ratio = self.norm() / unit_vector.norm()
        if self.sub(unit_vector).norm() > unit_vector.norm():
            # vectors are in opposite directions
            ratio = -ratio
        return ratio

    def __eq__(self, other: Vector) -> bool:
        return self._values == other._values

    def __ne__(self, other: Vector) -> bool:
        return self._values != other._values

    def __repr__(self) -> str:
        line = [self.get_component(col) for col in range(self.dimension)]
        return '(V)[' + ', '.join([str(field) for field in line]) + ']'

    def __hash__(self) -> int:
        return hash(tuple(self.get_data()))


def create_vector_from_coordinates(*coordinates: List[float]) -> Vector:
    vector = Vector(len(coordinates))
    vector.set_values(*coordinates)
    return vector


def zero(dimension: int) -> Vector:
    zeros = [0.0] * dimension
    return create_vector_from_coordinates(*zeros)
