"""
Toolbox for handling projections.

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

import logging
from typing import List

from linalg import Vector, Matrix, zero, create_vector_from_coordinates
import util

_h = util.NullHandler()
_logger = logging.getLogger('vecspace')
_logger.addHandler(_h)


def define_line(p0: Vector, p1: Vector) -> StraightLine:
    """
    Straight line in a n dim space given 2 points.
    """
    if p0.dimension != p1.dimension:
        raise ValueError(f'incompatible dimensions for {p0} and {p1}')

    return StraightLine(p0, p1)


class StraightLine(object):
    """
    Just a straight line.
    """

    def __init__(self, p1: Vector, p2: Vector):
        """
        """
        self._points = p1, p2

    def project(self, point: Vector) -> Projection:
        """
        Generalizing projection onto induced subspace.
        """
        vectors = [
            self._points[i].sub(self._points[0]) for i in range(1, len(self._points))
        ]
        subspace = VectorSpace(*vectors)
        proj = subspace.project(point.sub(self._points[0]))
        proj.projected = proj.projected.add(self._points[0])
        return proj

    def __repr__(self) -> str:
        return str(self._points)


class VectorSpace(object):
    """
    Defines a set on which we can project a point.
    """

    def __init__(self, *points: List[Vector]):
        """
        Defining a n-subspace generated by n points.
        """
        self._points = points

    def project(self, point: Vector) -> Projection:
        """
        The solution implies (x - x*) perpendicular to each y[i]
        with x* = sum( alpha[i] * y[i] )
        and y[i]: points generating the subspace.
        """
        space_dim = point.dimension
        subspace_dim = len(self._points)
        m = Matrix(subspace_dim)
        for row in range(subspace_dim):
            for column in range(row, subspace_dim):
                value = self._points[row].product(self._points[column])
                m.set_value(row, column, value)
                if column != row:
                    m.set_value(column, row, value)
        b = Vector(subspace_dim)
        for row in range(subspace_dim):
            value = point.product(self._points[row])
            b.set_component(row, value)
        alphas = util.system_solve(m.get_table(), b.get_data())
        result = zero(space_dim)
        for i in range(len(alphas)):
            component = self._points[i].scale(alphas[i])
            result = result.add(component)
        return Projection(result, point)


class Projection(object):
    """
    Simple container gathering details about projection.
    """

    def __init__(self, projected: Vector, start: Vector):
        self._projected = projected
        self._start = start
        self._projector = projected.sub(start)

    @property
    def projected(self) -> Vector:
        return self._projected

    @projected.setter
    def projected(self, value: Vector) -> None:
        self._projected = value

    @property
    def start(self) -> Vector:
        return self._start

    @property
    def projector(self) -> Vector:
        return self._projector
