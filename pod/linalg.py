"""
Basic linear algebra components and functions.

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

from __future__ import annotations

import math
import os
import logging
from typing import Optional, List

from util import NullHandler
from util import prod_scalar

_h = NullHandler()
_logger = logging.getLogger('linalg')
_logger.addHandler(_h)


class Matrix(object):

    def __init__(self, dim_row: int, dim_col: Optional[int] = None):
        if dim_col is None:
            dim_col = dim_row
        self.dim_row = dim_row
        self.dim_col = dim_col
        self._vectors = {}

    def transpose(self):
        m = Matrix(self.get_dim_col(), self.get_dim_row())
        for i in range(self.get_dim_row()):
            for j in range(self.get_dim_col()):
                m.set_value(j, i, self.get_value(i, j))
        return m

    def get_value(self, i, j):
        assert i < self.get_dim_row(), 'row %d exceeding dimension %d' % (i, self.get_dim_row())
        assert j < self.get_dim_col(), 'column %d exceeding dimension %d' % (j, self.get_dim_col())
        v = self._vectors[i] if i in self._vectors else self._create_vector()
        return v.get_component(j)

    def set_table(self, table):
        assert len(table) == self.get_dim_row(), 'expected %d rows instead of %d' % (self.get_dim_row(), len(table))
        for i in range(self.get_dim_row()):
            assert len(table[i]) == self.get_dim_col(), 'expected %d columns instead of %d' % (
            self.get_dim_col(), len(table[i]))
            for j in range(self.get_dim_col()):
                self.set_value(i, j, table[i][j])

    def get_table(self):
        table = []
        for i in range(self.get_dim_row()):
            row = [self.get_value(i, j) for j in range(self.get_dim_col())]
            table.append(row)
        return table

    def get_dimension(self):
        return (self.dim_row, self.dim_col)

    def get_dim_row(self):
        return self.get_dimension()[0]

    def get_dim_col(self):
        return self.get_dimension()[1]

    def _create_vector(self):
        return Vector(self.get_dimension()[1])

    def set_value(self, i, j, val):
        msg = str(self.get_dimension())
        assert i < self.get_dim_row(), 'row %d exceeding dimension %s' % (i, msg)
        assert j < self.get_dim_col(), 'column %d exceeding dimension %s' % (j, msg)
        if i in self._vectors:
            v = self._vectors[i]
        else:
            v = self._create_vector()
            self._vectors[i] = v
        try:
            v.set_component(j, float(val))
        except TypeError as e:
            logging.error(f'not a number: {val}')
            raise e

    def minor(self, r, c):
        """
        Sub-matrix excluding the specified row and column.
        """
        m = Matrix(self.get_dim_row() - 1, self.get_dim_col() - 1)
        dr = self.get_dim_row()
        dc = self.get_dim_col()
        for k, i in enumerate((n for n in range(dr) if n != r)):
            for l, j in enumerate((p for p in range(dc) if p != c)):
                v = self.get_value(i, j)
                m.set_value(k, l, v)
        return m

    def determinant(self):
        size = self.get_dim_row()
        det = 0.0
        # stopping condition
        if size == 1:
            det = self.get_value(0, 0)
        else:
            for i in range(size):
                minor = self.minor(0, i)
                sign = 1.0 if i % 2 == 0 else -1.0
                det += sign * self.get_value(0, i) * minor.determinant()
        return det

    def cofactors(self):
        m = Matrix(self.get_dim_row())
        for row in range(self.get_dim_row()):
            for col in range(self.get_dim_col()):
                sign = 1.0 if row % 2 == col % 2 else -1.0
                v = sign * self.minor(row, col).determinant()
                m.set_value(row, col, v)
        return m

    def multiply(self, m):
        return prod_matrix(self, m)

    def copy(self):
        c = Matrix(self.get_dim_row(), self.get_dim_col())
        for row in range(self.get_dim_row()):
            for col in range(self.get_dim_col()):
                c.set_value(row, col, self.get_value(row, col))
        return c

    def __repr__(self):
        out = '(M%dx%d)' % (self.get_dim_row(), self.get_dim_col()) + os.linesep
        for row in range(self.get_dim_row()):
            line = [self.get_value(row, col) for col in range(self.get_dim_col())]
            out += ', '.join([str(field) for field in line]) + os.linesep
        return out


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

    def symmetric(self) -> Vector:
        null_vector = Vector(self.dimension)
        return null_vector.sub(self)

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


def prod_matrix(m1: Matrix, m2: Matrix) -> Matrix:
    msg = f'incompatible dimensions: {m1.get_dimension()} x {m2.get_dimension()}'
    assert m1.get_dim_col() == m2.get_dim_row(), msg
    result = Matrix(m1.get_dim_row(), m2.get_dim_col())
    for i in range(m1.get_dim_row()):
        for j in range(m2.get_dim_col()):
            v = 0.0
            for k in range(m1.get_dim_col()):
                v += m1.get_value(i, k) * m2.get_value(k, j)
            result.set_value(i, j, v)
    return result
