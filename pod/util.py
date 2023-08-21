"""
Operations on matrices and various tools.

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


import math
import logging
from typing import List


class NullHandler(logging.Handler):
    """
    Null logging in order to avoid warning messages in client applications.
    """

    def emit(self, record):
        pass


_h = NullHandler()
_logger = logging.getLogger('util')
_logger.addHandler(_h)


def prod_scalar(v1: List[float], v2: List[float]) -> float:
    if len(v1) != len(v2):
        raise ValueError(f'input vectors {v1} and {v2} must be of the same size, currently {len(v1)} and {len(v2)} respectively)')
    prod = [x[0] * x[1] for x in zip(v1, v2)]
    return sum(prod)
