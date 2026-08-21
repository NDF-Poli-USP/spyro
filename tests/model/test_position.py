import pytest
from spyro.model.position import Position
import firedrake as fire


class TestPosition:

    def test_constructor_without_z(self):
        p = Position(x=1, y=1)

        assert p.x == 1
        assert p.y == 1
        assert p.z is None

    def test_full_constructor(self):
        p = Position(x=1, y=1, z=1)

        assert p.x == 1
        assert p.y == 1
        assert p.z == 1

    def test_get_position(self):
        p = Position(x=1, y=1, z=1)

        assert p.get_position() == (1, 1, 1)

    def test_get_position_without_z(self):
        p = Position(x=1, y=1)

        assert p.get_position() == (1, 1)
