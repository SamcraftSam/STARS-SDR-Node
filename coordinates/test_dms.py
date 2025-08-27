import pytest
from dms import Coordinate, CoordsNE, CoordsNW

def test_decimal_conversion():
    coord = Coordinate(73, 30, 0, negative=False)
    assert coord.to_decimal() == pytest.approx(73.5)

    coord_neg = Coordinate(73, 30, 0, negative=True)
    assert coord_neg.to_decimal() == pytest.approx(-73.5)

def test_coordinate_negation_copy():
    coord = Coordinate(10, 0, 0, negative=False)
    flipped = coord.copy_with_negation()
    assert flipped.deg == 10
    assert flipped.negative is True
    assert coord.negative is False  # оригинал остался как был

def test_coords_conversion_ne_to_nw():
    north = Coordinate(40, 0, 0)
    east = Coordinate(73, 45, 0, negative=False)
    coords_ne = CoordsNE(N=north, E=east)
    coords_nw = coords_ne.convert_to(CoordsNW)

    assert isinstance(coords_nw, CoordsNW)
    assert coords_nw.W.to_decimal() == pytest.approx(-73.75)

def test_coords_conversion_nw_to_ne():
    north = Coordinate(40, 0, 0)
    west = Coordinate(73, 45, 0, negative=True)
    coords_nw = CoordsNW(N=north, W=west)
    coords_ne = coords_nw.convert_to(CoordsNE)

    assert isinstance(coords_ne, CoordsNE)
    assert coords_ne.E.to_decimal() == pytest.approx(73.75)

def test_str_representation():
    coord = Coordinate(10, 15, 30, negative=True)
    assert str(coord) == "-10° 15' 30\""
