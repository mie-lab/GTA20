import numpy as np
import shapely
from shapely.geometry import Polygon
import pygeos


def pygeos_geometry_to_shapely(geom):
    if isinstance(geom, shapely.geometry.base.BaseGeometry):
        return geom
    elif type(geom) is pygeos.Geometry:
        coords = pygeos.get_coordinates(geom, 1)
        return Polygon([[c[0], c[1]] for c in coords])
    else:
        return np.nan
