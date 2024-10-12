from .elements import *


def transform(elem, se3):
    """Transform a geometry element by a SE3 transform.
    elem: the geometry element
    se3: the SE3 transform
    """
    if isinstance(elem, Vector):
        return Vector.from_numpy(se3 @ elem.to_homogeneous() - se3[:, 3])
    elif isinstance(elem, Line):
        return Line.from_numpy((se3 @ elem.to_homogeneous()).T)
    elif isinstance(elem, Point):
        return Point.from_numpy(se3 @ elem.to_homogeneous())
    elif isinstance(elem, Plane):
        return Plane(Point.from_numpy(se3 @ elem.p.to_homogeneous()), Vector.from_numpy(se3 @ elem.n.to_homogeneous() - se3[:, 3]))
    else:
        raise TypeError(f"Cannot transform {elem} of type {type(elem)}")