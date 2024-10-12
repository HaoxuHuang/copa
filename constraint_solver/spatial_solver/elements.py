"""Definition of 3D geometry elements: point, line, plane, etc."""
import dataclasses
import numpy as np


@dataclasses.dataclass
class Point:
    """A point in 3D space."""

    x: float
    y: float
    z: float

    @classmethod
    def from_numpy(cls, arr):
        """Create a point from a numpy array."""
        assert arr.shape == (3,) or arr.shape == (4,), "Array must have shape (3,) or (4,)"
        return cls(*arr[:3])
    
    def to_numpy(self):
        """Convert the point to a numpy array."""
        return np.array([self.x, self.y, self.z])
    
    def to_homogeneous(self):
        """Convert the point to homogeneous coordinates. Shape: (4,)"""
        return np.array([self.x, self.y, self.z, 1])


@dataclasses.dataclass
class Line:
    """A line in 3D space."""

    p1: Point
    p2: Point

    @classmethod
    def from_numpy(cls, arr):
        """Create a line from a numpy array."""
        assert arr.shape == (2, 3) or arr.shape == (2, 4), "Array must have shape (2, 3) or (2, 4)"
        return cls(Point.from_numpy(arr[0]), Point.from_numpy(arr[1]))
    
    def to_numpy(self):
        """Convert the line to a numpy array."""
        return np.array([self.p1.to_numpy(), self.p2.to_numpy()])
    
    def to_homogeneous(self):
        """Convert the line to homogeneous coordinates. Shape: (3, 2)"""
        return np.array([self.p1.to_homogeneous(), self.p2.to_homogeneous()]).T
    
    def to_vector(self):
        """Convert the line to a vector."""
        return Vector(self.p2.x - self.p1.x, self.p2.y - self.p1.y, self.p2.z - self.p1.z)


class Vector(Point):
    """A vector in 3D space."""

    pass


@dataclasses.dataclass
class Plane:
    """A plane in 3D space.
    Described by a central point and a normal vector.
    """
    p: Point
    n: Vector

    @classmethod
    def from_numpy(cls, arr):
        """Create a plane from a numpy array."""
        assert arr.shape == (2, 3) or arr.shape == (2, 4), "Array must have shape (2, 3) or (2, 4)"
        return cls(Point.from_numpy(arr[0,:]), Vector.from_numpy(arr[1,:]))
    
    def to_numpy(self):
        """Convert the plane to a numpy array. Shape: (2, 3)"""
        return np.array([self.p.to_numpy(), self.n.to_numpy()])
    
    def to_homogeneous(self):
        """Convert the plane to homogeneous coordinates. Shape: (4, 2)"""
        return np.array([self.p.to_homogeneous(), self.n.to_homogeneous()]).T
    