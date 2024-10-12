"""Definition of the Constraint class. Represent spatial constraints.
Including: colinear, parallel, perpendicular, etc."""
from .elements import *
from .transform import transform


class Constraint:
    """A constraint between two elements.
    For example, two lines are parallel, two planes are perpendicular, etc.
    """

    def __init__(self, e1, e2, type, **kwargs):
        """Create a constraint between two elements.
        e1: the element to be transformed
        e2: the fixed element
        type: the type of constraint
        """
        self.e1 = e1
        self.e2 = e2
        self.type = type
        self.check_constraint()
        self.obj_func = self.get_obj_func(**kwargs)

    def __repr__(self):
        return f"Constraint({self.e1}, {self.e2}, {self.type})"

    def __str__(self):
        return f"{self.e1} {self.type} {self.e2}"

    def __eq__(self, other):
        return (self.e1 == other.e1 and self.e2 == other.e2 and
                self.type == other.type)

    def __hash__(self):
        return hash((self.e1, self.e2, self.type))
    
    def check_constraint(self):
        """Check if the constraint is valid."""
        assert self.e1 is not self.e2 or self.type == "equal", "Two elements must be different"
        assert self.type in ["parallel", "perpendicular", "colinear",
                             "coplanar", "equal", "inverse", "distance", "online"], "Invalid constraint type"
        if self.type == "parallel":
            assert isinstance(self.e1, (Line, Vector)) and isinstance(self.e2, (Line, Vector)), \
                f"Parallel constraint must be between two lines or two vectors, but get {self.e1} and {self.e2}"
        elif self.type == "perpendicular":
            assert isinstance(self.e1, (Line, Vector, Plane)) and isinstance(self.e2, (Line, Vector, Plane)), \
                f"Perpendicular constraint must be between two lines or two vectors, but get {self.e1} and {self.e2}"
        elif self.type == "colinear":
            assert isinstance(self.e1, (Line, Plane)) and isinstance(self.e2, (Line, Plane)), \
                f"Colinear constraint must be between two lines, but get {self.e1} and {self.e2}"
        elif self.type == "coplanar":
            assert isinstance(self.e1, Plane) and isinstance(self.e2, Plane), \
                f"Coplanar constraint must be between two planes, but get {self.e1} and {self.e2}"
        elif self.type == "equal":
            assert isinstance(self.e1, (Point, Vector)) and isinstance(self.e2, (Point, Vector)), \
                f"Equal constraint must be between two points or two vectors, but get {self.e1} and {self.e2}"
        elif self.type == "inverse":
            assert isinstance(self.e1, (Point, Vector)) and isinstance(self.e2, (Point, Vector)), \
                f"Inverse constraint must be between two points or two vectors, but get {self.e1} and {self.e2}"
        elif self.type == "distance":
            assert isinstance(self.e1, Point) and isinstance(self.e2, Plane), \
                f"Distance constraint must be between a point and a plane, but get {self.e1} and {self.e2}"
        elif self.type == "online":
            assert isinstance(self.e1, Point) and isinstance(self.e2, (Line, Plane)), \
                f"Online constraint must be between a point and a line or plane, but get {self.e1} and {self.e2}"
    
    def get_obj_func(self, **kwargs):
        """Get the objective function of the constraint."""
        if self.type == "parallel":
            if isinstance(self.e1, Line) and isinstance(self.e2, Line):
                def obj_func(e1, e2):
                    return np.linalg.norm(np.cross(e1.p2.to_numpy() - e1.p1.to_numpy(), e2.p2.to_numpy() - e2.p1.to_numpy()))
                return obj_func
            else: # vector
                def obj_func(e1, e2):
                    return np.linalg.norm(np.cross(e1.to_numpy(), e2.to_numpy()))
                return obj_func
            
        elif self.type == "perpendicular":
            if isinstance(self.e1, (Line, Vector)) and isinstance(self.e2, (Line, Vector)):
                def obj_func(e1, e2):
                    if isinstance(e1, Line):
                        e1 = e1.to_vector()
                    if isinstance(e2, Line):
                        e2 = e2.to_vector()
                    return np.dot(e1.to_numpy(), e2.to_numpy())
                return obj_func
            elif isinstance(self.e1, Plane) and isinstance(self.e2, Plane):
                def obj_func(e1, e2):
                    return np.dot(e1.n.to_numpy(), e2.n.to_numpy())
                return obj_func
            else:   # line/vector and plane
                def obj_func(e1, e2):
                    if isinstance(e1, Plane):
                        e1 = e1.n
                        if isinstance(e2, Line):
                            e2 = e2.to_vector()
                    else:
                        e2 = e2.n
                        if isinstance(e1, Line):
                            e1 = e1.to_vector()
                    return np.dot(e1.to_numpy(), e2.to_numpy())
                return obj_func
        
        elif self.type == "colinear":   # evaluate by cross product, colinear doesn't necessarily mean same direction
            if isinstance(self.e1, Line) and isinstance(self.e2, Line):
                def obj_func(e1, e2):
                    v1 = e1.to_vector().to_numpy()
                    v2 = e2.to_vector().to_numpy()
                    v3 = e1.p1.to_numpy() - e2.p1.to_numpy()
                    l_direction = np.linalg.norm(np.cross(v1, v2))
                    l_distance = np.linalg.norm(np.cross(v3, v2)) / np.linalg.norm(v2)
                    return l_direction + l_distance
            elif isinstance(self.e1, Plane) and isinstance(self.e2, Plane): # here plane represents a vector with a base point
                def obj_func(e1, e2):
                    v1 = e1.n.to_numpy()
                    v2 = e2.n.to_numpy()
                    v3 = e1.p.to_numpy() - e2.p.to_numpy()
                    l_direction = np.linalg.norm(np.cross(v1, v2))
                    l_distance = np.linalg.norm(np.cross(v3, v2)) / np.linalg.norm(v2)
                    return l_direction + l_distance
            return obj_func
        
        elif self.type == "coplanar":
            def obj_func(e1, e2):
                return np.linalg.norm(np.cross(e1.n.to_numpy(), e2.n.to_numpy()))
            return obj_func
        
        elif self.type == "equal":
            def obj_func(e1, e2):
                return np.linalg.norm(e1.to_numpy() - e2.to_numpy())
            return obj_func
        
        elif self.type == "inverse":
            def obj_func(e1, e2):
                return np.linalg.norm(e1.to_numpy() + e2.to_numpy()) * 5
            return obj_func
        
        elif self.type == "distance":
            self.distance = distance = kwargs.get("distance", 0)
            def obj_func(e1, e2):
                return np.dot(e1.to_numpy() - e2.p.to_numpy(), e2.n.to_numpy()) - distance
            return obj_func
        
        elif self.type == "online":
            if isinstance(self.e2, Plane):  # the e1 point on central normal line of the plane
                def obj_func(e1, e2):
                    return np.linalg.norm(np.cross(e1.to_numpy() - e2.p.to_numpy(), e2.n.to_numpy()))
            else:  # the e1 point on the e2 line
                def obj_func(e1, e2):
                    return np.linalg.norm(np.cross(e1.to_numpy() - e2.p1.to_numpy(), e2.to_vector().to_numpy()))
            return obj_func
        
        else:
            raise NotImplementedError(f"Constraint type {self.type} not implemented")


    def objective(self, x):
        """Compute the objective loss of the constraint.
        x: a numpy array of the SE3 transform
        """
        transformed_e1 = transform(self.e1, x)
        obj = self.obj_func(transformed_e1, self.e2)
        return np.abs(obj)
    

    def test_objective(self, x):
        """Compute the objective function of the constraint and report.
        x: a numpy array of the SE3 transform
        """
        transformed_e1 = transform(self.e1, x)
        obj = self.obj_func(transformed_e1, self.e2)
        print(f"Constraint {self.type} objective: {obj}")
        return obj
