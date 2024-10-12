from scipy.optimize import minimize, Bounds, NonlinearConstraint
from scipy.spatial.transform import Rotation as R
import numpy as np

from .elements import *
from .constraint import *


def quaternion_norm_constraint(x):
    # 假设四元数是x中的前四个参数
    quaternion = x[:4]
    return np.linalg.norm(quaternion) - 1

def create_se3(x):
    se3 = np.zeros([4,4])
    se3[:3,:3] = R.from_quat(x[:4]).as_matrix()
    se3[:3,3] = x[4:]
    se3[3,3] = 1
    return se3

def create_x(se3):
    x = np.zeros(7)
    x[:4] = R.from_matrix(se3[:3,:3]).as_quat()
    x[4:] = se3[:3,3]
    return x

class SpatialSolver:
    """Spatial solver for 3D geometry constraints.
    Attributes:
        constraints: a list of constraints
    """

    def __init__(self, constraints=None, dist_factor=0):
        """Initialize the spatial solver.
        constraints: a list of constraints
        """
        self.constraints = constraints or []
        self.dist_factor = dist_factor

    def solve(self, max_iter=10000, tol=1e-6, initial_guess=[0,0,0,1,0,0,0], bounds=None):
        """Solve the constraints.
        max_iter: maximum number of iterations
        tol: tolerance for convergence
        """
        result = minimize(self.objective, initial_guess, method='trust-constr', # BFGS, trust-constr
                          bounds=bounds, options={'maxiter': max_iter}, tol=tol)
        # report convergence
        if result.success:
            print("Converged!")
        else:
            print("Failed to converge! Try increasing max_iter or tol.")
        print(f"Current solution: {result.x}")
        normalized_result = result.x
        normalized_result[:4] /= np.linalg.norm(normalized_result[:4])
        print(f"Normalized current solution: {normalized_result}")
        result_se3 = create_se3(result.x)
        print(f"Current solution SE3: \n{result_se3}")
        for constraint in self.constraints:
            constraint.test_objective(result_se3)
        solution_dist = np.linalg.norm(result.x - np.array([0,0,0,1,0,0,0]))
        print(f"Distance between start and end pose: {solution_dist}")
        return result

    def objective(self, x):
        """Compute the objective function of the constraints.
        x: the current solution
        """
        se3 = create_se3(x)
        obj = 0
        for constraint in self.constraints:
            obj += constraint.objective(se3)

        # distance between start and end pose
        if self.dist_factor > 0:
            obj += np.linalg.norm(x - np.array([0,0,0,1,0,0,0])) * self.dist_factor
        return obj
    
    def add_constraints(self, constraints):
        """Add constraints to the solver.
        constraints: a list of constraints
        """
        self.constraints.extend(constraints)

    def __str__(self) -> str:
        """Return a string representation of the solver."""
        return f"SpatialSolver({self.constraints})"
