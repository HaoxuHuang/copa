### Constraints parser ###
# Constraints include:
# 1. Vector A and Vector B are on the same line, with the same/opposite direction.
# 2. The target position of Point A is x cm along Vector B from Point C's current position.
# 3. Vector A is parallel to the table surface.
# 4. Vector A is perpendicular to the table surface, pointing downward/upward.
# 5. Point A is x cm above the table surface.

from .solver import SpatialSolver
from .constraint import Constraint
from .elements import *


class ConstraintsParser():
    """Parser for constraints."""

    def __init__(self, elements):
        """Initialize the parser.
        elements: a dict of elements (id: element)
        """
        self.elements = elements
        self.solver = SpatialSolver()

    def parse(self, text):
        """Parse the constraints.
        text: the paragraph of constraints
        """
        lines = text.split("\n")
        description = self.locate_constraints(lines)

        for line in description:
            self.parse_constraint(line)

        return self.solver
    
    def locate_constraints(self, lines):
        """Locate the description of the constraints.
        The description is wrapped in <Start Constraint>...<End Constraint>.
        lines: a list of lines
        """
        for i, line in enumerate(lines):
            line = line.strip()
            if line == "<Start Constraint>":
                start = i
            elif line == "<End Constraint>":
                end = i
        return lines[start+1:end]
    
    def parse_constraint(self, line) -> SpatialSolver:
        """Parse a line of description.
        line: a line of description
        """
        if 'Move' in line:
            return
        line = line.strip().strip('.')
        words = line.replace('\'', ' ').split()
        elems, distance = self.parse_elements(words)
        if "on the same line" in line:  # constraint 1
            constraints = [Constraint(elems[0], elems[1], "colinear")]
            if "same direction" in line:
                constraints.append(Constraint(elems[0].n, elems[1].n, "equal"))
            else: # opposite direction
                constraints.append(Constraint(elems[0].n, elems[1].n, "inverse"))

        elif "cm along" in line:  # constraint 2
            new_plane = Plane(elems[2].p, elems[1].n)
            constraints = [Constraint(elems[0].p, new_plane, "distance", distance=distance),
                           Constraint(elems[0].p, new_plane, "online")]
            
        elif "parallel to the table surface" in line:  # constraint 3
            if "Vector" in line:
                constraints = [Constraint(elems[0].n, elems[1].n, "perpendicular")]
            else:   # Plane
                constraints = [Constraint(elems[0].n, elems[1].n, "parallel")]
        
        elif "perpendicular to the table surface" in line:  # constraint 4
            if "downward" in line:
                constraints = [Constraint(elems[0].n, elems[1].n, "inverse")]
            else: # upward
                constraints = [Constraint(elems[0].n, elems[1].n, "equal")]

        elif "above the table surface" in line:  # constraint 5
            constraints = [Constraint(elems[0].p, elems[1], "distance", distance=distance)]

        else:
            raise NotImplementedError(f"Constraint type {line} not implemented")
        
        self.solver.add_constraints(constraints)


    def parse_elements(self, words):
        """Parse the elements in the description.
        words: a list of words
        """
        elems = []
        distance = None
        for i, word in enumerate(words):
            if word in ['Point', 'Vector', 'Surface']:
                elem_id = words[i+1]
                elems.append(self.elements[elem_id])
            if word == 'table':
                elems.append(self.elements['table'])
            if word == 'cm':
                distance = float(words[i-1]) / 100
        assert len(elems) > 0, "No element found in the description"
        return elems, distance
    
    def __str__(self) -> str:
        return f"ConstraintsParser({self.elements})\n {self.solver}"