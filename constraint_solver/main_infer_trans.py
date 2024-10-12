import argparse
import pickle
from pathlib import Path
import numpy as np
import open3d as o3d

from spatial_solver import ConstraintsParser, Plane, transform, create_se3, create_x
from draw_arrow import get_arrow
from action_parser import ActionParser

def create_elements(spatial_data, add_table=False):
    elements = {}
    for i, (ratio, center, norm) in enumerate(spatial_data):
        print('Element {}: ratio {:2f}, d1 {}, d2 {}'.format(i+1, ratio, center, norm))
        if ratio < 3:
            elements[str(i+1)] = Plane.from_numpy(np.stack([center, norm]))
        else:   # line connect start point (center) and end point (norm)
            elements[str(i+1)] = Plane.from_numpy(np.stack([norm, 
                                                            (norm - center) / np.linalg.norm(norm - center)]))
    if add_table:
        elements['table'] = Plane.from_numpy(np.stack([np.array([0.5, 0, 0.07]), np.array([0, 0, 1])]))
    print(elements)
    return elements

def visualize_transform(elements, se3, transform_obj_id=None):
    arrows = []
    for k, elem in elements.items():
        arrow = get_arrow(origin=elem.p.to_numpy(), vec=elem.n.to_numpy())
        arrow.paint_uniform_color([1, 0, 0])
        # print(elem)
        if transform_obj_id is None or k == transform_obj_id:    
            arrow.paint_uniform_color([0, 0, 1])
            trans_elem = transform(elem, se3)
            # print(trans_elem)
            trans_arrow = get_arrow(origin=trans_elem.p.to_numpy(), vec=trans_elem.n.to_numpy())
            trans_arrow.paint_uniform_color([0, 1, 0])
            arrows.append(trans_arrow)
        arrows.append(arrow)
        frame_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0,0,0])
    o3d.visualization.draw_geometries([*arrows, frame_axis])
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--spatial_data', type=Path, default=Path('spatial_data.pkl'))
    parser.add_argument('--constraints', type=Path, default=Path('response.txt'))
    parser.add_argument('--output', type=Path, default=Path('transform.npy'))
    args = parser.parse_args()

    # read data and constraints
    with open(args.spatial_data, 'rb') as f:
        spatial_data = pickle.load(f)
    with open(args.constraints, 'r') as f:
        constraints = f.read()

    elements = create_elements(spatial_data, add_table=True)
    constraints_parser = ConstraintsParser(elements)
    solver = constraints_parser.parse(constraints)
    # print(constraints_parser)
    solver.dist_factor = 0.02   # minimize distance between start and end pose
    result = solver.solve()
    se3 = create_se3(result.x)
    np.save(args.output, se3)

    subsequent_actions = ActionParser().parse(constraints)
    pickle.dump(subsequent_actions, open(args.output.parent / 'action.pkl', 'wb'))

    visualize_transform(elements, se3, '1')