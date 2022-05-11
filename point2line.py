import opengjkc as opengjk
from scipy.spatial.transform import Rotation as R
import numpy as np
import pytest

def settol():
    return 1e-12

def distance_point_to_line_3D(P1, P2, point):
    """
    distance from point to line
    """
    return np.linalg.norm(np.cross(P2-P1, P1-point))/np.linalg.norm(P2-P1)


def distance_point_to_plane_3D(P1, P2, P3, point):
    """
    Distance from point to plane
    """
    return np.abs(np.dot(np.cross(P2-P1, P3-P1) /
                         np.linalg.norm(np.cross(P2-P1, P3-P1)), point-P2))


# delta = 2

# line = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)
# print("line: \n", line)

# point_on_line = line[0] + 0 *(line[1]-line[0])
# print("point_on_line:", point_on_line)

# normal = np.cross(line[0], line[1])
# print("normal:", normal)

# point = point_on_line + delta * normal
# print("point:", point)

# distance = opengjk.gjk(line, point)
# actual_distance = distance_point_to_line_3D(line[0], line[1], point)
# print(distance, actual_distance)


cubes = [np.array(
	[
		[-1, -1, -1], 
	    [1, -1, -1], 
	    [-1, 1, -1], 
	    [1, 1, -1],
	    [-1, -1, 1], 
	    [1, -1, 1], 
	    [-1, 1, 1], 
	    [1, 1, 1]
	],dtype=np.float64)]

r = R.from_euler('z', 45, degrees=True)
cubes.append(r.apply(cubes[0]))
	
r = R.from_euler('y', np.arctan2(1.0, np.sqrt(2)))
cubes.append(r.apply(cubes[1]))
	
r = R.from_euler('y', 45, degrees=True)
cubes.append(r.apply(cubes[0]))

dx = cubes[0][:,0].max() - cubes[0][:,0].min()
cube0 = cubes[0]

# for delta in [1e8, 1.0, 1e-4, 1e-8, 1e-12]:
r_tempt = R.from_euler('z', 90, degrees=True)
cube1 = cubes[0] + np.array([dx + 10, 0, 0])
print(cube1)
distance = opengjk.gjk(cube0, cube1)
print(distance, 10)

cube1 = r_tempt.apply(cube1)
print(cube1)
distance = opengjk.gjk(cube0, cube1)
print(distance, 10)










