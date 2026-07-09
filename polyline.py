import numpy as np

def interpolate(a,b,alpha):
    return (1-alpha)*a + alpha*b

import numpy as np
from config import *

def project_along_y(point, vertices):
    """
    Project a point vertically onto a polyline.

    Returns
    -------
    projection : (2,) ndarray
    segment0 : int
    segment1 : int
    alpha : float
    """

    x = point[0]

    a = vertices[:-1]
    b = vertices[1:]

    x0 = a[:, 0]
    x1 = b[:, 0]

    # Segments whose x-range contains x
    mask = ((x0 <= x) & (x <= x1)) | ((x1 <= x) & (x <= x0))

    if not np.any(mask):
        raise ValueError("x-coordinate does not intersect the polyline.")

    # First intersecting segment
    seg = np.flatnonzero(mask)[0]

    dx = x1[seg] - x0[seg]

    if dx == 0:
        alpha = 0.0
        y = a[seg, 1]
    else:
        alpha = (x - x0[seg]) / dx
        y = a[seg, 1] + alpha * (b[seg, 1] - a[seg, 1])

    proj = np.array([x, y])
    dist = np.linalg.norm(proj - point)

    i_prev = seg
    i_next = seg+1

    if (i_next >= N-1):
        i_next = i_prev

    if (i_prev <= 0):
        i_prev = i_next

    return proj, dist, i_prev, i_next, alpha

# --------------------------------------------------------------------
# Example
# --------------------------------------------------------------------
if __name__ == "__main__":
    
    polyline = np.array([
        [0, 0],
        [1, 2],
        [3, 2],
        [4, 0]
    ])
    
    point = np.array([4, 5.0])
    
    proj, dist, iprev, inext, alpha = project_point_to_polyline(point, polyline)
    
    print("Projection:", proj)
    print("Distance:  ", dist)
    print("i_prev:    ", i_prev)
    print("i_next:    ", i_next)
    print("Alpha:     ", alpha)
