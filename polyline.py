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

    n,m = vertices.shape
    seg = np.searchsorted(vertices[:, 0], x, side="right") - 1

    if seg < 0 or seg >= len(vertices)-1:
        raise ValueError("x-coordinate does not intersect the polyline.")

    dx = vertices[seg+1,0] - vertices[seg,0]

    if dx == 0:
        alpha = 0.0
        y = vertices[seg, 1]
    else:
        alpha = (x - vertices[seg,0]) / dx
        y = vertices[seg, 1] + alpha * (vertices[seg+1, 1] - vertices[seg, 1])

    proj = np.array([x, y])
    dist = np.sqrt((proj[0] - point[0])**2 + (proj[1] - point[1])**2)

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
    
    point = np.array([2, 1.0])
    
    proj, dist, iprev, inext, alpha = project_along_y(point, polyline)
    
    print("Projection:", proj)
    print("Distance:  ", dist)
    print("i_prev:    ", iprev)
    print("i_next:    ", inext)
    print("Alpha:     ", alpha)
