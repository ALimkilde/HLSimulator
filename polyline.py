import numpy as np

def project_point_to_polyline(point, vertices):
    """
    Project a point onto a polyline.

    Parameters
    ----------
    point : (D,) array_like
        Point to project.
    vertices : (N, D) array_like
        Vertices of the polyline.

    Returns
    -------
    projection : (D,) ndarray
        Closest point on the polyline.
    distance : float
        Euclidean distance to the polyline.
    segment : int
        Index of the closest segment (between vertices[segment] and vertices[segment+1]).
    alpha : float
        Parameter along the segment in [0, 1].
    """

    point = np.asarray(point, dtype=float)
    vertices = np.asarray(vertices, dtype=float)

    if len(vertices) < 2:
        raise ValueError("Polyline must contain at least two vertices.")

    # Segment start/end points
    a = vertices[:-1]
    b = vertices[1:]

    # Segment vectors
    d = b - a

    # Squared lengths
    lengths_sq = np.sum(d * d, axis=1)

    # Handle zero-length segments
    valid = lengths_sq > 0

    alpha = np.zeros(len(d))
    alpha[valid] = np.sum((point - a[valid]) * d[valid], axis=1) / lengths_sq[valid]

    # Clamp to segment
    alpha = np.clip(alpha, 0.0, 1.0)

    # Closest point on each segment
    projections = a + alpha[:, None] * d

    # Squared distances
    dist_sq = np.sum((projections - point) ** 2, axis=1)

    # Best segment
    segment = np.argmin(dist_sq)

    return (
        projections[segment],
        np.sqrt(dist_sq[segment]),
        segment,
        alpha[segment],
    )

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
    
    proj, dist, seg, alpha = project_point_to_polyline(point, polyline)
    
    print("Projection:", proj)
    print("Distance:  ", dist)
    print("Segment:   ", seg)
    print("Alpha:     ", alpha)
