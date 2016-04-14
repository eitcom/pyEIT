# create mesh2d
# liubenyuan@gmail.com
# 2015-08-02

import numpy as np
import scipy.linalg as lp
from matplotlib.path import Path
import meshpy.triangle as triangle


# connect points to facet (standard code in meshpy)
def round_trip_connect(start, end):
    return [(i, i+1) for i in range(start, end)] + [(end, start)]


# refine (equivalent to the max_volume parameter)
def refinement_func_area(tri_points, area):
    max_area = 0.005
    return bool(area > max_area)


# [unused, skip it]
def refinement_func_location(tri_points, area):
    """
    refine around some locations.
    a tripoints is a ndarray of (x1, y1) (x2, y2) (x3, y3)
    we find its center and return a boolean if this triangle needs refined
    """
    center_tri = np.sum(np.array(tri_points), axis=0)/3.
    max_area = 0.005 + lp.norm(np.abs(center_tri) - 1.0) * 0.05
    return bool(area > max_area)


# [unused, skip it]
def refinement_func_anomaly(tri_points, area):
    """
    refine triangles within the anomaly regions.
    you have to specify the points which consist the polygon,
    you need to set the enclosing facet before refining,
    i.e., refinement_func.polygon = polygon

    this function is low-performance
    """
    polygon = Path(refinement_func_anomaly.polygon)
    center_tri = np.sum(np.array(tri_points), axis=0)/3.
    if area > 0.005:
        return True
    elif (area > 0.002) and polygon.contains_point(center_tri):
        return True
    else:
        return False


# a simple disc geometry for illustration purpose
def disc(numPoly):
    """
    draw a disc outline (circle

    <input>
    numPoly : number of nodes on the outer facet

    <output>
    points : points (x_i, y_i) on the facet
    npoints : the length of the facet
    """
    angles = np.linspace(0, 2*np.pi, numPoly, endpoint=False)
    points = [(np.cos(a), np.sin(a)) for a in angles]
    npoints = [np.size(points, 0)]
    return points, npoints


# create 2D mesh for EIT problem,
# numEl is the number of electrodes placed at the boundary
def create(numEl, max_area=0.01, curve=disc, refine=False):
    """
    inputs,
        numEl : number of electrodes
        curve : functions of generating curvature
    outputs,
        mesh : mesh object, including
            ['elements'] -> Mx3 ndarray
            ['node']     -> Nx2 ndarray
            ['alpha']    -> Mx1 ndarray
        elPos : the location of electrodes nodes
    """
    # number of interpolate boundary nodes, 4x
    numPoly = 4*numEl

    # the first #numPoly points of meshpy's outputs are just the facet
    elPos = np.arange(0, numPoly, 4)

    # generate 'points' and connect 'facets'
    if not hasattr(curve, '__call__'):
        exit('curvature is not callable, exit')
    points, npoints = curve(numPoly)

    # build facets (link structure l->r)
    lnode = 0
    facets = []
    for rnode in npoints:
        facets.extend(round_trip_connect(lnode, rnode-1))
        lnode = rnode

    # build triangle info
    info = triangle.MeshInfo()
    info.set_points(points)
    info.set_facets(facets)

    """
    assume the anomaly-region is convex.
    suppose you want to refine a region in a facet, you can simply specify
    a (any) point in that region, and the way goes :
    >>> points [x,y] in region, + region number, + regional area constraints
    >>> i.e., [0.3, 0.2] + [1] + [0.0001]
    so the 'only' facet that includes this point will be refined with
    [0.0001] area constraints and with 'point_markers'=1
    """
    if refine:
        num_regions = len(npoints) - 1
        info.regions.resize(num_regions)
        for i in range(num_regions):
            polygon = points[npoints[i]: npoints[i+1]]
            center_poly = list(np.mean(polygon, axis=0))
            # regional ID start from 1
            info.regions[i] = center_poly + [i+1] + [max_area/2.]

    """
    build mesh. min_angle can be tweaked, 32.5 is an optimal parameter,
    you may choose 26.67 for an alternate.
    you may also pass refinement_func= as your own tweaked refine function
    """
    mesh_struct = triangle.build(info,
                                 max_volume=max_area,
                                 volume_constraints=True,
                                 attributes=True,
                                 quality_meshing=True,
                                 min_angle=32.5)

    """
    mesh_structure :
        points, Nx2 ndarray
        point_markers, Nx1 ndarray, 1=boundary, 0=interior
        elements, Mx3 ndarray
        element_attributes (if refine==True), Mx1 ndarray, triangle markers
    """
    # build output dictionary, uniform element sigma
    alpha = 1. * np.ones(np.shape(mesh_struct.elements)[0])
    mesh = {'element': np.array(mesh_struct.elements),
            'node': np.array(mesh_struct.points),
            'alpha': alpha}
    return mesh, elPos


def set_alpha(mesh, anom=None, background=None):
    """
    update alphas of mesh structure, if specified,
    anom is a dictionary (or arrays of dictionary) contains :
    {'x': val, 'y': val, 'd': val, 'alpha': val}
    all alphas on triangles whose distance to (x,y) are less than (d)
    will be replaced with a new alpha.
    """
    el2no = mesh['element']
    no2xy = mesh['node']
    alpha = mesh['alpha']
    tri_centers = np.mean(no2xy[el2no], axis=1)
    """
    this code is equivalent to:
    >>> N = np.shape(el2no)[0]
    >>> for i in range(N):
    >>>     tri_centers[i] = np.mean(no2xy[el2no[i]], axis=0)
    >>> plt.plot(tri_centers[:,0], tri_centers[:,1], 'kx')
    """
    N = np.size(mesh['alpha'])

    # reset background if needed
    if background is not None:
        alpha = background * np.ones(N)

    if anom is not None:
        for i in range(len(anom)):
            cx = anom[i]['x']
            cy = anom[i]['y']
            diameter = anom[i]['d']
            alpha_anomaly = anom[i]['alpha']
            # find elements whose distance to (cx,cy) is smaller than d
            indice = np.sqrt((tri_centers[:, 0] - cx)**2 +
                             (tri_centers[:, 1] - cy)**2) < diameter
            alpha[indice] = alpha_anomaly

    mesh_new = {'node': no2xy,
                'element': el2no,
                'alpha': alpha}
    return mesh_new

# demo
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # simple
    mesh, elPos = create(16)

    # show elPos
    print(elPos)

    # extract 'node' and 'element'
    no2xy = mesh['node']
    el2no = mesh['element']

    # show the meshes
    plt.plot()
    plt.triplot(no2xy[:, 0], no2xy[:, 1], el2no)
    plt.plot(no2xy[elPos, 0], no2xy[elPos, 1], 'ro')
    plt.axis('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    title_src = 'number of triangles = ' + str(np.size(el2no, 0)) + ', ' + \
                'number of nodes = ' + str(np.size(no2xy, 0))
    plt.title(title_src)
    plt.show()
