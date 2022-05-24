import numpy as np
from numpy.typing import NDArray, ArrayLike

"""
render.py contains functions used to render unstructured 2D meshes into rectangular arrays of pixels
"""


def pt_in_triang(p_test, p0, p1, p2):
    """
    Test whether a point lies within a triangle

    Parameters
    ----------
    p_test: list: 2
        cartesian coordinates of the point under test
    p0: ndarray: (2,)
        cartesian coordinates of point 0 of the triangle
    p1: ndarray: (2,)
        cartesian coordinates of point 0 of the triangle
    p2: ndarray: (2,)
        cartesian coordinates of point 0 of the triangle

    Returns
    -------
    bool
        True if p_test lies within the triangle


    """
    dX = p_test[0] - p0[0]
    dY = p_test[1] - p0[1]

    dX20 = p2[0] - p0[0]
    dY20 = p2[1] - p0[1]
    dX10 = p1[0] - p0[0]
    dY10 = p1[1] - p0[1]

    s_p = (dY20 * dX) - (dX20 * dY)
    t_p = (dX10 * dY) - (dY10 * dX)
    D = (dX10 * dY20) - (dY10 * dX20)

    if D > 0:
        return (s_p >= 0) and (t_p >= 0) and (s_p + t_p) <= D
    else:
        return (s_p <= 0) and (t_p <= 0) and (s_p + t_p) >= D


def get_bounds(arr):
    """
    get the bounds of a Nx2 array

    Parameters
    ----------
    arr: array[Nx2]
        array to get the bounds of

    Returns
    -------
        bmin[0]: minimum bound, axis 0
        bmax[0]: maximum bound, axis 0
        bmin[1]: minimum bound, axis 1
        bmax[1]: maximum bound, axis 1

    """
    # gets the upper and lower
    bmax = np.ceil(np.max(arr, axis=0)).astype(int)
    bmin = np.floor(np.min(arr, axis=0)).astype(int)
    return bmin[0], bmax[0], bmin[1], bmax[1]


def model_inverse_uv(mesh, resolution, bounds=None, preserve_aspect_ratio=True):
    """
    Renders an unstructured triangular mesh into a rectangular array of pixels with the value of each pixel being the
    index of the triangle it lies within. An index of -1 refers to a pixel that does not lie within any of the triangles
    in the grid. Use map_image() to map desired values onto the pixels.

    *Note* I am not sure why this rotates the image by 90 degrees. Also I think the aspect ratio of the mesh should be retained
    when resolution is changed. Still, these issues shouldn't matter as long as the target mesh and reconstruction mesh are
    both processed by this function with the same resolution setting.

    Parameters
    ----------
    mesh: dict {element, node}
        mesh structure
            element: Nx3 array of indices to the node array. each row corresponds to one triangle
            node: Nx2 array of cartesian coordinates that make up the points of the triangles
    resolution: tuple(width, height)
        resolution of the rendered image
    bounds: tuple(tuple(x,y),tuple(x,y))
        bounds (in input mesh coordinate system) over which to render. Must contain entire mesh
        format: (minx, miny), (maxx, maxy)
    preserve_aspect_ratio: bool
        preserve aspect ratio

    Returns
    -------
    image: np.Array(width, height)
        rectangular array of pixels with the value of each pixel being the index of the triangle it lies within.


    """
    # iterate through all object values
    clist = np.array(mesh["element"])
    uv_list = np.array(mesh["node"])
    if bounds is not None:
        bounds = np.array(bounds)

    uv_list = scale_uv_list(uv_list, resolution, bounds, preserve_aspect_ratio)

    image = np.zeros(resolution) - 1
    yy, xx = np.meshgrid(range(resolution[1]), range(resolution[0]))

    # for every triangle
    # get the bounding box
    # for points in the bounding box, test the barycentric cords
    for step, inds in enumerate(clist):
        tri = np.asarray([uv_list[inds[0]], uv_list[inds[1]], uv_list[inds[2]]])

        min_x, max_x, min_y, max_y = get_bounds(tri)
        tri_fn = np.vectorize(
            lambda x, y: pt_in_triang([x, y], tri[0, :], tri[1, :], tri[2, :])
        )
        p_xx = xx[min_x:max_x, min_y:max_y]
        p_yy = yy[min_x:max_x, min_y:max_y]
        tri_in = tri_fn(p_xx, p_yy)

        image[min_x:max_x, min_y:max_y][tri_in] = (step * tri_in)[tri_in]

    return image


def scale_uv_list(
    uv_list: NDArray,
    resolution: ArrayLike,
    bounds: NDArray,
    preserve_aspect_ratio: bool,
) -> NDArray:
    """
    Prepare a uv_list (array of coordinates) for rendering by scaling it to the given resolution and bounds

    Parameters
    ----------
    uv_list
    resolution:
        resolution of the rendered image, (width, height)
    bounds:
        bounds (in input mesh coordinate system) over which to render. Must contain entire mesh.
        (minx, miny),(maxx, maxy)
    preserve_aspect_ratio: bool
        preserve aspect ratio

    Returns
    -------
    uv_list

    """
    uv_list = uv_list.copy()
    if bounds is None:
        bounds = np.array(
            [
                [np.min(uv_list[:, 0]), np.min(uv_list[:, 1])],
                [np.max(uv_list[:, 0]), np.max(uv_list[:, 1])],
            ]
        )

    if np.any(uv_list[:, 0] < 0):
        min = np.min(uv_list[:, 0])
        uv_list[:, 0] += min
        bounds[:, 0] += min

    if np.any(uv_list[:, 1] < 0):
        min = np.min(uv_list[:, 1])
        uv_list[:, 1] += min
        bounds[:, 1] += min

    # offset by min bound
    uv_list = uv_list - bounds[0]
    # scale by diff between bounds
    if preserve_aspect_ratio:
        scale = np.max((np.asarray(bounds[1]) - np.asarray(bounds[0])))
    else:
        scale = np.asarray(bounds[1]) - np.asarray(bounds[0])
    uv_list = uv_list / scale
    uv_list *= np.asarray(resolution)

    return uv_list


def map_image(image, values):
    """
    maps values onto the image generated by model_inverse_uv.

    Parameters
    ----------
    image: np.Array(width, height)
        image generated by model_inverse_uv with values of each pixel corresponding to the index of the triangle they
        lie within
    values: list [float]
        values to map to each triangle

    Returns
    -------
    vals np.Array(width, height)
        array representing an image with values mapped to it


    """
    vals = values[image.astype(int)]
    mask = image == -1
    vals[mask] = np.NaN

    return vals


# Why do we need this instead of fractional amplitude set?
# This uses an absolute threshold whereas fractional amplitude set uses a proportion. Could merge the two
def calc_absolute_threshold_set(image, threshold):
    """

    Parameters
    ----------
    image: np.Array(width,height)
    threshold: float

    Returns
    ---------
    image_set: np.Array(width,height)
    """

    image_set = np.full(np.shape(image), np.nan)

    if threshold < 0:
        with np.errstate(invalid="ignore"):
            image_set[image < threshold] = 1
            image_set[image >= threshold] = 0

    else:
        with np.errstate(invalid="ignore"):
            image_set[image < threshold] = 0
            image_set[image >= threshold] = 1

    return image_set
