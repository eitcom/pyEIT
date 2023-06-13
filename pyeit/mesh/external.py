from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np
import struct
import trimesh
import shapely
from shapely.geometry import Polygon, Point, MultiLineString, LineString
import shapely.affinity
from pyeit.mesh import PyEITMesh
from pathlib import Path


def load_mesh(filename: str, dims: int = 2) -> PyEITMesh:
    """
    Load mesh from file using trimesh. Output is a PyEITMesh

    Trimesh loads X,Y, and Z axes into columns 0,1, and 2 respectively. 2D meshes should be defined along the x and y
    axes. When plotted with Matplotlib, meshes will appear with the origin in the bottom left (except with imshow defaults).

    If the filetype is .ply, perm will be unpacked from color information. red, green, blue, and alpha are interpreted
    as 4 bytes in the IEEE 754 binary32 format. For more information, see the python docs for struct.pack, struct.unpack.

    Parameters
    ----------
    filename:
        Filename of mesh to load
    dims:
        If set to 2, the Z column will be deleted.

    Returns
    -------
    mesh

    """
    t_mesh = trimesh.load(filename, force="mesh")

    if Path(filename).suffix.casefold() == ".ply":
        ply_key = "_ply_raw"
        if ply_key not in t_mesh.metadata:
            ply_key = "ply_raw"
            if ply_key not in t_mesh.metadata:
                raise ValueError("Could not interpret mesh file as .ply")

        red = t_mesh.metadata[ply_key]["face"]["data"]["red"]
        green = t_mesh.metadata[ply_key]["face"]["data"]["green"]
        blue = t_mesh.metadata[ply_key]["face"]["data"]["blue"]
        alpha = t_mesh.metadata[ply_key]["face"]["data"]["alpha"]
        perm = np.array(
            [
                struct.unpack(
                    ">f",
                    bytearray(
                        [
                            red[i].item(),
                            green[i].item(),
                            blue[i].item(),
                            alpha[i].item(),
                        ]
                    ),
                )[0]
                for i in range(0, len(t_mesh.faces))
            ]
        )  # Use a.item() to support either binary or ASCII encoding
    else:
        perm = np.ones(t_mesh.faces.shape[0], dtype=float)

    if dims == 2:
        t_mesh.vertices = np.delete(t_mesh.vertices, 2, axis=1)

    mesh = PyEITMesh(node=t_mesh.vertices, element=t_mesh.faces, perm=perm)

    return mesh


def place_electrodes_equal_spacing(
    mesh: PyEITMesh,
    n_electrodes: int = 16,
    starting_angle: float = 0,
    starting_offset: float = 0.5,
    counter_clockwise: bool = False,
    chest_and_spine_ratio: float = 1,
    flat_plane: str = "z",
    output_obj: Optional[dict] = None,
) -> List[int]:
    """
    Creates a list of coordinate indices representing electrodes equally spaced around the perimeter of the 2D input mesh

    *NOTE* Clockwise/Counter clockwise is relative to the 2D mesh. If the mesh represents a radiological (i.e. feet up)
    view of a subject, the clockwiseness will be the reverse of the subject's perspective.

    Parameters
    ----------
    mesh
        PyEITMesh to use for electrode placement
    n_electrodes
        number of electrodes to space equally around the mesh perimeter
    starting_angle
        angle in radians from +y axis
        the starting point for the electrode placement is the intersection between the exterior of the mesh and a line
        drawn from the centroid at a given angle
    starting_offset
        offset between the starting point of electrode placement and the actual placement of the first electrode.
        units are exterior polygon perimeter/n_electrodes
    counter_clockwise
        place electrodes in a counter clockwise fashion. By default electrodes are placed in a clockwise fashion because
        the polygon exterior is always clockwise
    chest_and_spine_ratio
        spacing between electrodes 1 and N, and between electrodes (N/2)-1 and (N/2)+1 as a multiple of the remaining electrodes' spacing
    flat_plane
        plane in which to consider the mesh flat
    output_obj
        object to store optional output for plotting purposes
        structure: {centroid:trimesh object centroid,
                    exterior_polygon: shapely polygon,
                    intersecting_line: shapely linestring,
                    intersection: shapely point}

    Returns
    -------
    electrode_nodes
        node indices of equally spaced electrodes

    """
    if output_obj is None:
        output_obj = {}

    if flat_plane not in ["x", "y", "z"]:
        raise ValueError("Please select a flat plane from x, y, or z")
    flat_ind = {"x": 0, "y": 1, "z": 2}[flat_plane]

    trimesh_obj = trimesh.Trimesh(np.delete(mesh.node, flat_ind, axis=1), mesh.element)

    exterior_polygon = create_exterior_polygon(trimesh_obj)
    plotting_obj: dict = {}
    intersection = perimeter_point_from_centroid(
        exterior_polygon, starting_angle, plotting_obj
    )
    zero_offset = exterior_polygon.exterior.project(intersection, normalized=True)
    starting_offset_spacing = (1 / n_electrodes) * starting_offset

    if chest_and_spine_ratio != 1:
        electrode_spacing = 1 / (
            (n_electrodes - 2) + 2 * chest_and_spine_ratio
        )  # Normalized spacing
        chest_and_spine_spacing = 1 / (((n_electrodes - 2) / chest_and_spine_ratio) + 2)
        starting_offset_spacing = chest_and_spine_spacing * starting_offset
        spacing_list = [
            chest_and_spine_spacing
            if i == 0 or i == (n_electrodes / 2)
            else electrode_spacing
            for i in range(n_electrodes)
        ]

        electrode_points = [
            exterior_polygon.exterior.interpolate(distance, normalized=True)
            for distance in list_based_interpolate_distance(
                zero_offset,
                starting_offset_spacing,
                spacing_list,
                reverse=counter_clockwise,
            )
        ]

    else:
        electrode_points = [
            exterior_polygon.exterior.interpolate(
                # If counter clockwise is set, reverse the direction
                equal_spaced_interpolate_distance(
                    zero_offset,
                    starting_offset_spacing,
                    i,
                    n_electrodes,
                    reverse=counter_clockwise,
                ),
                normalized=True,
            )
            for i in range(0, n_electrodes)
        ]

    ex_poly_xy = exterior_polygon.exterior.xy
    exterior_polygon_points = np.array(list(zip(ex_poly_xy[0], ex_poly_xy[1])))
    electrode_nodes_exterior_polygon = [
        find_closest_point(np.array([point.x, point.y]), exterior_polygon_points)
        for point in electrode_points
    ]
    electrode_nodes = [
        find_closest_point(point, np.delete(mesh.node, flat_ind, axis=1))
        for point in exterior_polygon_points[electrode_nodes_exterior_polygon]
    ]

    output_obj["centroid"] = trimesh_obj.centroid
    output_obj["exterior_polygon"] = exterior_polygon
    output_obj["intersecting_line"] = plotting_obj["intersecting_line"]
    output_obj["intersection"] = intersection

    return electrode_nodes


def create_exterior_polygon(
    trimesh_obj: trimesh.Trimesh,
) -> Polygon:
    """
    Create a polygon representing the exterior edge of the input mesh. The polygon is created by identifying the edges
    of the input mesh which are only referenced by one triangle. These are the edges of the polygon.

    In order for this algorithm to run in O(n) time, we use a hash map (python dictionary) to store the number of triangles
    referring to each edge

    In case the mesh contains several separate polygons, this returns the one with the largest area

    Parameters
    ----------
    trimesh_obj
        object from which to create an exterior polygon

    Returns
    -------
    edge_polygon:
        Polygon representing the exterior edge of the input mesh

    """
    edge_dict: dict = {}
    for (
        edge
    ) in (
        trimesh_obj.edges_sorted
    ):  # trimesh_obj.edges_sorted removes directionality from edges (eg. (0,3) and (3,0) both become (0,3)
        edge_bytes = edge.tobytes()  # tobytes() makes the edge array hashable
        if edge_bytes in edge_dict:
            edge_dict[edge_bytes] = edge_dict[edge_bytes] + 1
        else:
            edge_dict[edge_bytes] = 1

    outer_edges = np.vstack(
        [np.frombuffer(k, dtype=np.int64) for k, v in edge_dict.items() if v == 1]
    )

    lines = MultiLineString(
        [LineString(line) for line in trimesh_obj.vertices[outer_edges]]
    )
    merged_line = shapely.ops.linemerge(lines)

    polygons = list(shapely.ops.polygonize(merged_line))

    areas = [polygon.area for polygon in polygons]
    index_max = areas.index(max(areas))

    return polygons[index_max]


def perimeter_point_from_centroid(
    polygon: Polygon, angle: float, output_obj: Optional[dict] = None
) -> Point:
    """
    Calculates a point on the perimeter of the input polygon which intersects with a line drawn from the centroid at a
    given angle

    Parameters
    ----------
    polygon
        polygon on which to find a point
    angle
        angle in radians from the +y axis
    output_obj
        object to store optional output for plotting purposes
        structure: {intersection_line: shapely.geometry.LineString}

    Returns
    -------
    perimeter_point
        point on the perimeter of the input polygon which intersects with a line drawn from the centroid

    """
    if output_obj is None:
        output_obj = {}

    max_distance = Point(polygon.bounds[0], polygon.bounds[1]).distance(
        Point(polygon.bounds[2], polygon.bounds[3])
    )  # The max distance is the diagonal line across the bounding box
    endpoint = [
        polygon.centroid.x + max_distance * np.sin(angle),
        polygon.centroid.y + max_distance * np.cos(angle),
    ]
    line = LineString((polygon.centroid, endpoint))
    perimeter_point = line.intersection(polygon.exterior)

    output_obj["intersecting_line"] = line

    return perimeter_point


def list_based_interpolate_distance(
    zero_offset, starting_offset_spacing, spacing_list, reverse=False
):
    """

    Parameters
    ----------
    zero_offset
    starting_offset_spacing
    spacing_list
    reverse

    Returns
    -------
    interpolate_distances

    """
    direction = -1 if reverse else 1

    # spacing_list[0] not used because extra offset is used instead
    interpolate_distances = [
        (
            zero_offset
            + (direction * (starting_offset_spacing + sum(spacing_list[1 : i + 1])))
        )
        % 1
        for i in range(len(spacing_list))
    ]

    return interpolate_distances


def equal_spaced_interpolate_distance(
    zero_offset: float,
    starting_offset_spacing: float,
    index: int,
    total: int,
    reverse: bool = False,
) -> float:
    """
    calculates distance for shapley interpolate function (normalized = True)

    Parameters
    ----------
    zero_offset
        offset from origin of shapely polygon to intersection with starting angle
    starting_offset_spacing
        offset from zero_offset to first point (if reverse is True, this is reversed, but zero offset is not)
    index
        which point to calculate distance for
    total
        total number of points
    reverse
        reverse the direction

    Returns
    -------
    interpolate_distance
        distance to be used as input to shapely interpolate function (with normalized=True)

    """
    if not reverse:
        interpolate_distance = (
            zero_offset + (starting_offset_spacing + index * (1 / total))
        ) % 1
    else:
        interpolate_distance = (
            zero_offset - (starting_offset_spacing + index * (1 / total))
        ) % 1

    return interpolate_distance


def find_closest_point(point: np.ndarray, point_list: np.ndarray) -> int:
    """
    Find the closest match between an input point and the elements of a point list

    Parameters
    ----------
    point
    point_list
    """

    index = np.argmin([np.linalg.norm(point - point_b) for point_b in point_list])
    return int(index)


def map_points_to_perimeter(
    mesh: PyEITMesh,
    points: List[Tuple[float, float]],
    output_obj: Optional[dict] = None,
    map_to_nodes: Optional[bool] = True,
) -> List[Point]:
    """
    Map a list of coordinates to points on the perimeter of a mesh. Coordinates are mapped by drawing a line
    from the centroid through each point, and finding the intersection between that line and the perimeter
    of the mesh.

    Parameters
    ----------
    mesh
    points
    output_obj
        object to store optional output for plotting purposes
        structure: {centroid:trimesh object centroid,
                    offset_points: list(tuple(x,y))
                        input points translated such that centroid is aligned with mesh centroid
                    exterior_polygon: shapely polygon,
                    intersecting_lines: list(shapely linestring)}
    map_to_nodes
        map resulting point to nodes on the mesh

    Returns
    -------
    intersections

    """
    if output_obj is None:
        output_obj = {}

    trimesh_obj = trimesh.Trimesh(mesh.node, mesh.element)

    exterior_polygon = create_exterior_polygon(trimesh_obj)

    # move points centroid to mesh centroid
    points_polygon_uncentered = Polygon(points)
    offset = (
        points_polygon_uncentered.centroid.x - exterior_polygon.centroid.x,
        points_polygon_uncentered.centroid.y - exterior_polygon.centroid.y,
    )
    points = [(point[0] - offset[0], point[1] - offset[1]) for point in points]
    points_polygon = Polygon(points)

    b1 = exterior_polygon.bounds
    b2 = points_polygon.bounds
    total_bounds = (
        min(b1[0], b2[0]),
        min(b1[1], b2[1]),
        max(b1[2], b2[2]),
        max(b1[3], b2[3]),
    )
    max_distance = Point(total_bounds[0], total_bounds[1]).distance(
        Point(total_bounds[2], total_bounds[3])
    )  # The max distance is the diagonal line across the bounding box

    # for each point, find intersection between line from point to centroid and exterior polygon
    intersections = []
    intersecting_lines = []
    for point in points:
        line = LineString((exterior_polygon.centroid, point))
        scale = max_distance / line.length
        line = shapely.affinity.scale(line, scale, scale, 1, exterior_polygon.centroid)
        intersecting_lines.append(line)
        intersection = line.intersection(exterior_polygon.exterior)
        intersections.append(intersection)

    if map_to_nodes:
        intersections = [
            find_closest_point(point.xy, mesh.node) for point in intersections
        ]

    output_obj["centroid"] = trimesh_obj.centroid
    output_obj["offset_points"] = points
    output_obj["exterior_polygon"] = exterior_polygon
    output_obj["intersecting_lines"] = intersecting_lines

    return intersections
