from pyeit.mesh.external import (
    find_closest_point,
    equal_spaced_interpolate_distance,
    create_exterior_polygon,
    load_mesh,
    perimeter_point_from_centroid,
    place_electrodes_equal_spacing,
    map_points_to_perimeter,
)
import numpy as np
from pathlib import Path
import trimesh
from shapely.geometry import LinearRing
from shapely.ops import polygonize

parent_dir = str(Path(__file__).parent)


def test_find_closest_point():
    point_coords = np.array([5.0, 6.0])
    node_coords = np.array([[0.0, 0.0], [0.0, 7.0], [7.0, 7.0], [7.0, 0.0]])
    correct_index = 2

    index = find_closest_point(point_coords, node_coords)

    assert index == correct_index


def test_equal_spaced_interpolate_distance():
    zero_offset = 0.25
    total = 8

    correct_2 = 0.5
    correct_7 = 0.125
    correct_2_reverse = 0

    calc_2 = equal_spaced_interpolate_distance(
        zero_offset=zero_offset, starting_offset_spacing=0, index=2, total=total
    )
    calc_7 = equal_spaced_interpolate_distance(
        zero_offset=zero_offset, starting_offset_spacing=0, index=7, total=total
    )
    calc_2_reverse = equal_spaced_interpolate_distance(
        zero_offset=zero_offset,
        starting_offset_spacing=0,
        index=2,
        total=total,
        reverse=True,
    )

    assert correct_2 == calc_2
    assert correct_7 == calc_7
    assert correct_2_reverse == calc_2_reverse


def test_create_exterior_polygon():
    mesh_obj = load_mesh(parent_dir + "/data/Rectangle.STL")
    trimesh_obj = trimesh.Trimesh(mesh_obj.node, mesh_obj.element)

    polygon = create_exterior_polygon(trimesh_obj)

    rectangle_coords = np.vstack(polygon.boundary.xy).transpose()

    correct_rectangle = [[0.0, 0.0], [0.0, 30.0], [50.0, 30.0], [50.0, 0.0], [0.0, 0.0]]

    assert np.all(rectangle_coords == correct_rectangle)
    assert not polygon.exterior.is_ccw


def test_create_exterior_polygons_double_circle():
    mesh_obj = load_mesh(parent_dir + "/data/Simple_Circle.ply")
    trimesh_obj = trimesh.Trimesh(mesh_obj.node, mesh_obj.element)

    polygon = create_exterior_polygon(trimesh_obj)

    polygon_coords = np.vstack(polygon.exterior.xy).transpose()

    correct_circle = [
        [0.00000000e00, 1.00000000e02],
        [5.13069272e-01, 1.10116798e02],
        [2.04700994e00, 1.20129898e02],
        [4.58607817e00, 1.29936295e02],
        [8.10422039e00, 1.39435593e02],
        [1.25653400e01, 1.48530197e02],
        [1.79236603e01, 1.57126801e02],
        [2.41241894e01, 1.65137207e02],
        [3.11033096e01, 1.72479294e02],
        [3.87893982e01, 1.79077606e02],
        [4.71035995e01, 1.84864395e02],
        [5.59605904e01, 1.89780502e02],
        [6.52694778e01, 1.93775208e02],
        [7.49347534e01, 1.96807693e02],
        [8.48572235e01, 1.98846802e02],
        [9.49350815e01, 1.99871704e02],
        [1.05064903e02, 1.99871704e02],
        [1.15142799e02, 1.98846802e02],
        [1.25065300e02, 1.96807693e02],
        [1.34730499e02, 1.93775208e02],
        [1.44039398e02, 1.89780502e02],
        [1.52896393e02, 1.84864395e02],
        [1.61210602e02, 1.79077606e02],
        [1.68896698e02, 1.72479294e02],
        [1.75875793e02, 1.65137207e02],
        [1.82076294e02, 1.57126801e02],
        [1.87434692e02, 1.48530197e02],
        [1.91895798e02, 1.39435593e02],
        [1.95413895e02, 1.29936295e02],
        [1.97953003e02, 1.20129898e02],
        [1.99486893e02, 1.10116798e02],
        [2.00000000e02, 1.00000000e02],
        [1.99486893e02, 8.98831711e01],
        [1.97953003e02, 7.98701477e01],
        [1.95413895e02, 7.00636902e01],
        [1.91895798e02, 6.05644188e01],
        [1.87434692e02, 5.14698105e01],
        [1.82076294e02, 4.28731804e01],
        [1.75875793e02, 3.48627586e01],
        [1.68896698e02, 2.75207195e01],
        [1.61210602e02, 2.09224300e01],
        [1.52896393e02, 1.51355801e01],
        [1.44039398e02, 1.02195396e01],
        [1.34730499e02, 6.22478914e00],
        [1.25065300e02, 3.19229102e00],
        [1.15142799e02, 1.15317094e00],
        [1.05064903e02, 1.28351197e-01],
        [9.49350815e01, 1.28351197e-01],
        [8.48572235e01, 1.15317094e00],
        [7.49347534e01, 3.19229102e00],
        [6.52694778e01, 6.22478914e00],
        [5.59605904e01, 1.02195396e01],
        [4.71035995e01, 1.51355801e01],
        [3.87893982e01, 2.09224300e01],
        [3.11033096e01, 2.75207195e01],
        [2.41241894e01, 3.48627586e01],
        [1.79236603e01, 4.28731804e01],
        [1.25653400e01, 5.14698105e01],
        [8.10422039e00, 6.05644188e01],
        [4.58607817e00, 7.00636902e01],
        [2.04700994e00, 7.98701477e01],
        [5.13069272e-01, 8.98831711e01],
        [0.00000000e00, 1.00000000e02],
    ]

    assert np.all(np.isclose(polygon_coords, correct_circle))


def test_perimeter_point_from_centroid():
    mesh_obj = load_mesh(parent_dir + "/data/Rectangle.STL")
    trimesh_obj = trimesh.Trimesh(mesh_obj.node, mesh_obj.element)
    rectangle = create_exterior_polygon(trimesh_obj)

    plotting_obj = {}
    point_pi_4 = perimeter_point_from_centroid(rectangle, np.pi / 4, plotting_obj)
    coords_pi_4 = [point_pi_4.x, point_pi_4.y]

    # # Plot calculation
    # fig, ax = plt.subplots()
    # create_mesh_plot(ax, mesh_obj)
    # ax.plot(trimesh_obj.centroid[0], trimesh_obj.centroid[1], marker="o", ls="", color="red")
    # ax.plot(*rectangle.exterior.xy, color="crimson")
    # ax.plot(*plotting_obj["intersecting_line"].intersection(rectangle).xy, color="crimson")
    # ax.plot(*plotting_obj["intersecting_line"].intersection(rectangle.boundary).xy, marker="o", color="purple")
    #
    # plt.show()

    correct_pi_4 = [40, 30]

    assert np.allclose(coords_pi_4, correct_pi_4)


def test_place_electrodes_equal_spacing():
    mesh_obj = load_mesh(parent_dir + "/data/Rectangle.STL")

    plotting_obj = {}
    electrode_nodes = place_electrodes_equal_spacing(
        mesh_obj,
        n_electrodes=8,
        starting_angle=np.pi,
        starting_offset=0,
        output_obj=plotting_obj,
    )
    ccw_electrode_nodes = place_electrodes_equal_spacing(
        mesh_obj,
        n_electrodes=8,
        starting_angle=np.pi,
        starting_offset=0,
        counter_clockwise=True,
    )

    # # Plot electrode placing data
    # fig, ax = plt.subplots()
    # create_mesh_plot(ax, mesh_obj)
    # ax.plot(plotting_obj["centroid"][0], plotting_obj["centroid"][1], marker="o", ls="", color="red")
    # ax.plot(*plotting_obj["exterior_polygon"].exterior.xy, color="crimson")
    # ax.plot(*plotting_obj["intersecting_line"].intersection(plotting_obj["exterior_polygon"]).xy, color="crimson")
    # ax.plot(*plotting_obj["intersection"].xy, marker="o", color="purple")
    #
    # electrode_points = [Point(node) for node in mesh_obj.node[electrode_nodes]]
    # for point in electrode_points:
    #     ax.plot(*point.xy, marker='o', color="green")
    #
    # plt.show()

    correct_nodes = [0, 0, 0, 1, 1, 3, 3, 2]
    correct_ccw_nodes = [0, 2, 3, 3, 1, 1, 0, 0]

    assert np.all(electrode_nodes == correct_nodes)
    assert np.all(ccw_electrode_nodes == correct_ccw_nodes)


def test_place_electrodes_equal_spacing_chest_and_spine():
    mesh_obj = load_mesh(
        parent_dir
        + "/data/mesha06_bumpychestslice_radiological_view_both_lungs_1_0-3.ply"
    )

    plotting_obj = {}
    electrode_nodes = place_electrodes_equal_spacing(
        mesh_obj, 8, 0, 0.5, chest_and_spine_ratio=2, output_obj=plotting_obj
    )

    # # Plot electrode placing data
    # fig, ax = plt.subplots()
    # create_mesh_plot(ax, mesh_obj)
    # ax.plot(plotting_obj["centroid"][0], plotting_obj["centroid"][1], marker="o", ls="", color="red")
    # ax.plot(*plotting_obj["exterior_polygon"].exterior.xy, color="crimson")
    # ax.plot(*plotting_obj["intersecting_line"].intersection(plotting_obj["exterior_polygon"]).xy, color="crimson")
    # ax.plot(*plotting_obj["intersection"].xy, marker="o", color="green")
    #
    # electrode_points = [Point(node) for node in mesh_obj.node[electrode_nodes]]
    # for point in electrode_points:
    #     ax.plot(*point.xy, marker='o', color="purple")
    # ax.set_title("Place electrodes equal spacing\nplus additional chest and spine spacing")
    # plt.show()

    correct_nodes_2 = [1562, 2948, 2061, 2326, 2949, 2394, 2698, 2908]

    assert np.all(electrode_nodes == correct_nodes_2)


def test_polygonize_cw():
    # Just making sure that shapely.ops.polygonize results in clockwise exteriors
    cw_ring = LinearRing([(0, 0), (1, 1), (1, 0)])
    ccw_ring = LinearRing([(1, 0), (1, 1), (0, 0)])

    p_cw = list(polygonize(cw_ring))
    p_ccw = list(polygonize(ccw_ring))

    assert not p_cw[0].exterior.is_ccw
    assert not p_ccw[0].exterior.is_ccw


def test_map_points_to_perimeter():
    mesh_obj = load_mesh(parent_dir + "/data/Rectangle.STL")
    angles = [(np.pi / 4) * i for i in range(8)]
    points = [(10 * np.sin(angle), 10 * np.cos(angle)) for angle in angles]

    plotting_obj = {}
    intersections = map_points_to_perimeter(
        mesh_obj, points, output_obj=plotting_obj, map_to_nodes=False
    )

    # # Plot intersections and mesh polygon
    # fig, ax = plt.subplots()
    # create_mesh_plot(ax, mesh_obj)
    # ax.plot(plotting_obj["centroid"][0], plotting_obj["centroid"][1], marker="o", ls="", color="red")
    # ax.plot(*plotting_obj["exterior_polygon"].exterior.xy, color="crimson")
    # for line, point in zip(plotting_obj["intersecting_lines"], intersections):
    #     ax.plot(*line.intersection(plotting_obj["exterior_polygon"]).xy, color="crimson")
    #     ax.plot(*point.xy, marker="o", color="purple")
    # for point in plotting_obj["offset_points"]:
    #     ax.plot(point[0], point[1], marker="o", color="green")
    # ax.set_title("Map points to perimeter")
    #
    # plt.show()

    correct_intersections_list = [
        (25.0, 30.0),
        (40.00000000000004, 30.0),
        (50.0, 15.0),
        (40.00000000000004, 0.0),
        (25.0, 0.0),
        (10.00000000000004, 0.0),
        (0.0, 15.0),
        (10.00000000000004, 30.0),
    ]

    intersections_list = [(i.x, i.y) for i in intersections]

    assert np.all(
        np.isclose(
            np.array(intersections_list),
            np.array(correct_intersections_list),
        )
    )
