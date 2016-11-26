""" add customized plot for 2D/3D mesh """
from .voronoi_plot import voronoi_plot

# TODO: update 3D plotting using vtk
try:
    from .tetplot import tetplot
except ImportError:
    print("mesh.plot: failed to import vispy")

__all__ = ['voronoi_plot', 'tetplot']
