#
from matplotlib.colors import LinearSegmentedColormap


cdict1 = {'red':   ((0.0, 0.0, 0.0),
                   (0.5, 0.0, 0.1),
                   (1.0, 1.0, 1.0)),

         'green':  ((0.0, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),

         'blue':   ((0.0, 0.0, 1.0),
                   (0.5, 0.1, 0.0),
                   (1.0, 0.0, 0.0))
         }
         

def blue_red():
    return LinearSegmentedColormap('BlueRed1', cdict1)

v = blue_red()

