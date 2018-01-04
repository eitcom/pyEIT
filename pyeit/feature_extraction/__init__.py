"""
The :mod:`pyeit.feature_extraction` module deals with feature extraction
from raw EIT data. It currently includes methods to extract features from
EIT dynamic images and static properties.
"""

from .static_r import ati, ati_df
from .mesh_geometry import SimpleMeshGeometry

__all__ = ['ati',
           'ati_df',
           'SimpleMeshGeometry']
