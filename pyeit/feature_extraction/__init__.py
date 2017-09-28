"""
The :mod:`pyeit.feature_extraction` module deals with feature extraction
from raw EIT data. It currently includes methods to extract features from
EIT dynamic images and static properties.
"""

from .static_r import ati
from .mesh_geometry import SimpleMeshGeometry

__all__ = ['ati',
           'SimpleMeshGeometry']
