import numpy as np
from .wrapper import create

def groundtruth_IMG_based(IMG, n_el=16, perm_empty_gnd = 1,perm_obj = 10, h0 = 0.1):
    """
    Wraps a image to the PyEITMesh unit circle area.
    
    Parameters
    ----------
    IMG : np.ndarray
        200x200 image
    n_el : int
        number of electrodes
    perm_empty_gnd : int
        perm of the empty ground
    perm_obj : int
        perm ob the object area
    h0 : float
        refinement of the mesh
    
    Returns
    -------
    PyEITMesh
        mesh object
    """
    mesh_obj = create(n_el=n_el, h0=h0)
    X_Y=np.array(np.where(IMG==1))
    X = X_Y[1,:] -100
    Y = (X_Y[0,:] -100)*-1
    pts = mesh_obj.element
    tri = mesh_obj.node
    perm = mesh_obj.perm.copy()
    tri_centers = np.mean(tri[pts], axis=1)
    mesh_x = np.round(tri_centers[:,0]*100)
    mesh_y = np.round(tri_centers[:,1]*100)
    Perm = np.ones(tri_centers.shape[0])*perm_empty_gnd
    for i in range(len(X)):
        for j in range(len(mesh_x)):
            if X[i] == mesh_x[j] and Y[i] == mesh_y[j]:
                Perm[j] = perm_obj
    mesh_obj.perm = Perm
    return mesh_obj