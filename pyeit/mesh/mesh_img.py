""" convert an image to pyeit mesh """
# https://github.com/JacobTh98/pyEIT_image_wrapper

# Heed requirements.txt
import numpy as np
import matplotlib.pyplot as plt
import pyeit.mesh as mesh
import math
import cv2

# Option to plot the generated mesh.
def plot_mesh(mesh_obj, el_pos):
    """
    Plot mesh
    
    input:  - mesh_obj: mesh_obj (pyEIT)
            - el_pos: electrode sequence (array)
    """
    plt.style.use('default')
    pts = mesh_obj['node']
    tri = mesh_obj['element']
    x, y = pts[:, 0], pts[:, 1]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.tripcolor(x, y, tri, np.real(mesh_obj['perm']),
                 edgecolors='k', shading='flat', alpha=0.5,
                 cmap=plt.cm.viridis)
    # draw electrodes
    ax.plot(x[el_pos], y[el_pos], 'ro')
    for i, e in enumerate(el_pos):
        ax.text(x[e], y[e], str(i+1), size=12)
    ax.set_title(r'mesh')
    ax.set_aspect('equal')
    ax.set_ylim([-1.2, 1.2])
    ax.set_xlim([-1.2, 1.2])
    fig.set_size_inches(6, 6)

# Not the actual function to mesh an Image. This function is provided to test the following wrapper.
def geometry_to_img_wrot(objct='circle',r=0.5,phi=0,d=0.5):
    """
    This function was used to generate the ground truth for a measurement on a rotationg plate below the SpectraEIT-Kit.
    geometry_to_img_wrot ... "geometry to image with rotation"
    
    input:  - object: 'circle', 'square', 'triangle'
            - r: radius for polar coordinate system
            - phi: angle for polar coordinate system
            - d: object diameter from 0-1. Zero equals 0% and one equals 100% of the unit circle diameter.
            - rot rotation on fixed position
            
    return: IMG (200x200 pixel picture)
    """
    def pol2cart(r, phi):
        x = r * np.cos(phi)+100
        y = r * np.sin(phi)+100
        return (int(x), int(y))

    def rotate_image(image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    def draw_circle(r,phi,d):
        IMG = np.zeros((200,200))
        center_coordinates = pol2cart(r,phi*0)#*0 für die Rotation
        color = (1,0,0)
        thickness = -1
        IMG = cv2.circle(IMG, center_coordinates, int(d*100), color, thickness)
        return IMG

    def draw_square(r,phi,d):
        IMG = np.zeros((200,200))
        center_coordinates = pol2cart(r,phi*0)#*0 für die Rotation
        start_point = (int(center_coordinates[0]-int(d*100)), center_coordinates[1]-int(d*100))
        end_point = (center_coordinates[0]+int(d*100), center_coordinates[1]+int(d*100))
        color = (1, 0, 0)
        thickness = -1
        IMG = cv2.rectangle(IMG, start_point, end_point, color, thickness)
        return IMG
    
    def draw_triangle(r,phi,d):
        IMG = np.zeros((200,200))
        center_coordinates = pol2cart(r,phi*0)
        pt1 = (int(center_coordinates[0]), int(center_coordinates[1]-int((d)*100)))
        pt2 = (int(center_coordinates[0]+int((d)*100)), int(center_coordinates[1])+int((d)*100))
        pt3 = (int(center_coordinates[0]-int((d)*100)), int(center_coordinates[1])+int((d)*100))
        tri_edges = np.array( [pt1, pt2, pt3] )
        IMG = cv2.drawContours(IMG, [tri_edges], 0, (1,0,0), -1)
        return IMG
    r = int(r*100)
    angle = phi #Needed for rotation
    phi = math.radians(phi)# Grad in Rad

    if objct == 'circle':
        IMG = draw_circle(r,phi,d)
    if objct == 'square':
        IMG = draw_square(r,phi,d)
    if objct == 'triangle':
        IMG = draw_triangle(r,phi,d)
        
    IMG = rotate_image(IMG,angle)    
    return IMG

#The wrapper function to transform a 200x200 picture to the pyEIT mesh.
def groundtruth_IMG_based(IMG, n_el=16, perm_empty_gnd = 1,perm_obj = 10):
    """
    Wraps a image to the pyEIT mesh area.
    
    input:  - 200x200 image
            - n_el: number of electrodes
            - perm_empty_gnd: perm of the empty ground
            - perm_obj: perm ob the object area
    return: mesh_obj (pyEIT)
    """
    def generate_mesh(n_el=n_el, perm=perm_obj):
        def _fd(pts):
            if pts.ndim == 1:
                pts = pts[np.newaxis]
            a, b = 1.0, 1.0
            return np.sum((pts/[a, b])**2, axis=1) - 1.0
        # Change h0 for a more filigree or rawer mesh.
        mesh_obj, el_pos = mesh.create(n_el, fd=_fd, h0=0.05)
        return mesh_obj
    X_Y=np.array(np.where(IMG==1))
    X = X_Y[1,:] -100
    Y = (X_Y[0,:] -100)*-1
    mesh_obj = generate_mesh()
    pts = mesh_obj["element"]
    tri = mesh_obj["node"]
    perm = mesh_obj["perm"].copy()
    tri_centers = np.mean(tri[pts], axis=1)
    mesh_x = np.round(tri_centers[:,0]*100)
    mesh_y = np.round(tri_centers[:,1]*100)
    Perm = np.ones(tri_centers.shape[0])*perm_empty_gnd
    for i in range(len(X)):
        for j in range(len(mesh_x)):
            if X[i] == mesh_x[j] and Y[i] == mesh_y[j]:
                Perm[j] = perm_obj
    mesh_obj['perm'] = Perm
    return mesh_obj