import numpy as np
import time
from numba import njit

#==================================#
# Rotation Matrix
#==================================#

def z_rotation(theta):
    """
    Rotation about the -z-axis. 
    (y in cam0 coordinates)
    """
    c = np.cos(-theta)
    s = np.sin(-theta) 

    Rot = np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])              
    return Rot

def label2corners(label, config):
    '''
    Task 1
    input
        label (N,7) 3D bounding box with (x,y,z,h,w,l,ry)
    output
        corners (N,8,3) corner coordinates in the rectified reference frame

        (8, 3) array of vertices for the 3D box in
        following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7

    '''
    corners = np.zeros((label.shape[0],8,3))
    #start = time.time()
    for i in range(label.shape[0]):
        # extract the dimensions of the box
        height = label[i,3] + config['delta']
        width = label[i,4] + config['delta']
        length = label[i,5] + config['delta']
        #center coordinates
        x =  label[i,0]
        y =  label[i,1]
        z =  label[i,2]

        # Corners location 3D in velodyne frame
        '''
        corners_x = [x+width/2, x-width/2, x-width/2, x+width/2,x+width/2, x-width/2, x-width/2, x+width/2]
        corners_y = [y+length/2, y+length/2, y-length/2, y-length/2, y+length/2, y+length/2, y-length/2, y-length/2]
        corners_z = [z+height, z+height, z+height, z+height, z, z, z, z]
        '''
        corners_x = [width/2,- width/2, -width/2, width/2, width/2, -width/2, -width/2, width/2]
        corners_y = [length/2, length/2, -length/2, -length/2, length/2, length/2, -length/2, -length/2]
        corners_z = [height, height, height, height, 0, 0, 0, 0]
        #Rotation around z-axis
        Rot = z_rotation(label[i,6])

        box_dim = Rot@np.vstack([corners_x, corners_y, corners_z])
        box_dim[0,:] += x
        box_dim[1,:] += y
        box_dim[2,:] += z
        # Dimensions of box_dim: 3 x 8 i.e. rows are (x,y,z) and columns are the corners
        corners[i,:,:] = np.transpose(box_dim) 
    #end = time.time()
    #print("time in the label2corners function is " , end-start)
    return corners 

#@njit
def points_in_box(projection, norm_u, norm_v, norm_w):
    '''
    input
        xyz (N,3) point coordinates in rectified reference frame
        box (7,) 3d bounding box label (x,y,z,h,w,l,ry)
    output
        flag (N,) bool vector: true if point is in bounding box
    '''
    flag = (projection[:,2] <= norm_w) & (projection[:,0] <= norm_u) & (projection[:,1] <= norm_v)
    return flag



#@njit
def indexInBox(directions, center2point, norm_u, norm_v, norm_w):
 
    projection = np.absolute(center2point@directions)   
    xyz_indic = np.flatnonzero(points_in_box(projection, norm_u, norm_v, norm_w))
    return xyz_indic

#@njit
def roi_pool(pred, xyz, feat, config):
    '''
    Task 2
    a. Enlarge predicted 3D bounding boxes by delta=1.0 meters in all directions.
       As our inputs consist of coarse detection results from the stage-1 network,
       the second stage will benefit from the knowledge of surrounding points to
       better refine the initial prediction.
    b. Form ROI's by finding all points and their corresponding features that lie 
       in each enlarged bounding box. Each ROI should contain exactly 512 points.
       If there are more points within a bounding box, randomly sample until 512.
       If there are less points within a bounding box, randomly repeat points until
       512. If there are no points within a bounding box, the box should be discarded.
    input
        pred (N,7) bounding box labels
        xyz (N,3) point cloud
        feat (N,C) features
        config (dict) data config
    output
        valid_pred (K',7)
        pooled_xyz (K',M,3)
        pooled_feat (K',M,C)
            with K' indicating the number of valid bounding boxes that contain at least
            one point
    useful config hyperparameters
        config['delta'] extend the bounding box by delta on all sides (in meters)
        config['max_points'] number of points in the final sampled ROI
    '''
    start = time.time()
    # IMPORTANT N = 100 for pred (boundig boxes) but N  =16384 for xyz (point cloud)
    pooled_xyz = []
    valid_pred = []
    pooled_feat = []
    max_pointsROI = config['max_points']
    #https://stackoverflow.com/questions/21037241/how-to-determine-a-point-is-inside-or-outside-a-cube
    #print("N is equal", pred.shape[0])
    corners = label2corners(pred, config)
    #print("corners shape", corners.shape[0])
    #vectores of shape (nb boxes = 100, 3)

    ##### Takes 1.2 ms #######
    u =  corners[:,5,:] - corners[:,6,:]
    norm_u = np.linalg.norm(u,axis=1)
    u = u/norm_u[:,None]
    v = (corners[:,7,:] - corners[:,6,:])
    norm_v = np.linalg.norm(v,axis = 1)
    v = v/norm_v[:,None]
    w = (corners[:,2,:] - corners[:,6,:])
    norm_w = np.linalg.norm(w,axis=1)
    w = w/norm_w[:,None]
    #box_center = (corners[:,6,:] + corners[:,0,:])/2.0
    #####################
   
    xmax = np.amax(corners[:,:,0])
    ymax = np.amax(corners[:,:,1])
    zmax = np.amax(corners[:,:,2])
    xmin = np.amin(corners[:,:,0])
    ymin = np.amin(corners[:,:,1])
    zmin = np.amin(corners[:,:,2])
    xyz_keep = np.flatnonzero((xyz[:,0] <= xmax) & (xyz[:,1] <= ymax) & (xyz[:,2] <= zmax) &
                               (xyz[:,0] >= xmin) & (xyz[:,1] >= ymin) & (xyz[:,2] >= zmin) )
    xyz = xyz[xyz_keep,:]
    # loop1 = time.time()
    #xyz_repeat = np.tile(xyz,(corners.shape[0],1,1))
    # loop2 = time.time()
    # print("sample", loop2-loop1) 
    #box_ref = corners[:,6,:].reshape(100,1,3)
    #center2point = np.subtract(xyz_repeat, box_ref, dtype=np.float32)
    # print(center2point.shape)

    #center2point = np.zeros((corners.shape[0], xyz.shape[0], xyz.shape[1]))
    #center2point = pointRectified(xyz,corners,center2point)
    
    
    for i in range(corners.shape[0]):
        
        #center2point = np.subtract(xyz, box_center[i,:],dtype=np.float32)

        center2point = np.subtract(xyz, corners[i,6,:],dtype=np.float32)
        
        directions = np.stack((u[i,:], v[i,:], w[i,:]), axis = 1)
        #print(directions.shape)
        #local = directions@(xyz.T)
        #check points which are not in the box is maybe faster ?
        xyz_indic = indexInBox(directions, center2point, norm_u[i], norm_v[i], norm_w[i])
        
        if 1 < len(xyz_indic) < max_pointsROI:
            idx = np.random.choice(xyz_indic, size=max_pointsROI, replace=True)
            boxPoints = xyz[idx,:]
            boxFeat = feat[idx,:]
    
        elif len(xyz_indic) > max_pointsROI:
            idx = np.random.choice(xyz_indic, size=max_pointsROI, replace=False)
            boxPoints = xyz[idx,:]
            boxFeat = feat[idx,:]
       
        else: # just one point in this box, discard
            continue
        
        ####### around 1.5 micro seconds per iter * 100 = 0.15 ms
        pooled_xyz.append(boxPoints)
        pooled_feat.append(boxFeat)
        valid_pred.append(pred[i,:])
        ###########################
    pooled_xyz = np.array(pooled_xyz)
    pooled_feat = np.array(pooled_feat)
    valid_pred = np.array(valid_pred)

    #print("valid pred shape", valid_pred.shape)
    # print("pooled_xyz shape", pooled_xyz.shape)
    # print("pooled_feat shape", pooled_feat.shape)
    return valid_pred, pooled_xyz, pooled_feat