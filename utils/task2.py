import numpy as np
import time
#from numba import njit
from utils.task1 import label2corners

#@njit
def points_in_box(projection, norm_u, norm_v, norm_w):
    '''
    output
        flag (N,) bool vector: true if point is in bounding box
    '''
    flag = (2*projection[:,2] <= norm_w) & (2*projection[:,0] <= norm_u) & (2*projection[:,1] <= norm_v)
    return flag

#@njit
def indexInBox(xyz, corners, max_points):
    '''
    input
        xyz (N,3) points in rectified reference frame
        corners (N',3,8) corner coordinates
        max_points sampling points number
    output
        valid_indices (K',M) indices of points that are in each k' bounding box
        valid (K') index vector showing valid bounding boxes, i.e. with at least
                   one point within the box
    '''
    u =  corners[:,5,:] - corners[:,6,:]
    norm_u = np.linalg.norm(u,axis=1)
    u = u/norm_u[:,None]
    v = (corners[:,7,:] - corners[:,6,:])
    norm_v = np.linalg.norm(v,axis = 1)
    v = v/norm_v[:,None]
    w = (corners[:,2,:] - corners[:,6,:])
    norm_w = np.linalg.norm(w,axis=1)
    w = w/norm_w[:,None]
    valid = []
    valid_indices = np.zeros((corners.shape[0],max_points),dtype=int)
    for i in range(corners.shape[0]):
        directions = np.stack((u[i,:], v[i,:], w[i,:]), axis = 1)   
        center2point = np.subtract(xyz, (corners[i,6,:]+corners[i,0,:])/2,dtype=np.float32)
        projection = np.absolute(np.matmul(center2point,directions,dtype=np.float32) )
        xyz_indic = np.flatnonzero(points_in_box(projection, norm_u[i], norm_v[i], norm_w[i]))
    
        if 1 < len(xyz_indic) < max_points:
            idx = np.random.choice(xyz_indic, size=max_points, replace=True)
            valid_indices[i,:] = idx
            valid.append(i)
        elif len(xyz_indic) > max_points:
            idx = np.random.choice(xyz_indic, size=max_points, replace=False)
            valid_indices[i,:] = idx
            valid.append(i)
        else: # just one point in this box, discard
            continue
    return valid_indices[valid,:], valid


def enlargeBox(label, delta):
    label[:,(4,5)] += 2*delta
    label[:,3] += delta
    return label
    
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
    corners = label2corners(enlargeBox(pred.copy(),config['delta']))
  
    xmax = np.amax(corners[:,:,0])
    ymax = np.amax(corners[:,:,1])
    zmax = np.amax(corners[:,:,2])
    xmin = np.amin(corners[:,:,0])
    ymin = np.amin(corners[:,:,1])
    zmin = np.amin(corners[:,:,2])
    # Filter points that will not be in the biggest bbox encircling all bbox
    xyz_keep = np.flatnonzero((xyz[:,0] <= xmax) & (xyz[:,1] <= ymax) & (xyz[:,2] <= zmax) &
                               (xyz[:,0] >= xmin) & (xyz[:,1] >= ymin) & (xyz[:,2] >= zmin) )
    xyz = xyz[xyz_keep,:]
    feat = feat[xyz_keep,:]
    # get indices satisfying conditions for points in boxes
    valid_indices, valid = indexInBox(xyz, corners, config['max_points'])
    pooled_xyz = np.zeros((len(valid),config['max_points'],xyz.shape[1]))
    pooled_feat = np.zeros((len(valid),config['max_points'],128))
    #extract points and features
    pooled_xyz = xyz[valid_indices,:]
    pooled_feat = feat[valid_indices,:]
    valid_pred = pred[valid,:]
 
    return valid_pred, pooled_xyz, pooled_feat