import numpy as np
import random
import time
import numba as nb

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
    start = time.time()
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
    end = time.time()
    print("time in the label2corners function is " , end-start)
    return corners 


#@nb.njit(cache = True, nogil = True)
def indexInBox(directions, center2point, norm_u, norm_v, norm_w):
    #projection = np.absolute(np.matmul(center2point,directions))
        #print(projection.shape)
    
    projection = np.absolute(center2point@directions)   
    #start = time.time()
    cond1 = np.flatnonzero(projection[:,2] <= norm_w )
    end = time.time()
    #print("cond1 time", end -start)
    cond2 = np.flatnonzero(projection[cond1,0] <= norm_u )
    xyz_indic = np.flatnonzero(projection[cond2,1] <= norm_v )

    return xyz_indic

# @nb.njit
# def loopBoxes(corners,xyz, pred, feat, max_pointsROI):
#     pooled_xyz = []
#     valid_pred = []
#     pooled_feat = []
#     # u =  corners[:,5,:] - corners[:,6,:]
#     # norm_u = np.linalg.norm(u,axis=1)
#     # u = u/norm_u[:,None]
#     # v = (corners[:,7,:] - corners[:,6,:])
#     # norm_v = np.linalg.norm(v,axis = 1)
#     # v = v/norm_v[:,None]
#     # w = (corners[:,2,:] - corners[:,6,:])
#     # norm_w = np.linalg.norm(w,axis=1)
#     # w = w/norm_w[:,None]
   
#     for i in range(corners.shape[0]):
#         u = corners[i,5,:] - corners[i,6,:]
#         norm_u = np.linalg.norm(u)
#         u = u/norm_u
#         v = corners[i,7,:] - corners[i,6,:]
#         norm_v = np.linalg.norm(v)
#         v = v/norm_v
#         w = corners[i,2,:] - corners[i,6,:]
#         norm_w = np.linalg.norm(w)

#         center2point = xyz - corners[i,6,:]

#         directions = np.stack((u, v, w), axis = 1)
    
#         #start = time.time()
       
#         # projection = np.absolute(center2point@directions)   
#         # cond1 = np.flatnonzero(projection[:,0] <= norm_u )
#         # cond2 = np.flatnonzero(projection[cond1,1] <= norm_v )
#         # xyz_indic = np.flatnonzero(projection[cond2,2] <= norm_w )
#         xyz_indic = indexInBox(directions, center2point, norm_u, norm_v, norm_w)
#        # end = time.time()
        
#         # boxPoints = np.zeros((max_pointsROI,3))
#         # boxFeat= np.zeros((max_pointsROI,feat.shape[1]))
       
      
#         if 1 < len(xyz_indic) < max_pointsROI:
#             idx = np.random.choice(xyz_indic, size=max_pointsROI, replace=True)
#             boxPoints = xyz[idx,:]
#             boxFeat = feat[idx,:]
    

#         elif len(xyz_indic) > max_pointsROI:
#             idx = np.random.choice(xyz_indic, size=max_pointsROI, replace=False)
#             boxPoints = xyz[idx,:]
#             boxFeat = feat[idx,:]
        
#         else: # just one point in this box, discard
#             continue
#         pooled_xyz.append(boxPoints)
#         pooled_feat.append(boxFeat)
#         valid_pred.append(pred[i,:])
#     return pooled_xyz, pooled_feat, valid_pred


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
    #pooled_xyz = np.zeros((1,512,3))
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
    #pooled_xyz, pooled_feat, valid_pred = loopBoxes(corners,xyz, pred, feat, max_pointsROI)
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
    
    for i in range(corners.shape[0]):
        
        #center2point = xyz - box_center[i,:]
        center2point = np.array(xyz - corners[i,6,:])
        directions = np.stack((u[i,:], v[i,:], w[i,:]), axis = 1)
        #start = time.time()
        xyz_indic = indexInBox(directions, center2point, norm_u[i], norm_v[i], norm_w[i])
        #end = time.time()
        #print("Find points execution time for conditions is ", end - start)
        # boxPoints = np.zeros((max_pointsROI,3))
        # boxFeat= np.zeros((max_pointsROI,feat.shape[1]))
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
        #xyz = np.delete(xyz, obj = xyz_indic, axis = 0)
        
    pooled_xyz = np.array(pooled_xyz)
    pooled_feat = np.array(pooled_feat)
    valid_pred = np.array(valid_pred)
    #print("valid pred shape", valid_pred.shape)
    # print("pooled_xyz shape", pooled_xyz.shape)
    # print("pooled_feat shape", pooled_feat.shape)
    return valid_pred, pooled_xyz, pooled_feat