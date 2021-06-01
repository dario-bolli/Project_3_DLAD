import numpy as np
import random
import time
#import numba as nb

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
    print("N is equal", pred.shape[0])
    corners = label2corners(pred, config)
    print("corners shape", corners.shape[0])
    """
    u =  corners[:,5,:] - corners[:,6,:]
    norm_u = np.linalg.norm(u)
    u = u/norm_u
    v = (corners[:,7,:] - corners[:,6,:])
    norm_v = np.linalg.norm(v)
    v = v/norm_v
    w = (corners[:,2,:] - corners[:,6,:])
    norm_w = np.linalg.norm(w)
    w = w/norm_w
    """
    
    for i in range(corners.shape[0]):
        
        u = np.array(corners[i,5,:] - corners[i,6,:])
        norm_u = np.linalg.norm(u)
        u = u/norm_u
        v = np.array(corners[i,7,:] - corners[i,6,:])
        norm_v = np.linalg.norm(v)
        v = v/norm_v
        w = np.array(corners[i,2,:] - corners[i,6,:])
        norm_w = np.linalg.norm(w)
        w = w/norm_w
        #box_center = (corners[6,:] + corners[0,:])/2.0
        
        #center2point = xyz - box_center
        center2point = np.array(xyz - corners[i,6,:])
        
        start = time.time()
        cond1 = set(np.where((np.absolute(np.dot(center2point, u))) <= norm_u )[0])
        cond2 = set(np.where((np.absolute(np.dot(center2point, v))) <= norm_v )[0])
        cond3 = set(np.where((np.absolute(np.dot(center2point, w))) <= norm_w )[0])
        xyz_indic = list(cond1.intersection( cond2, cond3 ))

        end = time.time()
        #print("Find points execution time for conditions is ", end - start)
        boxPoints = np.zeros((max_pointsROI,3))
        boxFeat= np.zeros((max_pointsROI,feat.shape[1]))
       
        
        """
        #xyz_indic = []
        #print(center2point.shape)
        #print(np.dot(center2point, u).shape)
        start = time.time()
        xyz_indic = []
        for k in range(xyz.shape[0]):
            if abs(np.dot(center2point[k,:], u)) <= norm_u and abs(np.dot(center2point[k,:], v)) and abs(np.dot(center2point[k,:], w)) <= norm_w: 
                #xyz_indic.append(k)
                xyz_indic.append(k)
        end = time.time()
        print("Find points execution time is ", end - start)
        
        """
        boxPoints = xyz[xyz_indic,:]
        boxFeat= feat[xyz_indic, :]
        
        # print(len(xyz_indic))
        if 1 < len(xyz_indic) < max_pointsROI:
            nb_fill = max_pointsROI-len(xyz_indic)
            fill_idx = np.random.randint(len(xyz_indic), size= nb_fill) # should we do uniform distribution ?
            fill_points = boxPoints[fill_idx,:]
            fill_feat = boxFeat[fill_idx,:]
            boxPoints = np.vstack((boxPoints, fill_points))
            boxFeat= np.vstack((boxFeat, fill_feat))

        elif len(xyz_indic) > max_pointsROI:
            nb_discard = len(xyz_indic) - max_pointsROI
            delete_idx = random.sample(range(0,len(xyz_indic)), nb_discard)
            boxPoints = np.delete(boxPoints, obj = delete_idx, axis=0)
            boxFeat = np.delete(boxFeat, obj = delete_idx, axis=0)
        
        else: # just one point in this box, discard
            continue
        pooled_xyz.append(boxPoints)
        pooled_feat.append(boxFeat)
        valid_pred.append(pred[i,:])
    
    pooled_xyz = np.array(pooled_xyz)
    pooled_feat = np.array(pooled_feat)
    valid_pred = np.array(valid_pred)
    # print("valid pred shape", valid_pred.shape)
    # print("pooled_xyz shape", pooled_xyz.shape)
    # print("pooled_feat shape", pooled_feat.shape)
    
   
    return valid_pred, pooled_xyz, pooled_feat


# if __name__ == '__main__':
#     pooled_xyz = []
#     valid_pred = []
#     boxPoints = np.array([[1,2,3], [4,5,6], [7,8,9]])
#     for i in range(5):
#         pooled_xyz.append(boxPoints)
#         valid_pred.append(np.array([1,2,3,4,5,6,7]))
#     pooled_xyz = np.array(pooled_xyz)
#     valid_pred = np.array(valid_pred)
#     print(pooled_xyz.shape)
#     print(valid_pred.shape)
#     #print(random.sample(boxPoints,2))
#     #print(np.random.choice(boxPoints,10))
#     idx = np.random.randint(3, size = 2)
#     nex_arr = boxPoints[idx,:]
#     boxPoints = np.vstack((boxPoints, nex_arr))
#     print(boxPoints)
#     delete = random.sample(range(0,4), 2)
#     print(delete)
#     boxPoints = np.delete(boxPoints, obj = delete, axis=0)
#     print(boxPoints)