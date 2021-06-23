import numpy as np
import shapely
from shapely.geometry import box, Polygon


def y_rotation(theta):
    """
    Rotation about the y-axis. 
    (y in cam0 coordinates)
    """
    c = np.cos(theta)
    s = np.sin(theta) 

    Rot = np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])              
    return Rot

def label2corners(label):
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

    for i in range(label.shape[0]):
        # extract the dimensions of the box
        height = label[i,3]
        width = label[i,4]
        length = label[i,5]
        
        # Corners location 3D in camera0 frame
        corners_x = [length/2,-length/2, -length/2, length/2, length/2, -length/2, -length/2, length/2]
        corners_y = [height, height, height, height, 0, 0, 0, 0]
        corners_z = [width/2, width/2, -width/2, -width/2, width/2, width/2, -width/2, -width/2]
        
        #Rotation around y-axis
        Rot = y_rotation(label[i,6])
        box_dim = Rot@np.vstack([corners_x, corners_y, corners_z])
        # center the box around (x,y,z)
        box_dim[0,:] += label[i,0]
        box_dim[1,:] += label[i,1]
        box_dim[2,:] += label[i,2]
        # Dimensions of box_dim: 3 x 8 i.e. rows are (x,y,z) and columns are the corners
        corners[i,:,:] = np.transpose(box_dim)   
    return corners

def get_volume(box):
    #box dimension (N,7)
    corners = label2corners(box)
    volume = np.zeros(box.shape[0])
    for i in range(box.shape[0]):
        u = corners[i,0,:] - corners[i,1,:]
        v = corners[i,0,:] - corners[i,3,:]
        w = corners[i,0,:] - corners[i,4,:]
        volume[i] = np.linalg.norm(u)*np.linalg.norm(v)*np.linalg.norm(w)
    return volume

def get_intersection_area(corners_pred, corners_target):
  
    box_p = shapely.geometry.Polygon([tuple(corners_pred[0,(0,2)]), tuple(corners_pred[1,(0,2)]), tuple(corners_pred[2,(0,2)]), tuple(corners_pred[3,(0,2)])])
    box_t = shapely.geometry.Polygon([tuple(corners_target[0,(0,2)]), tuple(corners_target[1,(0,2)]), tuple(corners_target[2,(0,2)]), tuple(corners_target[3,(0,2)])])
    if box_p.intersects(box_t):
        inter_area = box_p.intersection(box_t).area
    else:
        inter_area = 0
    return inter_area

def get_iou(pred, target):
    '''
    Task 1
    input
        pred (N,7) 3D bounding box corners
        target (M,7) 3D bounding box corners
    output
        iou (N,M) pairwise 3D intersection-over-union
    '''
    corners_pred = label2corners(pred)
    corners_target = label2corners(target)
    
    Vol_pred = get_volume(pred)
    Vol_target = get_volume(target)
    
    iou = np.zeros((pred.shape[0],target.shape[0]))
    for i in range(pred.shape[0]):
        for j in range(target.shape[0]):
            #Height of the intersection boxe
            max_pred = max(corners_pred[i,0,1], corners_pred[i,6,1])
            max_target = max(corners_target[j,0,1], corners_target[j,6,1])
            min_pred = min(corners_pred[i,0,1], corners_pred[i,6,1])
            min_target = min(corners_target[j,0,1], corners_target[j,6,1])
            y1 = max(min_pred, min_target)
            y2 = min(max_pred, max_target)
        
            interArea = get_intersection_area(corners_pred[i, 0:4,:], corners_target[j, 0:4,:])
            interVolume = interArea*max(0, y2 - y1)
            # compute the volume of union
            unionVolume = Vol_pred[i] + Vol_target[j] - interVolume

            #intersection over union
            iou[i,j] = interVolume/unionVolume
            
    return iou


def compute_recall(pred, target, threshold):
    '''
    Task 1
    input
        pred (N,7) proposed 3D bounding box labels
        target (M,7) ground truth 3D bounding box labels
        threshold (float) threshold for positive samples
    output
        recall (float) recall for the scene
    '''
    
    IoU = get_iou(pred, target)
    TP = np.zeros(1)
    FN = np.zeros(1)
    assigned_IoU = np.zeros(target.shape[0])
    for j in range(target.shape[0]):
        ind = np.argmax(IoU[:,j])
        assigned_IoU[j] = IoU[ind,j]
        if assigned_IoU[j] >= threshold:   
            TP += 1
        elif all(i<threshold for i in IoU[:,j]):
            FN += 1    
    recall = TP/(TP + FN)
    return recall
