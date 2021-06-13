import numpy as np
import shapely
from shapely.geometry import box, Polygon


def z_rotation(theta):
    """
    Rotation around the -z-axis. 
    """
    c = np.cos(-theta)
    s = np.sin(-theta) 

    Rot = np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])          
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
        #center coordinates
        x =  label[i,0]
        y =  label[i,1]
        z =  label[i,2]

        # Corners location 3D in velodyne frame
        '''
        corners_x = [width/2,- width/2, -width/2, width/2, width/2, -width/2, -width/2, width/2]
        corners_y = [length/2, length/2, -length/2, -length/2, length/2, length/2, -length/2, -length/2]
        corners_z = [height, height, height, height, 0, 0, 0, 0]

        corners_x = [length/2, -length/2, -length/2, length/2, length/2, -length/2, -length/2, length/2]
        corners_y = [width/2, width/2, -width/2, -width/2, width/2, width/2, -width/2, -width/2]
        corners_z = [height, height, height, height, 0, 0, 0, 0]
        '''
        corners_x = [height/2, -height/2, -height/2, height/2, height/2, -height/2, -height/2, height/2]
        corners_y = [width/2, width/2, -width/2, -width/2, width/2, width/2, -width/2, -width/2]
        corners_z = [length, length, length, length, 0, 0, 0, 0]
        #Rotation around z-axis
        Rot = z_rotation(label[i,6])

        box_dim = Rot@np.vstack([corners_x, corners_y, corners_z])
        box_dim[0,:] += x
        box_dim[1,:] += y
        box_dim[2,:] += z
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
        #volume[i] = (np.linalg.norm(u,2)+1)*(np.linalg.norm(v,2)+1)*(np.linalg.norm(w,2)+1)
        volume[i] = np.linalg.norm(u)*np.linalg.norm(v)*np.linalg.norm(w)
    return volume

def get_intersection_area(corners_pred, corners_target):
    '''
    #prediction's box
    max_corner_p_x = np.argmax(corners_pred[:,0])
    min_corner_p_x = np.argmin(corners_pred[:,0])
    max_corner_p_y = np.argmax(corners_pred[:,1])
    min_corner_p_y = np.argmin(corners_pred[:,1])
    #target's box
    max_corner_t_x = np.argmax(corners_target[:,0])
    min_corner_t_x = np.argmin(corners_target[:,0])
    max_corner_t_y = np.argmax(corners_target[:,1])
    min_corner_t_y = np.argmin(corners_target[:,1])

    box_p = shapely.geometry.box(min_corner_p_x, min_corner_p_y, max_corner_p_x, max_corner_p_y)
    box_t = shapely.geometry.box(min_corner_t_x, min_corner_t_y, max_corner_t_x, max_corner_t_y)
    '''
    
    box_p = shapely.geometry.Polygon([tuple(corners_pred[0,0:2]), tuple(corners_pred[1,0:2]), tuple(corners_pred[2,:]), tuple(corners_pred[3,:])])
    box_t = shapely.geometry.Polygon([tuple(corners_target[0,:]), tuple(corners_target[1,:]), tuple(corners_target[2,:]), tuple(corners_target[3,:])])
    if box_p.intersects(box_t):
        inter_area = box_p.intersection(box_t).area
    else:
        inter_area = 0
    return inter_area

def get_iou(pred, target, num):
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
            max_pred = max(corners_pred[i,4,2], corners_pred[i,2,2])
            max_target = max(corners_target[j,4,2], corners_target[j,2,2])
            min_pred = min(corners_pred[i,4,2], corners_pred[i,2,2])
            min_target = min(corners_target[j,4,2], corners_target[j,2,2])
            z1 = max(min_pred, min_target)
            z2 = min(max_pred, max_target)
            #print("z1", corners_pred[i,2,2], corners_target[j,2,2]) #should be 0
            # compute the volume of intersection
            #interVolume = max(0, x2 - x1 + 1)*max(0, y2 - y1 + 1)*max(0, z2 - z1 + 1)
            interArea = np.float16(get_intersection_area(corners_pred[i, 0:4,0:2], corners_target[j, 0:4,0:2]))
            interVolume = np.float16(interArea*max(0, z2 - z1))
            # compute the volume of union
            unionVolume = np.float16(Vol_pred[i] + Vol_target[j] - interVolume)

            #intersection over union
            iou[i,j] = interVolume/unionVolume
            
    return iou



def compute_recall(pred, target, threshold, num):
    '''
    Task 1
    input
        pred (N,7) proposed 3D bounding box labels
        target (M,7) ground truth 3D bounding box labels
        threshold (float) threshold for positive samples
    output
        recall (float) recall for the scene
    '''
    
    IoU = get_iou(pred, target, num)
    #assigned_targets, assigned_IoU = assigned_target(pred,target)

    TP = np.zeros(1)
    FN = np.zeros(1)
    assigned_IoU = np.zeros(target.shape[0])
    for j in range(target.shape[0]):
        ind = np.argmax(IoU[:,j])
        assigned_IoU[j] = IoU[ind,j]
        if assigned_IoU[j] >= threshold:    #-0.0001
            TP += 1
        elif all(i<threshold for i in IoU[:,j]):
            FN += 1
        if num == 88:
            print("FN, TP", FN, TP)
            print("assigned_IoU[j]", assigned_IoU[j])
        
    recall = TP/(TP + FN)
    return recall
