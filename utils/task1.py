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
    #print("rot:",Rot)             
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
        volume[i] = np.linalg.norm(u,2)*np.linalg.norm(v,2)*np.linalg.norm(w,2)
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
            #Intersection
            '''
            x1 = max(corners_pred[i,min_corner_x,0], corners_target[j,min_target_x,0]) #corner 6, x-coordinate
            x2 = min(corners_pred[i,max_corner_x,0], corners_target[j,max_target_x,0])
            y1 = max(corners_pred[i,min_corner_y,1], corners_target[j,min_target_y,1]) #corner 7, y-coordinate
            y2 = min(corners_pred[i,max_corner_y,1], corners_target[j,max_target_y,1])
            '''
            z1 = max(corners_pred[i,4,2], corners_target[j,4,2]) #corner 4, z-coordinate
            z2 = min(corners_pred[i,2,2], corners_target[j,2,2])
            # compute the volume of intersection
            #interVolume = max(0, x2 - x1 + 1)*max(0, y2 - y1 + 1)*max(0, z2 - z1 + 1)
            interArea = get_intersection_area(corners_pred[i, 0:4,0:2], corners_target[j, 0:4,0:2])
            interVolume = interArea*max(0, z2 - z1)
            # compute the volume of union
            unionVolume = Vol_pred[i] + Vol_target[j] - interVolume
            
            #intersection over union
            iou[i,j] = interVolume/unionVolume

            '''
            if i == pred.shape[0]-1:
                print("corners_pred[i, 0:4,0]", corners_pred[i, 0:4,0])
                print("corners_pred[i, 0:4,1]", corners_pred[i, 0:4,1])
                print("yaw rotation", pred[i,6])
                print("corners_target[i, 0:4,0]", corners_target[j, 0:4,0])
                print("corners_target[i, 0:4,1]", corners_target[j, 0:4,1])
                print("interArea", interArea)
                '''
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
    #assigned_targets, assigned_IoU = assigned_target(pred,target)

    TP = np.zeros(1)
    #TN = np.zeros(1)
    #FP = np.zeros(1)
    FN = np.zeros(1)
    assigned_IoU = np.zeros(target.shape[0])
    for j in range(target.shape[0]):
        ind = np.argmax(IoU[:,j])
        assigned_IoU[j] = IoU[ind,j]
        if assigned_IoU[j] >= threshold:
            TP += 1
        elif all(i<threshold for i in IoU[:,j]):
            FN += 1
        else:
            print("not FN not TP")
        print("FN, TP", FN, TP)
        
    recall = TP/(TP + FN)
    return recall
