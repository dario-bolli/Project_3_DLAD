import torch
import numpy as np
from utils.task1 import label2corners

def points_semseg_in_box(boxes, xyz):
    '''
    input:
    - target or prediction boxes
    - point cloud
    output:
    -point cloud label (1 if points inside boxes, 0 otherwise)
    '''
    corners = label2corners(boxes)
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
    xyz_label = np.zeros(xyz.shape[0])
    for i in range(corners.shape[0]):
        directions = np.stack((u[i,:], v[i,:], w[i,:]), axis = 1)   
        center2point = np.subtract(xyz, (corners[i,6,:]+corners[i,0,:])/2,dtype=np.float32)
        projection = np.absolute(np.matmul(center2point,directions,dtype=np.float32) )
        xyz_indic = np.flatnonzero(points_in_box(projection, norm_u[i], norm_v[i], norm_w[i]))
        xyz_label[xyz_indic] = 1
    return xyz_label


def focal_loss(target_boxes, pred_boxes, xyz):
    '''
    input:
    - point cloud ground truth label
    - point cloud pred label
    output:
    - focal loss
    '''
    xyz_gt_labels = torch.from_numpy(points_semseg_in_box(target_boxes, xyz))
    #As only 2 classes, xyz_gt_labels is already one_hot encoded
    xyz_pred_labels = torch.from_numpy(points_semseg_in_box(pred_boxes, xyz))

    alpha_t = 0.25
    gamma = 2

    pred_prob = torch.sigmoid(xyz_pred_labels)
    p_t = (xyz_gt_labels *pred_prob)+(1-xyz_gt_labels )*(1-pred_prob)
    
    L_f = -alpha_t*(1-p_t)**gamma*torch.log(p_t)

'''
add this loss to regression loss and classification loss in the training_step of train.py
'''