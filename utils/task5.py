import numpy as np

from utils.task1 import get_iou


def nms(pred, score, threshold):
    '''
    Task 5
    Implement NMS to reduce the number of predictions per frame with a threshold
    of 0.1. The IoU should be calculated only on the BEV.
    input
        pred (N,7) 3D bounding box with (x,y,z,h,w,l,ry)
        score (N,) confidence scores
        threshold (float) upper bound threshold for NMS
    output
        s_f (M,7) 3D bounding boxes after NMS
        c_f (M,1) corresopnding confidence scores
    '''
    pred_bev = pred.copy() #copy of prediction for BEV
    pred_bev[:,1] =0 # set y to zero (all points projected to the ground)
    pred_bev[:,3] = 1 #set height to 1
    #Final Set
    s_f = np.empty((0,7))
    c_f = np.empty((0,1))

    while pred.shape[0] > 0:
        # find max confidence score
        idx_max_score = np.argmax(score)
        s_f = np.append(s_f, pred[idx_max_score,:].reshape(-1,7),axis=0)
        c_f = np.append(c_f,score[idx_max_score].reshape(-1,1),axis=0)
        # discard elements with IoU > threshold
        IoU = get_iou(pred_bev,pred_bev)
        flag = IoU[idx_max_score,:] < threshold
        indices = np.flatnonzero(flag)
        pred = pred[indices,:]
        pred_bev = pred_bev[indices,:]
        score = score[indices]
        
    return s_f, c_f
    