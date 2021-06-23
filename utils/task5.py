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
    pred_bev = pred.copy()
    pred_bev[:,1] =0 # set height to zero
    pred_bev[:,3] = 1
    s_f = []
    c_f = []
    #print(pred_bev)
    # threshold = threshold + 0.0926
    # print(threshold)
    while pred.shape[0] > 0:
        print("pred shape",pred.shape[0])
        idx_max_score = np.argmax(score)
        s_f.append(pred[idx_max_score,:])
        c_f.append(score[idx_max_score])
        IoU = get_iou(pred_bev,pred_bev)
        flag = IoU[idx_max_score,:] < threshold
        #print(np.min(IoU))
        indices = np.flatnonzero(flag)
        print(indices)
        pred = pred[indices,:]
        pred_bev = pred_bev[indices,:]
        score = score[indices]
    s_f = np.array(s_f)
    c_f = np.array(c_f)
    print("real",s_f)
    return s_f, c_f
    