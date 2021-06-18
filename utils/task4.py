import torch
import torch.nn as nn
import numpy as np

class RegressionLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.loss = nn.SmoothL1Loss()

    def forward(self, pred, target, iou):
        '''
        Task 4.a
        We do not want to define the regression loss over the entire input space.
        While negative samples are necessary for the classification network, we
        only want to train our regression head using positive samples. Use 3D
        IoU ≥ 0.55 to determine positive samples and alter the RegressionLoss
        module such that only positive samples contribute to the loss.
        input
            pred (N,7) predicted bounding boxes
            target (N,7) target bounding boxes
            iou (N,) initial IoU of all paired proposal-targets
        useful config hyperparameters
            self.config['positive_reg_lb'] lower bound for positive samples
        '''
        index = []
        print("pred_size: ", pred.shape)
        print("target_size: ", target.shape)
        print("iou_size: ", iou.shape)
        for i in range(pred.shape[0]):
            #positive samples
            if iou[i] < self.config['positive_reg_lb']:  #self.config['positive_reg_lb'] instead of 0.55
                index = np.append(index,i)
        print("index",index.shape)
        if index.size == 0:
            loss = 0
        else:
            index = [int(i) for i in index]     #convert indices to int (only type accepted in delete), but why is it not int at first?
            filtered_pred= np.delete(pred,index,axis=0)#filtered_pred.append(pred[i,:])
            filtered_target = np.delete(target,index,axis=0)#filtered_target.append(target[i,:]) 
            #contribution of size parameters (3 times h,w,l in the average)
            filtered_pred = torch.hstack((filtered_pred, filtered_pred[:,3:6], filtered_pred[:,3:6]))
            filtered_target = torch.hstack((filtered_target, filtered_target[:,3:6],filtered_target[:,3:6]))      
            print("filtered_pred shape", filtered_pred.shape)
            loss = self.loss(filtered_pred, filtered_target)
        return loss

class ClassificationLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.loss = nn.BCELoss()

    def forward(self, pred, iou):
        '''
        Task 4.b
        Extract the target scores depending on the IoU. For the training
        of the classification head we want to be more strict as we want to
        avoid incorrect training signals to supervise our network.  A proposal
        is considered as positive (class 1) if its maximum IoU with ground
        truth boxes is ≥ 0.6, and negative (class 0) if its maximum IoU ≤ 0.45.
            pred (N,7) predicted bounding boxes
            iou (N,) initial IoU of all paired proposal-targets
        useful config hyperparameters
            self.config['positive_cls_lb'] lower bound for positive samples
            self.config['negative_cls_ub'] upper bound for negative samples
        '''
        
        index_p = []
        index_n = []
        print("pred_size: ", pred.shape)
        print("target_size: ", target.shape)
        print("iou_size: ", iou.shape)
        for i in range(pred.shape[0]):
            #positive samples
            if iou[i] >= self.config['positive_cls_lb']:  
                index_p = np.append(index_p,i)
            elif iou[i] >= self.config['negative_cls_ub']:
                index_n = np.append(index_n,i)
        print("index",index_p.shape, index_n.shape)
        index_p = [int(i) for i in index_p]     #convert indices to int (only type accepted in delete), but why is it not int at first?
        index_n = [int(i) for i in index_n]
        filtered_pred_p= np.delete(pred,index_p,axis=0)
        filtered_target_p = np.delete(target,index_p,axis=0)

        filtered_pred_n= np.delete(pred,index_n,axis=0)
        filtered_target_n = np.delete(target,index_n,axis=0)

        #How does BCELoss work with positive and negative samples
        loss = self.loss(filtered_pred_p, filtered_target_p)
        return loss