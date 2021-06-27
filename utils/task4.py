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
            pred (N,1) predicted bounding boxes
            target (N,7) target bounding boxes
            iou (N,) initial IoU of all paired proposal-targets
        useful config hyperparameters
            self.config['positive_reg_lb'] lower bound for positive samples
        '''
        if torch.cuda.is_available():
            device=torch.device("cuda:0")
        else:
            device=torch.device("cpu")
        index = []
        for i in range(pred.shape[0]):
            #positive samples
            if iou[i] >= self.config['positive_reg_lb']:  #self.config['positive_reg_lb'] instead of 0.55
                index = np.append(index,int(i))
<<<<<<< HEAD
        if len(index) == 0:
            loss = torch.zeros(1)
        else:
=======
        if index.size == 0:
            loss = 0
        else:
            #index = [int(i) for i in index]     #convert indices to int (only type accepted in delete), but why is it not int at first?
            
>>>>>>> 8376a4dee244fc5c30cf59144522da220e6b1749
            filtered_pred= pred[index].to(device) #np.delete(pred,index,axis=0)
            filtered_target = target[index].to(device) #np.delete(target,index,axis=0)    
            
            l_location = self.loss(filtered_pred[:,0:3], filtered_target[:,0:3])
            l_size = self.loss(filtered_pred[:,3:6], filtered_target[:,3:6])
            l_rotation = self.loss(filtered_pred[:,6], filtered_target[:,6])

            loss = l_location + 3*l_size + l_rotation
        return loss.to(device)

class ClassificationLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.loss = nn.BCELoss()
        self.m = nn.Sigmoid()

    def forward(self, pred, iou):
        '''
        Task 4.b
        Extract the target scores depending on the IoU. For the training
        of the classification head we want to be more strict as we want to
        avoid incorrect training signals to supervise our network.  A proposal
        is considered as positive (class 1) if its maximum IoU with ground
        truth boxes is ≥ 0.6, and negative (class 0) if its maximum IoU ≤ 0.45.
            pred (N,1) predicted bounding boxes
            iou (N,) initial IoU of all paired proposal-targets
        useful config hyperparameters
            self.config['positive_cls_lb'] lower bound for positive samples
            self.config['negative_cls_ub'] upper bound for negative samples
        '''
        if torch.cuda.is_available():
            device=torch.device("cuda:0")
        else:
            device=torch.device("cpu")

        index_p = []
        index_n = []

        for i in range(pred.shape[0]):
            #positive samples
            if iou[i] >= self.config['positive_cls_lb']:  
                index_p = np.append(index_p,int(i))
            if iou[i] <= self.config['negative_cls_ub']:
                index_n = np.append(index_n,int(i))

<<<<<<< HEAD
        if len(index_p) == 0 and len(index_n) == 0:
            loss = torch.zeros(1)
        else:
            filtered_pred_p= pred[index_p].to(device) 
            filtered_pred_n= pred[index_n].to(device) 

            label_n = np.zeros(filtered_pred_n.shape[0], dtype = np.float32).reshape(-1,1)
            label_p = np.ones(filtered_pred_p.shape[0], dtype = np.float32).reshape(-1,1)
            
            label = np.vstack((label_n,label_p))
            filtered_pred = torch.vstack((filtered_pred_n,filtered_pred_p)).to(device)
            
            loss = self.loss(filtered_pred, torch.from_numpy(label).to(device))
=======
        #index_p = [int(i) for i in index_p]     #convert indices to int (only type accepted in delete)
        #index_n = [int(i) for i in index_n]

        filtered_pred_p= pred[index_p].to(device) #np.delete(pred,index_p,axis=0)
        filtered_pred_n= pred[index_n].to(device) #np.delete(pred,index_n,axis=0)

        label_n = np.zeros(filtered_pred_n.shape[0], dtype = np.float32).reshape(-1,1)
        label_p = np.ones(filtered_pred_p.shape[0], dtype = np.float32).reshape(-1,1)
        
        label = np.vstack((label_n,label_p))
        filtered_pred = torch.vstack((filtered_pred_n,filtered_pred_p)).to(device)
        
        loss = self.loss(filtered_pred, torch.from_numpy(label).to(device))
>>>>>>> 8376a4dee244fc5c30cf59144522da220e6b1749
        return loss.to(device)