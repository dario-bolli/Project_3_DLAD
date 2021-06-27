import numpy as np

from .task1 import get_iou

def background_sampling(easy_background_index, hard_background_index, indices, start, stop):
    nb_fill = stop-start
    split1 = int(nb_fill/2)
    split2 = nb_fill -split1
    #only hard backgrounds
    if not easy_background_index:
        if len(hard_background_index)>=nb_fill:
            indices[start:stop] = np.random.choice(hard_background_index, size=nb_fill, replace=False)
        else:
            indices[start:stop] = np.random.choice(hard_background_index, size=nb_fill, replace=True)
    #only easy backgrounds
    elif not hard_background_index:
        if len(easy_background_index)>=nb_fill:
            indices[start:stop] = np.random.choice(easy_background_index, size=nb_fill, replace=False)
        else:
            indices[start:stop] = np.random.choice(easy_background_index, size=nb_fill, replace=True)
    #easy and hard backgrounds
    else:
        #50% easy 50% hard, attention odd numbers
        if len(hard_background_index)>=split1:
            indices[start:start+split1] = np.random.choice(hard_background_index, size=split1, replace=False)
        else:
            indices[start:start+split1] = np.random.choice(hard_background_index, size=split1, replace=True)
        if len(easy_background_index)>=split2:
            indices[start+split1:stop] = np.random.choice(easy_background_index, size=split2, replace=False)
        else:
            indices[start+split1:stop] = np.random.choice(easy_background_index, size=split2, replace=True)
    return indices[start:]

def sample_proposals(pred, target, xyz, feat, config, train=False):
    '''
    Task 3
    a. Using the highest IoU, assign each proposal a ground truth annotation. For each assignment also
       return the IoU as this will be required later on.
    b. Sample 64 proposals per scene. If the scene contains at least one foreground and one background
       proposal, of the 64 samples, at most 32 should be foreground proposals. Otherwise, all 64 samples
       can be either foreground or background. If there are less background proposals than 32, existing
       ones can be repeated.
       Furthermore, of the sampled background proposals, 50% should be easy samples and 50% should be
       hard samples when both exist within the scene (again, can be repeated to pad up to equal samples
       each). If only one difficulty class exists, all samples should be of that class.
    input
        pred (N,7) predicted bounding box labels
        target (M,7) ground truth bounding box labels
        xyz (N,512,3) pooled point cloud
        feat (N,512,C) pooled features
        config (dict) data config containing thresholds
        train (string) True if training
    output
        assigned_targets (64,7) target box for each prediction based on highest iou
        xyz (64,512,3) indices 
        feat (64,512,C) indices
        iou (64,) iou of each prediction and its assigned target box
    useful config hyperparameters
        config['t_bg_hard_lb'] threshold background lower bound for hard difficulty
        config['t_bg_up'] threshold background upper bound
        config['t_fg_lb'] threshold foreground lower bound
        config['num_fg_sample'] maximum allowed number of foreground samples
        config['bg_hard_ratio'] background hard difficulty ratio (#hard samples/ #background samples)
    '''
    ###Task a
    IoU = get_iou(pred, target)
    assigned_targets = np.zeros((pred.shape))
    assigned_IoU = np.zeros(pred.shape[0])
    for i in range(pred.shape[0]):
        ind = np.int(np.argmax(IoU[i,:]))
<<<<<<< HEAD
        assigned_targets[i,:] = target[ind,:]
=======
        assigned_targets[i,:] = target[ind]
>>>>>>> 8376a4dee244fc5c30cf59144522da220e6b1749
        assigned_IoU[i] = IoU[i,ind]

    ###Task b
    nb_fill = 64
<<<<<<< HEAD
    #pred_ind = np.arange(0,pred.shape[0]+1,1)
=======
    pred_ind = np.arange(0,pred.shape[0]+1,1)
>>>>>>> 8376a4dee244fc5c30cf59144522da220e6b1749

    indices = np.zeros(64, dtype = int)
    foreground_index = []
    easy_background_index = []
    hard_background_index = []
    #bounding box that has the highest IoU for a ground truth bounding box is considered an additional foreground sample
    for i in range(target.shape[0]):
        foreground_index.append(np.argmax(IoU[:,i]))
    
    #foreground and background regarding IoU
    for i in range(pred.shape[0]):
        #background
        if assigned_IoU[i] < 0.45:
            if assigned_IoU[i] < 0.05:
                easy_background_index.append(i)
            else:
                hard_background_index.append(i)
        #foreground
        elif assigned_IoU[i] >= 0.55:
            if i not in foreground_index:
                foreground_index.append(i)
    
    #no background proposals          (not == len(..)=0)
    if not easy_background_index and not hard_background_index:
        if len(foreground_index)>=nb_fill:
            indices = np.random.choice(foreground_index, size=nb_fill, replace=False)
        else:
            indices = np.random.choice(foreground_index, size=nb_fill, replace=True)
    #no foreground proposals            
    elif not foreground_index:
<<<<<<< HEAD
        indices = background_sampling(easy_background_index, hard_background_index, indices, 0, nb_fill)
=======
        indices = background_sampling(easy_background_index, hard_background_index, indices, 0, 64)
>>>>>>> 8376a4dee244fc5c30cf59144522da220e6b1749
    #foregrounds and backgrounds
    else:
        div = int(nb_fill/2)
        if len(foreground_index)>=div:
            indices[0:div] = np.random.choice(foreground_index, size=div, replace=False)
            indices[div:] = background_sampling(easy_background_index, hard_background_index, indices, div, nb_fill)
        else:
            split = len(foreground_index)
            indices[0:split] = np.random.choice(foreground_index, size=split, replace=False)
            indices[split:] = background_sampling(easy_background_index, hard_background_index, indices, split, nb_fill)
   
    xyz = np.take(xyz, indices, axis=0)
    assigned_targets = np.take(assigned_targets, indices, axis=0)
    feat = np.take(feat, indices, axis=0)
    assigned_IoU = np.take(assigned_IoU, indices, axis=0)

    return assigned_targets, xyz, feat, assigned_IoU
