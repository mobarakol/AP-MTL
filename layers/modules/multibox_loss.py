# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
#from data import coco as cfg
#from data import *
from ..box_utils import match, log_sum_exp


class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,
                 use_gpu=True):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        #cfg = instruments
        self.variance = [0.1, 0.2]

    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            targets (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        #print('self.num_classes',self.num_classes)
        #print('predictions', predictions[0].size(),predictions[1].size(), predictions[2].size(), 'targets', np.array(targets).shape)
        loc_data, conf_data, priors = predictions
        #print('lol',loc_data)
        num = loc_data.size(0) #batch
        priors = priors[:loc_data.size(1), :]
        #print('priors',len(priors))
        num_priors = (priors.size(0)) #prior num  8732
        num_classes = self.num_classes
        #print('num_priors', num_priors)

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            # if targets[idx].nelement() == 0:
            #     print('target',targets[idx].size())
            #print('num', num, len(targets))
            #print('targets',targets[idx].size())
            truths = targets[idx][:, :-1].data
            #truths = targets[idx][:, :-1].data
            #print(truths)
            labels = targets[idx][:, -1].data
            #print('labels',labels)
            defaults = priors.data
            #print('mobarakkkkkkkkkkkkk ',truths, self.variance, labels)
            match(self.threshold, truths, defaults, self.variance, labels,loc_t, conf_t, idx)
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        # wrap targets
        #print('conf_t match', conf_t.size(), conf_t.data.max(), num)
        #print('loct', loc_t)
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)

        #print('loct', loc_t)

        pos = conf_t > 0
        num_pos = pos.sum(dim=1, keepdim=True)
        #print('mobarak pos', pos)
        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        #print('mobarak pos_idx', pos)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        #print('lol', loc_p, loc_t)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)
        #print('loloo', loss_l)

        # Compute max conf across batch for hard negative mining
        #print('mobarak conf_data',conf_data.size(),conf_data)
        batch_conf = conf_data.view(-1, self.num_classes)
        #print('mobarak batch_conf', batch_conf.size(), batch_conf.data.max())
        #print('mobarak conf_t', conf_t.size(), conf_t.data.max())
        #print('mobarak conf_t.view(-1, 1)', conf_t.view(-1, 1).size(), conf_t.view(-1, 1).data.max())
        batch_conf_gat = batch_conf.gather(1, conf_t.view(-1, 1))
        #print('mobarak batch_conf_gat', batch_conf_gat.size(), batch_conf_gat)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))
        #print('mobarak loss_cccccc', loss_c)
        # Hard Negative Mining
        loss_c = loss_c.view(pos.size()[0], pos.size()[1])  # add line
        loss_c[pos] = 0  # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos.size(1) - 1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos + neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

        N = num_pos.data.sum().double()
        loss_l = loss_l.double()
        loss_c = loss_c.double()
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c
