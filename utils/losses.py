from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalization(data):

    for i in range(len(data)):

        _range = torch.max(data[i]) - torch.min(data[i])
        data[i] = (data[i] - torch.min(data[i])) / _range
    return data

class CapsuleLoss(nn.Module):

    def __init__(self, smooth=0.1, lamda=0.6):
        super(CapsuleLoss, self).__init__()
        self.smooth = smooth
        self.lamda = lamda

    def forward(self, input, target):
        one_hot = torch.zeros_like(input).to(input.device)
        one_hot = one_hot.scatter(1, target.unsqueeze(-1), 1)
        a = torch.max(torch.zeros_like(input).to(input.device), 1 - self.smooth - input)
        b = torch.max(torch.zeros_like(input).to(input.device), input - self.smooth)
        loss = one_hot * a * a + self.lamda * (1 - one_hot) * b * b
        loss = loss.sum(dim=1, keepdim=False)
        return loss.mean()

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, args, contrast_mode='one',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.args = args
        self.temperature = self.args.sup_temp
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 2:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 2:
            features = features.view(features.shape[0], features.shape[1], -1)

        # get batch_size
        batch_size = features.shape[0]

        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)     # 16*1
            if labels.shape[0] != batch_size:
                raise ValueError(
                    'Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)      # mask matrix: 16*16, torch.eq: Computes element-wise equality
        else:
            mask = mask.float().to(device)

        features = features.unsqueeze(dim=1) # (batch_size, 1, hidden_size)
        features = F.normalize(features, dim=2) 
        contrast_count = features.shape[1] # 1?
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) # (batch_size, hidden_size) contrast_feature is exactly the original features!

        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0] # original features
            anchor_count = 1 
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature) # (bs, bs)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits_min, _ = torch.min(logits, dim=1, keepdim=True)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        _range = logits_max - logits_min
        logits = torch.div(logits-logits_min, _range)

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # print("mask",mask)  # 16*16

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        ) # set diagonal-line elements to be zero
       
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1)+1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class SdConLoss(nn.Module):
    def __init__(self, args):
        super(SdConLoss, self).__init__()
        self.args = args
        self.temp = self.args.sd_temp
        self.ce_loss = nn.CrossEntropyLoss()
        self.sim = Similarity(temp=self.temp)

    def forward(self, feature_1, feature_2):
        cos_sim = self.sim(feature_1, feature_2) # (bs, hidden_size)
        tmp_labels = torch.arange(cos_sim.size(0)).long().to(self.args.device)
        loss = self.ce_loss(cos_sim, tmp_labels)
        return loss


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp
