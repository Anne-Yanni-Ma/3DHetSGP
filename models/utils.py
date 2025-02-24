import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable


class MLP(nn.Module):
    def __init__(self, mlp=[1024, 512, 512], dropout=False, log_sm=False, bn=False, ln=False, bias=True):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(mlp)-1):
            layers.append(nn.Linear(mlp[i], mlp[i+1], bias=bias))
            if bn and i != len(mlp) - 2:
                layers.append(nn.BatchNorm1d(mlp[i+1]))
            if ln and i != len(mlp) - 2:
                layers.append(nn.LayerNorm(mlp[i+1]))
            if i != len(mlp) - 2:
                layers.append(nn.ReLU())
            if dropout and i == 0:
                layers.append(nn.Dropout(0.2))
            if i == len(mlp)-2 and log_sm:
                layers.append(nn.LogSoftmax(dim=1))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        y = self.layers(x)
        return y


class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True, use_softmax=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num).cuda())
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
        self.use_softmax = use_softmax

    def forward(self, inputs, class_mask):
        ''' inputs: output from a linear layer
            class_mask: onehot matrix
        '''
        if self.use_softmax:
            P = F.softmax(inputs, dim=1)
        else:
            P = inputs
        class_mask = Variable(class_mask)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = (self.alpha * class_mask).sum(1).view(-1, 1)

        probs = (P*class_mask).sum(1).view(-1, 1)

        log_p = probs.log()

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

    def focal_loss(self, gamma, at, logpt, label):
        label = label.view(-1, 1).contiguous()
        logpt = logpt.gather(1, label)
        at = Variable(at.gather(0, label.data.view(-1))).cuda()
        pt = Variable(logpt.data.exp()).cuda()
        loss = torch.mean(-1 * (1 - pt) ** gamma * logpt)
        return loss
