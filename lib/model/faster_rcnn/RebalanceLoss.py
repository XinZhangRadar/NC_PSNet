import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class PerspectiveRebalance(nn.Module):

    def __init__(self, classNum: int , gamma: float = 0.5, pTildeResoures: str = None, temperature: int = 10, lambda_: float = 0.5, loss_type : int = 3 ):
        super(PerspectiveRebalance,self).__init__()
        #pdb.set_trace()

        if pTildeResoures is None:
            pTildeResoures = '/home/zhangxin/View-sensitive-conv/lib/datasets/pTilde_32.npy'
        self.pTilde = torch.from_numpy(np.load(pTildeResoures))
        self.gamma = gamma
        # self.pTilde = torch.randn(classNum)
        self.pTilde = nn.Parameter(self.pTilde, requires_grad=False)
        self.weight = nn.Parameter(self.weightInit(), requires_grad=False)
        # self.oneHotTaget = nn.Parameter(torch.zeros(classNum), requires_grad=False)
        self.T = temperature
        self.lambda_ = lambda_
        self.loss_type = loss_type
        

    def forward(self, scores, soft_target, hard_target):
        ''' INPUTS
                scores        [N, V]    view score
                soft_target   [N, V]    ground truth with Gaussian smoothed 
                hard_target   [N,  ]    view label (0~view_num-1)
                loss_type:    int       1 for hard loss; 2 for soft loss; 3 for distilling_loss
            OUTPUTS
                Loss          tesor     Rebalanced Loss
            FUNCTIONALITY 
        '''
        #pdb.set_trace()
        if self.loss_type == 1:  #soft loss
            weight = self.assignWeight(soft_target,False)
            RebalanceLoss = self.multinomialCrossEntropy(scores, soft_target, weight)
        elif self.loss_type == 2: #hard loss
            weight = self.assignWeight(soft_target,True)
            RebalanceLoss = self.multinomialCrossEntropyHardTarget(scores, hard_target, weight)

        elif self.loss_type == 3: #distilling_loss
            weight_soft  = self.assignWeight(soft_target,False)
            weight_hard = self.assignWeight(soft_target,True)
            RebalanceLoss = self.Distilling_loss(scores, soft_target, hard_target, weight_soft, weight_hard)

        return RebalanceLoss

        # RebalanceLoss1 = self.multinomialCrossEntropy(scores, target, weight)
        # RebalanceLoss2 = self.KLSimilarReplacement(scores, target, weight)
        # RebalanceLoss2 = self.KLSimilarReplacement(scores, target, weight)
        # import pdb; pdb.set_trace()
        # return RebalanceLoss1, RebalanceLoss2
    def Distilling_loss(self, scores, soft_target, hard_target, weight_soft, weight_hard):
        ''' INPUTS
                scores        [N, V]    view score
                soft_target   [N, V]    ground truth with Gaussian smoothed 
                hard_target   [N,  ]    view label (0~view_num-1)
                weight_soft   [N, 1]    soft loss weight used for class balance
                weight_hard   [1, V]    hard loss weight used for class balance
            OUTPUTS
                Loss          tesor     Distilling Loss
            FUNCTIONALITY 
        '''
        #pdb.set_trace()

        hard_loss = self.multinomialCrossEntropyHardTarget(scores, hard_target, weight_hard)

        soft_loss = self.multinomialCrossEntropy(scores/self.T,soft_target,weight_soft)

        loss = self.lambda_ * soft_loss + (1-self.lambda_ ) * hard_loss

        return loss

    def multinomialCrossEntropy(self, scores, target, weight):
        ''' INPUTS
                scores        [N, V]    view score
                target        [N, V]    ground truth with Gaussian smoothed 
                weight        [N, 1]    soft loss weight used for class balance
            OUTPUTS
                Loss          tesor     soft Loss
            FUNCTIONALITY 
        '''
        logProbs = F.log_softmax(scores, dim=1)
        lossReducedClass = torch.sum(target*logProbs, dim=1)
        lossWeighted = weight.view(-1)*lossReducedClass
        loss = -lossWeighted.mean()
        return loss
    
    def multinomialCrossEntropyHardTarget(self, scores, target, weight):
        ''' INPUTS
                scores        [N, V]    view score
                target        [N,  ]    view label (0~view_num-1)
                weight        [1, V]    hard loss weight used for class balance
            OUTPUTS
                Loss          tesor     hard Loss
            FUNCTIONALITY 
        '''


        #lossUnWeighted = F.cross_entropy(scores, target,reduction='none')
        loss = F.cross_entropy(scores, target,weight=weight,reduction='none')
        loss = loss.mean()
        return loss

    def KLSimilarReplacement(self, scores, target, weight):
        #pdb.set_trace()
        logProbs = F.log_softmax(scores, dim=1)
        lossReducedClass = F.kl_div(logProbs, target, reduction='none').sum(dim=1)
        lossWeighted = weight.view(-1)*lossReducedClass
        backward_num  = torch.nonzero(lossWeighted).size(0)
        loss = lossWeighted.sum()/backward_num
        return loss,lossReducedClass

    def KLSimilarReplacementV2(self, scores, target, weight):
        logProbs = F.log_softmax(scores, dim=1)
        lossReducedClass = (target*(torch.log(target)-logProbs)).sum(dim=1)
        lossWeighted = weight*lossReducedClass
        loss = lossWeighted.sum()
        return loss

    def weightInit(self):
        self.alpha = 1 # 0/1 whether ignore prior or not.

        # define uniform probability
        self.uni_probs = torch.zeros_like(self.pTilde)
        self.uni_probs[self.pTilde!=0] = 1.
        self.uni_probs = self.uni_probs/torch.sum(self.uni_probs)

        # convex combination of empirical prior and uniform distribution       
        self.priorMix = (1-self.gamma)*self.pTilde + self.gamma*self.uni_probs

        # set prior factor
        weight = self.priorMix**-self.alpha
        weight = weight/torch.sum(self.pTilde*weight) # re-normalize
        return weight.float()
    
    def assignWeight(self, target, hardTarget):
        #pdb.set_trace()

        if not hardTarget:
            weightIndex = torch.argmax(target, dim=1, keepdim=True)
            weight = self.weight[weightIndex]
        else:
            weight = self.weight.data.clone().unsqueeze(0)     
        return weight

if __name__ == "__main__":
    classNum = int(10)
    devices = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    testData = torch.randn((2,classNum), requires_grad=True).to(devices)
    TestTarget = torch.randn((2,classNum), requires_grad =False).to(devices)
    TestTarget = torch.abs(TestTarget)
    # TestTarget = torch.zeros((2,classNum,3,3)).scatter_(1, y, 1)
    TestTarget4CE = torch.argmax(TestTarget,dim=1)
    prLoss = PerspectiveRebalance(classNum, 0.5, '').to(devices)
    hardTarget = True
    loss  = prLoss(testData, TestTarget4CE, torch.randint(0,classNum,(3,)).long(), hardTarget)
    loss.backward()
    print('loss:{}'.format(loss))


