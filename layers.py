import numpy as np

import mindspore.nn as nn
import mindspore.ops.operations as P
import mindspore.common.dtype as mstype

class AsymmetricLoss(nn.Cell):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_mindspore_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_mindspore_grad_focal_loss = disable_mindspore_grad_focal_loss
        self.eps = eps

        self.sigmoid = P.Sigmoid()
        self.log = P.Log()
        self.reduce_sum = P.ReduceSum()
        self.clamp = P.Clamp()
        self.pow = P.Pow()

    def construct(self, x, y):
        # Calculating Probabilities
        x_sigmoid = self.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = self.clamp(xs_neg + self.clip, 0, 1)

        # Basic CE calculation
        los_pos = y * self.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * self.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_mindspore_grad_focal_loss:
                self.set_train(mode=False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = self.pow(1 - pt, one_sided_gamma)
            if self.disable_mindspore_grad_focal_loss:
                self.set_train(mode=True)
            loss *= one_sided_w

        return -self.reduce_sum(loss), -self.reduce_sum(loss)


class SoftMaxLoss(nn.Cell):
    def __init__(self):
        super(SoftMaxLoss, self).__init__()
        self.classify_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        self.log_softmax = nn.LogSoftmax(axis=1)

    def construct(self, output, labels, train=True):
        labels1 = labels.argmax(axis=1)
        out = self.log_softmax(output)
        cls = self.classify_loss(out, labels1)
        return cls, cls


class CEL(nn.Cell):
    def __init__(self):
        super(CEL, self).__init__()
        self.classify_loss = nn.BCELoss()
        self.sigmoid = nn.Sigmoid()

        # Prevent to overflow
        self.epsilon = 1e-32

    def construct(self, output, labels, train=True):
        cel_sum = 0
        batch_size = labels.shape[0]
        num_category = labels.shape[1]

        outs_sig = self.sigmoid(output[:, :num_category]).view([batch_size, num_category])
        reshape_out = outs_sig.view(batch_size * num_category, 1)
        reshape_lab = labels.view(batch_size * num_category, 1)

        pos_index = reshape_lab
        neg_index = 1 - reshape_lab

        pos_loss = pos_index * self.log(reshape_out + self.epsilon)
        neg_loss = neg_index * self.log(1 - reshape_out + self.epsilon)
        cel_sum = - pos_loss.mean() - neg_loss.mean()

        return cel_sum, cel_sum


class Loss(nn.Cell):
    def __init__(self, num_hard=0):
        super(Loss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.classify_loss = nn.BCELoss()
        self.regress_loss = nn.SmoothL1Loss()
        self.num_hard = num_hard

    def construct(self, output, labels, train=True):
        batch_size = labels.shape[0]
        outs = self.sigmoid(output[:, :1])
        cls = self.classify_loss(outs, labels)

        return cls, cls


import mindspore.nn as nn
import mindspore.ops.operations as P

class AUCPLoss(nn.Cell):
    def __init__(self, num_hard=0):
        super(AUCPLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.classify_loss = nn.BCELoss()
        self.regress_loss = nn.SmoothL1Loss()
        self.num_hard = num_hard
        self.alpha = 0.1

    def construct(self, output, labels, train=True):
        batch_size = labels.shape[0]
        outs = self.sigmoid(output[:, :1])

        out_pos = outs[labels == 1]
        out_neg = outs[labels == 0]
        penalty_term_sum = 0

        try:
            num_pos = out_pos.shape[0]
            num_neg = out_neg.shape[0]
            if num_pos == 0:
                trans_pos = 0
                trans_neg = out_neg
                penalty_term = P.ReduceMean()(P.Pow()(1 - (trans_pos - trans_neg), 2)) / 2
                print("pos")
            elif num_neg == 0:
                trans_pos = out_pos
                trans_neg = 0
                penalty_term = P.ReduceMean()(P.Pow()(1 - (trans_pos - trans_neg), 2)) / 2
                print("neg")
            else:
                trans_pos = P.Tile()(out_pos, (num_neg, 1))
                trans_neg = P.Tile()(P.Reshape()(out_neg, (1, num_neg)), (1, num_pos))
                penalty_term = P.ReduceMean()(P.Pow()(1 - (trans_pos - trans_neg), 2)) / 2
        except:
            import pdb;pdb.set_trace()

        cls = self.classify_loss(outs, labels) + 0.5 * penalty_term
        return cls, 0.5 * penalty_term


class AUCHLoss(nn.Cell):
    def __init__(self, alpha=0.1, lamb=1, num_hard=0):
        super(AUCHLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.classify_loss = nn.BCELoss()
        self.regress_loss = nn.SmoothL1Loss()
        self.num_hard = num_hard
        self.alpha = alpha
        self.lamb = lamb

    def construct(self, output, labels, train=True):
        batch_size = labels.shape[0]
        outs = self.sigmoid(output[:, :1])

        out_pos = outs[labels == 1]
        out_neg = outs[labels == 0]
        penalty_term_sum = 0

        try:
            num_pos = out_pos.shape[0]
            num_neg = out_neg.shape[0]
            if num_pos == 0:
                trans_pos = 0
                trans_neg = out_neg
                penalty_term = P.ReduceMean()(P.Pow()(1 - (trans_pos - trans_neg), self.lamb)) / self.lamb
                print("pos")
            elif num_neg == 0:
                trans_pos = out_pos
                trans_neg = 0
                penalty_term = P.ReduceMean()(P.Pow()(1 - (trans_pos - trans_neg), self.lamb)) / self.lamb
                print("neg")
            else:
                trans_pos = P.Tile()(out_pos, (num_neg, 1))
                trans_neg = P.Tile()(P.Reshape()(out_neg, (1, num_neg)), (1, num_pos))
                penalty_term = P.ReduceMean()(P.Pow()(1 - (trans_pos - trans_neg), self.lamb)) / self.lamb
        except:
            import pdb;pdb.set_trace()

        cls = self.classify_loss(outs, labels) + self.alpha * penalty_term
        return cls, self.alpha * penalty_term

    
class SoftmaxAUCHLoss(nn.Cell):
    def __init__(self):
        super(SoftmaxAUCHLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.classify_loss = SoftMaxLoss()
        self.regress_loss = nn.SmoothL1Loss()
        self.alpha = 0.1

    def construct(self, output, labels, train=True):
        batch_size = labels.shape[0]
        num_category = labels.shape[1]
        sum_term = 0
        outs_sig = self.sigmoid(output[:, :num_category]).view(batch_size, num_category)
        for k in range(num_category):
            outs = outs_sig[:, k]
            out_pos = outs[labels[:, k] == 1]
            out_neg = outs[labels[:, k] == 0]
            penalty_term_sum = 0

            try:
                num_pos = out_pos.shape[0]
                num_neg = out_neg.shape[0]
                if num_pos == 0:
                    trans_pos = 0
                    trans_neg = out_neg
                    penalty_term = P.ReduceMean()(P.Pow()(1 - (trans_pos - trans_neg), 2))
                elif num_neg == 0:
                    trans_pos = out_pos
                    trans_neg = 0
                    penalty_term = P.ReduceMean()(P.Pow()(1 - (trans_pos - trans_neg), 2))
                else:
                    trans_pos = P.Tile()(out_pos, (num_neg, 1))
                    trans_neg = P.Tile()(P.Reshape()(out_neg, (1, num_neg)), (1, num_pos))
                    penalty_term = P.ReduceMean()(P.Pow()(1 - (trans_pos - trans_neg), 2))
            except:
                import pdb;pdb.set_trace()

            sum_term += penalty_term
        cls = self.classify_loss(output, labels)[0] + 0.1 * (sum_term / 15)
        return cls, 0.1 * sum_term / 15

class CWAUCHLoss(nn.Cell):
    def __init__(self, alpha=1, lamb=1, num_hard=0):
        super(CWAUCHLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.classify_loss = CWCELoss()
        self.regress_loss = nn.SmoothL1Loss()
        self.num_hard = num_hard
        self.alpha = alpha
        self.lamb = lamb

    def construct(self, output, labels, train=True):
        batch_size = labels.shape[0]
        outs = self.sigmoid(output[:, :1])
        out_pos = outs[labels == 1]
        out_neg = outs[labels == 0]
        penalty_term_sum = 0

        try:
            num_pos = out_pos.shape[0]
            num_neg = out_neg.shape[0]
            if num_pos == 0:
                trans_pos = 0
                trans_neg = out_neg
                penalty_term = P.ReduceMean()(P.Pow()(1 - (trans_pos - trans_neg), self.lamb)) / self.lamb
            elif num_neg == 0:
                trans_pos = out_pos
                trans_neg = 0
                penalty_term = P.ReduceMean()(P.Pow()(1 - (trans_pos - trans_neg), self.lamb)) / self.lamb
            else:
                trans_pos = P.Tile()(out_pos, (num_neg, 1))
                trans_neg = P.Tile()(P.Reshape()(out_neg, (1, num_neg)), (1, num_pos))
                penalty_term = P.ReduceMean()(P.Pow()(1 - (trans_pos - trans_neg), self.lamb)) / self.lamb
        except:
            import pdb;pdb.set_trace()

        cls = self.classify_loss(outs, labels)[0] + self.alpha * penalty_term
        return cls, self.alpha * penalty_term

class FPLoss(nn.Cell):
    def __init__(self, num_hard=0):
        super(FPLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.classify_loss = nn.BCELoss()

    def construct(self, output, labels, train=True):
        outs = self.sigmoid(output[:, :1])

        neg_labels = 1 - labels
        neg_outs = 1 - self.sigmoid(output[:, :1])

        pos_loss = labels * P.Log()(outs)
        neg_loss = neg_labels * P.Log()(neg_outs)

        h_pos_loss = neg_outs * pos_loss
        h_neg_loss = outs * neg_loss

        fpcls = - P.ReduceMean()(h_pos_loss) - P.ReduceMean()(h_neg_loss)

        return fpcls
    
class FPLoss1(nn.Cell):
    def __init__(self, num_hard=0):
        super(FPLoss1, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.classify_loss = nn.BCELoss(reduction='mean')

    def construct(self, output, labels, train=True):
        outs = self.sigmoid(output[:, :1])
        neg_labels = 1 - labels
        neg_outs = 1 - self.sigmoid(output[:, :1])

        pos_loss = labels * P.Log()(outs + 1e-32)
        neg_loss = neg_labels * P.Log()(neg_outs + 1e-32)

        h_pos_loss = neg_outs * pos_loss
        h_neg_loss = outs * neg_loss

        fpcls = - P.ReduceMean()(h_pos_loss) - 2 * P.ReduceMean()(h_neg_loss)

        return fpcls

class RCLoss(nn.Cell):
    def __init__(self, num_hard=0):
        super(RCLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.classify_loss = nn.BCELoss(reduction='mean')

    def construct(self, output, labels, train=True):
        outs = self.sigmoid(output[:, :1])
        neg_labels = 1 - labels
        neg_outs = 1 - self.sigmoid(output[:, :1])

        pos_loss = labels * P.Log()(outs + 1e-32)
        neg_loss = neg_labels * P.Log()(neg_outs + 1e-32)

        h_pos_loss = neg_outs * pos_loss
        h_neg_loss = outs * neg_loss

        fpcls = - 2 * P.ReduceMean()(h_pos_loss) - P.ReduceMean()(h_neg_loss)

        return fpcls

class CWCELoss(nn.Cell):
    def __init__(self, num_hard=0):
        super(CWCELoss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.classify_loss = nn.BCELoss(reduction='mean')
        self.epsilon = 1e-32

    def construct(self, output, labels, train=True):
        outs = self.sigmoid(output[:, :1])
        neg_labels = 1 - labels
        neg_outs = 1 - self.sigmoid(output[:, :1])

        num_neg = P.ReduceSum()(neg_labels)
        num_pos = P.ReduceSum()(labels)

        Beta_P = num_pos / (num_pos + num_neg)
        Beta_N = num_neg / (num_pos + num_neg)

        pos_loss = labels * P.Log()(outs + self.epsilon)
        neg_loss = neg_labels * P.Log()(neg_outs + self.epsilon)

        fpcls = - Beta_N * P.ReduceMean()(pos_loss) - Beta_P * P.ReduceMean()(neg_loss)

        return fpcls, fpcls

class FocalLoss(nn.Cell):
    def __init__(self, num_hard=0):
        super(FocalLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.classify_loss = nn.BCELoss(reduction='mean')
        self.epsilon = 1e-32

    def construct(self, output, labels, train=True):
        batch_size = labels.shape[0]
        num_category = labels.shape[1]

        outs_sig = self.sigmoid(output[:, :num_category]).view([batch_size, num_category])
        reshape_out = outs_sig.view(batch_size * num_category, 1)
        reshape_lab = labels.view(batch_size * num_category, 1)

        pos_index = reshape_lab
        neg_index = 1 - reshape_lab

        pos_loss = pos_index * P.Log()(reshape_out + self.epsilon)
        pos_loss = (1 - reshape_out) * pos_loss

        neg_loss = neg_index * P.Log()(1 - reshape_out + self.epsilon)
        neg_loss = reshape_out * neg_loss

        cel_sum = - P.ReduceMean()(pos_loss) - P.ReduceMean()(neg_loss)

        return cel_sum, cel_sum

class CWFocalLoss(nn.Cell):
    def __init__(self, num_hard=0):
        super(CWFocalLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.classify_loss = nn.BCELoss(reduction='mean')
        self.epsilon = 1e-32

    def construct(self, output, labels, train=True):
        cel_sum = 0
        batch_size = labels.shape[0]
        num_category = labels.shape[1]

        outs_sig = self.sigmoid(output[:, :num_category]).view([batch_size, num_category])
        reshape_out = outs_sig.view(batch_size * num_category, 1)
        reshape_lab = labels.view(batch_size * num_category, 1)

        total_samples = batch_size * num_category

        num_P = P.ReduceSum()(reshape_lab)
        num_N = total_samples - num_P

        alpha_P = num_P / total_samples
        alpha_N = num_N / total_samples

        pos_index = reshape_lab
        neg_index = 1 - reshape_lab

        pos_loss = pos_index * P.Log()(reshape_out + self.epsilon)
        pos_loss = (1 - reshape_out) * pos_loss

        neg_loss = neg_index * P.Log()(1 - reshape_out + self.epsilon)
        neg_loss = reshape_out * neg_loss

        cel_sum = - alpha_N * P.ReduceMean()(pos_loss) - alpha_P * P.ReduceMean()(neg_loss)

        return cel_sum, cel_sum
    
class MCWCEL(nn.Cell):
    def __init__(self):
        super(MCWCEL, self).__init__()
        self.classify_loss = nn.BCELoss(reduction='mean')
        self.sigmoid = nn.Sigmoid()

        # Prevent to overflow
        self.epsilon = 1e-32

    def construct(self, output, labels, train=True):
        cel_sum = 0
        batch_size = labels.shape[0]
        num_category = labels.shape[1]

        outs_sig = self.sigmoid(output[:, :num_category]).view([batch_size, num_category])
        reshape_out = outs_sig.view(batch_size * num_category, 1)
        reshape_lab = labels.view(batch_size * num_category, 1)

        total_samples = batch_size * num_category

        num_P = P.ReduceSum()(reshape_lab)
        num_N = total_samples - num_P

        alpha_P = num_P / total_samples
        alpha_N = num_N / total_samples

        pos_index = reshape_lab
        neg_index = 1 - reshape_lab

        pos_loss = pos_index * P.Log()(reshape_out + self.epsilon)
        neg_loss = neg_index * P.Log()(1 - reshape_out + self.epsilon)

        cel_sum = - alpha_N * P.ReduceMean()(pos_loss) - alpha_P * P.ReduceMean()(neg_loss)

        return cel_sum, cel_sum  

class FGMCWCEL(nn.Cell):
    def __init__(self, alpha=1, lamb=1):
        super(FGMCWCEL, self).__init__()
        self.classify_loss = nn.BCELoss(reduction='mean')
        self.sigmoid = nn.Sigmoid()
        self.alpha = alpha
        self.lamb = lamb

        # Prevent to overflow
        self.epsilon = 1e-32

    def construct(self, output, labels, train=True):
        cel_sum = 0
        batch_size = labels.shape[0]
        num_category = labels.shape[1]

        outs_sig = self.sigmoid(output[:, :num_category]).view([batch_size, num_category])
        for i in range(num_category):
            reshape_out = outs_sig[:, i]
            reshape_lab = labels[:, i]

            total_samples = batch_size * num_category

            num_P = P.ReduceSum()(reshape_lab)
            num_N = total_samples - num_P

            alpha_P = num_P / total_samples
            alpha_N = num_N / total_samples

            pos_index = reshape_lab
            neg_index = 1 - reshape_lab

            pos_loss = pos_index * P.Log()(reshape_out + self.epsilon)
            neg_loss = neg_index * P.Log()(1 - reshape_out + self.epsilon)

            tmp_cel = - alpha_N * P.ReduceMean()(pos_loss) - alpha_P * P.ReduceMean()(neg_loss)

            if i == 0:
                cel_sum += 4 * tmp_cel
            else:
                cel_sum += tmp_cel

        return cel_sum, cel_sum  

class FGMCWAUCHLoss(nn.Cell):
    def __init__(self, alpha=1, lamb=1):
        super(FGMCWAUCHLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.classify_loss = FGMCWCEL()
        self.regress_loss = nn.SmoothL1Loss()
        self.alpha = alpha
        self.lamb = lamb

    def construct(self, output, labels, train=True):
        batch_size = labels.shape[0]
        num_category = labels.shape[1]

        sum_term = 0
        outs_sig = self.sigmoid(output[:, :num_category]).view([batch_size, num_category])
        for k in range(num_category):
            outs = outs_sig[:, k]
            out_pos = outs[labels[:, k] == 1]
            out_neg = outs[labels[:, k] == 0]

            try:
                num_pos = P.Shape()(out_pos)[0]
                num_neg = P.Shape()(out_neg)[0]

                if num_pos == 0:
                    trans_pos = 0
                    trans_neg = out_neg
                    penalty_term = P.ReduceMean()(1 - (trans_pos - trans_neg)).pow(self.lamb) / self.lamb
                elif num_neg == 0:
                    trans_pos = out_pos
                    trans_neg = 0
                    penalty_term = P.ReduceMean()(1 - (trans_pos - trans_neg)).pow(self.lamb) / self.lamb
                else:
                    trans_pos = P.Tile()(out_pos, (1, num_neg))
                    trans_neg = P.Tile()(P.Transpose()(out_neg, (1, 0)), (num_pos, 1))
                    penalty_term = P.ReduceMean()(1 - (trans_pos - trans_neg)).pow(self.lamb) / self.lamb

            except:
                penalty_term = 0

            sum_term += penalty_term

        cls = self.classify_loss(outs_sig, labels)[0] + self.alpha * (sum_term / num_category)

        return cls, self.alpha * penalty_term

class MCWAUCHLoss(nn.Cell):
    def __init__(self, alpha=1, lamb=1):
        super(MCWAUCHLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.classify_loss = MCWCEL()
        self.regress_loss = nn.SmoothL1Loss()
        self.alpha = 1
        self.lamb = 1

    def construct(self, output, labels, train=True):
        batch_size = labels.shape[0]
        num_category = labels.shape[1]

        sum_term = 0
        outs_sig = self.sigmoid(output[:, :num_category]).view([batch_size, num_category])
        for k in range(num_category):
            outs = outs_sig[:, k]
            out_pos = outs[labels[:, k] == 1]
            out_neg = outs[labels[:, k] == 0]

            try:
                num_pos = P.Shape()(out_pos)[0]
                num_neg = P.Shape()(out_neg)[0]

                if num_pos == 0:
                    trans_pos = 0
                    trans_neg = out_neg
                    penalty_term = P.ReduceMean()(1 - (trans_pos - trans_neg)).pow(self.lamb) / self.lamb
                elif num_neg == 0:
                    trans_pos = out_pos
                    trans_neg = 0
                    penalty_term = P.ReduceMean()(1 - (trans_pos - trans_neg)).pow(self.lamb) / self.lamb
                else:
                    trans_pos = P.Tile()(out_pos, (1, num_neg))
                    trans_neg = P.Tile()(P.Transpose()(out_neg, (1, 0)), (num_pos, 1))
                    penalty_term = P.ReduceMean()(1 - (trans_pos - trans_neg)).pow(self.lamb) / self.lamb

            except:
                penalty_term = 0

            sum_term += penalty_term

        cls = self.classify_loss(outs_sig, labels)[0] + 0.1 * (sum_term / num_category)

        return cls, 0.1 * penalty_term
    
class MAUCHLoss(nn.Cell):
    def __init__(self):
        super(MAUCHLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.classify_loss = CEL()
        self.regress_loss = nn.SmoothL1Loss()
        self.alpha = 0.1

    def construct(self, output, labels, train=True):
        batch_size = labels.shape[0]
        num_category = labels.shape[1]

        sum_term = 0
        outs_sig = self.sigmoid(output[:, :num_category]).view([batch_size, num_category])

        for k in range(num_category):
            outs = outs_sig[:, k]
            out_pos = outs[labels[:, k] == 1]
            out_neg = outs[labels[:, k] == 0]

            try:
                num_pos = P.Shape()(out_pos)[0]
                num_neg = P.Shape()(out_neg)[0]

                if num_pos == 0:
                    trans_pos = 0
                    trans_neg = out_neg
                    penalty_term = P.ReduceMean()(1 - (trans_pos - trans_neg))
                elif num_neg == 0:
                    trans_pos = out_pos
                    trans_neg = 0
                    penalty_term = P.ReduceMean()(1 - (trans_pos - trans_neg))
                else:
                    trans_pos = P.Tile()(out_pos, (1, num_neg))
                    trans_neg = P.Tile()(P.Transpose()(out_neg, (1, 0)), (num_pos, 1))
                    penalty_term = P.ReduceMean()(1 - (trans_pos - trans_neg))

            except:
                penalty_term = 0

            sum_term += penalty_term

        cls = self.classify_loss(outs_sig[:, :num_category], labels)[0] + 0.1 * (sum_term / 15)

        return cls, 0.1 * penalty_term

class FPSimilarityLoss(nn.Cell):
    def __init__(self, num_hard=0):
        super(FPSimilarityLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.classify_loss = nn.BCELoss()

    def construct(self, output, labels, train=True):
        batch_size = labels.shape[0]
        outs = self.sigmoid(output[:, :1])

        neg_labels = 1 - labels
        neg_outs = 1 - self.sigmoid(output[:, :1])

        pos_loss = labels * P.Log()(outs)
        neg_loss = neg_labels * P.Log()(neg_outs)

        h_pos_loss = neg_outs * pos_loss
        h_neg_loss = outs * neg_loss

        fpcls = - P.ReduceMean()(h_pos_loss) - 2 * P.ReduceMean()(h_neg_loss)

        return fpcls