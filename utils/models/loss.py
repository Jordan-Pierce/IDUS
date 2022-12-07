import torch
import torch.nn as nn
import numpy as np


def create_one_hot(mask, num_classes = 3):
    one_hot_mask = torch.zeros([mask.shape[0],
                                num_classes,
                                mask.shape[1],
                                mask.shape[2]],
                               dtype=torch.float32)
    if mask.is_cuda:
        one_hot_mask = one_hot_mask.cuda()
    one_hot_mask = one_hot_mask.scatter(1, mask.long().data.unsqueeze(1), 1.0)
    return one_hot_mask

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target, used_region = None):
        input = input.contiguous()
        target = target.contiguous()

        smooth = 1

        intersection = input * target

        if used_region is None:
            loss = (2 * (intersection.sum() + smooth)) / (input.sum() + target.sum() + smooth)
        else:
            loss = (2 * (intersection[used_region].sum() + smooth)) / (input[used_region].sum() + target[used_region].sum() + smooth)
        loss = 1 - loss

        return loss

class MulticlassDiceLoss(nn.Module):

    def __init__(self, n_classes = 8, ignore_index = None, weight = None):
        super(MulticlassDiceLoss, self).__init__()
        self.n_classes = n_classes
        self.ignore_index = ignore_index
        self.register_buffer('weight',weight)

        self.dice = DiceLoss()

    def forward(self,input, mask):
        input = torch.sigmoid(input)

        if self.ignore_index is None:
            one_hot_mask = create_one_hot(mask, num_classes=self.n_classes)
            used_region = None
        else:
            ignore_index = mask == self.ignore_index
            used_region = mask != self.ignore_index
            mask[ignore_index] = self.n_classes
            one_hot_mask = create_one_hot(mask, num_classes= self.n_classes + 1)
            one_hot_mask = one_hot_mask[:,0:self.n_classes]

        # if mask.is_cuda:
        #     one_hot_mask = one_hot_mask.cuda()

        # fig, ax = plt.subplots(1,2)
        # for lb, otm in zip(mask, one_hot_mask):
        #     lb = lb.cpu().data.numpy().astype('int32')
        #     for sub_otm in otm:
        #         sub_otm = sub_otm.cpu().data.numpy().astype('int32')
        #         ax[0].imshow(colormap[lb][:,:,::-1])
        #         ax[1].imshow(colormap[sub_otm][:,:,::-1])
        #         fig.show()
        #     pass

        losses = []
        totalLoss = 0
        if self.weight is None:
            for i in range(self.n_classes):
                diceLoss = self.dice(input[:, i], one_hot_mask[:, i], used_region=used_region)
                losses.append(diceLoss.cpu().data.numpy())
                totalLoss += diceLoss
            loss = totalLoss / self.n_classes
        else:
            for i in range(self.n_classes):
                diceLoss = self.dice(input[:,i], one_hot_mask[:,i], used_region = used_region)
                losses.append(diceLoss.cpu().data.numpy())
                diceLoss *= self.weight[i]
                totalLoss += diceLoss
            loss = totalLoss / torch.sum(self.weight)
        return loss, np.asarray(losses)

