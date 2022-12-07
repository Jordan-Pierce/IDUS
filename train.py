import torch, os
import numpy as np
import torch.nn as nn
from collections import defaultdict
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from utils.dataset import IDUS_Dataset
from utils.models.loss import MulticlassDiceLoss
from utils.models import UNet_Resnet18
from utils.evaluation import normalize_mutual_informaton, confusion_matrix_parallel
from utils.deep_tools import update_pseudo_mask


torch.cuda.set_device("cuda:1")
np.set_printoptions(precision=3, suppress=True)
np.set_printoptions(linewidth=400)



class MyLoss(nn.Module):
    def __init__(self,pseudo_masks,n_classes = 7,ignore_index = -1):
        super(MyLoss, self).__init__()
        self.n_classes = 7

        counts = np.bincount(pseudo_masks[pseudo_masks != ignore_index].reshape(-1), minlength=n_classes)
        ratios = counts/ np.sum(counts)



        self.dice_loss = MulticlassDiceLoss(
            n_classes = n_classes,
            weight = torch.from_numpy(1.0 / (np.sqrt(ratios))).cuda().float())

        self.cross_etp_loss = \
            nn.CrossEntropyLoss(
                weight=torch.from_numpy(1.0 / (np.sqrt(ratios) )).cuda().float())
    def forward(self, x, mask):

        loss1, losses = self.dice_loss(x,mask)
        loss2 = self.cross_etp_loss(x,mask.clone().long())

        return loss1 , loss2


def train_one_epoch(dataset, n_classes, max_epoch, save_path, train_encoder_epoch = 150):

    # load network

    net = UNet_Resnet18(n_channels=1, n_classes=n_classes, encoder_weights='imagenet')
    net.cuda()

    # gts = np.asarray([dataset[i]['ground_truth'].data.numpy() for i in range(len(dataset))]) - 1
    trainloader = DataLoader(dataset,
                             batch_size=16,
                             num_workers=0, shuffle=True, drop_last=True)

    # setup loss
    loss_function = MyLoss(dataset.pseudo_masks, n_classes=n_classes)

    # optimizer
    optimizer = optim.Adam([{'params': net.model.decoder.parameters()},
                            {'params': net.model.segmentation_head.parameters()}]
                           , lr=0.0001, weight_decay=1e-9)
    scheduler = MultiStepLR(optimizer, [100, 200], gamma=0.1)



    for e in range(max_epoch):

        # train
        net.train()
        t_losses = defaultdict(float)
        for step, batch in enumerate(trainloader):
            image = batch['image'].cuda()
            mask = batch['pseudo_mask'].cuda()



            outputs = net(image)
            loss1, loss2 = loss_function(outputs, mask)
            loss = 0.5*loss1 + 0.5*loss2
            loss.backward()
            optimizer.step()
            # print('Epoch:', e + 1, 'Loss:', loss.cpu().data.numpy(), 'learning rate:', scheduler.get_last_lr())
            t_losses['loss'] += loss.cpu().data.numpy()
            t_losses['loss1'] += loss1.cpu().data.numpy()
            t_losses['loss2'] += loss2.cpu().data.numpy()
            t_losses['batches'] += 1

        # round
        p_dict = defaultdict(float)
        p_dict['loss'] = np.around(t_losses['loss']/t_losses['batches'], decimals=3)
        p_dict['loss1'] = np.around(t_losses['loss1'] / t_losses['batches'], decimals=3)
        p_dict['loss2'] = np.around(t_losses['loss2'] / t_losses['batches'], decimals=3)



        scheduler.step()
        print('Epoch:',e + 1,'Loss:', p_dict['loss'], 'Loss1:', p_dict['loss1'], 'Loss2:', p_dict['loss1'], 'lr:',scheduler.get_last_lr())

        if (e+1) == train_encoder_epoch:
            optimizer.add_param_group({'params': net.model.encoder.parameters(), 'lr':1e-5})


        if (e + 1) % 10 == 0:
            # validation
            net.eval()

            torch.save(net.state_dict(), os.path.join(save_path, 'epoch_' + str(e + 1) + '.pth'))

        # update masks
    pseudo_masks = update_pseudo_mask(net, dataset.images(), n_classes, n_segments=256,
                                                  compactness=1.0,
                                                  img_seg_size=512,
                                                  seg_comp=-1)


    return pseudo_masks

if __name__ == '__main__':
    data_path = '/cvdata/yungchen/supervised_sonar_segentation/used_data/sonar_512x512.hdf5'
    names_path = '/cvdata/yungchen/idus/results/features/deep feature/wavelet_deep_texton_names.npy'
    pseudo_path = '/home/yungchen/idus_code/results/pseudo_mask.npy'
    save_path = '/cvdata/yungchen/test_data/model'

    n_classes = 7
    max_epoch = 200
    max_iterations = 5
    # load network
    dataset = IDUS_Dataset(data_path, names_path, pseudo_path)
    # print('************************************************************')
    for i in range(max_iterations):
        print('************************************************************')
        print('Iteration:',i + 1)
        print('************************************************************')
        model_dir = os.path.join(save_path,'iteration_' + str(i))
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)

        dataset.pseudo_masks = train_one_epoch(dataset, n_classes, max_epoch, model_dir)

