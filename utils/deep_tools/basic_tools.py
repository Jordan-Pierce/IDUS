import torch, cv2
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from utils.deep_tools.hook import register_hooks, extract_hook_features
from utils.image_process import regularize_img
from utils.dataset import load_patches, get_from_hdf5
from utils.models import UNet_Resnet18


def extract_center(feature, ratio=4):

    if len(feature.shape) == 3:

        ft = np.transpose(feature, (1, 2, 0))

        (ft_h, ft_w, ft_dim) = ft.shape

        pad_h, pad_w  = int(ft_h/ratio), int(ft_w/ratio)

        ft = ft[pad_h: 2* pad_h, pad_w: 2*pad_w]

    else:
        ft = feature

        (ft_h, ft_w) = ft.shape

        pad_h, pad_w = int(ft_h / ratio), int(ft_w / ratio)

        ft = ft[pad_h: 2 * pad_h, pad_w: 2 * pad_w]

    return ft


def extract_features_train(images,  net):

    net.cuda(), net.eval()

    features_all = []

    for i, image in tqdm(enumerate(images)):
        # pad image and preprocess image
        (h, w) = image.shape
        img = np.pad(image, ((h, h), (w, w)), mode='symmetric')
        img = cv2.equalizeHist((img * 255.0).astype('uint8')).astype('float32') / 255.0
        img = regularize_img(img)
        img = np.expand_dims(img, axis=2)
        # img = np.repeat(img, 3, 2)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).cuda()

        # forward
        outputs = torch.softmax(net(img), 1)
        # outputs = torch.sigmoid(net(img))
        outputs = torch.squeeze(outputs).cpu().data.numpy()
        outputs = extract_center(outputs, 3)
        features_all.append(outputs)

    return np.asarray(features_all)


def extract_features(patches_path, data_path, if_softmax = False,
                     ly_names = None, net = None, net_path = None):

    names = load_patches(patches_path, only_names=True) if \
        isinstance(patches_path, str) else patches_path

    if net is None:
        net = UNet_Resnet18(n_channels=1, n_classes=7, encoder_weights='imagenet')
    if net_path is not None:
        net.load_state_dict(torch.load(net_path))

    net.cuda(), net.eval()
    ly_hooks = None
    if ly_names is not None:
        if not isinstance(ly_names, list):
            ly_names = [ly_names]
        ly_hooks = register_hooks(ly_names,net)

    images = [get_from_hdf5(data_path,name,'data') for name in names]

    features_all = defaultdict(list)

    for i, image in tqdm(enumerate(images)):

        # pad image and preprocess image
        (h, w) = image.shape
        img = np.pad(image, ((h, h), (w, w)), mode='symmetric')
        img = cv2.equalizeHist((img * 255.0).astype('uint8')).astype('float32') / 255.0
        img = regularize_img(img)
        img = np.expand_dims(img, axis=2)
        # img = np.repeat(img, 3, 2)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).cuda()

        # forward
        # forward
        if if_softmax:
            outputs = torch.softmax(net(img), 1)
            # outputs = torch.sigmoid(net(img))
            outputs = torch.squeeze(outputs).cpu().data.numpy()
            outputs = extract_center(outputs, 3)
            features_all['softmax'].append(outputs)
        else:
            net(img)

        # get feature
        if ly_hooks is not None:
            features = extract_hook_features(ly_hooks)
            for key, ft in features.items():
                features_all[key].append(extract_center(ft,3))


    return features_all, names

