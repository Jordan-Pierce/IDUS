import torch, tqdm, cv2
import numpy as np
from collections import defaultdict
from utils.basic_tools import regularize
from utils.models.unet import UNet_Resnet18
from utils.deep_tools.hook import register_hooks, extract_hook_features
from utils.deep_tools.basic_tools import extract_center
from utils.dataset.tools import load_patches, get_from_hdf5

def extract_features(patches_path, data_path, ly_names,net_path = None):
    names = load_patches(patches_path, only_names=True)

    net = UNet_Resnet18(n_channels=1, n_classes=7, encoder_weights='imagenet')
    if net_path is not None:
        net.load_state_dict(torch.load(net_path))

    net.cuda(), net.eval()

    ly_hooks = register_hooks(ly_names,net)
    images = [get_from_hdf5(data_path,name,'data') for name in names]

    features_all = defaultdict(list)

    for i, image in tqdm(enumerate(images)):

        # pad image and preprocess image
        (h, w) = image.shape
        img = np.pad(image, ((h, h), (w, w)), mode='symmetric')
        img = cv2.equalizeHist((img * 255.0).astype('uint8')).astype('float32') / 255.0
        img = regularize(img)
        img = np.expand_dims(img, axis=2)
        # img = np.repeat(img, 3, 2)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).cuda()

        # forward
        net(img)

        # get feature
        features = extract_hook_features(ly_hooks)
        for key, ft in features.items():
            features_all[key].append(extract_center(ft,3))

    return features_all, names
