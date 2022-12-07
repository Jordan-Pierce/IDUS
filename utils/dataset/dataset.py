import torch, cv2
import numpy as np
from torch.utils.data import Dataset
from utils.dataset.tools import get_from_hdf5, load_patches
from utils.basic_tools import regularize, normalize


def img_preprocess(img):
    img = cv2.equalizeHist((img * 255.0).astype('uint8')).astype('float32') / 255.0
    img = regularize(img)
    img = np.expand_dims(img, axis=2)
    # img = np.repeat(img, 3, 2)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img)

    return img


class IDUS_Dataset(Dataset):
    def __init__(self, data_path, name_path,pseudo_path):
        self.data_path = data_path
        if isinstance(name_path, str):
            self.names = np.load(name_path)
        else:
            self.names = name_path

        self.pseudo_masks = np.load(pseudo_path) \
            if isinstance(pseudo_path, str) else pseudo_path

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):

        name = self.names[idx]
        image = get_from_hdf5(self.data_path, name, 'data')
        ground_truth = get_from_hdf5(self.data_path, name, 'label')

        pseudo_mask = self.pseudo_masks[idx]

        sample = {'name':name,
                  'image':img_preprocess(image),
                  'ground_truth':torch.from_numpy(ground_truth),
                  'pseudo_mask':torch.from_numpy(pseudo_mask)}

        return sample

    def images(self):
        return np.asarray([get_from_hdf5(self.data_path, name, 'data') for name in self.names])

    def ground_truths(self):
        return np.asarray([get_from_hdf5(self.data_path, name, 'label') for name in self.names])



class HDF5_Dataset(Dataset):

    def __init__(self, data_path, name_path):
        self.data_path = data_path
        if isinstance(name_path, str):
            self.names = np.load(name_path)
        else:
            self.names = name_path

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):

        name = self.names[idx]
        image = get_from_hdf5(self.data_path, name, 'data')
        ground_truth = get_from_hdf5(self.data_path, name, 'label')

        sample = {'name':name,
                  'image':img_preprocess(image),
                  'ground_truth':torch.from_numpy(ground_truth)}

        return sample

    def images(self):
        return np.asarray([get_from_hdf5(self.data_path, name, 'data') for name in self.names])

    def ground_truths(self):
        return np.asarray([get_from_hdf5(self.data_path, name, 'label') for name in self.names])

class Feature_Dataset2(Dataset):

    def __init__(self, features, ground_truths):
        self.ft = np.transpose(regularize(normalize(features)), (0, 3, 1, 2))
        self.gt = ground_truths

    def __len__(self):
        return len(self.ft)

    def __getitem__(self, idx):

        feature = self.ft[idx]

        ground_truth = self.gt[idx]

        sample = {'feature': torch.from_numpy(feature),
                  'ground_truth': torch.from_numpy(ground_truth)}

        return sample

    def ground_truths(self):
        return self.gt




class Feature_Dataset(Dataset):

    def __init__(self, data_path, feature_path ,name_path, train_patches):

        self.data_path = data_path
        used_names = [name for [name, _, _, _, _] in load_patches(train_patches)]
        used_names = set(used_names)
        self.features = np.load(feature_path)
        self.names = np.load(name_path)
        idx = []
        for i, name in enumerate(self.names):
            if name in used_names:
                idx.append(i)

        self.features = np.transpose(regularize(normalize(self.features[idx])),( 0, 3, 1,2))
        self.names = self.names[idx]



    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):

        name = self.names[idx]
        feature = self.features[idx]
        image = get_from_hdf5(self.data_path, name, 'data')
        ground_truth = get_from_hdf5(self.data_path, name, 'label')

        sample = {'name':name,
                  'image': img_preprocess(image),
                  'feature': torch.from_numpy(feature),
                  'ground_truth':torch.from_numpy(ground_truth)}

        return sample

    def images(self):
        return np.asarray([get_from_hdf5(self.data_path, name, 'data') for name in self.names])

    def ground_truths(self):
        return np.asarray([get_from_hdf5(self.data_path, name, 'label') for name in self.names])