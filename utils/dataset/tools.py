import h5py, csv
import numpy as np
import pandas as pd
from random import shuffle


def load_patches(path, only_names = False):

    patches = np.array(pd.read_csv(path, header=0)).tolist()

    if only_names:
        patches = [name for [name,_,_,_,_] in patches]

    return patches

def save_patches(path, patches):
    headers = ['name', 'top', 'left', 'h', 'w']
    with open(path, 'w', newline='')as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        shuffle(patches)
        f_csv.writerows(patches)

def get_names(path):

    with h5py.File(path,'r') as f:

        names = list(f.keys())

    return names

def get_from_hdf5(path,name,property):

    with h5py.File(path,'r') as f:
        if isinstance(name, str):
            result = f[name][property][()]
        else:
            result = []
            for n in name:
                result.append(f[n][property][()])
            result = np.asarray(result)
    return result

def count_patches_classes(patches, data_path):

    names = [name for [name, _, _,_,_] in patches]
    gts = np.asarray([get_from_hdf5(data_path, name, 'label') for name in names]).astype('int32')

    index = gts != 0

    count = np.bincount((gts[index] - 1).reshape(-1), minlength=7)
    ratios = count / np.sum(count)

    return ratios

