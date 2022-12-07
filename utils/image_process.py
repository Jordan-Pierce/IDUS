import numpy as np

def regularize_img(features):
    shape = features.shape

    ft_all = features.reshape((-1))
    (ft_all - np.mean(ft_all)) / np.std(ft_all)
    features = ft_all.reshape(shape)

    return features