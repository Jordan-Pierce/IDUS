import math, cv2
import numpy as np
from sklearn.decomposition import PCA


def features_combination(features_all, cb_settings, target_size=64):
    """
    This function takes in three inputs:

    features_all: a dictionary containing features of an image, where key is
    the name of feature and value is the feature. cb_settings: a dictionary
    containing information on how to split features_all target_size: an
    integer representing the size to which the features should be resized.
    It returns an array which is a combination of all the resized features
    from the input dictionary.
    """
    # Initialize a new dictionary for storing the decomposed and resized
    # features
    new_ft = dict()

    # Iterate over the input dictionary and decompose the features using the
    # cb_settings
    for name, ft_all in features_all.items():
        ft_all = np.asarray(ft_all)
        ft_all = decomposed(ft_all, ft_all.shape[-1] // cb_settings[name])

        # Resize the decomposed features and store them in the new dictionary
        new_ft[name] = np.asarray([cv2.resize(ft, (target_size, target_size),
                                              interpolation=cv2.INTER_LINEAR)
                                   for ft in ft_all])

    # Concatenate all the resized features and return the combined array
    return np.concatenate(tuple(new_ft.values()), axis=3)


class CountBin(object):

    def __init__(self, minlength):
        self.minlength = minlength

    def __call__(self, window):
        window = window.astype('int32')
        count = np.bincount(window.reshape(-1), minlength=self.minlength)
        return count


def normalize(feature):
    shape = feature.shape

    feature = feature.reshape((-1, shape[-1]))

    result = (feature - np.min(feature, axis=0)) / (
                np.ptp(feature, axis=0) + 1e-6)

    return result.reshape(shape)


def sliding_window(image, size, func=None):

    (h, w) = size

    h1_pad = math.floor(h / 2)
    h2_pad = math.ceil(h / 2) - 1
    w1_pad = math.floor(w / 2)
    w2_pad = math.ceil(w / 2) - 1

    if len(image.shape) == 2:
        img_pad = np.pad(image, ((h1_pad, h2_pad), (w1_pad, w2_pad)),
                         'reflect')
    else:
        img_pad = np.pad(image, ((h1_pad, h2_pad), (w1_pad, w2_pad), (0, 0)),
                         'reflect')

    outputs = []

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = img_pad[i:i + h, j:j + w]
            outputs.append(func(window))

    # transfer to array with shape NxMxC
    outputs = np.asarray(outputs)
    outputs.reshape(image.shape[0:2] + tuple([outputs.shape[-1]]))

    return outputs.reshape(image.shape[0:2] + tuple([outputs.shape[-1]]))


def regularize(features):

    shape = features.shape

    if len(shape) >= 3:
        ft_all = features.reshape((-1, shape[-1]))
    else:
        ft_all = features.reshape(-1)

    (ft_all - ft_all.mean(axis=0)) / (ft_all.std(axis=0) + 1e-6)
    features = ft_all.reshape(shape)

    return features


def decomposed(features, n_components):

    if features.shape[-1] == n_components or n_components == -1:
        return features

    decompsed = PCA(n_components=n_components)
    shape = features.shape

    ft_all = features.reshape((-1, shape[-1]))
    # (ft_all - ft_all.mean(axis=0)) / (ft_all.std(axis=0) + 1e-6)
    ft_all = decompsed.fit_transform(ft_all)
    features = ft_all.reshape(shape[0:len(shape) - 1] + tuple([n_components]))

    return features


def decomposed_and_resize(features, resize, components):

    features = decomposed(features, components)
    features = np.asarray([cv2.resize(feature,
                                      (resize, resize),
                                      interpolation=cv2.INTER_LINEAR) for
                           feature in features])

    return features


def average(window):
    shape = window.shape
    return np.mean(window.reshape(-1, shape[-1]), axis=0)


def features_combination(features_all, cb_settings, target_size=64):
    new_ft = dict()
    for name, ft_all in features_all.items():
        ft_all = np.asarray(ft_all)
        ft_all = decomposed(ft_all, ft_all.shape[-1] // cb_settings[name])
        new_ft[name] = np.asarray([cv2.resize(ft, (target_size, target_size),
                                              interpolation=cv2.INTER_LINEAR)
                                   for ft in ft_all])

    return np.concatenate(tuple(new_ft.values()), axis=3)


def extend_array(arrays):
    array_extend = arrays[0]
    index = [[0, len(arrays[0])]]
    start = len(arrays[0])
    end = None

    for i in range(1, len(arrays), 1):
        end = start + len(arrays[i])

        array_extend = np.concatenate((array_extend, arrays[i]), axis=0)
        index.append([start, end])

        start = end

    return array_extend, index


def unextend_array(array, index):
    arrays = []

    for [start, end] in index:
        arrays.append(array[start:end])

    return arrays
