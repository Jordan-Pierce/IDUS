import numpy as np
import cv2

if __name__ == '__main__':

    """This script concatenates wavelet and deep texton features, resizing 
    them to the given size and saving the resulting features and names to 
    the specified paths. """
    
    # paths to wavelet and deep texton features and names
    wavelet = './results/wavelet_features.npy'
    wavelet_names = './results/wavelet_names.npy'
    deep_feature = './results/deep_texton.npy'
    deep_feature_name = './results/deep_texton_names.npy'

    # path to save concatenated features and names
    save_path = './results/wavelet_deep_texton.npy'
    name_path = './results/wavelet_deep_texton_names.npy'

    # size to resize features to
    size = 256

    # load wavelet features and names
    wv = np.load(wavelet)
    w_names = np.load(wavelet_names)

    # load deep texton features and names
    df = np.load(deep_feature)
    d_names = np.load(deep_feature_name)

    # create dictionaries of wavelet and deep texton features
    w = dict(zip(w_names,wv))
    d = dict(zip(d_names,df))

    # initialize lists for concatenated features and names
    wavelet_deep = []
    wavelet_deep_names = []

    # loop through wavelet features
    for key, value in w.items():
        # retrieve corresponding deep texton feature
        df_value = cv2.resize(d[key],(size,size),interpolation=cv2.INTER_LINEAR)
        # resize wavelet feature to specified size
        value = cv2.resize(value,(size,size), interpolation=cv2.INTER_LINEAR)
        # concatenate wavelet and deep texton features
        wavelet_deep.append(np.concatenate((value,df_value), axis=2))
        # add image name to list
        wavelet_deep_names.append(key)

    # save concatenated features and names
    np.save(save_path,np.asarray(wavelet_deep).astype('float32'))
    np.save(name_path,np.asarray(wavelet_deep_names))
