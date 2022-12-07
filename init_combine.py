import numpy as np
import cv2
if __name__ == '__main__':
    wavelet = '/home/yungchen/idus_code/results/wavelet_features.npy'
    wavelet_names = '/home/yungchen/idus_code/results/wavelet_names.npy'
    deep_feature = '/home/yungchen/idus_code/results/deep_texton.npy'
    deep_feature_name = '/home/yungchen/idus_code/results/deep_texton_names.npy'

    save_path = '/home/yungchen/idus_code/results/wavelet_deep_texton.npy'
    name_path = '/home/yungchen/idus_code/results/wavelet_deep_texton_names.npy'

    size = 256
    wv = np.load(wavelet)
    w_names = np.load(wavelet_names)

    df = np.load(deep_feature)
    d_names = np.load(deep_feature_name)

    w = dict(zip(w_names,wv))
    d = dict(zip(d_names,df))

    wavelet_deep = []
    wavelet_deep_names = []
    for key, value in w.items():

        df_value = cv2.resize(d[key],(size,size),interpolation=cv2.INTER_LINEAR)
        value = cv2.resize(value,(size,size), interpolation=cv2.INTER_LINEAR)
        wavelet_deep.append(np.concatenate((value,df_value), axis=2))
        wavelet_deep_names.append(key)


    np.save(save_path,np.asarray(wavelet_deep).astype('float32'))
    np.save(name_path,np.asarray(wavelet_deep_names))
