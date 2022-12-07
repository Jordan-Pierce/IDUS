import cv2, pywt
import numpy as np
from joblib import  Parallel,delayed
from utils.basic_tools import sliding_window
from utils.dataset import get_from_hdf5, get_names
from time import time
def wavelet_coeff(window):
    LL = window
    features = []
    for i in range(5):
        coeffs2 = pywt.dwt2(LL, 'bior1.3')
        LL, scales = coeffs2

        for s in scales:
            rms = np.sqrt(np.mean(s ** 2))
            features.append(rms)
    rms = np.sqrt(np.mean(LL ** 2))
    features.append(rms)

    return np.asarray(features)

def wavelet(img, window_size = 31):

    res = sliding_window(img, (window_size, window_size),func=wavelet_coeff)

    return res

def extract_feature(images, func, func_params=None, pad_size = 0, pre_size = 256, aft_size = 512,
                    n_jobs = -1,
                    verbose = 0,
                    max_nbytes = '10M',
                    print_info = True):

    if func_params == None:
        func_params = {}

    images = [cv2.resize(image, (pre_size, pre_size),interpolation=cv2.INTER_LINEAR) for image in images]

    images = [np.pad(image, ((pad_size, pad_size), (pad_size, pad_size)), 'reflect') for image in images]

    start = time()
    feature_all = Parallel(n_jobs= n_jobs, max_nbytes= max_nbytes, verbose= verbose)(
        delayed(func)(image,**func_params) for image in images)
    end = time()

    if print_info:
        print('Extact Feature Time:', end - start)

    feature_all = [feature[ pad_size:pad_size + pre_size, pad_size:pad_size + pre_size]
                   for feature in feature_all]

    # resize
    feature_all = [cv2.resize(feature, (aft_size, aft_size), interpolation=cv2.INTER_LINEAR) for feature in feature_all]

    feature_all = np.asarray(feature_all).astype('float32')

    return feature_all



if __name__ == '__main__':
    data_path = '/cvdata/yungchen/supervised_sonar_segentation/used_data/sonar_512x512.hdf5'
    ft_path = '/home/yungchen/idus_code/results/wavelet_features.npy'
    name_path = '/home/yungchen/idus_code/results/wavelet_names.npy'

    pad_size = 10
    pre_size = 256
    aft_size = 512

    wavelet_param = {'window_size': 16}

    names = get_names(data_path)

    images = [get_from_hdf5(data_path, name, 'data') for name in names]

    ft_all = extract_feature(images, wavelet,
                             func_params= wavelet_param,
                             pad_size=pad_size,
                             pre_size= pre_size,
                             aft_size= aft_size,
                             n_jobs=-1)



    np.save(name_path, names)
    np.save(ft_path, ft_all)

