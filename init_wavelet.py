import cv2, pywt
import numpy as np
from joblib import  Parallel,delayed
from utils.basic_tools import sliding_window
from utils.dataset import get_from_hdf5, get_names
from time import time

import pywt
import numpy as np

def wavelet_coeff(window):
    """
    Computes the wavelet coefficients for a given window.

    This function uses the bi-orthogonal wavelet with a decomposition level of 5 and a wavelet filter with
    coefficients [1, 3, 3, 1] / 8. The root mean square (RMS) of each wavelet subband is computed and returned as a
    feature vector.

    Args:
        window (np.ndarray): The window to compute wavelet coefficients for.

    Returns:
        np.ndarray: The wavelet coefficient feature vector.
    """
    LL = window
    features = []

    # Decompose the window using the specified wavelet with a decomposition level of 5.
    for i in range(5):
        coeffs2 = pywt.dwt2(LL, 'bior1.3')
        LL, scales = coeffs2

        # Compute the RMS of each wavelet subband.
        for s in scales:
            rms = np.sqrt(np.mean(s ** 2))
            features.append(rms)

    # Compute the RMS of the low-low subband.
    rms = np.sqrt(np.mean(LL ** 2))
    features.append(rms)

    return np.asarray(features)


def extract_feature(images, func, func_params=None, pad_size = 0, pre_size = 256, aft_size = 512,
                    n_jobs = -1,
                    verbose = 0,
                    max_nbytes = '10M',
                    print_info = True):
    """
    This function extracts features from a list of images using the specified function and its parameters.

    Args:
    images: a list of images as numpy arrays
    func: the function to be used for feature extraction
    func_params (optional): a dictionary of parameters to be passed to the function
    pad_size: an integer representing the size of padding to be applied to the images
    pre_size: an integer representing the size of the images before padding
    aft_size: an integer representing the size of the images after padding
    n_jobs: an integer representing the number of jobs to run in parallel (default=-1)
    verbose: an integer representing the level of verbosity in the output (default=0)
    max_nbytes: a string representing the maximum number of bytes to be used (default='10M')
    print_info: a boolean indicating whether to print information about the feature extraction process (default=True)

    Returns:
    a numpy array of extracted features
    """

    # If func_params is not provided, set it to an empty dictionary
    if func_params == None:
        func_params = {}

    # Resize the images to the specified pre_size
    images = [cv2.resize(image, (pre_size, pre_size),interpolation=cv2.INTER_LINEAR) for image in images]

    # Pad the images using the specified pad_size
    images = [np.pad(image, ((pad_size, pad_size), (pad_size, pad_size)), 'reflect') for image in images]

    # Extract features using the provided function in parallel
    start = time()
    feature_all = Parallel(n_jobs= n_jobs, max_nbytes= max_nbytes, verbose= verbose)(
        delayed(func)(image,**func_params) for image in images)
    end = time()

    # Print the extraction time if print_info is set to True
    if print_info:
        print('Extact Feature Time:', end - start)

    # Crop the padded area from the extracted features
    feature_all = [feature[ pad_size:pad_size + pre_size, pad_size:pad_size + pre_size]
                   for feature in feature_all]

    # Resize the features to the specified aft_size
    feature_all = [cv2.resize(feature, (aft_size, aft_size), interpolation=cv2.INTER_LINEAR) for feature in feature_all]

    # Convert the list of features to a numpy array
    feature_all = np.asarray(feature_all).astype('float32')

    return feature_all




if __name__ == '__main__':

    """
    Extracts wavelet features from a dataset stored in an HDF5 file and saves the feature vectors and names to npy files.

    Args:
    data_path (str): path to the HDF5 file containing the dataset
    ft_path (str): path to save the extracted wavelet feature vectors to
    name_path (str): path to save the names of the dataset images to
    pad_size (int): padding size for the wavelet transform
    pre_size (int): pre-processing size for the wavelet transform
    aft_size (int): post-processing size for the wavelet transform

    Returns:
    None
    """
    data_path = '/cvdata/yungchen/supervised_sonar_segentation/used_data/sonar_512x512.hdf5'
    ft_path = '/home/yungchen/idus_code/results/wavelet_features.npy'
    name_path = '/home/yungchen/idus_code/results/wavelet_names.npy'

    pad_size = 10
    pre_size = 256
    aft_size = 512

    wavelet_param = {'window_size': 16}

    names = get_names(data_path)

    # extract images from HDF5 file
    images = [get_from_hdf5(data_path, name, 'data') for name in names]

    # extract wavelet features from images
    ft_all = extract_feature(images, wavelet,
                             func_params= wavelet_param,
                             pad_size=pad_size,
                             pre_size= pre_size,
                             aft_size= aft_size,
                             n_jobs=-1)

    # save names and feature vectors to npy files
    np.save(name_path, names)
    np.save(ft_path, ft_all)
