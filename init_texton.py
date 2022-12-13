import numpy as np
from time import time
from utils.texton import two_level_deep_texton


if __name__ == '__main__':

    """
    Extracts deep texton features from an image

    This function extracts deep texton features from a given image using a pre-trained model. It first applies a sliding window on the image to compute global textons, and then uses the global textons to compute local textons. The final output is a list of deep texton features and their corresponding names.

    Args:
    patches_path (str): path to the patches csv file
    data_path (str): path to the image data hdf5 file
    ly_names (list): list of names of layers to extract features from
    cb_settings (list): list of settings for the convolutional blocks
    local_texton_num (int): number of local textons to compute
    global_texton_num (int): number of global textons to compute
    sld_winsize (int): size of the sliding window for computing global textons
    target_size (int): target size for rescaling the extracted features

    Returns:
    tuple: a tuple containing the list of deep texton features and their corresponding names
    """

    start = time()
    
    # data and patches path
    data_path = '/cvdata/yungchen/supervised_sonar_segentation/used_data/sonar_512x512.hdf5'
    patches_path = '/cvdata/yungchen/supervised_sonar_segentation/used_data/patches_all.csv'
    save_path = '/home/yungchen/idus_code/results/deep_texton.npy'
    names_path = '/home/yungchen/idus_code/results/deep_texton_names.npy'

    # net_path = None

    # settings for extract layer features
    ly_names = ['model.encoder.layer1', 'model.encoder.layer3']
    cb_settings =[8, 16]
    
    # feature rescale size
    target_size = 128

    # local and global texton number
    local_texton_num = 128
    global_texton_num = 128
    
    # sliding window size for computiing global textons
    sld_winsize = 10

    probs, names = two_level_deep_texton(patches_path, data_path, ly_names, cb_settings,
                          local_texton_num = local_texton_num, global_texton_num=global_texton_num,
                          sld_winsize=sld_winsize, target_size=target_size)


    np.save(save_path,probs)
    np.save(names_path, np.asarray(names))

    print(time()-start)

