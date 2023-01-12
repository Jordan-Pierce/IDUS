import numpy as np
from time import time
from utils.texton import two_level_deep_texton


if __name__ == '__main__':

    """
    Extracts deep texton features from an image

    This function extracts deep texton features from a given image using a 
    pre-trained model. It first applies a sliding window on the image to 
    compute global textons, and then uses the global textons to compute 
    local textons. The final output is a list of deep texton features and 
    their corresponding names. 
    """

    start = time()
    
    # data and patches path
    data_path = './dataset/sonar_512x512.hdf5'
    patches_path = './dataset/patches_all.csv'
    save_path = './results/deep_texton.npy'
    names_path = './results/deep_texton_names.npy'

    # ?
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

    probs, names = two_level_deep_texton(patches_path,
                                         data_path,
                                         ly_names,
                                         cb_settings,
                                         local_texton_num = local_texton_num,
                                         global_texton_num=global_texton_num,
                                         sld_winsize=sld_winsize,
                                         target_size=target_size)

    # save names and feature vectors to npy files
    np.save(save_path, probs)
    np.save(names_path, np.asarray(names))

    print(time()-start)

