import numpy as np
from time import time
from utils.texton import two_level_deep_texton
if __name__ == '__main__':

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

