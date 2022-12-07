import cv2
import numpy as np
from joblib import Parallel, delayed
from utils.superpixel import superpixel_clustering, superpixel_quatization_parallel, superpixel_mapping
from utils.basic_tools import sliding_window, average
from utils.deep_tools import extract_features_train



def update_pseudo_mask(net,
                       images,
                       n_clusters,
                       n_segments = 256,
                       compactness = 1.0,
                       img_seg_size = 512,
                       seg_comp = -1,
                       sld_winsize = 5):
    # combine settings
    ft_all = extract_features_train(images, net)
    # ft_all = [cv2.resize(feature, (target_size,target_size),interpolation=cv2.INTER_LINEAR) for feature in ft_all]

    # ft_all = features_combination(ft_all,
    #                               dict(zip(ly_names,cb_settings)),
    #                               target_size=target_size)
    # ft_all = regularize(np.asarray(ft_all))




    features = np.asarray( Parallel(n_jobs=-1, max_nbytes='10M')(
        delayed(sliding_window)(
            ft,
            (sld_winsize, sld_winsize),
            func = average) for ft in ft_all))

    img_sizes = images.shape[1:3]
    features = np.asarray([cv2.resize(feature,
                           img_sizes,
                           interpolation=cv2.INTER_LINEAR) for feature in features])

    sp_fts, sp_ids, segments = superpixel_quatization_parallel(features,
                                    n_segments = n_segments,
                                    pre_size = img_seg_size,
                                    seg_comp = seg_comp,
                                    compactness = compactness,
                                    if_normalize= True,
                                    if_regularize = True,
                                    n_jobs = -1,
                                    max_bytes = '10M',
                                    verbose = 0,
                                    if_decomposed = False)



    pseudo_masks = superpixel_clustering(sp_fts, sp_ids, segments, n_clusters,
                                         n_jobs=-1, max_nbytes='10M', verbose=0)


    return np.asarray(pseudo_masks)

