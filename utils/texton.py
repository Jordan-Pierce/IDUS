import numpy as np
from sklearn.cluster import KMeans
from joblib import Parallel, delayed
from utils.basic_tools import extend_array, unextend_array
from utils.deep_tools.basic_tools import extract_features
from utils.basic_tools import features_combination, regularize, sliding_window, CountBin

def local_texton(feature, **kwargs):
    #n_clusters=n_clusters, max_iter=20, n_init=n_init
    shape = feature.shape

    feature = feature.reshape((-1,shape[-1]))

    cluster_model = KMeans(**kwargs)

    texton = cluster_model.fit_predict(feature).reshape(shape[0:2])


    return texton, cluster_model.cluster_centers_

def local_texton_parallel(feature_all, **kwargs):

    return zip(*Parallel(n_jobs=-1,
                       max_nbytes='10M')(delayed(local_texton)(feature, **kwargs)
                                         for feature in feature_all))

def global_texton(local_textons, centrodis,**kwargs):

    ct_ext, ct_ext_idx = extend_array(centrodis)

    ct_lbs = unextend_array(KMeans(**kwargs).fit_predict(np.asarray(ct_ext)), ct_ext_idx)

    global_textons = []
    for lc_txt, ct_lb in zip(local_textons, ct_lbs):

        gl_txt = np.zeros(lc_txt.shape).astype('int32') # global texton

        for txt_lb, glb in enumerate(ct_lb):
            gl_txt[lc_txt == txt_lb] = glb
        global_textons.append(gl_txt)

    return np.asarray(global_textons)


def two_level_deep_texton(patches_path, data_path, ly_names, cb_settings,
                          local_texton_num=128, global_texton_num=128,
                          sld_winsize=3, target_size=256, net_path=None):
    # combine settings

    ft_all, names = extract_features(patches_path, data_path, if_softmax=False,
                     ly_names=ly_names, net=None, net_path=net_path)


    ft_all = features_combination(ft_all,
                                  dict(zip(ly_names, cb_settings)),
                                  target_size=target_size)
    ft_all = regularize(np.asarray(ft_all))

    local_textons, centroids = local_texton_parallel(ft_all,
                                                     n_clusters=local_texton_num,
                                                     max_iter=20,
                                                     n_init=1)

    global_textons = global_texton(local_textons,
                                   centroids,
                                   n_clusters=global_texton_num,
                                   max_iter=20,
                                   n_init=10)

    hists = np.asarray(Parallel(n_jobs=-1, max_nbytes='10M')(
        delayed(sliding_window)(g_texton, (sld_winsize, sld_winsize),
                                func=CountBin(global_texton_num)) for g_texton in global_textons))

    # histogram to probability
    hists_sum = np.sum(hists, axis=3)
    hists_sum = np.repeat(hists_sum[:, :, :, np.newaxis], hists.shape[-1], axis=3)
    probs = hists / hists_sum

    return probs, names