import cv2
import numpy as np
from utils.evaluation import confusion_matrix_parallel
from utils.dataset import get_from_hdf5
from utils.superpixel import superpixel_quatization_parallel, superpixel_clustering
np.set_printoptions(precision=3, suppress=True)
np.set_printoptions(linewidth=400)

if __name__ == '__main__':

    data_path = '/cvdata/yungchen/supervised_sonar_segentation/used_data/sonar_512x512.hdf5'
    n_clusters = 7

    deep_wavelet_texton_params = {
        'feature_path': '/cvdata/yungchen/idus/results/features/deep feature/wavelet_deep_texton.npy',
        'names_path': '/cvdata/yungchen/idus/results/features/deep feature/wavelet_deep_texton_names.npy',
        'n_segments':100,
        'compactness': 0.25,
        'seg_comp': 16,
        'pre_size': 512,
        'mask_path': '/home/yungchen/idus_code/results/pseudo_mask.npy',
        'if_decomposed':True,
    }



    feature_params = [
        deep_wavelet_texton_params,


    ]



    for i, params in enumerate(feature_params):
        print('**************************************************')
        print('Iteration',i+1,'Start!')
        print('**************************************************')
        try:
            # preparing
            names = np.load(params['names_path'])
            features = np.load(params['feature_path'])
            ground_truths = np.asarray([get_from_hdf5(data_path, name, 'label') for name in names])

            if_decomposed = params['if_decomposed'] if 'if_decomposed' in params else False
            # superpixel quatization
            sp_fts, sp_ids, segments = superpixel_quatization_parallel(
                features, n_segments = params['n_segments'], compactness = params['compactness'],
                seg_comp = params['seg_comp'], pre_size = params['pre_size'],
                if_normalize= True, if_regularize = True,
                max_bytes = '10M', verbose = 10, n_jobs = -1, if_decomposed=if_decomposed)

            preds = superpixel_clustering(sp_fts, sp_ids, segments, n_clusters)
            # ground truths will not be used in the train, only used for test
            preds = np.asarray([cv2.resize(pred, (512, 512), interpolation=cv2.INTER_NEAREST )
                      for pred in preds])




            if 'mask_path' in params:
                np.save(params['mask_path'], preds)


        except:

            print('Number:',i+1, 'failed!')





