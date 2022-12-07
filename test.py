import cv2
import numpy as np
from utils.evaluation import confusion_matrix_parallel
from utils.dataset import get_from_hdf5
from utils.superpixel import superpixel_quatization_parallel, superpixel_clustering
from utils.deep_tools import extract_features
from utils.dataset import get_from_hdf5

np.set_printoptions(precision=3, suppress=True)
np.set_printoptions(linewidth=400)
columns = [0, 1, 2, 3, 4, 5, 6]
best_cfm = None

best_m = 0


def re_sort(cfm, arr, columns):
    if len(arr) >= 7:
        mean = np.mean(np.diag(cfm[:,arr]))
        global best_m
        global best_cfm
        if mean >= best_m:
            best_m = mean
            best_cfm = cfm[:,arr]

        return
    for i in range(len(columns)):
        arr.append(columns[i])
        re_sort(cfm, arr, columns[0:i] + columns[i+1:len(columns)])
        arr.pop()



if __name__ == '__main__':

    data_path = '/cvdata/yungchen/supervised_sonar_segentation/used_data/sonar_512x512.hdf5'
    patches_path = '/cvdata/yungchen/supervised_sonar_segentation/used_data/patches_all.csv'
    net_path = '/home/yungchen/idus_code/models/idus.pth'
    save_path = '/home/yungchen/idus_code/results/idus_cfm.npy'

    features, names = extract_features(patches_path, data_path, if_softmax = True, net_path = net_path)

    preds = np.argmax(np.asarray(features['softmax']), axis=3)

    gts = np.asarray([ get_from_hdf5(data_path,name,'label') for name in names ])

    cfm = confusion_matrix_parallel(gts, preds, 7, 7)

    re_sort(cfm, [], columns)


    np.save(save_path, best_cfm)


    print(best_cfm)