import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import confusion_matrix, normalized_mutual_info_score

def normalize_mutual_informaton(pred, gt):

    index = gt != 0

    return normalized_mutual_info_score(gt[index].reshape(-1) - 1, pred[index].reshape(-1))

def cfm_ratio(cfm):

    cf_sum = cfm.sum(axis=1)
    cf_sum = np.expand_dims(cf_sum, axis=1)
    cf_sum = np.repeat(cf_sum, cfm.shape[1], 1)

    return cfm / (cf_sum + 1)

def cf_matrix(gt, pred, ignore_index = None, pdlb = None, gtlb = None):


    if pdlb != None and gtlb != None:
        labels = np.arange(max(pdlb,gtlb))

    else:
        labels = None

    if ignore_index != None:
        index = gt != ignore_index

        return confusion_matrix(gt[index].reshape(-1) - 1, pred[index].reshape(-1), labels=labels)
    else:
        return confusion_matrix(gt - 1, pred, labels=labels)

def confusion_matrix_parallel(gts, preds,pdlb,gtlb):

    cfms = np.asarray(Parallel(n_jobs=-1, max_nbytes='10M')(
        delayed(cf_matrix)(gt,pred, ignore_index = 0, pdlb = pdlb, gtlb = gtlb )
                for gt, pred in zip(gts, preds)))

    return cfm_ratio(np.sum(cfms,axis=0))