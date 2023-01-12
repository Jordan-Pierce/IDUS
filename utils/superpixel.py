import cv2
import numpy as np
from sklearn.cluster import KMeans
from skimage.segmentation import slic
from joblib import Parallel, delayed
from utils.basic_tools import normalize, regularize, decomposed_and_resize, extend_array, unextend_array
from utils.dataset import get_from_hdf5
from time import time
from sklearn.utils.extmath import squared_norm
from tqdm import tqdm
def superpixel_center_position(segment):

    sp_num = np.max(segment)


    sp_pos = []

    for sp_id in range(sp_num + 1):

        # if have this superpixel
        sp_area = segment == sp_id
        if np.sum(sp_area) > 0:

            pos = np.where(segment==sp_id)
            pr = np.mean(pos[0])
            pc = np.mean(pos[1])

            sp_pos.append(np.asarray([pr, pc]))

    return np.asarray(sp_pos)


def superpixel_quatization(feature, segment):

    shape = feature.shape[0:2]
    sp_num = np.max(segment)

    segment = cv2.resize(segment, shape, interpolation = cv2.INTER_NEAREST)

    sp_fts = []
    sp_ids = []

    for sp_id in range(sp_num + 1):

        # if have this superpixel
        sp_area = segment == sp_id
        if np.sum(sp_area) > 0:
            sp_fts.append(np.mean(feature[sp_area], axis=0))
            sp_ids.append(sp_id)


    return np.asarray(sp_fts), np.asarray(sp_ids)

def superpixel_quatization_parallel(features,
                                    n_segments = 256,
                                    pre_size = 512,
                                    seg_comp = -1,
                                    compactness = 1.0,
                                    if_normalize= True,
                                    if_regularize = True,
                                    n_jobs = -1,
                                    max_bytes = '10M',
                                    verbose = 10,
                                    if_decomposed = False):
    # pre-process feature
    if if_normalize:
        features = normalize(features)
    if if_regularize:
        features = regularize(features)

    # get superpixel segments



    start = time()
    sp_features = decomposed_and_resize(features, pre_size, seg_comp)

    segments = Parallel(n_jobs=n_jobs, max_nbytes=max_bytes, verbose= verbose)(
        delayed(slic)(feature.astype('float64'), n_segments=n_segments,
                                                compactness=compactness,
                                                multichannel=True)
        for feature in sp_features)

    print('Superpixel Quatization:', time() - start)
    # decomposed feature
    if if_decomposed:
        features = sp_features

    # superpixel quantization
    results = Parallel(n_jobs=n_jobs, max_nbytes=max_bytes, verbose= verbose)(
        delayed(superpixel_quatization)(feature, segment)
        for feature, segment in zip(features, segments))
    sp_fts, sp_ids = zip(*results)

    return sp_fts, sp_ids, segments


def superpixel_mapping(sp_lb, sp_id, segment):

    mask = np.zeros(segment.shape).astype('int32')

    for (lb, id) in zip(sp_lb, sp_id):
        mask[segment == id] = lb

    return mask

def superpixel_clustering(superpixel_features,
                          superpixel_ids,
                          segments,
                          n_clusters,
                          n_jobs=-1, max_nbytes='10M', verbose = 0):
    sp_fts_ext, sp_fts_extid = extend_array(superpixel_features)
    sp_lbs_ext = KMeans(n_clusters=n_clusters, algorithm='full').fit_predict(sp_fts_ext)
    # sp_lbs_ext = KMeans(n_clusters=n_clusters, n_jobs=-1, algorithm='full').fit_predict(sp_fts_ext)
    # sp_lbs_ext = BayesianGaussianMixture(n_components = n_clusters).fit_predict(sp_fts_ext)

    sp_lbs = unextend_array(sp_lbs_ext, sp_fts_extid)

    preds = Parallel(n_jobs=n_jobs, max_nbytes=max_nbytes, verbose=verbose)(
        delayed(superpixel_mapping)(sp_lb, sp_id, segment)
        for (sp_lb, sp_id, segment) in zip(sp_lbs, superpixel_ids, segments))

    return np.asarray(preds)

def eps(x, epsilon = 1e-6):

    return np.where(x != 0, x, epsilon)

def pdist(pos):
    posDist = []
    for p in pos:
        d = []
        for posn in p:
            dn = np.linalg.norm(posn[np.newaxis, :] - p, axis= 1)
            d.append(dn)
        posDist.append(eps(np.asarray(d)))

    return np.asarray(posDist)

def getPos(ext_id):
    posIdx = []
    imgIdx = 0
    for rg in ext_id:
        pi_id = np.asarray(range(rg[1] - rg[0]))
        imgId = np.full(pi_id.shape,imgIdx)
        posIdx.append(np.concatenate((imgId[:,np.newaxis], pi_id[:,np.newaxis]), axis=1))

        imgIdx += 1

    return np.concatenate(tuple(posIdx), axis=0)

def plfcim_constant(x, pos, ext_id,C = 7,
           a=14,
           b = 0.3,
           m= 1.8,
           q=2.2,
           epsilon = 1e-6,
           iter=100):


    # initialization
    # randomly select input data points as the C initial cluster
    idx = np.random.choice(x.shape[0], C, replace=False)
    center = x[idx]

    # distance between data points and centers
    D = np.zeros((x.shape[0], C))
    # typical values
    T = np.ones((x.shape[0], C))
    # membership
    U = np.ones((x.shape[0],C)) / C

    Uold = np.copy(U)

    # pbar = tqdm(total=iter)
    for i in range(iter):
        # pbar.update(1)

        # compute distant (||x - c||2)^2
        for k in range(C):
            norm2 = np.linalg.norm(x - np.expand_dims(center[k, :],axis=0), axis=1)
            D[:,k] =  np.power(norm2,2)
        D = eps(D)


        # compute membership
        for n, xn in enumerate(x):
            for c in range(C):
                dnc = D[n,c]
                eSum = 0.0
                for k in range(C):
                    dnk = D[n,k]
                    eSum +=  (dnc /dnk )**(1.0/(m - 1))

                U[n,c] = 1.0 / eps(eSum)


        # compute gamma value for computing typicality values
        Um = U**m
        gamma = np.zeros(C)
        for c in range(C):
            gammaC = np.sum(Um[:,c]*D[:,c]) / np.sum(Um[:,c])
            gamma[c] = gammaC

        # compute typical value
        for n, xn in enumerate(x):
            for c in range(C):
                dnc = D[n, c]
                tcn = 1.0 / (1.0 + ( (b/gamma[c]) * dnc  )**( 1 / (q-1) ) )
                T[n,c] = tcn

        # compute center

        Uold = np.copy(U)

        centerOld = center
        center = np.zeros((C, x.shape[1]))

        aUbT = a*U + b*T
        for c in range(C):
            aUbTc = np.expand_dims(aUbT[:,c],axis=1)

            center[c] =  np.sum(aUbT[:,c,np.newaxis]*x, axis=0) / np.sum(aUbTc, axis=0)

        residual = squared_norm( centerOld - center)
        # pbar.set_description("residual: %f.8" % residual)
        if residual < epsilon:
            break


    map = U*T

    return np.argmax(map,axis=1)



def plfcim(x, pos, ext_id,C = 7,
           a=14,
           b = 0.3,
           m= 1.8,
           q=2.2,
           epsilon = 1e-6,
           iter=100):

    # compute superpixel distances
    posDist = pdist(pos)
    posIdx = getPos(ext_id)


    # initialization
    # randomly select input data points as the C initial cluster
    idx = np.random.choice(x.shape[0], C, replace=False)
    center = x[idx]

    # distance between data points and centers
    D = np.zeros((x.shape[0], C))
    # typical values
    T = np.ones((x.shape[0], C))
    # membership
    U = np.ones((x.shape[0],C)) / C

    Uold = np.copy(U)

    # pbar = tqdm(total=iter)
    for i in range(iter):
        # pbar.update(1)

        # compute distant (||x - c||2)^2
        for k in range(C):

            norm2 = np.linalg.norm(x - np.expand_dims(center[k, :],axis=0), axis=1)
            D[:,k] =  np.power(norm2,2)
        D = eps(D)

        # compute local information term G
        G = np.zeros((x.shape[0], C) )
        Uuext = unextend_array(Uold, ext_id)
        Duext = unextend_array(D, ext_id)
        for n, xn in enumerate(x):
            for c in range(C):
                pId = posIdx[n]
                pnk = posDist[pId[0]][pId[1]]
                pnk = np.concatenate((pnk[0:pId[1]], pnk[pId[1] + 1:]), axis=0)
                uck = Uuext[pId[0]][:,c]
                uck = np.concatenate((uck[0:pId[1]], uck[pId[1] + 1:]), axis=0)
                dck = Duext[pId[0]][:,c]
                dck = np.concatenate((dck[0:pId[1]], dck[pId[1] + 1:]), axis=0)


                Gcn = np.sum((1.0 / (1.0 + pnk))*((1.0-uck)**m)*dck)
                G[n,c] = Gcn



        # compute membership
        for n, xn in enumerate(x):
            for c in range(C):
                dnc = D[n,c]
                gnc = G[n,c]
                eSum = 0.0
                for k in range(C):
                    dnk = D[n,k]
                    gnk = G[n,k]
                    eSum +=  ((dnc + gnc)/(dnk + gnk))**(1.0/(m - 1))

                U[n,c] = 1.0 / eps(eSum)


        # compute gamma value for computing typicality values
        Um = U**m
        gamma = np.zeros(C)
        for c in range(C):
            gammaC = np.sum(Um[:,c]*D[:,c]) / np.sum(Um[:,c])
            gamma[c] = gammaC

        # compute typical value
        for n, xn in enumerate(x):
            for c in range(C):
                dnc = D[n, c]
                tcn = 1.0 / (1.0 + ( (b/gamma[c]) * dnc  )**( 1 / (q-1) ) )
                T[n,c] = tcn

        # compute center

        Uold = np.copy(U)

        centerOld = center
        center = np.zeros((C, x.shape[1]))

        aUbT = a*U + b*T
        for c in range(C):
            aUbTc = np.expand_dims(aUbT[:,c],axis=1)

            center[c] =  np.sum(aUbT[:,c,np.newaxis]*x, axis=0) / np.sum(aUbTc, axis=0)

        residual = squared_norm( centerOld - center)
        # pbar.set_description("residual: %f.8" % residual)
        if residual < epsilon:
            break


    map = U*T

    return np.argmax(map,axis=1)








def plfcim_superpixel(superpixel_features,
                          superpixel_ids,
                          segments,
                          n_clusters,
                          n_jobs=-1, max_nbytes='10M', verbose = 0):

    pos = [superpixel_center_position(segment) for segment in segments]

    sp_fts_ext, sp_fts_extid = extend_array(superpixel_features)

    sp_lbs_ext = plfcim(sp_fts_ext, pos, sp_fts_extid, C=n_clusters)
    # sp_lbs_ext = BayesianGaussianMixture(n_components = n_clusters).fit_predict(sp_fts_ext)

    sp_lbs = unextend_array(sp_lbs_ext, sp_fts_extid)

    preds = Parallel(n_jobs=n_jobs, max_nbytes=max_nbytes, verbose=verbose)(
        delayed(superpixel_mapping)(sp_lb, sp_id, segment)
        for (sp_lb, sp_id, segment) in zip(sp_lbs, superpixel_ids, segments))

    return np.asarray(preds)