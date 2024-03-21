import os
import math 
import numpy as np 
import pandas as pd 
from scipy.sparse import dok_matrix 
import faiss 
#from .parse_parm import parse_parm
from .io import load_data, load_data_label, froot_prototype_learn
from annoy import AnnoyIndex


def find_BMU(X, W, nBMU=2, nTrees=50):
    N = X.shape[0]
    d = X.shape[1]
    M = W.shape[0]
    
    ## Build ANNoy Index
    t = AnnoyIndex(d, 'euclidean')
    for i in range(M):
        if i % 100 == 0 or i==(M-1): print("\rBuilding Annoy Index: adding prototype %d of %d" % (i+1, M), end="")
        t.add_item(i, W[i,:])
    print("\n")
    t.build(nTrees)

    ## Search index 
    BMU = np.zeros([N,nBMU])
    QE = np.zeros([N,nBMU])
    for i in range(N):
        if i % 1000 == 0 or i==(N-1): print("\rFinding BMUs: searching datum %d of %d" % (i+1, N), end="")
        tmpBMU, tmpQE = t.get_nns_by_vector(X[i,:], nBMU, include_distances=True)
        BMU[i,:] = tmpBMU
        QE[i,:] = tmpQE 
    print("\n")
    BMU = BMU.astype('int')
    
    return BMU, QE



def prototype_recall(M, BMU, XL=None):
    # M = # of prototypes (int)
    # BMU = 2d array (N x nBMU) whose rows contain 1st, 2nd, ... BMUs of each data point
    # XL = array of labels of each data point, if available. If given, len(XL) must equal nrow(BMU)

    # Input checks 
    N = BMU.shape[0]
    if BMU.shape[1] < 2:
        raise ValueError("Cannot perform recall with < 2 BMUs")
    if XL is not None and len(XL) != N:
        raise ValueError("len(XL) != nrow(BMU)")
    
    # Label checks 
    if XL is None:
        label_dict = {}
    else:
        unq_labels = np.unique(XL)
        label_dict = dict(zip(unq_labels, [0]*len(unq_labels)))
        
    ## Recall 
    # Loop through data to add data contribution to its respective: 
    # 1) RF
    # 2) RFSize 
    # 3) CADJ bin 
    # 4) RFLabel dist
    CADJ = dok_matrix((M,M), dtype='int')
    RF = [[] for x in range(M)]
    RFSize = np.zeros([M,])
    RFLDist = [label_dict.copy() for x in range(M)]
    for i in range(N):
        if i % 1000 == 0 or i==(N-1): print("\rRecall: processing datum %d of %d" % (i+1, N), end="")
        RF[BMU[i,0]] = RF[BMU[i,0]] + [i]
        RFSize[BMU[i,0]] += 1
        CADJ[BMU[i,0], BMU[i,1]] += 1
        if XL is not None:
            RFLDist[BMU[i,0]][XL[i]] += 1
    print("\n")
    CADJ = CADJ.tocoo()


    # If labels are given, loop through each populated RF to compute: 
    # 1) RF label distribution
    # 2) RF winning label (plurality vote)
    # 3) RF label purity score 
    RFL = [None for x in range(M)]
    RFLPurity = np.zeros([M,])
    RFLPurityUOA = 0.0 
    RFLPurityWOA = 0.0
    if XL is not None:
        denominatorUOA = 0.0    
        for i in range(M):
            if i % 100 == 0 or i==(M-1): print("\rRecall: processing labels for RF %d of %d" % (i+1, M), end="")
            # If RF empty, skip 
            if RFSize[i] == 0:
                continue
            # Normalize RFLDist to sum to 1 
            #RFLDist[i] = [RFLDist[i][x] / RFSize[i] for x in RFLDist[i]]
            RFLDist[i] = {k:v/RFSize[i] for k,v in RFLDist[i].items()}
            # Find winner
            RFL[i] = max(RFLDist[i], key=RFLDist[i].get)
            # Compute RF Purity 
            RFLPurity[i] = 1.0 - math.sqrt(1.0 - math.sqrt(RFLDist[i][RFL[i]])) # 1 - Hellinger Distance(ideal, observed)
            # Add Purity contribution to weighted & unweighted averages 
            RFLPurityUOA += RFLPurity[i]
            denominatorUOA += 1.0
            RFLPurityWOA += RFSize[i] * RFLPurity[i]
        print("\n")
        RFLPurityUOA /= denominatorUOA
        RFLPurityWOA /= np.sum(RFSize)
    
    return CADJ, RF, RFSize, RFLDist, RFL, RFLPurity, RFLPurityUOA, RFLPurityWOA

def faiss_kmeans(X, M, niter_monitor=20, niter_max = 1000, BMU_conv_tol = 0.01, seed=123, num_threads=-1):
    # X = data
    # M = number of centroids 
    # niter_monitor = number of iterations between monitoring rounds
    # niter_max = max number of iterations
    # BMU_conv_tol = proportion of BMUs that stop changing at convergence 
    # seed = random seed 

    # Store sample dimensions 
    N = X.shape[0]
    d = X.shape[1]

    ## Setup FAISS object for Kmeans 
    verbose = True
    if num_threads > 0:
        faiss.omp_set_num_threads(num_threads)
    km = faiss.Kmeans(d, M, niter=niter_monitor, verbose=verbose, min_points_per_centroid = 1, max_points_per_centroid = N, seed = 123)
    
    # First round of training 
    print("*** FAISS K-Means ***")

    km.train(X)
    cum_iter = niter_monitor
    prev_QE2, prev_BMU = km.index.search(X, 2) # squared L2 quantization error, BMU 
    delBMU = 1.0 
    print("\nIter %d: delBMU = %.3f" % (cum_iter, delBMU))

    # Retrain until convergence 
    while cum_iter <= niter_max and delBMU > BMU_conv_tol:
        km.train(x=X, init_centroids=km.centroids)
        cum_iter += niter_monitor
        QE2, BMU = km.index.search(X, 2) # squared L2 quantization error, BMU 
        delBMU = np.sum((prev_BMU - BMU) != 0) / np.prod(BMU.shape)
        print("\nIter %d: delBMU = %.3f" % (cum_iter, delBMU))
        #if delBMU < proto_conv_tol: flag_conv = True 
        prev_QE2 = QE2
        prev_BMU = BMU 
    print("*** END FAISS K-Means ***\n")

    
    return km.centroids, BMU, np.sqrt(QE2)


def prototype_learn(parm):

    # Load data, store sample dimensions 
    X = load_data(parm)
    XL = load_data_label(parm)
    N = X.shape[0]
    d = X.shape[1]

    # Extract number of prototypes from parm 
    M = parm["proto_m"]

    W, BMU, QE2 = faiss_kmeans(X, M, 
                               niter_monitor=parm["proto_iter_monitor"], 
                               seed=parm["proto_seed"], 
                               num_threads=parm['num_threads'])

    # ## Setup FAISS object for Kmeans 
    # verbose = True
    # km = faiss.Kmeans(d, M, niter=parm["proto_iter_monitor"], verbose=verbose, min_points_per_centroid = 1, max_points_per_centroid = N, seed = parm["proto_seed"])
    
    # # First round of training 
    # print("*** FAISS K-Means ***")

    # km.train(X)
    # cum_iter = parm["proto_iter_monitor"]
    # prev_QE2, prev_BMU = km.index.search(X, 2) # squared L2 quantization error, BMU 
    # delBMU = 1.0 
    # print("\nIter %d: delBMU = %.3f" % (cum_iter, delBMU))

    # # Retrain until convergence 
    # while cum_iter <= parm["proto_iter_max"] and delBMU > parm["proto_conv_tol"]:
    #     km.train(x=X, init_centroids=km.centroids)
    #     cum_iter += parm["proto_iter_monitor"]
    #     QE2, BMU = km.index.search(X, 2) # squared L2 quantization error, BMU 
    #     delBMU = np.sum((prev_BMU - BMU) != 0) / np.prod(BMU.shape)
    #     print("\nIter %d: delBMU = %.3f" % (cum_iter, delBMU))
    #     #if delBMU < proto_conv_tol: flag_conv = True 
    #     prev_QE2 = QE2
    #     prev_BMU = BMU 
    # print("*** END FAISS K-Means ***\n")

    
    ## Recall 
    CADJ, RF, RFSize, RFLDist, RFL, RFLPurity, RFLPurityUOA, RFLPurityWOA = prototype_recall(M, BMU, XL)
    # CADJ = dok_matrix((M,M), dtype='int')
    # RF = [[] for x in range(M)]
    # RFSize = np.zeros([M,])
    # for i in range(N):
    #     if i % 1000 == 0 or i==(N-1): print("\rRecall: processing datum %d of %d" % (i+1, N), end="")
    #     CADJ[BMU[i,0], BMU[i,1]] += 1
    #     RF[BMU[i,0]] = RF[BMU[i,0]] + [i]
    #     RFSize[BMU[i,0]] += 1
    # print("\n")

    ## Save 
    #froot = os.path.join(parm["output_dir"], parm["project_root"] + "_" + parm['proto_method'] + str(M))
    froot = froot_prototype_learn(parm)

    # Prototypes 
    fname = froot + "_W.pkl"
    #pd.to_pickle(km.centroids, fname)
    pd.to_pickle(W, fname)
    print("Prototypes written:\n %s" % fname)

    # BMU 
    fname = froot + "_BMU.pkl"
    pd.to_pickle(BMU, fname)
    print("BMUs written:\n %s" % fname)
    # QE 
    fname = froot + "_QE.pkl"
    pd.to_pickle(np.sqrt(QE2), fname)
    print("Quantization Errors written:\n %s" % fname)
    # CADJ 
    fname = froot + "_CADJ.pkl"
    pd.to_pickle(CADJ, fname)
    print("CADJ written:\n %s" % fname)
    # RF 
    fname = froot + "_RF.pkl"
    pd.to_pickle(RF, fname)
    print("Receptive Fields written:\n %s" % fname)
    # RF Size 
    fname = froot + "_RFSize.pkl"
    pd.to_pickle(RFSize, fname)
    print("Receptive Field Sizes written:\n %s" % fname)
    # RF Label Dist
    fname = froot + "_RFLDist.pkl"
    pd.to_pickle(RFLDist, fname)
    print("Receptive Field Label Distribution written:\n %s" % fname)
    # RF Winning Labels
    fname = froot + "_RFL.pkl"
    pd.to_pickle(RFL, fname)
    print("Receptive Field Labels written:\n %s" % fname)
    # RF Label Purities 
    fname = froot + "_RFLPurity.pkl"
    pd.to_pickle(RFLPurity, fname)
    print("Receptive Field Label Distribution Purities:\n %s" % fname)
    # RF Label Unweighted Overall Purities 
    fname = froot + "_RFLPurityUOA.pkl"
    pd.to_pickle(RFLPurityUOA, fname)
    print("Receptive Field Label Purity UOA written:\n %s" % fname)
    # RF Label Weighted Overall Purities 
    fname = froot + "_RFLPurityWOA.pkl"
    pd.to_pickle(RFLPurityWOA, fname)
    print("Receptive Field Label Purity WOA written:\n %s" % fname)

    return 

def prototype_fwdmap_mean(X, RF):
    N = X.shape[0]
    M = len(RF)
    d = 1 
    if len(X.shape) > 1:
        d = X.shape[1]
    else:
        X = X.reshape((N,1))
    
    W = np.zeros(shape=(M,d), dtype=X.dtype)
    
    for i in range(M):
        if RF[i] is not None and len(RF[i])>0:
            W[i,:] = X[RF[i],:].mean(axis=0)
        
    
    return W
    
