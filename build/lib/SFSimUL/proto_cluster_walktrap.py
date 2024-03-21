# Cluster 
import numpy as np 
import pandas as pd 
import os 
import igraph 
import math 
#from .proto_learn_faiss import froot_prototype_learn
from .io import load_prototype_learn_products, froot_cluster_walktrap

def prototype_cluster_walktrap(parm):
    #parm = parse_parm(parm_file)

    print("*** Prototype Clustering via Walktrap ***")

    print("Loading prototype learning products ... ", end="")
    plearn = load_prototype_learn_products(parm)
    print("done")

    #froot = os.path.join(parm["output_dir"], parm["project_root"] + "_" + parm["proto_method"] + str(parm["proto_m"]))
    # froot = froot_prototype_learn(parm)

    # # Construct CADJ file path, load 
    # print("Loading CADJ file ... ", end="")
    # cadj_file = froot + "_CADJ.pkl"
    # try:
    #     ADJ = pd.read_pickle(cadj_file)
    #     print("done")
    # except:
    #     raise ValueError("\nCADJ file not found")
    
    # # Construct BMU file path, load 
    # print("Loading BMU file ... ", end="")
    # bmu_file = froot + "_BMU.pkl"
    # try:
    #     BMU = pd.read_pickle(bmu_file)
    #     print("done")
    # except:
    #     raise ValueError("\nBMU file not found")

    # Determine whether we are clustering on CADJ or CONN. 
    # If CONN, symmetrize CADJ 
    ADJ = plearn['CADJ']
    if parm["clus_wt_directed"]==0:
        igraph_mode = 'undirected'
        ADJ = ADJ + ADJ.transpose()
    else:
        igraph_mode = 'directed'
        
    # Determine number of WT steps, if needed 
    if parm["clus_wt_steps"]==0:
        print("Determining number of WT steps ... ", end="")
        g = igraph.Graph.Adjacency(ADJ, mode=igraph_mode)
        #spaths = igraph.Graph.shortest_paths(g)
        spaths = igraph.Graph.distances(g)
        spaths = np.array([item for sublist in spaths for item in sublist])
        spaths = spaths[np.where(np.isfinite(spaths))]
        spaths = spaths[np.where(spaths)]
        wt_steps = math.ceil(spaths.mean())
        print(str(wt_steps))
    else:
        wt_steps = parm["clus_wt_steps"]
        print("Using WT steps = " + str(wt_steps))

    # Cluster, project cluster labels to data via BMU 
    print("WT community detection ... ", end="")
    g = igraph.Graph.Weighted_Adjacency(ADJ, mode=igraph_mode)
    clus_wt_W = np.asarray(igraph.Graph.community_walktrap(g, weights='weight', steps=wt_steps).as_clustering().membership, dtype='int')
    clus_wt_X = clus_wt_W[plearn['BMU'][:,0]]
    print("done")

    # Report number of clusterings 
    print(str(len(np.unique(clus_wt_W))) + " clusters found")

    # Save 
    froot = froot_cluster_walktrap(parm)
    fname = froot + "_WL.pkl"                     
    pd.to_pickle(clus_wt_W, fname)
    print("Prototype clustering written:\n %s" % fname)

    fname = froot + "_XL.pkl"                     
    pd.to_pickle(clus_wt_X, fname)
    print("Data clustering written:\n %s" % fname)

    fname = froot + "_cfg.txt"  
    with open(fname, 'w') as f:
            f.write('steps = %d\n' % wt_steps)
            f.write('nclus = %d' % len(np.unique(clus_wt_W)))
    print("Config file written:\n %s" % fname)

    print("*** END Prototype Clustering via Walktrap ***")

    return 
