import os
import numpy as np 
import pandas as pd 

    


def froot_prototype_learn(parm):
    fpath = os.path.join(parm["output_dir"], "prototype_learn")
    if not os.path.exists(fpath):
        os.makedirs(fpath)
    froot = os.path.join(fpath, parm["project_root"] + "_" + parm['proto_method'] + str(parm["proto_m"]))
    return froot


def froot_cluster_walktrap(parm):
    fpath = os.path.join(parm["output_dir"], "cluster")
    if not os.path.exists(fpath):
        os.makedirs(fpath)
    froot = os.path.join(fpath, parm["project_root"] + "_" + parm['proto_method'] + str(parm["proto_m"]))
    froot = froot + "_clusWT" + "-dir" + str(parm["clus_wt_directed"]) + "-steps" + str(parm["clus_wt_steps"])
    return froot

def froot_umap_embed(parm):
    fpath = os.path.join(parm["output_dir"], "embed") 
    if not os.path.exists(fpath):
        os.makedirs(fpath)
    froot = os.path.join(fpath, parm["project_root"] + "_UMAP-k" + str(parm["umap_n_neighbors"]))
    if parm["umap_target_weight"] > 0.0:
        froot = froot + "-t" + parm["umap_target_name"] + str(parm["umap_target_weight"])
    
    return froot

# def froot_umap_project(parm):
#     fpath = os.path.join(parm["output_dir"], "embed") 
#     if not os.path.exists(fpath):
#         os.makedirs(fpath)
#     froot = os.path.join(fpath, parm["project_root"] + "_UMAP-k" + str(parm["umap_n_neighbors"]))
#     if parm["umap_target_weight"] > 0.0:
#         froot = froot + "-t" + parm["umap_target_name"] + str(parm["umap_target_weight"])
    
#     return froot



def load_data(parm):
    # Load data, store sample dimensions 
    X = pd.read_pickle(parm["data_file"])
    if not isinstance(X, np.ndarray):
        X = X.to_numpy()
    return X


def load_data_label(parm):
    if parm['data_label_file'] is not None:
        XL = pd.read_pickle(parm['data_label_file'])
        if not isinstance(XL, np.ndarray):
            XL = XL.to_numpy()
    else:
        XL = None 
    return XL 

def load_prototype_learn_products(parm):
    froot = froot_prototype_learn(parm)
    out = {} 
    out['W'] = pd.read_pickle(froot + "_W.pkl")
    out['BMU'] = pd.read_pickle(froot + "_BMU.pkl")
    out['QE'] = pd.read_pickle(froot + "_QE.pkl")
    out['CADJ'] = pd.read_pickle(froot + "_CADJ.pkl")
    out['RF'] = pd.read_pickle(froot + "_RF.pkl")
    out['RFSize'] = pd.read_pickle(froot + "_RFSize.pkl")
    out['RFLDist'] = pd.read_pickle(froot + "_RFLDist.pkl")
    out['RFL'] = pd.read_pickle(froot + "_RFL.pkl")
    out['RFLPurity'] = pd.read_pickle(froot + "_RFLPurity.pkl")
    out['RFLPurityUOA'] = pd.read_pickle(froot + "_RFLPurityUOA.pkl")
    out['RFLPurityWOA'] = pd.read_pickle(froot + "_RFLPurityWOA.pkl")
    return out


