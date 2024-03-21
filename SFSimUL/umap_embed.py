import os
import numpy as np 
import pandas as pd 
import umap
import pickle

from .io import froot_umap_embed, load_prototype_learn_products, load_data

def umap_embed(parm):

    print("*** UMAP Data Embedding ***")
    
    # Load data and labels 
    print("Loading data ... ", end = "")
    X = load_data(parm)
    if parm["umap_target_weight"] > 0.0:
        XL = np.array(pd.read_pickle(parm["umap_target_file"]))
    else:
        XL = None
    print("done")

    ## UMAP 
    print("Initializing UMAP object ... ", end = "")
    u = umap.UMAP(n_neighbors=parm["umap_n_neighbors"], \
                  n_components=parm["umap_n_components"], \
                  init = parm["umap_init"], \
                  min_dist = parm["umap_min_dist"], \
                  spread = parm["umap_spread"], \
                  n_jobs = parm["umap_n_jobs"], 
                  set_op_mix_ratio=1.0, local_connectivity=1.0, repulsion_strength=1.0, 
                  negative_sample_rate = parm["umap_negative_sample_rate"], \
                  transform_queue_size=4.0, \
                  random_state=parm["umap_random_state"], \
                  angular_rp_forest=False, \
                  target_n_neighbors=-1, \
                  target_weight=parm["umap_target_weight"], \
                  target_metric=parm["umap_target_metric"], \
                  target_metric_kwds=None, \
                  transform_seed=42, transform_mode='embedding', force_approximation_algorithm=False, verbose=True, tqdm_kwds=None, unique=False, \
                  densmap=False, dens_lambda=2.0, dens_frac=0.3, dens_var_shift=0.1, output_dens=False, disconnection_distance=None, precomputed_knn=(None, None, None))
    print("done")

    print("Training UMAP object")
    UX = u.fit_transform(X=X, y=XL) # next property supervised 

    print("Projecting prototypes ... ", end = "")
    plearn = load_prototype_learn_products(parm)
    UW = u.transform(plearn['W'], force_all_finite=True)
    print("done")
    
    
    froot = froot_umap_embed(parm)
    
    fname =  froot + "_embed-X.pkl"
    pd.to_pickle(UX, fname)
    print("Embedded coordinates saved:\n %s" % fname)

    fname =  froot + "_embed-W.pkl"
    pd.to_pickle(UW, fname)
    print("Embedded prototype coordinates saved:\n %s" % fname)

    # fname = froot + "_UOBJ.pkl"    
    # pickle.dump(u, open(fname, 'wb'))
    # print("Fitted UMAP object saved:\n %s" % fname)

    fname = froot + "_cfg.txt"
    with open(fname, 'w') as f:
        f.write('n_neighbors = %d\n' % parm["umap_n_neighbors"])
        f.write('n_components = %d\n' % parm["umap_n_components"])
        f.write('random_state = %d\n' % parm["umap_random_state"])
        f.write('init = %s\n' % parm['umap_init'])
        f.write('min_dist = %.2f\n' % parm['umap_min_dist'])
        f.write('spread = %.2f\n' % parm['umap_spread'])
        f.write('n_jobs = %d\n' % parm['umap_n_jobs'])
        f.write('negative_sample_rate = %d\n' % parm['umap_negative_sample_rate'])
        f.write('target = %s\n' % parm['umap_target_file'])
        f.write('target_weight = %.2f\n' % parm['umap_target_weight'])
        f.write('target_name = %s\n' % parm['umap_target_name'])
        f.write('target_metric = %s' % parm['umap_target_metric'])        
    print("Config file written:\n %s" % fname)

    return 

