import os 

def ensure_parmkey_exists(dict, key):
    if not dict.get(key, False):
        raise ValueError("keyword " + key + " missing")
    return 

def ensure_parmval_numeric(dict, key):
    try:
        tmp = float(dict[key])
    except:
        raise ValueError("keyword " + key + " not numeric")
    return tmp 




def parse_parm(parm_file):
    # Initialize emtpy dictionary to contain kwd/val pairs of parm file 
    parm = {"num_threads": None, 
            "data_file": None, 
            "data_label_file": None,
            "output_dir": None, 
            "project_root": None, 
            "proto_method": None, 
            "proto_m": None, 
            "proto_seed": 123, 
            "proto_iter_monitor": 20, 
            "proto_iter_max": 1000, 
            "proto_conv_tol": 0.01, 
            "clus_wt_directed": 0, 
            "clus_wt_steps": 0, 
            "umap_n_neighbors": 15,
            "umap_random_state": 123, 
            "umap_n_components": 2, 
            "umap_init": "random", 
            "umap_min_dist": 0.1, 
            "umap_spread": 1.0,
            "umap_n_jobs": -1, 
            "umap_negative_sample_rate": 5, 
            "umap_target_weight": 0.0, 
            "umap_target_name": None,
            "umap_target_file": None,
            "umap_target_metric": None
            }

    # Read parm 
    lines = [line.strip() for line in open(parm_file)]

    # Fill up dictionary 
    for l in lines:
        if len(l)==0: continue
        if l[0][0]=="#": continue 
        tmp = l.split("#")[0] # Toss anything after a # symbol (comment)
        tmp = tmp.split() # Remove whitespace 
        kwd = tmp[0].lower() # Ensure lowercase kwd
        val = tmp[1] # Extract val 
        if kwd in parm: parm[kwd] = val 



    ## Checks on MANDATORY dictionary keys 
    ensure_parmkey_exists(parm, "num_threads")
    parm["num_threads"] = int(ensure_parmval_numeric(parm, "num_threads"))


    # data_file
    ensure_parmkey_exists(parm, "data_file")
    # data_file should be valid file path  
    if not os.path.isfile(parm["data_file"]):
        raise ValueError("data_file not found:\n %s" % parm["data_file"])
    
    # OPTIONAL: data_label_file 
    # If exists, should be a valid pickle file 
    if not parm.get("data_label_file", False):
        parm["data_label_file"] = ""
    else:
        if not os.path.isfile(parm["data_label_file"]):
            raise ValueError("data_label_file not found:\n %s" % parm["data_label_file"])

    # output_dir must exist
    ensure_parmkey_exists(parm, "output_dir")
    # create it if path doesn't exist 
    if not os.path.isdir(parm["output_dir"]):
        os.makedirs(parm["output_dir"])
        Warning("output_dir not found, created:\n %d" % parm["output_dir"])
    
    # project_root must not be blank 
    ensure_parmkey_exists(parm, "project_root")
    if parm["project_root"]=="":
        raise ValueError("keyword project_root must not be blank")
    
    # proto_method must exist and be valid 
    # currently, can only be kmeans 
    ensure_parmkey_exists(parm, "proto_method")
    valid_proto_methods = ["KM"]
    if not parm["proto_method"] in valid_proto_methods:
        raise ValueError("keyword proto_method invalid")

    # proto_m must exist and be a integer 
    ensure_parmkey_exists(parm, "proto_m")
    parm["proto_m"] = int(ensure_parmval_numeric(parm, "proto_m"))


    # OPTIONAL: 
    # if proto_seed exists, must be numeric. 
    # default = 123
    parm["proto_seed"] = int(ensure_parmval_numeric(parm, "proto_seed"))
    
    # proto_iter_monitor must exist and be a integer 
    parm["proto_iter_monitor"] = int(ensure_parmval_numeric(parm, "proto_iter_monitor"))
    
    # proto_iter_max must exist and be a integer 
    parm["proto_iter_max"] = int(ensure_parmval_numeric(parm, "proto_iter_max"))
    
    # proto_conv_tol must be numeric, in [0,1]
    parm["proto_conv_tol"] = float(ensure_parmval_numeric(parm, "proto_conv_tol"))
    if parm["proto_conv_tol"] < 0.0 or parm["proto_conv_tol"] > 1.0:
        raise ValueError("keyword proto_conv_tol must be in range [0,1]")
    

    # clus_wt_directed must exist and be either 0 or 1 
    parm["clus_wt_directed"] = int(ensure_parmval_numeric(parm, "clus_wt_directed"))
    if not(parm["clus_wt_directed"] == 0 or parm["clus_wt_directed"] == 1):
        raise ValueError("keyword clus_wt_directed must be 0 (False) or 1 (True)")
  
    # clus_wt_steps can be either the string 'auto' or an integer 
    parm["clus_wt_steps"] = int(ensure_parmval_numeric(parm, "clus_wt_steps"))


    ## UMAP      
    parm["umap_n_neighbors"] = int(ensure_parmval_numeric(parm, "umap_n_neighbors"))
    parm["umap_random_state"] = int(ensure_parmval_numeric(parm, "umap_random_state"))
    parm["umap_n_components"] = int(ensure_parmval_numeric(parm, "umap_n_components"))
    parm["umap_min_dist"] = float(ensure_parmval_numeric(parm, "umap_min_dist"))
    parm["umap_spread"] = float(ensure_parmval_numeric(parm, "umap_spread"))
    parm["umap_n_jobs"] = int(ensure_parmval_numeric(parm, "umap_n_jobs"))
    parm["umap_negative_sample_rate"] = int(ensure_parmval_numeric(parm, "umap_negative_sample_rate"))
    valid_inits = ['spectral', 'random']
    if not parm["umap_init"] in valid_inits:
        raise ValueError("keyword umap_init invalid")
    

    
    ## OPTIONAL 
    # umap_target_weight must be numeric, in [0,1], if it exists 
    parm["umap_target_weight"] = float(ensure_parmval_numeric(parm, "umap_target_weight"))
    if parm["umap_target_weight"] < 0.0 or parm["umap_target_weight"] > 1.0:
        raise ValueError("keyword umap_target_weight must be in range [0,1]")
    
    # If umap_target_weight > 0, then umap_target_name, umap_target_file, and umap_target_metric are required 
    if parm["umap_target_weight"] > 0.0: 
        if not parm.get("umap_target_name", False):
            raise ValueError("keyword umap_target_name missing. Must be present with umap_target_weight > 0.0")
        
        if not parm.get("umap_target_file", False):
            raise ValueError("keyword umap_target_file missing. Must be present with umap_target_weight > 0.0")
        if not os.path.isfile(parm["umap_target_file"]):
            raise ValueError("umap_target_file not found:\n %s" % parm["umap_target_file"])
        
        if not parm.get("umap_target_metric", False):
            raise ValueError("keyword umap_target_metric missing. Must be present with umap_target_weight > 0.0")
        valid_target_metrics = ["categorical", "euclidean"]
        if not parm["umap_target_metric"] in valid_target_metrics:
            raise ValueError("keyword umap_target_metric invalid")
    else:
        parm["umap_target_name"] = None
        parm["umap_target_file"] = None
        parm["umap_target_metric"] = None


        
    return parm
    
