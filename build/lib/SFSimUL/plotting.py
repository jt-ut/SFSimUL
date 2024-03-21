import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt 
from .io import froot_umap_embed
from .proto_learn_faiss import prototype_fwdmap_mean

def plot_categorical(x, y, label, s=0.5, title=None, xlabel=None, ylabel=None, cmap='tab20'):

    # Ensure labels are strings 
    label = [str(x) for x in label]
    # Get unique set of labels, encode as integers
    unq_labels, label_int = np.unique(label, return_inverse=True)
    n_labels = len(unq_labels)

    # Build colormap for labels
    cmap = plt.get_cmap(cmap, n_labels)

    # Plot, add title & axis labels 
    fig, ax = plt.subplots()
    cs = ax.scatter(x=x, y=y, c=label_int, s = s, cmap=cmap)
    ax.set_title(title, fontweight='bold')
    ax.set(xlabel=xlabel, ylabel=ylabel)

    # Build custom colorbar denoting categories
    cbar = fig.colorbar(cs)
    tick_locs = (np.arange(n_labels) + 0.5)*(n_labels-1)/n_labels
    cbar.set_ticks(tick_locs)
    cbar.set_ticklabels(unq_labels)

    return fig

def plot_continuous(x, y, z, log=False, s=0.5, title=None, xlabel=None, ylabel=None, cmap='hot_r'):
    # Ensure z is a numpy array 
    z = np.array(z)
    if log:
        z = np.log10(z)

    fig, ax = plt.subplots()
    
    cs = ax.scatter(x=x, y=y, c=z, s = s, cmap = cmap)
    ax.set_title(title, fontweight='bold')
    cbar = fig.colorbar(cs)
    
    return fig


def plot_embedding(x, y, z, categorical=False, log=False, s=0.5, title=None, xlabel=None, ylabel=None, cmap='tab20'):

    if categorical:
        # Ensure labels are strings 
        z = [str(x) for x in z]
        # Get unique set of labels, encode as integers
        unq_labels, z, cnt_labels = np.unique(z, return_inverse=True, return_counts=True)
        n_labels = len(unq_labels)

        # Build colormap for labels
        cmap = plt.get_cmap(cmap, n_labels)
    else:
        # Ensure z is a numpy array 
        z = np.array(z)
        if log:
            z = np.log10(z)



    # Plot, add title & axis labels 
    fig, ax = plt.subplots()
    if categorical:
        plot_order = np.flip(np.argsort(cnt_labels))
        reorder_idx = list()
        for i in plot_order:
            reorder_idx.append(np.where(z==i)[0])
        reorder_idx = np.concatenate(reorder_idx).ravel()
        
        x = x[reorder_idx]
        y = y[reorder_idx]
        z = z[reorder_idx]        
    
    cs = ax.scatter(x=x, y=y, c=z, s = s, cmap=cmap)
    ax.set_title(title, fontweight='bold')
    ax.set(xlabel=xlabel, ylabel=ylabel)

    # Build custom colorbar denoting categories
    cbar = fig.colorbar(cs)

    if categorical:
        tick_locs = (np.arange(n_labels) + 0.5)*(n_labels-1)/n_labels
        cbar.set_ticks(tick_locs)
        cbar.set_ticklabels(unq_labels)
    


    return fig


def umap_project(parm, varname, z, s=0.5, title=None, cmap='tab20', categorical=False, log=False):

    # Load UMAP coordinates 
    fname = froot_umap_embed(parm) + "_embed-X.pkl"
    UX = pd.read_pickle(fname)
    
    # if categorical:
    #     fig = plot_categorical(x=UX[:,0], y=UX[:,1], label=z, s=s, title=title, xlabel='$U_1$', ylabel='$U_2$', cmap=cmap)
    # else:
    #     fig = plot_continuous(x=UX[:,0], y=UX[:,1], z=z, log=log, s=s, title=title, xlabel='$U_1$', ylabel='$U_2$', cmap=cmap)

    fig = plot_embedding(x=UX[:,0], y=UX[:,1], z=z, categorical=categorical, log=log, s=s, title=title, xlabel='$U_1$', ylabel='$U_2$', cmap=cmap)    
    
    #fname = froot_umap_embed(parm) + "embed-X_proj-" + str(varname) + ".svg"
    fname = froot_umap_embed(parm) + "_embed-X_proj-" + str(varname) + ".png"
    #plt.savefig(fname, format='pdf')
    plt.savefig(fname, format='png', dpi=1200)
    plt.clf()

    return 

def umap_project_prototypes(parm, varname, z, s=0.5, title=None, cmap='tab20', categorical=False, log=False, fwdmap=False, RF=None):

    # Load UMAP coordinates 
    fname = froot_umap_embed(parm) + "_embed-W.pkl"
    UW = pd.read_pickle(fname)

    if fwdmap:
        z = np.array(z) # Ensure z is a numpy array 
        if log:
            z = prototype_fwdmap_mean(np.log10(z), RF)
            z = 10**z
        else:
            z = prototype_fwdmap_mean(z, RF)
    
    fig = plot_embedding(x=UW[:,0], y=UW[:,1], z=z, categorical=categorical, log=log, s=s, title=title, xlabel='$U_1$', ylabel='$U_2$', cmap=cmap)    
    
    fname = froot_umap_embed(parm) + "_embed-W_proj-" + str(varname) + ".png"
    plt.savefig(fname, format='png', dpi=1200)
    plt.clf()

    return 
