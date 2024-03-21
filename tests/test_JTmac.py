import SFSimUL as sfsul
import pandas as pd 

# Specify parameter file, load it 
parm_file = "/Users/jtaylor/Dropbox/research/starforge_cores/M2e4_R10_B0.1/M2e4_R10_B0.1.parm"
parm = sfsul.parse_parm(parm_file)

# Prototype learning 
sfsul.prototype_learn(parm)

# Walktrap Clustering of prototypes 
sfsul.prototype_cluster_walktrap(parm)

# UMAP embedding of data 
sfsul.umap_embed(parm)

## Plots 
# Load Protostellar label, project 
XL = sfsul.load_data_label(parm)
sfsul.umap_project(parm, varname='PS', z=XL, s=0.5, title='Protostellar', cmap='tab20', categorical=True)

# Load unnormalized data file, for variable projection 
data = pd.read_pickle('/Users/jtaylor/Dropbox/research/starforge_cores/M2e4_R10_B0.1/ds/M2e4_R10_B0.1_data_Bulk.pkl')

sfsul.umap_project(parm, varname='logReff', z=data['Reff'], log=True, s=0.5, title='log(Reff)', cmap='hot_r', categorical=False)
sfsul.umap_project(parm, varname='logLeafDisp', z=data['LeafDisp'], log=True, s=0.5, title='log(LeafDisp)', cmap='hot_r', categorical=False)
sfsul.umap_project(parm, varname='logLeafMass', z=data['LeafMass'], log=True, s=0.5, title='log(LeafMass)', cmap='hot_r', categorical=False)
sfsul.umap_project(parm, varname='CoherentRadius', z=data['CoherentRadius'], log=False, s=0.5, title='CoherentRadius', cmap='hot_r', categorical=False)
sfsul.umap_project(parm, varname='DensityIndex', z=data['DensityIndex'], log=False, s=0.5, title='DensityIndex', cmap='hot_r', categorical=False)
sfsul.umap_project(parm, varname='logVBulk', z=data['VBulk'], log=True, s=0.5, title='log(VBulk)', cmap='hot_r', categorical=False)
sfsul.umap_project(parm, varname='logLeafKe', z=data['LeafKe'], log=True, s=0.5, title='log(LeafKe)', cmap='hot_r', categorical=False)
sfsul.umap_project(parm, varname='logLeafGrav', z=data['LeafGrav'], log=True, s=0.5, title='log(LeafGrav)', cmap='hot_r', categorical=False)
sfsul.umap_project(parm, varname='logLeafVR', z=data['LeafVR'], log=True, s=0.5, title='log(LeafVR)', cmap='hot_r', categorical=False)
sfsul.umap_project(parm, varname='logMaxDen', z=data['MaxDen'], log=True, s=0.5, title='log(MaxDen)', cmap='hot_r', categorical=False)
sfsul.umap_project(parm, varname='ShapeC', z=data['ShapeC'], log=False, s=0.5, title='ShapeC', cmap='hot_r', categorical=False)
sfsul.umap_project(parm, varname='ShapeB', z=data['ShapeB'], log=False, s=0.5, title='ShapeB', cmap='hot_r', categorical=False)
sfsul.umap_project(parm, varname='logShapeNorm', z=data['ShapeNorm'], log=True, s=0.5, title='log(ShapeNorm)', cmap='hot_r', categorical=False)
sfsul.umap_project(parm, varname='logHalfMassR', z=data['HalfMassR'], log=True, s=0.5, title='log(HalfMassR)', cmap='hot_r', categorical=False)
sfsul.umap_project(parm, varname='logMeanB', z=data['MeanB'], log=True, s=0.5, title='log(MeanB)', cmap='hot_r', categorical=False)
sfsul.umap_project(parm, varname='logMagEnergy', z=data['MagEnergy'], log=True, s=0.5, title='log(MagEnergy)', cmap='hot_r', categorical=False)
sfsul.umap_project(parm, varname='logSoundSpeed', z=data['SoundSpeed'], log=True, s=0.5, title='log(SoundSpeed)', cmap='hot_r', categorical=False)
sfsul.umap_project(parm, varname='logLeafKeOnly', z=data['LeafKeOnly'], log=True, s=0.5, title='log(LeafKeOnly)', cmap='hot_r', categorical=False)
sfsul.umap_project(parm, varname='logMeanOutMassFrac', z=data['MeanOutMassFrac'], log=True, s=0.5, title='log(MeanOutMassFrac)', cmap='hot_r', categorical=False)
sfsul.umap_project(parm, varname='logMeanWindMassFrac', z=data['MeanWindMassFrac'], log=True, s=0.5, title='log(MeanWindMassFrac)', cmap='hot_r', categorical=False)




