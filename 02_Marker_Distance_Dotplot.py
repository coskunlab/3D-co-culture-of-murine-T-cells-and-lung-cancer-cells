# %%
import os
import h5py
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import skimage
import imageio

import tifffile as tf
import cv2

from skimage.exposure import match_histograms, rescale_intensity
from skimage import exposure, util, filters, restoration

import pandas as pd

from cellpose import models,core

from joblib import Parallel, delayed
import seaborn as sns
from statannotations.Annotator import Annotator
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder

# %%
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import random
import numpy as np
import pandas as pd
from matplotlib.pyplot import scatter
from matplotlib.pyplot import Line2D
import matplotlib as mpl
import matplotlib.colors as mcolors
mpl.use('Agg') # Use a non-interactive backend to prevent figures from displaying
mpl.rcParams['figure.dpi'] = 600
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import textwrap

# %%
# Define plot save directory
try:
    # Get the directory of the current script
    script_dir = Path(__file__).parent.resolve()
except NameError:
    # Fallback for interactive environments (e.g., Jupyter)
    script_dir = Path.cwd().resolve()

plots_save_path = script_dir.parent / "Plots"
plots_save_path.mkdir(exist_ok=True)
print(f"All plots will be saved in subdirectories within: {plots_save_path}")


# %%
def contrast_str(img, n_min=10, n_max=100):
    p2, p98 = np.percentile(img, (n_min, n_max))
    img_rescale = rescale_intensity(img, in_range=(p2, p98))
    img_rescale = util.img_as_ubyte(img_rescale)
    return img_rescale

# %%
def extract_numeric(filename):
    return int(filename.split('_')[-1].split('.')[0])

# %%
def load_and_process_hdf5(path):
    """
    Loads and processes a single HDF5 file.
    This function reads image data and markers, then applies a series of
    image processing steps.
    """
    with h5py.File(path, "r") as f:
        imgs = f['imgs'][:]
        markers = f['imgs'].attrs['Marker']
    imgs_processed = skimage.exposure.adjust_log(imgs)
    imgs_processed = np.clip(imgs_processed, a_min=0, a_max=imgs_processed.max())
    imgs_processed = contrast_str(imgs_processed, n_min=0, n_max=100)
    return imgs_processed, markers

# %%
if core.use_gpu()==False: 
    use_GPU=False
else:
    print('GPU applied')
    use_GPU=True

# %%
# cytotoxic T cell - Parallel Loading
print("Loading and processing cytotoxic T cell HDF5 files in parallel...")
paths_cytotoxic = [
    r"Y:\coskun-lab\Thomas\20_MouseCocultureMatrigel\data\CytoCoc\hdf5\processed\CD8T_pos1_5.hdf5",
    r"Y:\coskun-lab\Thomas\20_MouseCocultureMatrigel\data\CytoCoc\hdf5\processed\CD8T_pos2_16.hdf5"
]
# Use n_jobs=-1 to use all available CPU cores
results_cytotoxic = Parallel(n_jobs=-1)(delayed(load_and_process_hdf5)(p) for p in paths_cytotoxic)
imgs_processed_all_cytotoxic = [res[0] for res in results_cytotoxic]
# Assuming markers are the same across files, we take them from the first result
markers_cytotoxic = results_cytotoxic[0][1] if results_cytotoxic else None
print("Cytotoxic T cell data loaded.")

# %%
# naive T cell - Parallel Loading
print("Loading and processing naive T cell HDF5 files in parallel...")
paths_naive = [
    r"Y:\coskun-lab\Thomas\20_MouseCocultureMatrigel\data\NaiveCoC\hdf5\processed\CD8T_pos1_10.hdf5",
    r"Y:\coskun-lab\Thomas\20_MouseCocultureMatrigel\data\NaiveCoC\hdf5\processed\CD8T_pos2_10.hdf5"
]
results_naive = Parallel(n_jobs=-1)(delayed(load_and_process_hdf5)(p) for p in paths_naive)
imgs_processed_all_naive = [res[0] for res in results_naive]
markers_naive = results_naive[0][1] if results_naive else None
print("Naive T cell data loaded.")


# %%
Cyto_holder = imgs_processed_all_cytotoxic.copy()
naive_holder = imgs_processed_all_naive.copy()

# %%
from skimage import measure
from functools import reduce

### Helper function
def read_intensity_per_cell(img, mask):
    props = skimage.measure.regionprops_table(
        mask, img, properties=["label", "mean_intensity", "area"]
    )
    df_prop = pd.DataFrame(props)
    df_prop["mean_intensity"] = df_prop["mean_intensity"]
    df_prop.drop("area", axis=1, inplace=True)

    df_prop["mean_intensity"] = np.arcsinh(df_prop[["mean_intensity"]])

    return df_prop

def create_cell_mean_expression(mask, cell2int, **kwargs):
    img = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)

    for cell, int in cell2int.items():
        img = np.where(mask == cell, int, img)
    return img

# %%
from skimage.io import imread
naive_cyto_mask_pos1 = imread("Y:\coskun-undergrads\Mingshuang\Thomas\CoCulture\imgs\masks\CD8T_pos1_10_cp_masks.png")
naive_cyto_mask_pos2 = imread("Y:\coskun-undergrads\Mingshuang\Thomas\CoCulture\imgs\masks\CD8T_pos2_10_cp_masks.png")

cytotoxic_cyto_mask_pos1 = tf.imread(r"Y:\coskun-lab\Thomas\20_MouseCocultureMatrigel\data\CytoCoc\masks\pos1_5.tif")
cytotoxic_cyto_mask_pos2 = tf.imread(r"Y:\coskun-lab\Thomas\20_MouseCocultureMatrigel\data\CytoCoc\masks\pos2_16.tif")

# %%
# Cytotoxic

# Extract mean intensity per mask
cyto_masks_cytotoxic = [cytotoxic_cyto_mask_pos1, cytotoxic_cyto_mask_pos2]
# df_appended_list_cytotoxic = pd.DataFrame()
df_appended_list_cytotoxic_all = []
for i, img_processed in enumerate(imgs_processed_all_cytotoxic):
    cyto_mask = cyto_masks_cytotoxic[i]
    df_appended_list = []
    for i, img in enumerate(img_processed):
        if markers_cytotoxic[i] == 'DNA' and i != 0:
            continue 
        df_prop = read_intensity_per_cell(img, cyto_mask)
        df_prop.columns = ["Cell_label", markers_cytotoxic[i]]
        df_appended_list.append(df_prop)
    df_appended_list_cytotoxic_all.append(df_appended_list)


# %%
# Naive

# Extract mean intensity per mask
cyto_masks_naive = [naive_cyto_mask_pos1, naive_cyto_mask_pos2]
df_appended_list_naive_all = []
for i, img_processed in enumerate(imgs_processed_all_naive):
    cyto_mask = cyto_masks_naive[i]
    df_appended_list = []
    for i, img in enumerate(img_processed):
        if markers_naive[i] == 'DNA' and i != 0:
            continue 
        df_prop = read_intensity_per_cell(img, cyto_mask)
        df_prop.columns = ["Cell_label", markers_naive[i]]
        df_appended_list.append(df_prop)
    df_appended_list_naive_all.append(df_appended_list)


# %%
# cytotoxic
df_cell_intensity_pos1_cyto = reduce(
    lambda left, right: pd.merge(left, right, on=["Cell_label"]),
    df_appended_list_cytotoxic_all[0],
)

df_cell_intensity_pos2_cyto = reduce(
    lambda left, right: pd.merge(left, right, on=["Cell_label"]),
    df_appended_list_cytotoxic_all[1],
)

# %%
# naive
df_cell_intensity_pos1_naive = reduce(
    lambda left, right: pd.merge(left, right, on=["Cell_label"]),
    df_appended_list_naive_all[0],
)

df_cell_intensity_pos2_naive = reduce(
    lambda left, right: pd.merge(left, right, on=["Cell_label"]),
    df_appended_list_naive_all[1],
)

# %%
# cytotoxic
props = skimage.measure.regionprops_table(
    cytotoxic_cyto_mask_pos1, properties=["label", "centroid"]
)
rows = props["centroid-0"] 
cols = props["centroid-1"] 
centroid_cytotoxic = np.array(list(zip(cols, rows)))
df_cell_intensity_pos1_cyto['Row'] = rows
df_cell_intensity_pos1_cyto['Cols'] = cols
df_cell_intensity_pos1_cyto['FOV'] = 'pos1'
df_prop_cytotoxic = df_cell_intensity_pos1_cyto

props = skimage.measure.regionprops_table(
    cytotoxic_cyto_mask_pos2, properties=["label", "centroid"]
)
rows = props["centroid-0"] 
cols = props["centroid-1"] 
centroid_cytotoxic = np.concatenate([centroid_cytotoxic, np.array(list(zip(cols, rows)))])
df_cell_intensity_pos2_cyto['Row'] = rows
df_cell_intensity_pos2_cyto['Cols'] = cols
df_cell_intensity_pos2_cyto['FOV'] = 'pos2'
# df_prop_cytotoxic = df_prop_cytotoxic.append(df_cell_intensity_pos2_cyto)
df_prop_cytotoxic = pd.concat([df_prop_cytotoxic, df_cell_intensity_pos2_cyto])

# %%
df_prop_cytotoxic = df_prop_cytotoxic.drop(['Concavalin A', 'WGA'], axis = 1)

# %%
# naive
props = skimage.measure.regionprops_table(
    naive_cyto_mask_pos1, properties=["label", "centroid"]
)
rows = props["centroid-0"] 
cols = props["centroid-1"] 
centroid_naive = np.array(list(zip(cols, rows)))
df_cell_intensity_pos1_naive['Row'] = rows
df_cell_intensity_pos1_naive['Cols'] = cols
df_cell_intensity_pos1_naive['FOV'] = 'pos1'
df_prop_naive = df_cell_intensity_pos1_naive

props = skimage.measure.regionprops_table(
    naive_cyto_mask_pos2, properties=["label", "centroid"]
)
rows = props["centroid-0"] 
cols = props["centroid-1"] 
centroid_naive = np.concatenate([centroid_naive, np.array(list(zip(cols, rows)))])
df_cell_intensity_pos2_naive['Row'] = rows
df_cell_intensity_pos2_naive['Cols'] = cols
df_cell_intensity_pos2_naive['FOV'] = 'pos2'
# df_prop_naive = df_prop_naive.append(df_cell_intensity_pos2_naive)
df_prop_naive = pd.concat([df_prop_naive, df_cell_intensity_pos2_naive])

# %%
# import seaborn as sns
# df_subset = df_prop[['CD44','E-cadherin']].copy()
# sns.scatterplot(data=df_subset, x="CD44", y='E-cadherin')

df_DNA_cyto = df_prop_cytotoxic[['DNA']].copy()
df_marker_cyto = df_prop_cytotoxic[['CD44','E-cadherin']].copy()

df_DNA_naive = df_prop_naive[['DNA']].copy()
df_marker_naive = df_prop_naive[['CD44','E-cadherin']].copy()

# %%
centroid_cytotoxic = centroid_cytotoxic[df_prop_cytotoxic['DNA'] > 2, :]
df_prop_cytotoxic = df_prop_cytotoxic[df_prop_cytotoxic['DNA'] > 2]

centroid_naive = centroid_naive[df_prop_naive['DNA'] > 2, :]
df_prop_naive = df_prop_naive[df_prop_naive['DNA'] > 2]

# %%
# Define cell type
df_prop_cytotoxic['Cell Type'] = 0
df_prop_cytotoxic.loc[df_prop_cytotoxic['E-cadherin'] > 0.1, 'Cell Type'] = 1
df_prop_cytotoxic.loc[df_prop_cytotoxic['E-cadherin'] <= 0.1, 'Cell Type'] = 2

# %%
# Define cell type
df_prop_naive['Cell Type'] = 0
df_prop_naive.loc[df_prop_naive['E-cadherin'] > 0.1, 'Cell Type'] = 1
df_prop_naive.loc[df_prop_naive['E-cadherin'] <= 0.1, 'Cell Type'] = 2

# %%
df_cytotoxic_value = df_prop_cytotoxic.iloc[:,1:32]
df_cytotoxic_value_norm = (df_cytotoxic_value - df_cytotoxic_value.min()) / (df_cytotoxic_value.max() - df_cytotoxic_value.min())
# print(df_cytotoxic_value_norm)
df_prop_cytotoxic.iloc[:,1:32] = df_cytotoxic_value_norm

# %%
df_naive_value = df_prop_naive.iloc[:,1:32]
df_naive_value_norm = (df_naive_value - df_naive_value.min()) / (df_naive_value.max() - df_naive_value.min())
df_prop_naive.iloc[:,1:32] = df_naive_value_norm

# %%
markers_cytotoxic = markers_cytotoxic[:41]

# %%
df_prop_cytotoxic['Cell_type_name'] = df_prop_cytotoxic['Cell Type'].apply(lambda x: {0:'Other', 1: 'cancer cell interacting with CytoT', 2:'Cytotoxic T cell'}.get(x,x))
df_cytotoxic = df_prop_cytotoxic.loc[:,np.concatenate([markers_cytotoxic, ['Cell_type_name']])].copy()
df_cytotoxic = df_cytotoxic[df_cytotoxic.Cell_type_name != 'Other']

# %%
df_prop_naive['Cell_type_name'] = df_prop_naive['Cell Type'].apply(lambda x: {0:'Other', 1: 'cancer cell interacting with NaiveT', 2:'Naive T cell'}.get(x,x))
df_naive = df_prop_naive.loc[:,np.concatenate([markers_naive, ['Cell_type_name']])].copy()
df_naive = df_naive[df_naive.Cell_type_name != 'Other']

# %%
def plot(plotting, save_path=None, figsize=(24,7)):
    """
    Generates and optionally saves a boxplot with statistical annotations.
    """
    c1, c2 = plotting['data'][plotting['hue']].unique()
    pairs = [((e, c1), (e, c2)) for e in plotting['data'][plotting['x']].unique()]

    with sns.plotting_context('talk', font_scale=1.5):
        fig, ax = plt.subplots(figsize=figsize)
        ax = sns.boxplot(**plotting, showfliers=False, ax=ax)
        annot = Annotator(ax, pairs, **plotting)
        annot.configure(test='Mann-Whitney', text_format='star', comparisons_correction="bonferroni",loc='inside', verbose=0)
        annot.apply_test().annotate()
        plt.xticks(rotation=30, ha='right')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
            plt.close(fig) # Close the figure to free memory
        else:
            plt.show()

# %%
df_melt_cytotoxic = df_cytotoxic.melt(id_vars=['Cell_type_name'])
df_melt_cytotoxic.columns = ['Cell Type', 'Marker', 'Expression']
df_melt_cytotoxic.head()

# %%
df_melt_naive = df_naive.melt(id_vars=['Cell_type_name'])
df_melt_naive.columns = ['Cell Type', 'Marker', 'Expression']
df_melt_naive.head()


# %%
df_melt_all = pd.concat([df_melt_naive, df_melt_cytotoxic])

# %%
df_melt_all_corrected = df_melt_all[(~(df_melt_all['Marker'] == 'WGA'))]
df_melt_all_corrected = df_melt_all_corrected[(~(df_melt_all_corrected['Marker'] == 'Concavalin A'))]

# %%
df_melt_all_corrected = df_melt_all_corrected[~((df_melt_all_corrected['Cell Type'] == 'cancer cell interacting with NaiveT') | (df_melt_all_corrected['Cell Type'] == 'cancer cell interacting with CytoT'))]

# %%
df_melt_all_corrected['Marker'] = df_melt_all_corrected['Marker'].replace('ASNA', 'ASNS')

# %%
df_melt_all_corrected.to_csv('sns_testing2.csv')

# %%
# Boxplot per cell
print("Generating initial comparison plot...")
plotting = {
    "data": df_melt_all_corrected,
    "x": "Marker",
    "y": "Expression",
    "hue": "Cell Type"
}

plot(plotting, save_path=plots_save_path / "initial_marker_expression_comparison.png")
sns.despine()

# %%
def get_distance(centroid_fit, centroid_search):
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(centroid_fit)

    d, n = neigh.kneighbors(centroid_search, return_distance=True)

    return d

# %%
centroid_cytotoxic_copy = centroid_cytotoxic.copy()
centroid_naive_copy = centroid_naive.copy()

# %%
df_pos1_cytotoxic = df_prop_cytotoxic[df_prop_cytotoxic['FOV'] == 'pos1']
df_pos2_cytotoxic = df_prop_cytotoxic[df_prop_cytotoxic['FOV'] == 'pos2']

df_pos1_naive = df_prop_naive[df_prop_naive['FOV'] == 'pos1']
df_pos2_naive = df_prop_naive[df_prop_naive['FOV'] == 'pos2']

# %%
#cytotoxic

# Get distance
centroid_all_pos1_cyto = centroid_cytotoxic[df_prop_cytotoxic.FOV == 'pos1']
centroid_tcell_pos1_cyto = centroid_all_pos1_cyto[df_pos1_cytotoxic.Cell_type_name == 'Cytotoxic T cell']
d = get_distance(centroid_tcell_pos1_cyto, centroid_all_pos1_cyto)

# Assign to dataframe
df_pos1_cytotoxic['Distance to T-cell'] = d

# Get distance
centroid_all_pos2_cyto = centroid_cytotoxic[df_prop_cytotoxic.FOV == 'pos2']
centroid_tcell_pos2_cyto = centroid_all_pos2_cyto[df_pos2_cytotoxic.Cell_type_name == 'Cytotoxic T cell']
d = get_distance(centroid_tcell_pos2_cyto, centroid_all_pos2_cyto)

# Assign to dataframe
df_pos2_cytotoxic['Distance to T-cell'] = d

# %%
#naive

# Get distance
centroid_all_pos1_naive = centroid_naive[df_prop_naive.FOV == 'pos1']
centroid_tcell_pos1_naive = centroid_all_pos1_naive[df_pos1_naive.Cell_type_name == 'Naive T cell']
d = get_distance(centroid_tcell_pos1_naive, centroid_all_pos1_naive)
# Assign to dataframe
df_pos1_naive['Distance to T-cell'] = d

# Get distance
centroid_all_pos2_naive = centroid_naive[df_prop_naive.FOV == 'pos2']
centroid_tcell_pos2_naive = centroid_all_pos2_naive[df_pos2_naive.Cell_type_name == 'Naive T cell']
d = get_distance(centroid_tcell_pos2_naive, centroid_all_pos2_naive)

# Assign to dataframe
df_pos2_naive['Distance to T-cell'] = d

# %%
df_both_cytotoxic = pd.concat([df_pos1_cytotoxic, df_pos2_cytotoxic])
df_both_naive = pd.concat([df_pos1_naive, df_pos2_naive])

# %%
df_cancer_cyto = df_both_cytotoxic[df_both_cytotoxic.Cell_type_name == 'cancer cell interacting with CytoT'].copy()
df_cancer_cyto.drop(['Row', 'Cols', 'FOV', 'Cell Type', 'Cell_type_name', 'Cell_label'], axis=1, inplace=True)
df_cancer_cyto['neighbor'] = 'Cytotoxic T cell'


df_cancer_naive = df_both_naive[df_both_naive.Cell_type_name == 'cancer cell interacting with NaiveT'].copy()
df_cancer_naive.drop(['Row', 'Cols', 'FOV', 'Cell Type', 'Cell_type_name', 'Cell_label'], axis=1, inplace=True)
df_cancer_naive['neighbor'] = 'Naive T cell'

df_all_cancer = pd.concat([df_cancer_cyto, df_cancer_naive])


# %%
df_all_cancer[f'Distance T-cell'] = pd.qcut(df_all_cancer['Distance to T-cell'], 5, duplicates='drop')
LE = LabelEncoder()
df_all_cancer[f'Distance T-cell bucket'] = LE.fit_transform(df_all_cancer[f'Distance T-cell'])

# %%
df_all_cancer
df_cytotoxic = df_all_cancer[df_all_cancer['neighbor'] == 'Cytotoxic T cell']
df_naive = df_all_cancer[df_all_cancer['neighbor'] == 'Naive T cell']

# %%
df_all_cancer_corrected = pd.concat([df_naive, df_cytotoxic])

# %%
#cytotoxic

# Get distance
centroid_all_pos1_cyto = centroid_cytotoxic[df_prop_cytotoxic.FOV == 'pos1']
centroid_can_pos1_cyto = centroid_all_pos1_cyto[df_pos1_cytotoxic.Cell_type_name == 'cancer cell interacting with CytoT']
d = get_distance(centroid_can_pos1_cyto, centroid_all_pos1_cyto)

# Assign to dataframe
df_pos1_cytotoxic['Distance to cancer-cell'] = d

# Get distance
centroid_all_pos2_cyto = centroid_cytotoxic[df_prop_cytotoxic.FOV == 'pos2']
centroid_can_pos2_cyto = centroid_all_pos2_cyto[df_pos2_cytotoxic.Cell_type_name == 'cancer cell interacting with CytoT']
d = get_distance(centroid_can_pos2_cyto, centroid_all_pos2_cyto)

# Assign to dataframe
df_pos2_cytotoxic['Distance to cancer-cell'] = d

# %%
#naive

# Get distance
centroid_all_pos1_naive = centroid_naive[df_prop_naive.FOV == 'pos1']
centroid_can_pos1_naive = centroid_all_pos1_naive[df_pos1_naive.Cell_type_name == 'cancer cell interacting with NaiveT']
d = get_distance(centroid_can_pos1_naive, centroid_all_pos1_naive)
# Assign to dataframe
df_pos1_naive['Distance to cancer-cell'] = d

# Get distance
centroid_all_pos2_naive = centroid_naive[df_prop_naive.FOV == 'pos2']
centroid_can_pos2_naive = centroid_all_pos2_naive[df_pos2_naive.Cell_type_name == 'cancer cell interacting with NaiveT']
d = get_distance(centroid_can_pos2_naive, centroid_all_pos2_naive)

# Assign to dataframe
df_pos2_naive['Distance to cancer-cell'] = d

# %%
df_both_cytotoxic = pd.concat([df_pos1_cytotoxic, df_pos2_cytotoxic])
df_both_naive = pd.concat([df_pos1_naive, df_pos2_naive])

# %%
# Tumor cell: marker expression versus distance with respect to T cell:
df_cancer_cyto = df_both_cytotoxic[df_both_cytotoxic.Cell_type_name == 'Cytotoxic T cell'].copy()
df_cancer_cyto.drop(['Row', 'Cols', 'FOV', 'Cell Type', 'Cell_label'], axis=1, inplace=True)


df_cancer_naive = df_both_naive[df_both_naive.Cell_type_name == 'Naive T cell'].copy()
df_cancer_naive.drop(['Row', 'Cols', 'FOV', 'Cell Type', 'Cell_label'], axis=1, inplace=True)

df_all_T = pd.concat([df_cancer_cyto, df_cancer_naive])

# %%
df_all_T[f'Distance cancer-cell'] = pd.qcut(df_all_T['Distance to cancer-cell'], 5, duplicates='drop')
LE = LabelEncoder()
df_all_T[f'Distance cancer-cell bucket'] = LE.fit_transform(df_all_T[f'Distance cancer-cell'])

# %%
df_cytotoxic = df_all_T[df_all_T['Cell_type_name'] == 'Cytotoxic T cell']
df_naive = df_all_T[df_all_T['Cell_type_name'] == 'Naive T cell']

# %%
df_all_T_corrected = pd.concat([df_cytotoxic, df_naive])

# %%
# Boxplot per cell for cancer cell comparison
print("\nGenerating and saving cancer cell comparison plots...")
save_path = plots_save_path / "cancer_cell_comparison"
save_path.mkdir(exist_ok=True)

for i in df_all_cancer_corrected.columns:
    if i in ['neighbor', 'Distance T-cell bucket', 'Distance to T-cell', 'Distance T-cell']:
        continue
    print(f"  Plotting for marker: {i}")
    A = df_all_cancer_corrected[[i, 'neighbor',"Distance T-cell"]]
    plotting = {
        "data": A,
        "x": "Distance T-cell",
        "y": i,
        "hue": "neighbor"
    }
    # Add .png extension for saving
    full_path = os.path.join(save_path, f'cancer_{i}.png')
    # Correctly pass save_path as a keyword argument
    plot(plotting, save_path=full_path)
    sns.despine()

# %%
# Boxplot per cell for T-cell comparison
print("\nGenerating and saving T-cell comparison plots...")
save_path = plots_save_path / "T_cell_comparison"
save_path.mkdir(exist_ok=True)
    
for i in df_all_T_corrected.columns:
    if i in ['Cell_type_name', 'Distance cancer-cell', 'Distance to cancer-cell', 'Distance cancer-cell bucket']:
        continue
    print(f"  Plotting for marker: {i}")
    A = df_all_T_corrected[[i, 'Cell_type_name',"Distance cancer-cell"]]
    plotting = {
        "data": A,
        "x": "Distance cancer-cell",
        "y": i,
        "hue": "Cell_type_name"
    }
    # Add .png extension for saving
    full_path = os.path.join(save_path, f'T_{i}.png')
    # Correctly pass save_path as a keyword argument
    plot(plotting, save_path=full_path)
    sns.despine()

# %%
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

all_ratio_df = pd.DataFrame(columns = ['0', '1', '2', '3', '4','marker'])
all_p_val_df = pd.DataFrame(columns = ['0', '1', '2', '3', '4','marker'])
for i in df_all_cancer_corrected:
    
    if (i == 'neighbor') | (i == 'Distance T-cell') | (i == 'Distance to T-cell') | (i == 'Distance T-cell bucket'):
        continue
#     print(i)
    df = df_all_cancer_corrected[[i, 'neighbor',"Distance T-cell bucket"]]
    groups = df.groupby(['Distance T-cell bucket'])


    medians_cytotoxic = []
    medians_naive = []
    ratio = []
    p_values = []
    distances = []


    for (distance), data in groups:
        cytotoxic_data = data[data['neighbor'] == 'Cytotoxic T cell'][i]
        naive_data = data[data['neighbor'] == 'Naive T cell'][i]
        stat, p_value = mannwhitneyu(cytotoxic_data, naive_data)
        medians_cytotoxic.append(cytotoxic_data.median() + 0.01)
        medians_naive.append(naive_data.median() + 0.01)
        ratio.append((cytotoxic_data.median() + 0.01)/(naive_data.median() + 0.01))
        p_values.append(p_value)
        distances.append(distance)
    reject, corrected_p_values, _, _ = multipletests(p_values, method='bonferroni')
    ratio.append(i)
    corrected_p_values = corrected_p_values.tolist()
    corrected_p_values.append(i)
    
    new_ratio_row = pd.DataFrame([ratio], columns=all_ratio_df.columns)
    all_ratio_df = pd.concat([all_ratio_df, new_ratio_row], ignore_index=True)
    
    new_pval_row = pd.DataFrame([corrected_p_values], columns=all_p_val_df.columns)
    all_p_val_df = pd.concat([all_p_val_df, new_pval_row], ignore_index=True)

# %%
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

all_df_rows = []
for i in df_all_cancer_corrected:
    
    if (i == 'neighbor') | (i == 'Distance T-cell') | (i == 'Distance to T-cell') | (i == 'Distance T-cell bucket'):
        continue
#     print(i)
    df = df_all_cancer_corrected[[i, 'neighbor',"Distance T-cell bucket"]]
    groups = df.groupby(['Distance T-cell bucket'])

    df_temp_rows = []
    p_values = []
    for (distance), data in groups:
        cytotoxic_data = data[data['neighbor'] == 'Cytotoxic T cell'][i]
        naive_data = data[data['neighbor'] == 'Naive T cell'][i]
        stat, p_value = mannwhitneyu(cytotoxic_data, naive_data)
        df_temp_rows.append({
            'Marker': i, 
            'Distance': distance, 
            'Cytotoxic Median': cytotoxic_data.median(), 
            'Naive Median': naive_data.median()
        })
        p_values.append(p_value)
    
    df_temp = pd.DataFrame(df_temp_rows)
    reject, corrected_p_values, _, _ = multipletests(p_values, method='bonferroni')
    df_temp['p-val'] = corrected_p_values
    all_df_rows.append(df_temp)

all_df = pd.concat(all_df_rows, ignore_index=True)

# %%
all_p_val_df.index = all_p_val_df['marker']
all_p_val_df = all_p_val_df.drop(['marker'], axis = 1)
all_ratio_df.index = all_ratio_df['marker']
all_ratio_df = all_ratio_df.drop(['marker'], axis = 1)

# %%
def size_function(value):
    if value > 0.05:
        return 0.03
    elif value > 0.01:
        return 0.15
    elif value > 0.005:
        return 0.25
    else:
        return 0.35
# df = df.applymap(size_function)


# %%
from matplotlib.collections import PatchCollection
import matplotlib
from matplotlib import pyplot as plt, patches

plt.style.use("classic")
plt.style.use("seaborn-v0_8-white")

xlabels = all_ratio_df.columns.tolist()
ylabels = all_ratio_df.index.tolist()
M = len(xlabels)
N = len(ylabels)

x, y = np.meshgrid(np.arange(M), np.arange(N))
s = all_p_val_df.applymap(size_function).values
c = all_ratio_df.values
print(c)
# norm = mcolors.CenteredNorm(halfrange = 1.5)
norm = mcolors.CenteredNorm(vcenter = 1)
c = norm(c)
print(c)

# Get size and color
# R = s / s.max(axis=1)[:, np.newaxis] / 2.5
R = s

circles = [plt.Circle((j, i), radius=r, ec='none') for r, j, i in zip(R.flat, x.flat, y.flat)]
col = PatchCollection(circles, array=c.flatten(), cmap="RdBu_r")

# Plot image
fig, ax = plt.subplots(figsize=(3.3, 18))
ax.add_collection(col)
ax.set(
    xticks=np.arange(M), yticks=np.arange(N), xticklabels=xlabels, yticklabels=ylabels
)
ax.set_xticks(np.arange(M + 1) - 0.2, minor=True)
ax.set_yticks(np.arange(N + 1) - 0.1, minor=True)
ax.set_xticklabels(xlabels,  ha="center")
ax.tick_params(which="major", length=5)
ax.tick_params(top=False, right=False)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
ax.invert_yaxis()

# Color legend
cbar = fig.colorbar(col, cax=fig.add_axes([0.92, 0.1, 0.02, 0.2]), ticks=[0.1, 1, 2])
cbar.ax.set_yticklabels(["0.1", '1', "2"])
cbar.set_label("log2 (cancer_cyto/ cancer_naive)")

thresholds = [0.05, 0.01, 0.005]

h = [
    ax.plot([], [], color="gray", marker="o", ms=i , ls="", 
            label=("> " + str(threshold)) if threshold != thresholds[-1] else ("< " + str(threshold)))[0]
    for i, threshold in zip([3, 10, 15, 20], thresholds + [thresholds[-1]])
]


leg = ax.legend(
    handles=h, loc="lower left", bbox_to_anchor=(1.0, 0.3, 0.02, 0.4), fontsize="medium"
)
leg.set_title("adjusted p-value", prop={"size": 14})

plt.savefig(plots_save_path / "marker_ratio_pvalue_dotplot.png", bbox_inches='tight')
plt.close(fig) # Close the figure to free memory


# %%
def dotplot_v2(df, key1, key2, title, save_path=None):
    def custom_wrap(text, width):
        lines = []
        for line in text.split('\\'):
            lines.extend(textwrap.wrap(line, width=width))
        return '\n'.join(lines)

    color_dict = {0: 'seashell', 3.33: 'lightsalmon', 6.66: 'tomato', 10: 'red'}
    df['ratio'] = df[key1]/df[key2]
    all_usage = df['Marker'].unique()

    fig, ax = plt.subplots(figsize=(3, len(all_usage) * 0.1))

    color_bar_para = df['p-val']
    y_width = 0.4
    x_width = 0.4
    y_pad = 0.1
    x_pad = 0.2
    scaling_factor = 20.0
    
    y_base = 0
    colors = df['p-val'].values

    all_cell_counts = 0
    for i in all_usage:
        curr_usage = df[df['Marker'] == i]
        cell_type = np.array(curr_usage['Distance'])
        function = np.array(curr_usage['Marker'])

        num_cells = len(curr_usage)
        num_rows = 1
        num_cols = num_cells

        cell_counts = 1

        for column in range(num_cols):
            for inner_row in range(num_rows):
                x_left = (2 * x_width + x_pad) * column
                x_right = x_left + x_width + 0.01
                y = y_base + inner_row * y_width

                if cell_counts <= num_cells:
                    ax.add_patch(Rectangle((x_right, y), x_width, y_width, color = 'steelblue') )
                    ax.add_patch(Rectangle((x_left, y), x_width, y_width, color = 'darkorange') )

                    s1 = scaling_factor * (curr_usage.iloc[cell_counts - 1, :])[key1]
                    s2 = scaling_factor * (curr_usage.iloc[cell_counts - 1, :])[key2]
                    c1 = color_dict[colors[all_cell_counts]]
                    c2 = color_dict[colors[all_cell_counts]]

                    ax.scatter(x_left + 0.5 * x_width, y + 0.5 * y_width, s = s1, color =  c1, edgecolors=c1)
                    ax.scatter(x_right + 0.5 * x_width, y + 0.5 * y_width, s = s2, color = c2, edgecolors=c2)
                    if all_cell_counts < 5:
                        ax.text(x_right, y - 0.4,cell_type[cell_counts - 1], ha = 'center', fontsize = 3, weight = 'bold')
                cell_counts = cell_counts + 1
                all_cell_counts = all_cell_counts + 1

        func = custom_wrap(function[0], 25)
        ax.text(-0.1, y_base + 0.3 * y_width,func, ha = 'right', fontsize = 3, wrap = True, weight = 'bold')
        y_base = y_base + y_width * num_rows 
        y_base = y_base + y_pad
    
    legend_colors = [Patch(facecolor='darkorange', edgecolor='darkorange',label='Cytotoxic T'),
                    Patch(facecolor='steelblue', edgecolor='steelblue',label='Naive T'),
                     Line2D([0], [0], marker='o', color="w", label='> 0.05', markerfacecolor=color_dict[0], markersize=5),
                     Line2D([0], [0], marker='o', color="w", label='> 0.01', markerfacecolor=color_dict[3.33], markersize=5),
                     Line2D([0], [0], marker='o', color="w", label='> 0.005', markerfacecolor=color_dict[6.66], markersize=5),
                    Line2D([0], [0], marker='o', color="w", label='< 0.005', markerfacecolor=color_dict[10], markersize=5)]
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    leg_color = plt.legend(handles=legend_colors, loc='upper left', prop={'size': 4},bbox_to_anchor=(1.05, 1))
    leg_color.get_frame().set_linewidth(0.0)

    plt.title(title, fontsize = 8)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()

# %%
def size_function(value):
    if value > 0.05:
        return 0
    elif value > 0.01:
        return 3.33
    elif value > 0.005:
        return 6.66
    else:
        return 10

# %%
all_df['p-val'] = pd.DataFrame(all_df['p-val']).applymap(size_function)
all_df['Marker'] = all_df['Marker'].replace('ASNA', 'ASNS')
dotplot_v2(all_df, 'Cytotoxic Median', 'Naive Median', 'Cancer Cell Comparison', save_path=plots_save_path / "cancer_cell_comparison_dotplot.png")

# %%
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

all_df_rows = []
for i in df_all_T_corrected:
    
    if (i == 'Cell_type_name') | (i == 'Distance cancer-cell') | (i == 'Distance to cancer-cell') | (i == 'Distance cancer-cell bucket'):
        continue
    df = df_all_T_corrected[[i, 'Cell_type_name',"Distance cancer-cell bucket"]]
    groups = df.groupby(['Distance cancer-cell bucket'])

    df_temp_rows = []
    p_values = []
    for (distance), data in groups:
        cytotoxic_data = data[data['Cell_type_name'] == 'Cytotoxic T cell'][i]
        naive_data = data[data['Cell_type_name'] == 'Naive T cell'][i]
        stat, p_value = mannwhitneyu(cytotoxic_data, naive_data)
        df_temp_rows.append({
            'Marker': i, 
            'Distance': distance, 
            'Cytotoxic Median': cytotoxic_data.median(), 
            'Naive Median': naive_data.median()
        })
        p_values.append(p_value)

    df_temp = pd.DataFrame(df_temp_rows)
    reject, corrected_p_values, _, _ = multipletests(p_values, method='bonferroni')
    df_temp['p-val'] = corrected_p_values
    all_df_rows.append(df_temp)

all_df = pd.concat(all_df_rows, ignore_index=True)

# %%
all_df['p-val'] = pd.DataFrame(all_df['p-val']).applymap(size_function)

# %%
all_df['Marker'] = all_df['Marker'].replace('ASNA', 'ASNS')

# %%
dotplot_v2(all_df, 'Cytotoxic Median', 'Naive Median', 'T Cell Comparison', save_path=plots_save_path / "t_cell_comparison_dotplot.png")



