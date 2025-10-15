# %%
# #############################################################################
# Script modified to run in parallel and save all plots automatically.
# Key changes:
# 1. Matplotlib backend is set to 'Agg' for non-interactive plotting.
# 2. A 'Plots' directory is created to store all generated figures.
# 3. HDF5 file reading and initial image processing are parallelized.
# 4. Intensity calculation per image channel is parallelized.
# 5. All plotting calls are modified to save the figure instead of displaying it.
# 6. Added scaled versions for every boxplot for comparison.
# 7. Normalization order changed to: 1. Mean Center, 2. Min-Max Scale.
# 8. Increased boxplot line thickness and annotation font size for clarity.
# 9. Added plot for Figure 2c: General cancer cell expression by co-culture condition.
# 10. FIXED: Added data type coercion to prevent 'isnan' TypeError during statistical tests.
# 11. MODIFIED: Combined neighbor vs. non-neighbor plots into a single comparative plot.
# #############################################################################

import os
import h5py
from pathlib import Path
import numpy as np
import pandas as pd
from functools import reduce

# --- Setup for Non-Interactive Plotting ---
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent plots from displaying
import matplotlib.pyplot as plt
import seaborn as sns

import skimage
import imageio
import tifffile as tf
import cv2

from skimage.exposure import match_histograms, rescale_intensity
from skimage import exposure, util, filters, restoration, measure
from cellpose import models, core
from joblib import Parallel, delayed
from sklearn.neighbors import NearestNeighbors, BallTree
from sklearn.preprocessing import LabelEncoder
import scanpy as sc
from statannotations.Annotator import Annotator

# --- Create a directory to save plots ---
# This will create a "Plots" folder in the parent directory of the script's location
try:
    script_path = Path(__file__).resolve()
    plots_dir = script_path.parent.parent / "Plots"
except NameError:
    # If __file__ is not defined (e.g., in an interactive environment)
    # use the current working directory.
    script_path = Path.cwd()
    plots_dir = script_path.parent / "Plots"

os.makedirs(plots_dir, exist_ok=True)
print(f"Plots will be saved to: {plots_dir}")


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
# ################### PARALLEL HELPER FUNCTIONS ###################

def process_hdf5(path):
    """Reads an HDF5 file, processes the images, and returns them with markers."""
    with h5py.File(path, "r") as f:
        markers = f['imgs'].attrs['Marker']
        imgs = f['imgs'][:]
    imgs_processed = skimage.exposure.adjust_log(imgs)
    imgs_processed = np.clip(imgs_processed, a_min=0, a_max=imgs_processed.max())
    imgs_processed = contrast_str(imgs_processed, n_min=0, n_max=100)
    return imgs_processed, markers

def get_channel_df(img, mask, marker):
    """Processes one channel and returns a dataframe with the correct column name."""
    props = skimage.measure.regionprops_table(
        mask, img, properties=["label", "mean_intensity", "area"]
    )
    df_prop = pd.DataFrame(props)
    df_prop["mean_intensity"] = df_prop["mean_intensity"]
    df_prop.drop("area", axis=1, inplace=True)
    df_prop["mean_intensity"] = np.arcsinh(df_prop[["mean_intensity"]])
    df_prop.columns = ["Cell_label", marker]
    return df_prop

# ###############################################################

# %%
if core.use_gpu() == False:
    use_GPU = False
else:
    print('GPU applied')
    use_GPU = True

# %%
# cytotoxic T cell - PARALLEL LOADING
print("Processing cytotoxic T cell HDF5 files in parallel...")
cytotoxic_paths = [
    r"Y:\coskun-lab\Thomas\20_MouseCocultureMatrigel\data\CytoCoc\hdf5\processed\CD8T_pos1_5.hdf5",
    r"Y:\coskun-lab\Thomas\20_MouseCocultureMatrigel\data\CytoCoc\hdf5\processed\CD8T_pos2_16.hdf5"
]
# Use n_jobs=-1 to use all available CPU cores
results_cytotoxic = Parallel(n_jobs=-1)(delayed(process_hdf5)(p) for p in cytotoxic_paths)
imgs_processed_all_cytotoxic = [res[0] for res in results_cytotoxic]
markers_cytotoxic = results_cytotoxic[0][1]  # Assuming markers are the same for all files

# %%
# naive T cell - PARALLEL LOADING
print("Processing naive T cell HDF5 files in parallel...")
naive_paths = [
    r"Y:\coskun-lab\Thomas\20_MouseCocultureMatrigel\data\NaiveCoC\hdf5\processed\CD8T_pos1_10.hdf5",
    r"Y:\coskun-lab\Thomas\20_MouseCocultureMatrigel\data\NaiveCoC\hdf5\processed\CD8T_pos2_10.hdf5"
]
results_naive = Parallel(n_jobs=-1)(delayed(process_hdf5)(p) for p in naive_paths)
imgs_processed_all_naive = [res[0] for res in results_naive]
markers_naive = results_naive[0][1]  # Assuming markers are the same for all files


# %%
from skimage.io import imread
naive_cyto_mask_pos1 = imread("Y:\coskun-undergrads\Mingshuang\Thomas\CoCulture\imgs\masks\CD8T_pos1_10_cp_masks.png")
naive_cyto_mask_pos2 = imread("Y:\coskun-undergrads\Mingshuang\Thomas\CoCulture\imgs\masks\CD8T_pos2_10_cp_masks.png")

cytotoxic_cyto_mask_pos1 = tf.imread(r"Y:\coskun-lab\Thomas\20_MouseCocultureMatrigel\data\CytoCoc\masks\pos1_5.tif")
cytotoxic_cyto_mask_pos2 = tf.imread(r"Y:\coskun-lab\Thomas\20_MouseCocultureMatrigel\data\CytoCoc\masks\pos2_16.tif")

# %%
# Cytotoxic - PARALLEL INTENSITY EXTRACTION
print("Extracting cytotoxic T cell intensities in parallel...")
cyto_masks_cytotoxic = [cytotoxic_cyto_mask_pos1, cytotoxic_cyto_mask_pos2]
df_appended_list_cytotoxic_all = []
for i, img_processed in enumerate(imgs_processed_all_cytotoxic):
    cyto_mask = cyto_masks_cytotoxic[i]
    tasks = [
        delayed(get_channel_df)(img, cyto_mask, markers_cytotoxic[j])
        for j, img in enumerate(img_processed)
        if not (markers_cytotoxic[j] == 'DNA' and j != 0)
    ]
    df_appended_list = Parallel(n_jobs=-1)(tasks)
    df_appended_list_cytotoxic_all.append(df_appended_list)

# %%
# Naive - PARALLEL INTENSITY EXTRACTION
print("Extracting naive T cell intensities in parallel...")
cyto_masks_naive = [naive_cyto_mask_pos1, naive_cyto_mask_pos2]
df_appended_list_naive_all = []
for i, img_processed in enumerate(imgs_processed_all_naive):
    cyto_mask = cyto_masks_naive[i]
    tasks = [
        delayed(get_channel_df)(img, cyto_mask, markers_naive[j])
        for j, img in enumerate(img_processed)
        if not (markers_naive[j] == 'DNA' and j != 0)
    ]
    df_appended_list = Parallel(n_jobs=-1)(tasks)
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
# Add centroid and FOV info
# cytotoxic
props = skimage.measure.regionprops_table(cytotoxic_cyto_mask_pos1, properties=["label", "centroid"])
rows, cols = props["centroid-0"], props["centroid-1"]
centroid_cytotoxic = np.array(list(zip(cols, rows)))
df_cell_intensity_pos1_cyto['Row'], df_cell_intensity_pos1_cyto['Cols'] = rows, cols
df_cell_intensity_pos1_cyto['FOV'] = 'pos1'
df_prop_cytotoxic = df_cell_intensity_pos1_cyto

props = skimage.measure.regionprops_table(cytotoxic_cyto_mask_pos2, properties=["label", "centroid"])
rows, cols = props["centroid-0"], props["centroid-1"]
centroid_cytotoxic = np.concatenate([centroid_cytotoxic, np.array(list(zip(cols, rows)))])
df_cell_intensity_pos2_cyto['Row'], df_cell_intensity_pos2_cyto['Cols'] = rows, cols
df_cell_intensity_pos2_cyto['FOV'] = 'pos2'
df_prop_cytotoxic = pd.concat([df_prop_cytotoxic, df_cell_intensity_pos2_cyto], ignore_index=True)


# %%
# naive
props = skimage.measure.regionprops_table(naive_cyto_mask_pos1, properties=["label", "centroid"])
rows, cols = props["centroid-0"], props["centroid-1"]
centroid_naive = np.array(list(zip(cols, rows)))
df_cell_intensity_pos1_naive['Row'], df_cell_intensity_pos1_naive['Cols'] = rows, cols
df_cell_intensity_pos1_naive['FOV'] = 'pos1'
df_prop_naive = df_cell_intensity_pos1_naive

props = skimage.measure.regionprops_table(naive_cyto_mask_pos2, properties=["label", "centroid"])
rows, cols = props["centroid-0"], props["centroid-1"]
centroid_naive = np.concatenate([centroid_naive, np.array(list(zip(cols, rows)))])
df_cell_intensity_pos2_naive['Row'], df_cell_intensity_pos2_naive['Cols'] = rows, cols
df_cell_intensity_pos2_naive['FOV'] = 'pos2'
df_prop_naive = pd.concat([df_prop_naive, df_cell_intensity_pos2_naive], ignore_index=True)

# %%
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
# Plot and save E-cadherin distribution
plt.figure(figsize=(8, 6))
plt.hist(df_marker_naive['E-cadherin'], bins=50, density=True, alpha=0.7, color='skyblue')
title = 'Kernel_Density_Estimate_of_E-cadherin'
plt.title(title.replace('_', ' '))
plt.xlabel('values')
plt.ylabel('Density')
plt.grid(True)
save_path = plots_dir / f"{title}.png"
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.close()
print(f"Saved plot: {save_path}")


# %%
# Plot and save DNA distribution
plt.figure(figsize=(8, 6))
plt.hist(df_DNA_naive['DNA'], bins=30, density=True, alpha=0.7, color='skyblue')
title = 'Kernel_Density_Estimate_of_DNA'
plt.title(title.replace('_', ' '))
plt.xlabel('DNA values')
plt.ylabel('Density')
plt.grid(True)
save_path = plots_dir / f"{title}.png"
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.close()
print(f"Saved plot: {save_path}")


# %%
# Define cell type
df_prop_cytotoxic['Cell Type'] = 0
df_prop_cytotoxic.loc[df_prop_cytotoxic['E-cadherin'] > 0.1, 'Cell Type'] = 1
df_prop_cytotoxic.loc[df_prop_cytotoxic['E-cadherin'] <= 0.1, 'Cell Type'] = 2

df_prop_naive['Cell Type'] = 0
df_prop_naive.loc[df_prop_naive['E-cadherin'] > 0.1, 'Cell Type'] = 1
df_prop_naive.loc[df_prop_naive['E-cadherin'] <= 0.1, 'Cell Type'] = 2

# %%
df_prop_cytotoxic['Cell_type_name'] = df_prop_cytotoxic['Cell Type'].apply(lambda x: {0:'Other', 1: 'Cancer cell', 2:'Cytotoxic T cell'}.get(x,x))
df_cytotoxic = df_prop_cytotoxic.loc[:,np.concatenate([markers_cytotoxic, ['Cell_type_name']])].copy()
df_cytotoxic = df_cytotoxic.loc[:, ~df_cytotoxic.columns.duplicated()].copy() # Remove duplicate columns
df_cytotoxic = df_cytotoxic[df_cytotoxic.Cell_type_name != 'Other']

df_prop_naive['Cell_type_name'] = df_prop_naive['Cell Type'].apply(lambda x: {0:'Other', 1: 'Cancer cell', 2:'Naive T cell'}.get(x,x))
df_naive = df_prop_naive.loc[:,np.concatenate([markers_naive, ['Cell_type_name']])].copy()
df_naive = df_naive.loc[:, ~df_naive.columns.duplicated()].copy() # Remove duplicate columns
df_naive = df_naive[df_naive.Cell_type_name != 'Other']

# FIX: Proactively convert all marker columns to numeric and drop invalid rows
marker_cols_cyto_list = [col for col in df_cytotoxic.columns if col != 'Cell_type_name']
for col in marker_cols_cyto_list:
    df_cytotoxic[col] = pd.to_numeric(df_cytotoxic[col], errors='coerce')
df_cytotoxic.dropna(subset=marker_cols_cyto_list, how='any', inplace=True)

marker_cols_naive_list = [col for col in df_naive.columns if col != 'Cell_type_name']
for col in marker_cols_naive_list:
    df_naive[col] = pd.to_numeric(df_naive[col], errors='coerce')
df_naive.dropna(subset=marker_cols_naive_list, how='any', inplace=True)


# %%
# ################### PLOTTING FUNCTIONS (MODIFIED TO SAVE) ###################
# Define a consistent color palette for all plots to use
master_color_palette = {
    'Naive T cell': '#1f77b4',                # Blue
    'Cytotoxic T cell': '#ff7f0e',            # Orange
    'Interacting with Naive T': '#1f77b4',     # Blue
    'Non-Interacting with Naive T': '#aec7e8', # Light Blue
    'Interacting with CytoT': '#ff7f0e',       # Orange
    'Non-Interacting with CytoT': '#ffbb78'    # Light Orange
}


def plot(plotting, save_filename, title=None, figsize=(40,10)):
    # Check if there are enough unique values in hue to create pairs for annotation
    unique_hues = plotting['data'][plotting['hue']].unique()

    # Ensure there's a defined order for comparison if possible
    # New order: Cytotoxic (Orange) first, then Naive (Blue)
    ordered_hues = [h for h in ['Interacting with CytoT', 'Non-Interacting with CytoT',
                               'Interacting with Naive T', 'Non-Interacting with Naive T',
                               'Cytotoxic T cell', 'Naive T cell',
                               'Interacting w/ Cytotoxic T', 'Interacting w/ Naive T'
                               ] if h in unique_hues]
    if len(ordered_hues) < len(unique_hues):
        ordered_hues.extend([h for h in unique_hues if h not in ordered_hues])

    if len(unique_hues) < 2:
        pairs = []
    else:
        # Create pairs for statistical annotation based on the interaction type
        pairs = []
        markers = plotting['data'][plotting['x']].unique()

        # This logic now covers all plot types correctly
        if 'Interacting with Naive T' in unique_hues and 'Non-Interacting with Naive T' in unique_hues:
            pairs.extend([((m, 'Interacting with Naive T'), (m, 'Non-Interacting with Naive T')) for m in markers])
        if 'Interacting with CytoT' in unique_hues and 'Non-Interacting with CytoT' in unique_hues:
            pairs.extend([((m, 'Interacting with CytoT'), (m, 'Non-Interacting with CytoT')) for m in markers])
        if 'Cytotoxic T cell' in unique_hues and 'Naive T cell' in unique_hues:
             pairs.extend([((m, 'Cytotoxic T cell'), (m, 'Naive T cell')) for m in markers])
        # Added logic for the new combined interacting plot
        if 'Interacting w/ Naive T' in unique_hues and 'Interacting w/ Cytotoxic T' in unique_hues:
            pairs.extend([((m, 'Interacting w/ Cytotoxic T'), (m, 'Interacting w/ Naive T')) for m in markers])


    with sns.plotting_context('talk', font_scale=1.5):
        fig, ax = plt.subplots(figsize=figsize)
        # Use the master palette for consistent coloring and add linewidth
        ax = sns.boxplot(**plotting, showfliers=False, ax=ax, palette=master_color_palette, order=plotting['data'][plotting['x']].unique(), hue_order=ordered_hues, linewidth=3)

        if title:
            ax.set_title(title, fontsize=24, pad=20)

        # Only add annotations if there are pairs to compare
        if pairs:
            try:
                annot = Annotator(ax, pairs, **plotting, hue_order=ordered_hues)
                # Configure annotation font size
                annot.configure(test='Mann-Whitney', text_format='star',comparisons_correction="bonferroni",loc='inside', verbose=0, fontsize=24)
                result = annot.apply_test().annotate()
            except TypeError:
                print(f"Warning: Could not perform statistical annotation for plot '{title or save_filename}'. "
                      "This is likely due to data type issues or insufficient data in one of the categories. "
                      "Plot will be generated without annotations.")
                pass

        plt.xticks(rotation=45, ha='right')
        # increase font sizes of tick labels
        ax.tick_params(axis='x', labelsize=48)
        ax.tick_params(axis='y', labelsize=48)
        ax.xaxis.label.set_size(56)
        ax.yaxis.label.set_size(56)
        # Customize legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=labels, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title=plotting['hue'].replace('_', ' ').title())

        save_path = plots_dir / save_filename
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f"Saved plot: {save_path}")

# ##########################################################################

# %%
# --- Generate Un-normalized version of the first plot for comparison ---
df_melt_cytotoxic_unscaled = df_cytotoxic.melt(id_vars=['Cell_type_name'], value_name='Expression', var_name='Marker')
df_melt_cytotoxic_unscaled.columns = ['Cell Type', 'Marker', 'Expression']

df_melt_naive_unscaled = df_naive.melt(id_vars=['Cell_type_name'], value_name='Expression', var_name='Marker')
df_melt_naive_unscaled.columns = ['Cell Type', 'Marker', 'Expression']

df_melt_all_unscaled = pd.concat([df_melt_naive_unscaled, df_melt_cytotoxic_unscaled])
df_melt_all_corrected_unscaled = df_melt_all_unscaled[
    (~(df_melt_all_unscaled['Marker'] == 'WGA')) &
    (~(df_melt_all_unscaled['Marker'] == 'Concavalin A')) &
    (~(df_melt_all_unscaled['Cell Type'] == 'Cancer cell'))
]

# FIX: Ensure the expression column is numeric before plotting
df_melt_all_corrected_unscaled['Expression'] = pd.to_numeric(df_melt_all_corrected_unscaled['Expression'], errors='coerce')
df_melt_all_corrected_unscaled.dropna(subset=['Expression'], inplace=True)


# Boxplot per cell (Original Arcsinh Transformed Data)
plotting_unscaled = {
    "data": df_melt_all_corrected_unscaled,
    "x": "Marker",
    "y": "Expression",
    "hue": "Cell Type"
}
plot(plotting_unscaled, "Expression_by_Marker_and_Cell_Type_original.png", title="T-Cell Expression (Original)")
sns.despine()


# %%
# Normalize data: Mean center with Phalloidin first, then Min-Max scale
df_cytotoxic_value = df_cytotoxic.copy() # Use .copy() to avoid SettingWithCopyWarning
# Step 1: Mean centering
phalloidin_mean_cyto = df_cytotoxic_value['Phalloidin'].mean()
for column in df_cytotoxic_value.columns:
    if column in df_cytotoxic_value.columns and df_cytotoxic_value[column].dtype in ['float64', 'int64']:
        df_cytotoxic_value[column] -= phalloidin_mean_cyto
# Step 2: Min-Max scaling
marker_cols_to_norm = [col for col in df_cytotoxic_value.columns if df_cytotoxic_value[col].dtype in ['float64', 'int64']]
df_cytotoxic_value_norm = df_cytotoxic_value.copy()
df_cytotoxic_value_norm[marker_cols_to_norm] = (df_cytotoxic_value[marker_cols_to_norm] - df_cytotoxic_value[marker_cols_to_norm].min()) / (df_cytotoxic_value[marker_cols_to_norm].max() - df_cytotoxic_value[marker_cols_to_norm].min())
df_cytotoxic_value_norm = df_cytotoxic_value_norm.loc[:,~df_cytotoxic_value_norm.columns.duplicated()].copy()


df_naive_value = df_naive.copy() # Use .copy() to avoid SettingWithCopyWarning
# Step 1: Mean centering
phalloidin_mean_naive = df_naive_value['Phalloidin'].mean()
for column in df_naive_value.columns:
     if column in df_naive_value.columns and df_naive_value[column].dtype in ['float64', 'int64']:
        df_naive_value[column] -= phalloidin_mean_naive
# Step 2: Min-Max scaling
marker_cols_to_norm_naive = [col for col in df_naive_value.columns if df_naive_value[col].dtype in ['float64', 'int64']]
df_naive_value_norm = df_naive_value.copy()
df_naive_value_norm[marker_cols_to_norm_naive] = (df_naive_value[marker_cols_to_norm_naive] - df_naive_value[marker_cols_to_norm_naive].min()) / (df_naive_value[marker_cols_to_norm_naive].max() - df_naive_value[marker_cols_to_norm_naive].min())
df_naive_value_norm = df_naive_value_norm.loc[:,~df_naive_value_norm.columns.duplicated()].copy()


# %%
df_melt_cytotoxic = df_cytotoxic_value_norm.melt(id_vars=['Cell_type_name'], value_name='Expression', var_name='Marker')
df_melt_cytotoxic.columns = ['Cell Type', 'Marker', 'Expression']

df_melt_naive = df_naive_value_norm.melt(id_vars=['Cell_type_name'], value_name='Expression', var_name='Marker')
df_melt_naive.columns = ['Cell Type', 'Marker', 'Expression']

df_melt_all = pd.concat([df_melt_naive, df_melt_cytotoxic])
df_melt_all_corrected = df_melt_all[
    (~(df_melt_all['Marker'] == 'WGA')) &
    (~(df_melt_all['Marker'] == 'Concavalin A')) &
    (~(df_melt_all['Cell Type'] == 'Cancer cell'))
]

# FIX: Ensure the expression column is numeric before plotting
df_melt_all_corrected['Expression'] = pd.to_numeric(df_melt_all_corrected['Expression'], errors='coerce')
df_melt_all_corrected.dropna(subset=['Expression'], inplace=True)


# %%
# Boxplot per cell (T-Cells only)
plotting = {
    "data": df_melt_all_corrected,
    "x": "Marker",
    "y": "Expression",
    "hue": "Cell Type"
}
plot(plotting, "Expression_by_Marker_and_Cell_Type_normalized.png", title="T-Cell Expression (Normalized)")
sns.despine()

# %%
# Define columns that are NOT markers to exclude them before melting
non_marker_cols = ['Cell_label', 'Row', 'Cols', 'FOV', 'Cell Type', 'Cell_type_name',
                   'Distance to T-cell', 'T_cell_neighbors', 'Tumour_neighbors']
# Get the list of actual marker columns by excluding the non-marker ones
marker_cols_cyto = [col for col in df_cytotoxic.columns if col not in non_marker_cols]
marker_cols_naive = [col for col in df_naive.columns if col not in non_marker_cols]


# ################### NEW PLOT FOR FIGURE 2c ###################

# --- Prepare data for general cancer cell comparison by condition ---
# Filter for cancer cells from the scaled cytotoxic data
df_cancer_cyto_all = df_cytotoxic_value_norm[df_cytotoxic_value_norm['Cell_type_name'] == 'Cancer cell'].copy()
df_cancer_cyto_all.rename(columns={'Cell_type_name': 'Cell Type'}, inplace=True)
df_cancer_cyto_all['Cell Type'] = 'Cytotoxic T cell'

# Filter for cancer cells from the scaled naive data
df_cancer_naive_all = df_naive_value_norm[df_naive_value_norm['Cell_type_name'] == 'Cancer cell'].copy()
df_cancer_naive_all.rename(columns={'Cell_type_name': 'Cell Type'}, inplace=True)
df_cancer_naive_all['Cell Type'] = 'Naive T cell'

# Combine and melt
df_cancer_all_conditions = pd.concat([df_cancer_cyto_all, df_cancer_naive_all])

# Get a list of all unique marker columns from both conditions to use for melting
all_marker_cols = list(set(marker_cols_cyto) | set(marker_cols_naive))
valid_markers = [col for col in all_marker_cols if col in df_cancer_all_conditions.columns]
df_melt_cancer_conditions = df_cancer_all_conditions.melt(id_vars=['Cell Type'], value_vars=valid_markers, value_name='Intensity', var_name='Marker')

# FIX: Ensure the intensity column is numeric before plotting
df_melt_cancer_conditions['Intensity'] = pd.to_numeric(df_melt_cancer_conditions['Intensity'], errors='coerce')
df_melt_cancer_conditions.dropna(subset=['Intensity'], inplace=True)

# --- Generate and save the plots ---
# Scaled version
plotting_fig2c_scaled = {
    "data": df_melt_cancer_conditions,
    "x": "Marker",
    "y": "Intensity",
    "hue": "Cell Type"
}
plot(plotting_fig2c_scaled, "Cancer_Expression_by_Condition_scaled.png", title="Cancer Cell Expression by Co-Culture Condition (Scaled)")

# Original (unscaled) version
df_cancer_cyto_orig = df_cytotoxic[df_cytotoxic['Cell_type_name'] == 'Cancer cell'].copy()
df_cancer_cyto_orig.rename(columns={'Cell_type_name': 'Cell Type'}, inplace=True)
df_cancer_cyto_orig['Cell Type'] = 'Cytotoxic T cell'

df_cancer_naive_orig = df_naive[df_naive['Cell_type_name'] == 'Cancer cell'].copy()
df_cancer_naive_orig.rename(columns={'Cell_type_name': 'Cell Type'}, inplace=True)
df_cancer_naive_orig['Cell Type'] = 'Naive T cell'

df_cancer_orig_conditions = pd.concat([df_cancer_cyto_orig, df_cancer_naive_orig])
valid_markers_orig = [col for col in all_marker_cols if col in df_cancer_orig_conditions.columns]
df_melt_cancer_orig_conditions = df_cancer_orig_conditions.melt(id_vars=['Cell Type'], value_vars=valid_markers_orig, value_name='Intensity', var_name='Marker')

# FIX: Ensure the intensity column is numeric before plotting
df_melt_cancer_orig_conditions['Intensity'] = pd.to_numeric(df_melt_cancer_orig_conditions['Intensity'], errors='coerce')
df_melt_cancer_orig_conditions.dropna(subset=['Intensity'], inplace=True)

plotting_fig2c_orig = {
    "data": df_melt_cancer_orig_conditions,
    "x": "Marker",
    "y": "Intensity",
    "hue": "Cell Type"
}
plot(plotting_fig2c_orig, "Cancer_Expression_by_Condition_original.png", title="Cancer Cell Expression by Co-Culture Condition (Original)")


# %%
# Further analysis and plotting
def get_distance(centroid_fit, centroid_search):
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(centroid_fit)
    d, n = neigh.kneighbors(centroid_search, return_distance=True)
    return d

df_pos1_cytotoxic = df_prop_cytotoxic[df_prop_cytotoxic['FOV'] == 'pos1'].copy()
df_pos2_cytotoxic = df_prop_cytotoxic[df_prop_cytotoxic['FOV'] == 'pos2'].copy()
df_pos1_naive = df_prop_naive[df_prop_naive['FOV'] == 'pos1'].copy()
df_pos2_naive = df_prop_naive[df_prop_naive['FOV'] == 'pos2'].copy()

#cytotoxic
centroid_all_pos1_cyto = centroid_cytotoxic[df_prop_cytotoxic.FOV == 'pos1']
centroid_tcell_pos1_cyto = centroid_all_pos1_cyto[df_pos1_cytotoxic.Cell_type_name == 'Cytotoxic T cell']
d = get_distance(centroid_tcell_pos1_cyto, centroid_all_pos1_cyto)
df_pos1_cytotoxic['Distance to T-cell'] = d

centroid_all_pos2_cyto = centroid_cytotoxic[df_prop_cytotoxic.FOV == 'pos2']
centroid_tcell_pos2_cyto = centroid_all_pos2_cyto[df_pos2_cytotoxic.Cell_type_name == 'Cytotoxic T cell']
d = get_distance(centroid_tcell_pos2_cyto, centroid_all_pos2_cyto)
df_pos2_cytotoxic['Distance to T-cell'] = d

#naive
centroid_all_pos1_naive = centroid_naive[df_prop_naive.FOV == 'pos1']
centroid_tcell_pos1_naive = centroid_all_pos1_naive[df_pos1_naive.Cell_type_name == 'Naive T cell']
d = get_distance(centroid_tcell_pos1_naive, centroid_all_pos1_naive)
df_pos1_naive['Distance to T-cell'] = d

centroid_all_pos2_naive = centroid_naive[df_prop_naive.FOV == 'pos2']
centroid_tcell_pos2_naive = centroid_all_pos2_naive[df_pos2_naive.Cell_type_name == 'Naive T cell']
d = get_distance(centroid_tcell_pos2_naive, centroid_all_pos2_naive)
df_pos2_naive['Distance to T-cell'] = d

df_both_cytotoxic = pd.concat([df_pos1_cytotoxic, df_pos2_cytotoxic])
df_both_naive = pd.concat([df_pos1_naive, df_pos2_naive])

df_cancer_cyto = df_both_cytotoxic[df_both_cytotoxic.Cell_type_name == 'Cancer cell'].copy()
df_cancer_cyto.drop(['Row', 'Cols', 'FOV', 'Cell Type', 'Cell_type_name', 'Cell_label'], axis=1, inplace=True)
df_cancer_cyto['neighbor'] = 'Cytotoxic T cell'

df_cancer_naive = df_both_naive[df_both_naive.Cell_type_name == 'Cancer cell'].copy()
df_cancer_naive.drop(['Row', 'Cols', 'FOV', 'Cell Type', 'Cell_type_name', 'Cell_label'], axis=1, inplace=True)
df_cancer_naive['neighbor'] = 'Naive T cell'

df_all_cancer = pd.concat([df_cancer_cyto, df_cancer_naive])

# %%
# Scatter plot for Ki67
plt.figure(figsize=(10, 8))
sns.scatterplot(data=df_all_cancer, x="Distance to T-cell", y="Ki67", hue='neighbor')
title = 'Ki67_Expression_vs_Distance_to_T-cell'
plt.title(title.replace('_', ' '))
save_path = plots_dir / f"{title}.png"
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.close()
print(f"Saved plot: {save_path}")

# %%
# Heatmap plots
df_cancer_cyto_dist = df_both_cytotoxic[df_both_cytotoxic.Cell_type_name == 'Cancer cell'].copy()
df_cancer_cyto_dist.sort_values('Distance to T-cell', inplace=True)
df_cancer_cyto_dist[f'Distance T-cell'] = pd.qcut(df_cancer_cyto_dist['Distance to T-cell'], 10, duplicates='drop')
df_cancer_cyto_dist[f'Distance T-cell'] = LabelEncoder().fit_transform(df_cancer_cyto_dist[f'Distance T-cell'])
df_sort_cyto = df_cancer_cyto_dist.drop(['DNA', 'Row', 'Cols', 'FOV', 'Cell Type', 'Cell_label', 'Cell_type_name'], axis=1).groupby('Distance T-cell').median()

adata = sc.AnnData(df_sort_cyto.iloc[:,1:])
adata.var_names = df_sort_cyto.iloc[:,1:].columns.tolist()
adata.obs['Proximity'] = df_sort_cyto.index.tolist()
adata.X = (adata.X - np.min(adata.X, axis=0)) / (np.max(adata.X, axis=0) - np.min(adata.X, axis=0))

with sns.plotting_context('poster', font_scale=1):
    sc.pl.heatmap(adata, adata.var_names, groupby='Proximity', swap_axes=True, cmap='bwr', figsize=(12,12), show=False)
    save_path = plots_dir / "Proximity_Heatmap_Cytotoxic.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved plot: {save_path}")


# %%
df_cancer_naive_dist = df_both_naive[df_both_naive.Cell_type_name == 'Cancer cell'].copy()
df_cancer_naive_dist.sort_values('Distance to T-cell', inplace=True)
df_cancer_naive_dist[f'Distance T-cell'] = pd.qcut(df_cancer_naive_dist['Distance to T-cell'], 10, duplicates='drop')
df_cancer_naive_dist[f'Distance T-cell'] = LabelEncoder().fit_transform(df_cancer_naive_dist[f'Distance T-cell'])
df_sort_naive = df_cancer_naive_dist.drop(['DNA', 'Row', 'Cols', 'FOV', 'Cell Type', 'Cell_label', 'Cell_type_name'], axis=1).groupby('Distance T-cell').median()

adata = sc.AnnData(df_sort_naive.iloc[:,1:])
adata.var_names = df_sort_naive.iloc[:,1:].columns.tolist()
adata.obs['Proximity'] = df_sort_naive.index.tolist()
adata.X = (adata.X - np.min(adata.X, axis=0)) / (np.max(adata.X, axis=0) - np.min(adata.X, axis=0))

with sns.plotting_context('poster', font_scale=1):
    sc.pl.heatmap(adata, adata.var_names, groupby='Proximity', swap_axes=True, cmap='bwr', figsize=(12,12), show=False)
    save_path = plots_dir / "Proximity_Heatmap_Naive.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved plot: {save_path}")


# %%
# ################### NEIGHBOR ANALYSIS SECTION ###################

def get_neighbors_in_radius(centroid_fit, centroid_search, r = 70):
    """Finds all neighbors within a given radius using BallTree."""
    kdt = BallTree(centroid_fit, metric='euclidean')
    # Find all points within radius r of each point in centroid_search
    ind = kdt.query_radius(centroid_search, r=r, return_distance=False)

    # For each point, remove itself from its list of neighbors if present
    for i in range(len(ind)):
        # Check if the point's own index is in its neighbor list and remove it
        self_index_mask = np.isin(ind[i], i) # More robust check for self-index
        if np.any(self_index_mask):
             ind[i] = np.delete(ind[i], np.where(self_index_mask))

    neighbours = pd.Series([len(x) > 0 for x in ind]) # Return a 1D Series instead of a DataFrame
    return neighbours

# --- Cytotoxic Neighbor Analysis ---
group_cytotoxic = df_prop_cytotoxic.groupby(['FOV'])
dfs_list_cytotoxic = []
for fov, df_group in group_cytotoxic:
    df_group_copy = df_group.copy()
    
    # Get centroids for each cell type
    centroid_tcell = df_group_copy[df_group_copy['Cell Type'] == 2][['Row', 'Cols']].values
    centroid_tumour = df_group_copy[df_group_copy['Cell Type'] == 1][['Row', 'Cols']].values
    
    # Find T-cell neighbors for Cancer cells
    if len(centroid_tcell) > 0 and len(centroid_tumour) > 0:
        neighbours_t_for_cancer = get_neighbors_in_radius(centroid_tcell, centroid_tumour, r=145)
        df_group_copy.loc[df_group_copy['Cell Type'] == 1, 'T_cell_neighbors'] = neighbours_t_for_cancer.values

    # Find Cancer cell neighbors for T-cells
    if len(centroid_tumour) > 0 and len(centroid_tcell) > 0:
        neighbours_cancer_for_t = get_neighbors_in_radius(centroid_tumour, centroid_tcell, r=145)
        df_group_copy.loc[df_group_copy['Cell Type'] == 2, 'Tumour_neighbors'] = neighbours_cancer_for_t.values

    dfs_list_cytotoxic.append(df_group_copy)

df_final_cytotoxic = pd.concat(dfs_list_cytotoxic).fillna(False)


# --- Naive Neighbor Analysis ---
group_naive = df_prop_naive.groupby(['FOV'])
dfs_list_naive = []
for fov, df_group in group_naive:
    df_group_copy = df_group.copy()

    # Get centroids for each cell type
    centroid_tcell = df_group_copy[df_group_copy['Cell Type'] == 2][['Row', 'Cols']].values
    centroid_tumour = df_group_copy[df_group_copy['Cell Type'] == 1][['Row', 'Cols']].values

    # Find T-cell neighbors for Cancer cells
    if len(centroid_tcell) > 0 and len(centroid_tumour) > 0:
        neighbours_t_for_cancer = get_neighbors_in_radius(centroid_tcell, centroid_tumour, r=143)
        df_group_copy.loc[df_group_copy['Cell Type'] == 1, 'T_cell_neighbors'] = neighbours_t_for_cancer.values

    # Find Cancer cell neighbors for T-cells
    if len(centroid_tumour) > 0 and len(centroid_tcell) > 0:
        neighbours_cancer_for_t = get_neighbors_in_radius(centroid_tumour, centroid_tcell, r=143)
        df_group_copy.loc[df_group_copy['Cell Type'] == 2, 'Tumour_neighbors'] = neighbours_cancer_for_t.values

    dfs_list_naive.append(df_group_copy)

df_final_naive = pd.concat(dfs_list_naive).fillna(False)


# %%
# ################### PREPARE ORIGINAL (UNSCALED) DATA FOR FINAL PLOTS ###################

# Define columns that are NOT markers to exclude them before melting
non_marker_cols = ['Cell_label', 'Row', 'Cols', 'FOV', 'Cell Type', 'Cell_type_name',
                   'Distance to T-cell', 'T_cell_neighbors', 'Tumour_neighbors']

# Get the list of actual marker columns by excluding the non-marker ones
marker_cols_cyto = [col for col in df_final_cytotoxic.columns if col not in non_marker_cols]
marker_cols_naive = [col for col in df_final_naive.columns if col not in non_marker_cols]


# Cancer cell analysis (neighboring T-cells)
df_subset_cyto_cancer = df_final_cytotoxic[df_final_cytotoxic['Cell Type'] == 1][marker_cols_cyto + ['T_cell_neighbors']].copy()
df_melt_cyto_cancer = df_subset_cyto_cancer.melt(id_vars='T_cell_neighbors', var_name='Marker', value_name='Intensity')
df_melt_cyto_cancer = df_melt_cyto_cancer[df_melt_cyto_cancer['T_cell_neighbors'] == True]
df_melt_cyto_cancer['Neighbor_T_cell'] = 'Cytotoxic T cell'

df_subset_naive_cancer = df_final_naive[df_final_naive['Cell Type'] == 1][marker_cols_naive + ['T_cell_neighbors']].copy()
df_melt_naive_cancer = df_subset_naive_cancer.melt(id_vars='T_cell_neighbors', var_name='Marker', value_name='Intensity')
df_melt_naive_cancer = df_melt_naive_cancer[df_melt_naive_cancer['T_cell_neighbors'] == True]
df_melt_naive_cancer['Neighbor_T_cell'] = 'Naive T cell'

df_melt_all_cancer_neighbors = pd.concat([df_melt_cyto_cancer, df_melt_naive_cancer])
df_melt_all_cancer_neighbors = df_melt_all_cancer_neighbors[
    (~df_melt_all_cancer_neighbors['Marker'].isin(['WGA', 'Concavalin A']))
]

# T-cell analysis (neighboring Tumour cells)
df_subset_cyto_tcell = df_final_cytotoxic[df_final_cytotoxic['Cell Type'] == 2][marker_cols_cyto + ['Tumour_neighbors']].copy()
df_melt_cyto_tcell = df_subset_cyto_tcell.melt(id_vars='Tumour_neighbors', var_name='Marker', value_name='Intensity')
df_melt_cyto_tcell = df_melt_cyto_tcell[df_melt_cyto_tcell['Tumour_neighbors'] == True]
df_melt_cyto_tcell['cell_type'] = 'Cytotoxic T cell'

df_subset_naive_tcell = df_final_naive[df_final_naive['Cell Type'] == 2][marker_cols_naive + ['Tumour_neighbors']].copy()
df_melt_naive_tcell = df_subset_naive_tcell.melt(id_vars='Tumour_neighbors', var_name='Marker', value_name='Intensity')
df_melt_naive_tcell = df_melt_naive_tcell[df_melt_naive_tcell['Tumour_neighbors'] == True]
df_melt_naive_tcell['cell_type'] = 'Naive T cell'

df_melt_all_tcell_neighbors = pd.concat([df_melt_cyto_tcell, df_melt_naive_tcell])
df_melt_all_tcell_neighbors = df_melt_all_tcell_neighbors[
    (~df_melt_all_tcell_neighbors['Marker'].isin(['WGA', 'Concavalin A']))
]

# Neighbor vs. Non-Neighbor Data Prep
# Cytotoxic
df_subset_cyto_neighbor_comp = df_final_cytotoxic[df_final_cytotoxic['Cell Type'] == 1][marker_cols_cyto + ['T_cell_neighbors']].copy()
df_subset_cyto_neighbor_comp['Interaction'] = df_subset_cyto_neighbor_comp['T_cell_neighbors'].map({ True: 'Interacting with CytoT', False: 'Non-Interacting with CytoT' })
df_melt_cyto_neighbor_comp = df_subset_cyto_neighbor_comp.drop('T_cell_neighbors', axis=1).melt(id_vars='Interaction', var_name='Marker', value_name='Intensity')
df_melt_cyto_neighbor_comp = df_melt_cyto_neighbor_comp[(~df_melt_cyto_neighbor_comp['Marker'].isin(['WGA', 'Concavalin A']))]
# Naive
df_subset_naive_neighbor_comp = df_final_naive[df_final_naive['Cell Type'] == 1][marker_cols_naive + ['T_cell_neighbors']].copy()
df_subset_naive_neighbor_comp['Interaction'] = df_subset_naive_neighbor_comp['T_cell_neighbors'].map({ True: 'Interacting with Naive T', False: 'Non-Interacting with Naive T' })
df_melt_naive_neighbor_comp = df_subset_naive_neighbor_comp.drop('T_cell_neighbors', axis=1).melt(id_vars='Interaction', var_name='Marker', value_name='Intensity')
df_melt_naive_neighbor_comp = df_melt_naive_neighbor_comp[(~df_melt_naive_neighbor_comp['Marker'].isin(['WGA', 'Concavalin A']))]
# Combined
df_combined_neighbor_comp = pd.concat([df_melt_cyto_neighbor_comp, df_melt_naive_neighbor_comp])

# %%
# ################### GENERATE AND SAVE FINAL ORIGINAL (UNSCALED) BOXPLOTS ###################

# FIX: Ensure intensity columns are numeric before plotting
df_melt_all_cancer_neighbors['Intensity'] = pd.to_numeric(df_melt_all_cancer_neighbors['Intensity'], errors='coerce')
df_melt_all_cancer_neighbors.dropna(subset=['Intensity'], inplace=True)
plotting_cancer = { "data": df_melt_all_cancer_neighbors, "x": "Marker", "y": "Intensity", "hue": "Neighbor_T_cell" }
plot(plotting_cancer, "Cancer_Expression_Near_TCells_original.png", title="Cancer Expression Near T-Cells (Original)")

df_melt_all_tcell_neighbors['Intensity'] = pd.to_numeric(df_melt_all_tcell_neighbors['Intensity'], errors='coerce')
df_melt_all_tcell_neighbors.dropna(subset=['Intensity'], inplace=True)
plotting_tcell = { "data": df_melt_all_tcell_neighbors, "x": "Marker", "y": "Intensity", "hue": "cell_type" }
plot(plotting_tcell, "TCell_Expression_Near_Cancer_original.png", title="T-Cell Expression Near Cancer (Original)")

df_melt_cyto_neighbor_comp['Intensity'] = pd.to_numeric(df_melt_cyto_neighbor_comp['Intensity'], errors='coerce')
df_melt_cyto_neighbor_comp.dropna(subset=['Intensity'], inplace=True)
plotting_cyto_neighbor = { "data": df_melt_cyto_neighbor_comp, "x": "Marker", "y": "Intensity", "hue": "Interaction" }
plot(plotting_cyto_neighbor, "Cancer_Neighbor_vs_NonNeighbor_Cytotoxic_original.png", title="Cancer Expression: Interacting vs. Non-Interacting with Cytotoxic T-Cells (Original)")

df_melt_naive_neighbor_comp['Intensity'] = pd.to_numeric(df_melt_naive_neighbor_comp['Intensity'], errors='coerce')
df_melt_naive_neighbor_comp.dropna(subset=['Intensity'], inplace=True)
plotting_naive_neighbor = { "data": df_melt_naive_neighbor_comp, "x": "Marker", "y": "Intensity", "hue": "Interaction" }
plot(plotting_naive_neighbor, "Cancer_Neighbor_vs_NonNeighbor_Naive_original.png", title="Cancer Expression: Interacting vs. Non-Interacting with Naive T-Cells (Original)")

df_combined_neighbor_comp['Intensity'] = pd.to_numeric(df_combined_neighbor_comp['Intensity'], errors='coerce')
df_combined_neighbor_comp.dropna(subset=['Intensity'], inplace=True)
plotting_combined_neighbor = { "data": df_combined_neighbor_comp, "x": "Marker", "y": "Intensity", "hue": "Interaction" }
plot(plotting_combined_neighbor, "Cancer_Neighbor_vs_NonNeighbor_Combined_original.png", title="Cancer Expression: Interacting vs. Non-Interacting with T-Cells (Original, Combined)")


# %%
# ################### PREPARE SCALED (MIN-MAX) DATA FOR FINAL PLOTS ###################

# Create Min-Max Scaled versions of the final dataframes
df_final_cytotoxic_scaled = df_final_cytotoxic.copy()
df_final_naive_scaled = df_final_naive.copy()

# --- Step 1: Apply Mean Centering using original Phalloidin values ---
phalloidin_mean_cyto_orig = df_final_cytotoxic_scaled['Phalloidin'].mean()
for column in marker_cols_cyto:
    if column in df_final_cytotoxic_scaled:
        df_final_cytotoxic_scaled[column] -= phalloidin_mean_cyto_orig

phalloidin_mean_naive_orig = df_final_naive_scaled['Phalloidin'].mean()
for column in marker_cols_naive:
    if column in df_final_naive_scaled:
        df_final_naive_scaled[column] -= phalloidin_mean_naive_orig

# --- Step 2: Apply Min-Max Scaling on the mean-centered data ---
cyto_markers_data = df_final_cytotoxic_scaled[marker_cols_cyto]
df_final_cytotoxic_scaled[marker_cols_cyto] = (cyto_markers_data - cyto_markers_data.min()) / (cyto_markers_data.max() - cyto_markers_data.min())

naive_markers_data = df_final_naive_scaled[marker_cols_naive]
df_final_naive_scaled[marker_cols_naive] = (naive_markers_data - naive_markers_data.min()) / (naive_markers_data.max() - naive_markers_data.min())


# Cancer cell analysis (neighboring T-cells) - SCALED
df_subset_cyto_cancer_s = df_final_cytotoxic_scaled[df_final_cytotoxic_scaled['Cell Type'] == 1][marker_cols_cyto + ['T_cell_neighbors']].copy()
df_melt_cyto_cancer_s = df_subset_cyto_cancer_s.melt(id_vars='T_cell_neighbors', var_name='Marker', value_name='Intensity')
df_melt_cyto_cancer_s = df_melt_cyto_cancer_s[df_melt_cyto_cancer_s['T_cell_neighbors'] == True]
df_melt_cyto_cancer_s['Neighbor_T_cell'] = 'Cytotoxic T cell'

df_subset_naive_cancer_s = df_final_naive_scaled[df_final_naive_scaled['Cell Type'] == 1][marker_cols_naive + ['T_cell_neighbors']].copy()
df_melt_naive_cancer_s = df_subset_naive_cancer_s.melt(id_vars='T_cell_neighbors', var_name='Marker', value_name='Intensity')
df_melt_naive_cancer_s = df_melt_naive_cancer_s[df_melt_naive_cancer_s['T_cell_neighbors'] == True]
df_melt_naive_cancer_s['Neighbor_T_cell'] = 'Naive T cell'

df_melt_all_cancer_neighbors_s = pd.concat([df_melt_cyto_cancer_s, df_melt_naive_cancer_s])
df_melt_all_cancer_neighbors_s = df_melt_all_cancer_neighbors_s[(~df_melt_all_cancer_neighbors_s['Marker'].isin(['WGA', 'Concavalin A']))]

# T-cell analysis (neighboring Tumour cells) - SCALED
df_subset_cyto_tcell_s = df_final_cytotoxic_scaled[df_final_cytotoxic_scaled['Cell Type'] == 2][marker_cols_cyto + ['Tumour_neighbors']].copy()
df_melt_cyto_tcell_s = df_subset_cyto_tcell_s.melt(id_vars='Tumour_neighbors', var_name='Marker', value_name='Intensity')
df_melt_cyto_tcell_s = df_melt_cyto_tcell_s[df_melt_cyto_tcell_s['Tumour_neighbors'] == True]
df_melt_cyto_tcell_s['cell_type'] = 'Cytotoxic T cell'

df_subset_naive_tcell_s = df_final_naive_scaled[df_final_naive_scaled['Cell Type'] == 2][marker_cols_naive + ['Tumour_neighbors']].copy()
df_melt_naive_tcell_s = df_subset_naive_tcell_s.melt(id_vars='Tumour_neighbors', var_name='Marker', value_name='Intensity')
df_melt_naive_tcell_s = df_melt_naive_tcell_s[df_melt_naive_tcell_s['Tumour_neighbors'] == True]
df_melt_naive_tcell_s['cell_type'] = 'Naive T cell'

df_melt_all_tcell_neighbors_s = pd.concat([df_melt_cyto_tcell_s, df_melt_naive_tcell_s])
df_melt_all_tcell_neighbors_s = df_melt_all_tcell_neighbors_s[(~df_melt_all_tcell_neighbors_s['Marker'].isin(['WGA', 'Concavalin A']))]

# Neighbor vs. Non-Neighbor Data Prep - SCALED
# Cytotoxic
df_subset_cyto_neighbor_comp_s = df_final_cytotoxic_scaled[df_final_cytotoxic_scaled['Cell Type'] == 1][marker_cols_cyto + ['T_cell_neighbors']].copy()
df_subset_cyto_neighbor_comp_s['Interaction'] = df_subset_cyto_neighbor_comp_s['T_cell_neighbors'].map({ True: 'Interacting with CytoT', False: 'Non-Interacting with CytoT' })
df_melt_cyto_neighbor_comp_s = df_subset_cyto_neighbor_comp_s.drop('T_cell_neighbors', axis=1).melt(id_vars='Interaction', var_name='Marker', value_name='Intensity')
df_melt_cyto_neighbor_comp_s = df_melt_cyto_neighbor_comp_s[(~df_melt_cyto_neighbor_comp_s['Marker'].isin(['WGA', 'Concavalin A']))]
# Naive
df_subset_naive_neighbor_comp_s = df_final_naive_scaled[df_final_naive_scaled['Cell Type'] == 1][marker_cols_naive + ['T_cell_neighbors']].copy()
df_subset_naive_neighbor_comp_s['Interaction'] = df_subset_naive_neighbor_comp_s['T_cell_neighbors'].map({ True: 'Interacting with Naive T', False: 'Non-Interacting with Naive T' })
df_melt_naive_neighbor_comp_s = df_subset_naive_neighbor_comp_s.drop('T_cell_neighbors', axis=1).melt(id_vars='Interaction', var_name='Marker', value_name='Intensity')
df_melt_naive_neighbor_comp_s = df_melt_naive_neighbor_comp_s[(~df_melt_naive_neighbor_comp_s['Marker'].isin(['WGA', 'Concavalin A']))]
# Combined
df_combined_neighbor_comp_s = pd.concat([df_melt_cyto_neighbor_comp_s, df_melt_naive_neighbor_comp_s])


# %%
# ################### GENERATE AND SAVE FINAL SCALED (MIN-MAX) BOXPLOTS ###################

# FIX: Ensure intensity columns are numeric before plotting
df_melt_all_cancer_neighbors_s['Intensity'] = pd.to_numeric(df_melt_all_cancer_neighbors_s['Intensity'], errors='coerce')
df_melt_all_cancer_neighbors_s.dropna(subset=['Intensity'], inplace=True)
plotting_cancer_s = { "data": df_melt_all_cancer_neighbors_s, "x": "Marker", "y": "Intensity", "hue": "Neighbor_T_cell" }
plot(plotting_cancer_s, "Cancer_Expression_Near_TCells_scaled.png", title="Cancer Expression Near T-Cells (Scaled)")

df_melt_all_tcell_neighbors_s['Intensity'] = pd.to_numeric(df_melt_all_tcell_neighbors_s['Intensity'], errors='coerce')
df_melt_all_tcell_neighbors_s.dropna(subset=['Intensity'], inplace=True)
plotting_tcell_s = { "data": df_melt_all_tcell_neighbors_s, "x": "Marker", "y": "Intensity", "hue": "cell_type" }
plot(plotting_tcell_s, "TCell_Expression_Near_Cancer_scaled.png", title="T-Cell Expression Near Cancer (Scaled)")


# --- Combined plot for Cancer Cells Interacting with Naive vs. Cytotoxic T-Cells ---

# Ensure data is clean and numeric before filtering
df_melt_cyto_neighbor_comp_s['Intensity'] = pd.to_numeric(df_melt_cyto_neighbor_comp_s['Intensity'], errors='coerce')
df_melt_cyto_neighbor_comp_s.dropna(subset=['Intensity'], inplace=True)

df_melt_naive_neighbor_comp_s['Intensity'] = pd.to_numeric(df_melt_naive_neighbor_comp_s['Intensity'], errors='coerce')
df_melt_naive_neighbor_comp_s.dropna(subset=['Intensity'], inplace=True)

# Filter for ONLY the cancer cells that are interacting with T-cells
df_interacting_cyto = df_melt_cyto_neighbor_comp_s[df_melt_cyto_neighbor_comp_s['Interaction'] == 'Interacting with CytoT'].copy()
df_interacting_naive = df_melt_naive_neighbor_comp_s[df_melt_naive_neighbor_comp_s['Interaction'] == 'Interacting with Naive T'].copy()

# Combine the two 'interacting' dataframes
df_cancer_interacting_combined = pd.concat([df_interacting_cyto, df_interacting_naive])

# Map new hue labels for clarity in the legend and to trigger the correct statistical test
# Blue for Naive T interaction, Orange for Cytotoxic T interaction
df_cancer_interacting_combined['Interaction'] = df_cancer_interacting_combined['Interaction'].replace({
    'Interacting with Naive T': 'Interacting w/ Naive T',
    'Interacting with CytoT': 'Interacting w/ Cytotoxic T'
})

# Update the master color palette to match the new labels
master_color_palette['Interacting w/ Naive T'] = '#1f77b4' # Blue
master_color_palette['Interacting w/ Cytotoxic T'] = '#ff7f0e' # Orange


# Generate the combined plot
plotting_cancer_interacting_combined = {
    "data": df_cancer_interacting_combined,
    "x": "Marker",
    "y": "Intensity",
    "hue": "Interaction"
}
plot(plotting_cancer_interacting_combined, "Cancer_Interacting_TCells_Comparison_scaled.png", title="Cancer Expression: Interacting with Naive vs. Cytotoxic T-Cells (Scaled)")


# %%
# ################### Supplementary Figure 2: Scatter Plot ###################
print("\nGenerating Supplementary Figure 2: Scatter Plot...")

# Check if required columns exist, otherwise skip plotting
required_cols = ['CD8', 'CD44']
if all(col in df_cytotoxic.columns for col in required_cols) and all(col in df_naive.columns for col in required_cols):
    # Extract the relevant data for T-cells from the original (unscaled) dataframes
    df_cyto_t_cells = df_cytotoxic[df_cytotoxic['Cell_type_name'] == 'Cytotoxic T cell'][required_cols].copy()
    # The 'Condition' column will be used for the legend
    df_cyto_t_cells['Condition'] = 'CytoTcells'

    df_naive_t_cells = df_naive[df_naive['Cell_type_name'] == 'Naive T cell'][required_cols].copy()
    df_naive_t_cells['Condition'] = 'NaiveTcells'

    # Combine into a single dataframe for plotting
    df_scatter = pd.concat([df_cyto_t_cells, df_naive_t_cells])

    # The raw data was arcsinh transformed. To plot on a log scale, we should reverse the transformation.
    df_scatter['CD8'] = np.sinh(df_scatter['CD8'])
    df_scatter['CD44'] = np.sinh(df_scatter['CD44'])

    # Use a specific plotting context to increase font sizes for better readability
    with sns.plotting_context("talk", font_scale=1.2):
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 12))

        # Plot each condition separately to handle mixed marker styles
        sns.scatterplot(
            data=df_scatter[df_scatter['Condition'] == 'CytoTcells'],
            x='CD8',
            y='CD44',
            color='orange',
            marker='o',
            s=100,
            ax=ax,
            label='CytoTcells'
        )

        sns.scatterplot(
            data=df_scatter[df_scatter['Condition'] == 'NaiveTcells'],
            x='CD8',
            y='CD44',
            color='steelblue',
            marker='x',
            s=100,
            ax=ax,
            label='NaiveTcells'
        )

        # Set axes to log scale
        ax.set_xscale('log')
        ax.set_yscale('log')

        # Set title and labels
        title_text = "Supplementary Figure 2: Scatter plot showing the difference between the cytotoxic\nand the naïve T cell populations based on the expression of CD8 and CD44"
        ax.set_title(title_text, pad=20)
        ax.set_xlabel('CD8')
        ax.set_ylabel('CD44')
        
        # Set axis limits to match the example plot
        ax.set_xlim(left=0.1)
        ax.set_ylim(bottom=0.1)

        # Add the legend and move it outside the plot
        ax.legend(title='Condition', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Save the figure to the plots directory
    save_path = plots_dir / "Supplementary_Figure_2_Scatter_Plot.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig) # Explicitly close the figure
    print(f"Saved plot: {save_path}")
else:
    print("Skipping Supplementary Figure 2: 'CD8' or 'CD44' not found in data.")


print("\nScript finished. All plots saved.")


