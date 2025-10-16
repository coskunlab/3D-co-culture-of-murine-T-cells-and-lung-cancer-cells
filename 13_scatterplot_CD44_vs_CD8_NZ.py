# %%
import itertools
import os
import sys
from pathlib import Path

# import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage.io

from collections import defaultdict
from tqdm.notebook import trange, tqdm, tqdm_notebook
from joblib import Parallel, delayed
import re
import h5py
import tifffile as tiff
from natsort import natsorted, natsort_keygen
import cv2

# %%
# --- Create a directory to save plots ---
# This logic is copied from script 12 to ensure plots are saved in the same location.
try:
    # Try to get path of the script to save plots in ../Plots
    script_path = Path(__file__).resolve()
    plots_dir = script_path.parent.parent / "Plots"
except NameError:
    # Fallback for interactive environments (like Jupyter)
    script_path = Path.cwd()
    plots_dir = script_path.parent / "Plots"

os.makedirs(plots_dir, exist_ok=True)
print(f"Plots will be saved to: {plots_dir}")


# %%
data_dir = (Path().cwd().parents[0] / 'data').absolute()
# data_processed = data_dir / 'processed'
# data_raw = r'Y:\coskun-lab\Mayar\3D_culture_experiments\101823_cyclicIFonTcellsMatrigel'


# %%
# Save dataframe information 
# df_path = data_dir / 'Tcell'/ 'metadata' / 'imgs_reg.csv'
df_path = Path(r"Y:\coskun-lab\Thomas\20_MouseCocultureMatrigel\data\Tcell\metadata\imgs_reg.csv")
df_prop = pd.read_csv(df_path)

# %%
df_prop.Marker.unique()

# %% [markdown]
# # Visual examples

# %%


# %% [markdown]
# # Cell type definition

# %% [markdown]
# ## Naive vs Cytotoxic

# %%
import seaborn as sns
sns.set(font_scale = 2)
sns.set_style("whitegrid")
# Define a consistent color palette: blue for Naive T cells, orange for Cyto T cells
color_palette = {'NaiveTcells': '#377eb8', 'CytoTcells': 'darkorange'}

# %%
# Compare cell level 
dfCell = df_prop[df_prop.Type == 'Cell']
dfCellPivot = dfCell.pivot_table(values='Intensity', columns='Marker', index=['Conditon', 'FOV', 'Cell_label'])
dfCellPivot.reset_index(inplace=True)

# %%
dfCellPivot[['CD8', 'CD44']].describe()

# %%
fig, ax = plt.subplots(figsize=(10,10))
sns.scatterplot(data=dfCellPivot, x='CD8', y='CD44', hue='Conditon', ax=ax, style="Conditon", palette=color_palette)
ax.set_xlim(0,80)
ax.set_ylim(0,50)
ax.set_title('CD44 vs. CD8 Expression (Linear Scale)')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
save_path = plots_dir / "scatterplot_CD44_vs_CD8_linear.png"
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.close(fig)
print(f"Saved plot: {save_path}")


# %%
fig, ax = plt.subplots(figsize=(10,10))
pl = sns.scatterplot(data=dfCellPivot, x='CD8', y='CD44', hue='Conditon', ax=ax, style="Conditon", s=30, palette=color_palette)
ax.set_xlim(0.1,80)
ax.set_ylim(0.1,50)
plt.xscale('log')
plt.yscale('log')
ax.set_title('CD44 vs. CD8 Expression (Log Scale)')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
save_path = plots_dir / "scatterplot_CD44_vs_CD8_log.png"
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.close(fig)
print(f"Saved plot: {save_path}")

# %% [markdown]
# ## CD8 marker expression distribution

# %%
dfCD8Naive = dfCellPivot[['Conditon', 'CD8']]

# %%
fig, ax = plt.subplots(figsize=(10,10))
sns.kdeplot(dfCD8Naive, x='CD8', hue='Conditon', bw_adjust=0.5, ax=ax, palette=color_palette)
ax.set_title('CD8 Marker Expression Distribution')
ax.legend(title='Condition', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
save_path = plots_dir / "kdeplot_CD8_distribution.png"
fig.savefig(save_path, bbox_inches='tight', dpi=300)
plt.close(fig)
print(f"Saved plot: {save_path}")

# %% [markdown]
# ## Marker expression comparison

# %%
from statannotations.Annotator import Annotator

def plot(plotting, save_path, figsize=(20,7)):
    c1, c2 = plotting['data'][plotting['hue']].unique()
    pairs = [((e, c1), (e, c2)) for e in plotting['data'][plotting['x']].unique()]

    with sns.plotting_context('talk', font_scale=2):
        fig, ax = plt.subplots(figsize=figsize)
        ax = sns.boxplot(**plotting, showfliers=False, ax=ax)
        # ax = sns.swarmplot(**plotting, ax=ax, dodge=True, edgecolor='k', size=5)
        ax.set(yscale='log')
        annot = Annotator(ax, pairs, **plotting)
        annot.configure(test='Mann-Whitney', text_format='star', loc='outside', verbose=0)
        result = annot.apply_test().annotate()
        plt.xticks(rotation=30, ha='right')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
        plt.title('Marker Expression Comparison (Box Plot)')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close(fig)
            print(f"Saved plot: {save_path}")


def plot2(plotting, save_path, figsize=(20,7)):
    c1, c2 = plotting['data'][plotting['hue']].unique()
    pairs = [((e, c1), (e, c2)) for e in plotting['data'][plotting['x']].unique()]

    with sns.plotting_context('talk', font_scale=2):
        fig, ax = plt.subplots(figsize=figsize)
        ax = sns.violinplot(**plotting, ax=ax, bw_adjust=1, cut=0,)
        ax.set_ylim(0,50)
        # ax = sns.swarmplot(**plotting, ax=ax, dodge=True, edgecolor='k', size=5)
        annot = Annotator(ax, pairs, **plotting)
        annot.configure(test='Mann-Whitney', text_format='star', loc='outside', verbose=0)
        result = annot.apply_test().annotate()
        plt.xticks(rotation=30, ha='right')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
        plt.title('Marker Expression Comparison (Violin Plot)')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close(fig)
            print(f"Saved plot: {save_path}")
        

# %%
df_prop['Intensity'] += 1

# %%
# Boxplot per cell
plotting = {
    "data": df_prop[~df_prop.Marker.isin(['DNA'])],
    "x": "Marker",
    "y": "Intensity",
    "hue": "Conditon",
    "palette": color_palette
}

plot(plotting, save_path=plots_dir / "boxplot_marker_expression.png")

# %%
# Violin plot per cell
plotting = {
    "data": df_prop[~df_prop.Marker.isin(['DNA'])],
    "x": "Marker",
    "y": "Intensity",
    "hue": "Conditon",
    "palette": color_palette
}

plot2(plotting, save_path=plots_dir / "violinplot_marker_expression.png")

