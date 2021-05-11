# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 2020

@author: Alexandre Maciel-Guerra
"""

import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from matplotlib import cm
import colorcet as cc
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category= FutureWarning)
simplefilter(action='ignore', category= UserWarning)
simplefilter(action='ignore', category= DeprecationWarning)

def update_progress(progress):
    barLength = 100 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), round(progress*100, 2), status)
    sys.stdout.write(text)
    sys.stdout.flush()  

if __name__ == "__main__":
    method = "2000"
    folder = "Saureus" 
    name_dataset = "food_673samples"
    results_folder = "PopulationCorrection3"
    antibiotic_df = pd.read_csv(folder+"/"+name_dataset+'_AMR_data_RSI.csv', header = [0])
    delimiter = ' '

    n_lines = antibiotic_df.shape[0]
    samples = np.array(antibiotic_df[antibiotic_df.columns[0]])

    print(antibiotic_df.columns[3:])
    for name_antibiotic in antibiotic_df.columns[3:]:
        print("Antibiotic: {}".format(name_antibiotic))

        target_str = np.array(antibiotic_df[name_antibiotic])
        
        target = np.zeros(len(target_str)).astype(int)
        idx_S = np.where(target_str == 'S')[0]
        idx_R = np.where(target_str == 'R')[0]
        idx_I = np.where((target_str != 'R') & (target_str != 'S'))[0]
        target[idx_R] = 1
        target[idx_I] = 2

        idx = np.hstack((idx_S,idx_R))
        
        if len(idx) == 0:
            print("Empty")
            continue
        
        samples_name = np.delete(samples,idx_I, axis=0)

        target = target[idx]
        target_str = target_str[idx]
        
        count_class = Counter(target)
        print(count_class)

        if count_class[0] < 12 or count_class[1] < 12:
            continue

        file_name = folder+"/"+results_folder+"/data_2000_"+name_dataset+"_"+name_antibiotic+'.pickle'
        my_file = Path(file_name)

        try:
            my_abs_path = my_file.resolve(strict=True)
        except FileNotFoundError:
            continue
        else:
            with open(file_name, 'rb') as f:
                data = pickle.load(f)

        print(data.shape)

        data = MinMaxScaler().fit_transform(data)
        
        df_meta = pd.read_csv(folder+"/"+name_dataset+'_metadata.csv', header = [0],encoding = 'unicode_escape')
        df_meta = df_meta.loc[idx,:]

        # Get data from pandas dataframe
        target_unique = np.unique(target)
        sample_classes_unique = np.array(["Resistant", "Sensitive"])
        sample_id = np.array(df_meta[df_meta.columns[0]])
        sample_source = np.array(df_meta[df_meta.columns[5]])
        sample_source_unique = np.unique(sample_source)
        sample_year = np.array(df_meta[df_meta.columns[6]])
        sample_year_unique = np.unique(sample_year)
        sample_CC = np.array(df_meta[df_meta.columns[7]])
        sample_CC_unique = np.unique(sample_CC)

        # Create colormaps
        cmap_class = cm.get_cmap('Paired',len(target_unique))
        colormap_class = []
        for it in range(len(target_unique)):
            colormap_class.append(cmap_class(it))

        cmap_source = cm.get_cmap('Set1',3)
        colormap_source = []
        for it in range(len(sample_source_unique)):
            colormap_source.append(cmap_source(it))

        cmap_year = cm.get_cmap('Set3',len(sample_year_unique))
        colormap_year = []
        for it in range(len(sample_year_unique)):
            colormap_year.append(cmap_year(it))

        cmap_CC = cc.cm["glasbey_light"]
        colormap_CC = []
        for it in range(len(sample_CC_unique)):
            colormap_CC.append(cmap_CC(it))

        # Create row colors array
        df_class = pd.DataFrame({'class': target_str})
        df_class['class'] = pd.Categorical(df_class['class'])
        my_color_class = df_class['class'].cat.codes
        lut_class = dict(zip(set(target), colormap_class))
        row_colors_class = my_color_class.map(lut_class)
        
        df_source = pd.DataFrame({'Source': sample_source})
        df_source['Source'] = pd.Categorical(df_source['Source'])
        my_color_source = df_source['Source'].cat.codes
        lut_source = dict(zip(np.arange(len(sample_source_unique)), colormap_source))
        row_colors_source = my_color_source.map(lut_source)

        df_year = pd.DataFrame({'Year': sample_year})
        df_year['Year'] = pd.Categorical(df_year['Year'])
        my_color_year = df_year['Year'].cat.codes
        lut_year = dict(zip(np.arange(len(sample_year_unique)), colormap_year))
        row_colors_year = my_color_year.map(lut_year)

        
        df_CC = pd.DataFrame({'CC': sample_CC})
        df_CC['CC'] = pd.Categorical(df_CC['CC'])
        my_color_CC = df_CC['CC'].cat.codes
        lut_CC = dict(zip(np.arange(len(sample_CC_unique)), colormap_CC))
        row_colors_CC = my_color_CC.map(lut_CC)
        
        # Create clustermap plot
        sns_plot = sns.clustermap(data, cmap='tab20b', vmin=0, vmax=1,
            cbar_kws={"shrink": .5, 'label': 'Normalized k-mer count'}, row_colors=[row_colors_class, row_colors_source, row_colors_CC, row_colors_year], 
            yticklabels=False, xticklabels=False)

        # Create Excel file based on clustermap plot
        columns_name_df = ['kmer '+str(x+1) for x in range(data.shape[1])]
        df_data = pd.DataFrame(data=data, columns=columns_name_df)
        data_excel = pd.DataFrame({'ID': sample_id, 'class': target_str, 'Source': sample_source, 'Year': sample_year, 'CC': sample_CC})
        data_excel = pd.concat([data_excel, df_data], axis=1)
        
        order_rows = sns_plot.dendrogram_row.reordered_ind
        data_excel = data_excel.reindex(order_rows)
        
        data_excel.to_excel(folder+'/'+results_folder+'/SupplementaryTable_'+method+'_'+name_dataset+'_'+name_antibiotic+'.xlsx',sheet_name = name_antibiotic, index=False)  
        
        # Define legends for clustermaps
        l1_patch = []
        for count, label in enumerate(sample_classes_unique):
            l1_patch.append(mpatches.Patch(color=lut_class[count], label=label))

        l2_patch = []
        for count, label in enumerate(sample_source_unique):
            l2_patch.append(mpatches.Patch(color=lut_source[count], label=label))

        l3_patch = []
        for count, label in enumerate(sample_CC_unique):
            l3_patch.append(mpatches.Patch(color=lut_CC[count], label=label))

        l4_patch = []
        for count, label in enumerate(sample_year_unique):
            l4_patch.append(mpatches.Patch(color=lut_year[count], label=label))

        l1 = sns_plot.ax_col_dendrogram.legend(handles = l1_patch,title='Phenotype', loc="center", ncol=1, bbox_to_anchor=(-0.3, 1.5), prop={'size': 6})
        l1.get_title().set_fontsize('6')
        sns_plot.ax_col_dendrogram.add_artist(l1)

        l2 = sns_plot.ax_col_dendrogram.legend(handles = l2_patch, title='Source', loc="center", ncol = 1, bbox_to_anchor=(-0.1, 1.5), prop={'size': 6})
        l2.get_title().set_fontsize('6')
        sns_plot.ax_col_dendrogram.add_artist(l2)

        l3 = sns_plot.ax_col_dendrogram.legend(handles = l3_patch,title='CC Type', loc="center",ncol=5, bbox_to_anchor=(0.3, 1.5), prop={'size': 6})
        l3.get_title().set_fontsize('6')
        sns_plot.ax_col_dendrogram.add_artist(l3)

        l4 = sns_plot.ax_col_dendrogram.legend(handles = l4_patch,title='Year', loc="center",ncol=2, bbox_to_anchor=(0.75, 1.5), prop={'size': 6})
        l4.get_title().set_fontsize('6')


        sns_plot.savefig(folder+'/'+results_folder+'/Clustermap_kmers_Test_'+method+'_'+name_dataset+'_'+name_antibiotic+'.pdf', bbox_inches='tight')
        
        del sns_plot

        input("cont")