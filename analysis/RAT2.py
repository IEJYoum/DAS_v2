# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 10:43:39 2024

@author: youm
"""

import os
import numpy as np
import pandas as pd
import math
import csv
import gzip
import scipy.io
#import IFprocessing7 as ifp
#import IFvisualization0 as ifv
#import cmifAnalysis49 as cm
#import SVM5 as sv
import time
#import scipy as sp
import matplotlib as mpl
import torchCluster5 as tC



DATA = r'\\accsmb.ohsu.edu\CEDAR\archive\seq\JungsunKim\in_house_data\Visium'

PATHS = ['//accsmb.ohsu.edu/CEDAR/archive/seq/JungsunKim/in_house_data/Visium/3011/Results/3011_Adjacent_1/outs/raw_feature_bc_matrix',
          '//accsmb.ohsu.edu/CEDAR/archive/seq/JungsunKim/in_house_data/Visium/3011/Results/3011_Adjacent_1/outs/spatial/tissue_positions.csv',
          '//accsmb.ohsu.edu/CEDAR/archive/seq/JungsunKim/in_house_data/Visium/3011/Results/3011_Distal_1/outs/raw_feature_bc_matrix',
          '//accsmb.ohsu.edu/CEDAR/archive/seq/JungsunKim/in_house_data/Visium/3011/Results/3011_Distal_1/outs/spatial/tissue_positions.csv']

PATHS = ['//accsmb.ohsu.edu/CEDAR/archive/seq/JungsunKim/in_house_data/Visium/2223/Results/2223_Adjacent_2/outs/raw_feature_bc_matrix',
          '//accsmb.ohsu.edu/CEDAR/archive/seq/JungsunKim/in_house_data/Visium/2223/Results/2223_Adjacent_2/outs/spatial/tissue_positions.csv',
          '//accsmb.ohsu.edu/CEDAR/archive/seq/JungsunKim/in_house_data/Visium/2223/Results/2223_Distal_2/outs/raw_feature_bc_matrix',
          '//accsmb.ohsu.edu/CEDAR/archive/seq/JungsunKim/in_house_data/Visium/2223/Results/2223_Distal_2/outs/spatial/tissue_positions.csv']

KEYS = ['2223_Adjacent','2223_Distal']

#\\accsmb.ohsu.edu\CEDAR\archive via windows
#\\accsmb.ohsu.edu\CEDAR\archive\seq\JungsunKim\in_house_data\Visium\2223\Results\2223_Adjacent_1\outs\raw_feature_bc_matrix
TSTEM = '3011_test' #'rna_3011_raw'
TPATH = os.getcwd()
SAVEFOLDER = os.getcwd()


def main(n=9,nn=9,nnn=9):
    options = ["Import and clean data","load prepared data","use: "+TSTEM]
    functions = [impor,load,preload]
    df,obs,dfxy = menu(options,functions)
    df = df.loc[dfxy.index,:]
    df,obs,dfxy = loadingMenu(df,obs,dfxy)
    return(df,obs,dfxy)


def loadingMenu(df,obs,dfxy):
    options = ["save","cluster and aggregate genes"]
    functions = [save,gCluster]
    df,obs,dfxy = menu(options,functions,df,obs,dfxy)
    return(df,obs,dfxy)





'''
clustering biomarkers functions
'''
#need a function to pick subset of markers from one cluster


def gCluster(df,obs,dfxy):
    ncl = int(input('number of clusters'))
    nit = int(input('number of iterations'))
    lra = eval(input('learning rate 10**-12 or less suggested'))
    sch = input('save?')
    if sch == 'y':
        fnam = input('filename?')
    df = df.transpose()
    print(df.shape)
    #input('')
    clusters,centroids = tC.main(df,ncl=ncl,MAXI=nit,LR=lra)
    cdf = pd.DataFrame(data=clusters,index = df.index)
    cluster_counts = cdf.iloc[:,0].astype(str).value_counts().sort_index().to_dict()
    print("cluster counts:", cluster_counts)
    if sch == 'y':
        cdf.to_csv(fnam+"_clusters.csv")
        with open(fnam+"_torch_cluster.summary.txt","w") as f:
            f.write("Summary\n")
            f.write("RNA gene torch clustering completed.\n\n")
            f.write("Facts\n")
            f.write("n_clusters: "+str(ncl)+"\n")
            f.write("iterations: "+str(nit)+"\n")
            f.write("learning_rate: "+str(lra)+"\n")
            f.write("cluster_counts: "+str(cluster_counts)+"\n")
            f.write("cluster_table: "+fnam+"_clusters.csv\n")
    cdf = cdf.transpose()
    #print(cdf,'cdf')
    ucl = sorted(list(cdf.iloc[0,:].unique()))
    print(ucl)
    odf = pd.DataFrame(columns = ucl,index=df.columns)
    #print(odf.shape)
    #input('')
    for uc in ucl:
        print(uc)
        key= cdf.iloc[:,0] == uc
        if key.sum() > cdf.shape[0]/2:
            continue
        sdf = cdf.loc[key,:]
        prots = []
        proteins = sdf.index
        for pro in list(proteins):
            if pro not in df.index:
                print(pro,' not found')
            else:
                prots.append(pro)
        sdat = df.loc[prots,:]

        counts = sdat.sum(axis=0)
        #print(counts)
        #vals = list(counts.values)
        #print(vals)
        #print(counts,'counts')
        #print(odf,'odf')
        odf.loc[:,uc] = counts
    #odf.to_csv(str(ncl)+'_'+str(nit)+'_'+str(lra)+'_processed.csv')
    if sch == 'y':
        save(odf,obs,dfxy,filename = fnam)
    return(odf,obs,dfxy)


'''
importing functions
'''
def impor(n=9,nn=9,nnn=9):
    try:
        paths = PATHS
        print('loading from globally defined PATHS')
    except:
        paths = []
        path = DATA
        while True:
            path= navigate(path)
            print(path,'path 1')
            if type(path) == int:
                break
            paths.append(path)
            print(path,'path 2')
            path = '/'.join(path.split('/')[:-1])
    print(paths)
    '''
    keyS = []
    while True:
        ch = input('unique key string found in path and used to name slide_scene:')
        if ch == '':
            break
        keyS.append(ch)
    '''
    keyS = KEYS
    dfs,obs = [],[]
    for KS in keyS:
        for path in sorted(paths):
            if KS not in path:
                continue
            if '.csv' in path:
                ob = pd.read_csv(path,index_col=0)
                ind = pd.Series(ob.index)
                ind = ind +'_'+KS
                ob.index = ind
                obs.append(ob)
            else:
                df = GZtoDF(path)
                key = "DEPRECATED" !=  pd.Series(df.index,index=df.index).apply(lambda n: str(n).split('_')[0])
                df = df.transpose()
                ind = pd.Series(df.index)
                ind = ind +'_'+KS

                df.index = ind
                dfs.append(df)
    df = pd.concat(dfs,axis=0) #.drop_duplicates() #works
    print(ob.shape)
    ob = pd.concat(obs,axis=0).drop_duplicates() #doesn't work - actually idk
    print(ob.shape)

    ob = ob.drop_duplicates()#ob[~ob.index.duplicated(keep='first')]
    print(ob.shape,'after drop')
    #ob.to_csv("debug_obs.csv")
    #df = df.loc[obs.index,:] #'not identical df objects'
    i0 = list(df.index)[0]
    print(df.loc[i0,:].iloc[0],'\n',df.loc[i0,:].iloc[1],'i0')
    df = df.drop_duplicates()
    key2 = ob.loc[:,"in_tissue"].astype(int) == 1
    print(key2.sum())

    ob = ob.loc[key2,:]
    df = df.loc[key2,:]
    dfxy = ob.loc[:,['pxl_row_in_fullres','pxl_col_in_fullres']]
    print(df,ob)
    '''
    i = 0
    for ind in df.index:
        if ind not in ob.index:
            i += 1
            print(ind,'not found',i)
    '''
    df = df.loc[ob.index,:]
    print(df.index == ob.index,'matching index')
    return(df,ob,dfxy)





def GZtoDF(matrix_dir):
    print(matrix_dir,' .... loading')
    mat_filtered = scipy.io.mmread(os.path.join(matrix_dir, "matrix.mtx.gz"))

    # list of transcript ids, e.g. 'ENSG00000187634'
    features_path = os.path.join(matrix_dir, "features.tsv.gz")
    feature_ids = [row[0] for row in csv.reader(gzip.open(features_path, mode="rt"), delimiter="\t")]

    # list of gene names, e.g. 'SAMD11'
    gene_names = [row[1] for row in csv.reader(gzip.open(features_path, mode="rt"), delimiter="\t")]

    # list of feature_types, e.g. 'Gene Expression'
    feature_types = [row[2] for row in csv.reader(gzip.open(features_path, mode="rt"), delimiter="\t")]

    # list of barcodes, e.g. 'AAACATACAAAACG-1'
    barcodes_path = os.path.join(matrix_dir, "barcodes.tsv.gz")
    barcodes = [row[0] for row in csv.reader(gzip.open(barcodes_path, mode="rt"), delimiter="\t")]



    # transform table to pandas dataframe and label rows and columns
    matrix = pd.DataFrame.sparse.from_spmatrix(mat_filtered)
    matrix.columns = barcodes
    matrix.insert(loc=0, column="feature_id", value=feature_ids)
    matrix.insert(loc=1, column="gene", value=gene_names)
    matrix.insert(loc=2, column="feature_type", value=feature_types)

    dist = matrix.drop(["feature_id","feature_type"],axis=1)
    dist.index = list(dist["gene"])
    dist = dist.drop(["gene"],axis=1)
    return(dist)


def navigate(path,ch=None):
    print(path)
    path = path.replace('\\','/')
    ldir = sorted(os.listdir(path))
    for i,file in enumerate(ldir):
        print(i,file)
    if ch == None:
        ch = input('select which. "done" to return, blank to go to parent dir')
    if ch == '':
        path = '/'.join(path.split('/')[:-1])
        path = navigate(path)
        return(path)
    elif ch == 'done':
        return(9)
    else:
        try:
            ch = int(ch)
            path = path + '/' + ldir[ch]
            if os.path.isdir(path) and 'feature_bc_matrix' not in path:
                path = navigate(path)
            print(path,'path 0')
            return(path)
        except Exception as e:
            print('invalid path:',e)
    print(path,'path3')
    return('script finished should never')


'''
general functions
'''

def menu(options,functions,df=9,obs=9,dfxy=9): #MANUAL MENU
    print("menu")
    while True:
        print("\n")
        for i,op in enumerate(options):
            print(i,op)
        try:
            print("send non-int when done (return df)")
            ch = int(input("number: "))
        except:
            return(df,obs,dfxy)
        df,obs,dfxy=functions[ch](df,obs,dfxy)
        print(all(obs.index==df.index),"all index the same")
        try:
            print(obs.columns)
        except:
            print("no obs")
        try:
            print(df.shape,"df shape")
        except:
            pass

def load(bl1,bl2,bl3,path = "none"):
    if path == "none":
        path = SAVEFOLDER
    print(path)
    #"C:/Users/youm/.spyder-py3/src"
    while True:
        path = navigate(path)
        print(path,"out of navigate")
        if not os.path.isdir(path):
            break

    name = "_".join(path.split("_")[:-1])
    #print(name)
    name=name.split("/")[-1]+"_"
    #print(name,path)
    path = "/".join(path.split("/")[:-1])
    for file in os.listdir(path):
        fn = "_".join(file.split("_")[:-1])
        fn=fn.split("/")[-1]+"_"
        if fn == name:
            print(file)
            if "dfxy" in file:
                dfxy = pd.read_csv(path+"/"+file,index_col=0)
            elif "df" in file:
                df = pd.read_csv(path+"/"+file,index_col=0)
            elif "obs" in file:
                obs = pd.read_csv(path+"/"+file,index_col=0)
    #ser = pd.Series(df.index).apply(lambda x: x.split('.1')[0])
    #df.index = ser
    #obs.index = ser
    #dfxy = dfxy.loc[df.index,:]
    return(df,obs,dfxy)

def preload(bl1,bl2,bl3,path = TPATH):
    if path == "none" or path == "":
        path = SAVEFOLDER
    print(path)
    for file in os.listdir(path):
        if TSTEM == "_".join(file.split("_")[:-1]):
            if "dfxy" in file:
                dfxy = pd.read_csv(path+"/"+file,index_col=0)
            elif "df" in file:
                df = pd.read_csv(path+"/"+file,index_col=0)
            elif "obs" in file:
                obs = pd.read_csv(path+"/"+file,index_col=0)
    obs.name = obs.columns[-1]
    obs = obs.loc[list(df.index),:]
    dfxy = dfxy.loc[list(df.index),:]
    print(all(obs.index==df.index),"all index the same")
    print(df.index,obs.index)
    return(df,obs,dfxy)



def save(df,obs,dfxy,filename = None):
    if not filename:
        filename = input("filename: ")
    if filename != '':
        df.to_csv(filename+"_df.csv")
        obs.to_csv(filename+"_obs.csv")
        dfxy.to_csv(filename+"_dfxy.csv")
    return(df,obs,dfxy)

if __name__ == '__main__':
    main()
