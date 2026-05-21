# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 18:08:37 2021

@author: youm
"""


import warnings
warnings.simplefilter(action='ignore')#, category=FutureWarning)

import numpy as np
import pandas as pd
import time
import os
import matplotlib
import matplotlib.pyplot as plt
import math
import seaborn as sns
import copy
###import squidpy as sq

#import phenograph  #problem with igraph again
#from scipy import sparse
#from sklearn.metrics import adjusted_rand_score
#import sklearn as skl
import re
import scipy as sp
import statistics as stat
import random
#import bokehClusterMap1 as bcm
import allcolors as allc
import sys
import orthogonal7 as ort
import combat1 as combat1
#import recropTma as rec
#import lithresh1 as lithresh
#import RESTORE as RES
#import IFanalysisPackage0 as IF
from skimage import io
#import napari7 as NP
#import napari as NAPARI
from sklearn.mixture import BayesianGaussianMixture as GMM
import matplotlib.style
import matplotlib as mpl
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import skimage
import PIL
import tifffile
from tqdm import tqdm
import scipy
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu
from scipy.stats import zscore as ZSC
#from GraphST import GraphST
#from GraphST.utils import clustering
try:
    import pygame
except Exception:
    pygame = None
from shapely.geometry import Point, Polygon
import colorsys

_NEW_DAS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "support"))
if _NEW_DAS_DIR not in sys.path:
    sys.path.append(_NEW_DAS_DIR)
from shared_utils import checkChange as shared_check_change

DEVMODE = True


def _require_pygame():
    if pygame is None:
        raise ImportError("pygame is required for ROI selector functions.")
    return pygame


def _load_scanpy_stack(action_label="this action"):
    try:
        import anndata
        import scanpy as sc
        return(sc, anndata)
    except Exception as exc:
        print(action_label + " requires scanpy and anndata.")
        print("Install them to use this option.")
        print(type(exc).__name__, exc)
        return(None, None)

mpl.style.use('default')

ODF = 9
OOBS = 9
OXY = 9

PXSIZE = 0.325#0.65


#srun --pty --time=1-0 --mem=64G --gres=gpu:1 --partition=gpu bash -i
# Y:\  ==  /home/groups/graylab_share/Chin_Lab/ChinData
# T:\ == /home/groups/CEDAR/ChinData/


def main(df,obs,dfxy):
    global ODF
    global OOBS
    global OXY
    print("NOT LOGGED !!!")
    if isinstance(ODF,int):
        ODF = copy.deepcopy(df)
        OOBS = copy.deepcopy(obs)
        OXY = copy.deepcopy(dfxy)
        df.index = pd.Series(df.index).astype(str)
        obs.index = pd.Series(obs.index).astype(str)
        dfxy.index = pd.Series(dfxy.index).astype(str)
    op = ["processing","visualization"]#,"aggregate","split"]
    fn = [processing,visu]
    try:
        print(obs.name,'obs will cluster by')
    except:
        pass
    df,obs,dfxy=menu(op,fn,df,obs,dfxy)
    return(df,obs,dfxy,['OLD TOOL NOT LOGGED'])




def menu(options,functions,df=9,obs=9,dfxy=9,cdf=9):
    while True:

        if DEVMODE:
            print("\n")
            for i,op in enumerate(options):
                print(i,":",op)
            try:
                print("send non-int when done (return df)")
                ch = int(input("number: "))
            except:
                return(df,obs,dfxy)
            if isinstance(cdf,int):
                df,obs,dfxy=functions[ch](df,obs,dfxy)
                print(all(obs.index==df.index),"all index the same")
            else:
                df,obs,dfxy=functions[ch](df,obs,dfxy,cdf)
                print(all(obs.index==df.index),"all index the same")

        else:
            try:
                print("\n")
                for i,op in enumerate(options):
                    print(i,":",op)
                try:
                    print("send non-int when done (return df)")
                    ch = int(input("number: "))
                    if ch >= len(functions):
                        1/0
                except:
                    return(df,obs,dfxy)
                if isinstance(cdf,int):
                    df,obs,dfxy=functions[ch](df,obs,dfxy)
                    print(all(obs.index==df.index),"all index the same")
                else:
                    df,obs,dfxy=functions[ch](df,obs,dfxy,cdf)
                    print(all(obs.index==df.index),"all index the same")
            except KeyboardInterrupt:
                break
            except Exception as e:
                print('Error: ',e)
                return(df,obs,dfxy)
                #break




def processing(df,obs,dfxy):
    op = ["start over with raw data","log2","scale from -1 to 1", "z-score","multiply columns (see coexpression)","trim outliers or subtract channel thresholds",
          "make control TMA sample sizes the same","combat",
          "apply TMA combat to other dataset","equalizeBiomLevel","adjust for negative values",
          "save to csv", "pick subset of data", "manually threshold",
          "cluster by obs catagory","Leiden cluster","GMM cluster","K-means","aggregate",
          "manually celltype random training set","auto-cell-type",
          "convert df to fractions in obs categories","convert to superbiom-only df",
          "remove non-primary biomarkers","calculate biomarker expression in region around each cell",
          "count label fractions in neighborhood","calculate entropy in neighborhood",
          "select ROI","remove cells expressing certain biomarker combinations","pick random subset","clag","clauto","get per-cell expression ratio",
          "normalize round by housekeeping marker","add distance between each cell and nearest cell of specified type",
          "expand phenotype into binary categories","get fraction of cells by annotation",
          'get list of unique annotations','stats tests','test markers for level of background signal']
    fn = [revert,log2,scale1,zscore,multCols,outliers,equalizeTMA,combat,TMAcombat,equalizeBiomLevel,remNegatives,save,pick,
          manThresh,obCluster,leiden,gmm,kmeans,aggregate,celltype,autotype,countObs,superBiomDF,
          onlyPrimaries,regionAverage,neighborhoodFractions,neighborhoodEntropy,roi,simulateTherapy,subset,clag,clauto,biomRatio,
          housekeeping,nearestOfType,phenoToObs,fractionTable,listAnnots,statTest,BGtest]
    df,obs,dfxy = menu(op,fn,df,obs,dfxy)
    return(df,obs,dfxy)


def visu(df,obs,dfxy):
    mpl.style.use('default')
    op = ["heatmap","cluster-bar plot","bar plot","box plot","correlation matrix","scatterplot","umap","show spatial",
          "scanpy visuals","built in pie chart","silhouette score for clusters","napari visualization","sorted single-cell heatmap",
          "save obs columns as annotation png set","histogram","volcano plot","Ripley's L"]#,"send cluster df to processing"]
    fn = [heatmap,clusterbar,barplot,boxplot,correlationMatrix,scatterplot,showUmap,spatial,scanpyv,neighPie,silh,nap,sortedMap,
          savepngs,hist,volcanoPlot,ripleyl]#graphST]#,cdfToProcessing]
    clusterA,ucl,obs = clusterMeans(df,obs,dfxy)
    cdf = pd.DataFrame(clusterA,index=ucl,columns = df.columns)
    cdf.name = obs.name
    df,obs,dfxy = menu(op,fn,df,obs,dfxy,cdf = cdf)
    return(df,obs,dfxy)



def countObs(df,obs,dfxy,cdf=None):
    if input("this will overwrite your working dataframe and will make some functions crash. Continue? (y)") != "y":
        return(df,obs,dfxy)
    obs = obs.astype(str)
    ch,uch = obMenu(obs,title="sort by:")

    cols = []
    for ob in obs.columns:
        uos = obs[ob].unique()
        if len(uos) > 30 or len(uos) < 2:
            continue
        for uo in uos:
            cols.append(str(ob)+"   "+str(uo))
    A = np.zeros((len(uch),len(cols)))
    #O = np.zeros((len(uch),len(cols)),dtype=object)
    for i,cat in enumerate(uch):
        sobs = obs.loc[obs[obs.columns[ch]] == cat,:]
        for j,c in enumerate(cols):
            clis = c.split("   ")
            ob = clis[0]
            uo = clis[1]
            key = sobs[ob] == uo
            A[i,j] = key.sum()/sobs.shape[0]
    print(A)
    ndf = pd.DataFrame(index=uch,columns=cols,data=A)
    obs = pd.DataFrame(index=uch,columns=[obs.columns[ch]],data=uch)
    xy = pd.DataFrame(index=uch,columns=["x","y"],data=np.zeros((len(uch),2)))
    return(ndf,obs,xy)



def aggregate(df,obs,dfxy,cdf=None):
    for i,ob in enumerate(obs.columns):
        print(i,":",ob)
    print("aggregate by which?")
    ch1 = int(input("enter number:"))
    ch2 = int(input("0:average biomarker values or 1:count fraction positive"))
    color = obs.columns[ch1]
    bins = np.unique(np.array(sorted(obs[obs.columns[ch1]])))
    newA = []
    newObs = []
    newXY = []
    for bi in bins:
        #print(bi)
        arow = []
        orow = []
        xrow = []
        key = obs[color] == bi
        sdf = df.loc[key,:]
        sobs = obs.loc[key,:]
        sxy = dfxy.loc[key,:]
        if ch2 == 1:
            for col in sdf.columns:
                tdf = sdf.loc[sdf[col]>0,:]
                arow.append(tdf.shape[0]/sdf.shape[0])
        else:
            arow = list(sdf.mean(axis=0))
        for col in sobs.columns:
            orow.append(sp.stats.mode(list(sobs[col]))[0][0])
        for col in sxy.columns:
            xrow.append(np.mean(list(sxy[col])))
            #xrow.append(float(sxy[col].mean()))
        newA.append(arow)
        newObs.append(orow)
        newXY.append(xrow)
    newDF = pd.DataFrame(data=newA, columns=df.columns,index=bins)
    newObs = pd.DataFrame(data=newObs,columns = obs.columns,index=bins)
    newXY = pd.DataFrame(data=newXY,columns=dfxy.columns,index=bins)
    print(newObs)
    return(newDF,newObs,newXY)



def clusterMeans(DF,obs,dfxy,oname = 'slide'):
    clusterA = []
    try:
        oname = obs.name
    except:
        pass
    print(oname,'oname')
    try:
        try:
            clName = oname
            obs.name = clName
        except:
            try:
                clName = "slide_scene"
                obs.name = clName
            except:
                clName = obs.columns[-1]
                obs.name = clName
        ucl = np.unique(obs.loc[:,clName].values)

    except Exception as e:
        print("clustermeans",e)
        DF,obs,dfxy= obCluster(DF, obs, dfxy)
        clName = obs.name
        ucl = np.unique(obs.loc[:,clName].values)
        #obs.name = clName
    nClusters = len(ucl)
    clusterA = np.zeros((nClusters,DF.shape[1]))
    print("ucl",ucl)
    for i,c in enumerate(ucl):
        cl = DF.loc[obs[clName] == c,:]
        try:
            markerMeans = np.mean(cl.values,axis = 0)
        except:
            markerMeans = np.mean(cl.values.astype(float),axis = 0)
        clusterA[i,:] = markerMeans
    #print(clusterA,"clusterA")
    return(clusterA,ucl,obs)

def clusterMeans1(DF,obs,dfxy):
    clusterA = []
    try:
        ucl = np.unique(obs.loc[:,'Cluster'].values)
        clName = "old clustermeans"
    except:
        DF,obs,dfxy,clName=obCluster(DF, obs, dfxy)
        ucl = np.unique(obs.loc[:,'Cluster'].values)
    nClusters = len(ucl)
    clusterA = np.zeros((nClusters,DF.shape[1]))
    print("ucl",ucl)
    for i,c in enumerate(ucl):
        cl = DF.loc[obs['Cluster'] == c,:]
        try:
            markerMeans = np.mean(cl.values,axis = 0)
        except:
            markerMeans = np.mean(cl.values.astype(float),axis = 0)
        clusterA[i,:] = markerMeans
    #print(clusterA,"clusterA")
    return(clusterA,ucl,obs,clName)

def silh(df,obs,dfxy,cdf):
    #print(obs.columns,'into silh')
    obs = obs.fillna("nan")
    print(obs.columns)
    cols = []
    scores = []
    keys = input("string required in every annotation to compare (send blank to skip)")
    for col in sorted(obs.columns):
        if keys != "" and keys not in col:
            continue
        ncat = len(obs.loc[:,col].unique())
        if ncat < 2 or ncat > 20:
            continue
        try:
            sco = silhouette_score(df,obs.loc[:,col])
            silhouette_vals = silhouette_samples(df,obs.loc[:,col])
            #silhH1(silhouette_vals,obs.loc[:,col])
            #print(col,sco)
            cols.append(col)
            scores.append(sco)
        except Exception as e:
            print("could not score",col,"    ",e)
    plt.scatter(cols,scores)
    plt.xticks(rotation = 90)
    plt.grid(visible=True,axis='x')

    plt.show()
    return(df,obs,dfxy)

def silhH1(silhouette_vals,labels):
    fig,ax1 = plt.subplots()
    y_ticks = []
    y_lower, y_upper = 0, 0
    for i, cluster in enumerate(np.unique(labels)):
        cluster_silhouette_vals = silhouette_vals[labels == cluster]
        cluster_silhouette_vals.sort()
        y_upper += len(cluster_silhouette_vals)
        ax1.barh(range(y_lower, y_upper), cluster_silhouette_vals, edgecolor='none', height=1)
        ax1.text(-0.03, (y_lower + y_upper) / 2, str(i + 1))
        y_lower += len(cluster_silhouette_vals)
    plt.show()

'''
PROCESSING
'''

#def cdfToProcessing(df,obs,dfxy,cdf):
    #cobs =
'''
Endothelial: CD31+ and others.
From others in 1), Tumor: any of Ecad+, CK19+, CK7+, CK8+, CK5+, CK14+, CK17+, MUC1+ and others.
From others in 2), Immune: any of CD20+, CD45+, CD4+, CD8+, CD68+, CD3+ and others.
From others in 3), Active fibroblast: any of aSMA+, VIM+, CAV1 and others.
From others in 4) is Inactive fibroblast.
Tumor in 2), tumor proliferative: Ki67+, PCNA+, pHH3+.
Tumor in 2), luminal: CK19+, CK7+, CK8+, but none of CK5+, CK14+, CK17+.
Tumor in 2), luminal-basal: any of Ecad+, CK19+, CK7+, CK8+, and any of CK5+, CK14+, CK17+.
Tumor in 2), luminal-basal: any of CK5+, CK14+, CK17+, but none of CK19+, CK7+, CK8+.
Tumor in 2), HR+ tumor: any of ER+, PgR+.
Tumor in 2), HER2+ tumor: any of HER2+.
Tumor in 2), none of HER2+, ER+, PgR+, so exclusive 10) and 11).
Immune in 3), immune proliferative: Ki67+, PCNA+, pHH3+.
Immune in 3), T-cell: CD4+, CD8+, CD3+.
Immune in 3), B-cell: CD20+.
Immune in 3), macrophage: CD68+, then others are “other immune”.
'''

#calculate the average distance to 25 nearest neighbors
#for each biomarker, find fraction of X+ cells nearby (X being every other biomarker)
#implement chi-squared testing function


def subset(df,obs,dfxy,n=False,replace=False):
    if type(n) != int:
        n = int(input("how many cells?"))
    inds = []
    sh = df.shape[0]
    while len(inds) <= n:
        ind = random.randint(0,sh-1)
        if ind not in inds:
            inds.append(ind)
    if replace:
        df= df.iloc[inds,:]
        obs = obs.iloc[inds,:]
        dfxy=dfxy.iloc[inds,:]
    else:
        obs['subset'] = 'out'
        obs.loc[obs.index[inds],'subset'] = 'in'


    return(df,obs,dfxy)

def simulateTherapy(df,obs,dfxy):
    maxCombs = 2
    ch,uch = obMenu(obs,title="celltype selector")
    ob = obs.columns[ch]
    print(uch)
    celltype = []
    for i,uc in enumerate(uch):
        print(i,":",uc)
    ch = int(input("number of celltype to try to eliminate:"))
    celltype.append(uch[ch])
    print(celltype,"type to compare")


    zdfq = input("zscore dataframe? (y)")
    thresholdSD = float(input("threshold (in SD if zscored)"))

    #get only non-primary DF (substitute later for non-essential-to-life df): disabled
    pDf,n,nn = onlyPrimaries(df,obs,dfxy)
    nP = []
    for biom in df.columns:
        if biom not in pDf.columns:
            nP.append(biom)
    print(nP,"non primary")
    npDf = df.loc[:,nP]


    #threshA = np.where(npDf.values>thresholdSD,0,1)


    #find every combination of biomarkers up to maxCombs
    combinations = []
    for i in range(maxCombs):
        if len(combinations) == 0:
            combinations = list(npDf.columns)
        else:        #len(combinations) < maxCombs*len(bioms):
            for comb in combinations:
                if len(comb.split("#")) == maxCombs:
                    continue
                for biom in npDf.columns:
                    if biom not in comb:
                        combinations.append(comb+"#"+biom)

    #make threshold matrix matching dataframe
    if zdfq == "y":
        npDf,n,nn = zscore(npDf,obs,dfxy,ax=0)
    npDf= pd.DataFrame(np.where(npDf.values>thresholdSD,0,1),index=npDf.index,columns=npDf.columns)
    keptFrames = []
    scores = []
    for comb in combinations:
        frames = [df.copy(),obs.copy(),dfxy.copy()]
        for frame in frames:
            frame.index = pd.Series(frame.index) +"_"+comb
        #print("\n\n",comb)
        key = None
        for biom in comb.split("#"):
            try:
                key = key * npDf.loc[:,biom].values
            except:
                #print("starting key")
                key = npDf.loc[:,biom].values

        #print(key)#),~key <-?)
        key = pd.Series(key)==1
        #key.index= npDf.index
        #print(key,key.shape[0])
        #print(df.index,key.index)
        for i,frame in enumerate(frames):
            #print(frame.shape[0],key.shape[0],"frame,key")
            #print(frame.index==key.index)  SUPER WEIRD THE STRS ARE THE SAME BUT  == is [False False False False]
            key.index=frame.index
            frames[i] = frame.loc[key,:]
        totalRem = frames[1].shape[0]/obs.shape[0]*100
        cellt = celltype[0]
        tumKey = obs.loc[:,ob] == cellt
        tumO = obs.loc[tumKey,:]
        otherO = obs.loc[~tumKey,:]
        thTumKey = frames[1].loc[:,ob] == cellt
        thTumRem = frames[1].loc[thTumKey,:]
        thOtherRem = frames[1].loc[~thTumKey,:]
        typeRem = thTumRem.shape[0]/tumO.shape[0]*100
        otherRem = thOtherRem.shape[0]/otherO.shape[0]*100
        #score = (100-typeRem)**2/(100-totalRem)*totalRem**2 #PDPN
        #score = (100-typeRem)/(100-totalRem) #gH2AX
        score = otherRem/typeRem
        scores.append(score)
    Mind = scores.index(max(scores))
    bestComb = combinations[Mind]
    print(bestComb,"Best combination of biomarkers to remove, score:",max(scores))
    for i,biom in enumerate(bestComb.split("#")):
        if i==0:
            key1 = npDf.loc[:,biom].values
        else:
            key1 = key1 * npDf.loc[:,biom].values


    key1 = pd.Series(key1)==1
    #print(key,key1,"key key1")
    key1.index=df.index
    try:
        obs["slide"]+="_"+bestComb
        obs["slide_scene"]+="_"+bestComb
    except IndexError:
        print("slide and'or slide_scene missing from obs annotations")
    return(df.loc[key1,:],obs.loc[key1,:],dfxy.loc[key1,:])






def neighborhoodEntropy(df,obs,dfxy):

    #calculate sample entropy based on inputs, print sample entropy, ask user if binarize into high-low based on sample total
    radii = []
    while True:
        try:
            radii.append(float(input("radius (in um) to consider neighbors: (send blank when done)"))/PXSIZE)
        except:
            if len(radii) < 1:
                print("invalid radius, using 25 as default")
                radii = [25/PXSIZE]
            break
    ch,uch = obMenu(obs,title="category to add neighborhood fractions to dataframe")
    print(uch)
    goodsts = []
    while True:
        inp = input("string to include in celltypes to consider as centers (blank to include all, blank when done. Leaves NAN values if not all): ")
        if inp == "":
            break
        else:
            goodsts.append(inp)
    #much = len(uch)
    obcol = obs.columns[ch]
    if len(goodsts) == 0:
        goodsts = uch
    for radius in radii:
        ecol = obcol+"_entropy_"+str(radius*PXSIZE)
        df[ecol] = 0
        for us in tqdm(sorted(list(obs["slide_scene"].unique())),us):
            key0 = obs["slide_scene"] == us
            tdfxy = dfxy.loc[key0,:]
            tobs = obs.loc[key0,:]
            for i in range(tdfxy.shape[0]):
                ind = tdfxy.index[i]
                if len(goodsts) > 0:
                    check = False
                    for gs in goodsts:
                        if gs in tobs.loc[ind,obcol]:
                            check = True
                if not check:
                    continue

                #if i % 1000 == 1:
                #    print(i/tdfxy.shape[0]*100,"% done with",us)
                #neighbors = []
                x,y = tdfxy.iloc[i,0],tdfxy.iloc[i,1]
                nx,ny = tdfxy.iloc[:,0],tdfxy.iloc[:,1]
                distanceV = ((x-nx)**2+(y-ny)**2)**.5
                key = distanceV < radius
                neighbors = tobs.loc[key,:]
                neighbors = neighbors.drop(pd.Series(tdfxy.index).iloc[i])
                nnei = neighbors.shape[0]
                entropy = 0
                if nnei > 1:
                    for uc in uch:
                        inNei = neighbors.loc[neighbors.loc[:,obcol] == uc,obcol]
                        Pi = inNei.shape[0]/nnei
                        if Pi != 0:
                            entropy -= Pi * math.log2(Pi)

                        #df.loc[ind,uc+"_"+obcol+"_neighbors_"+str(radius)] = 1/much #add equal fraction of each category
                df.loc[ind,ecol] = entropy

    return(df,obs,dfxy)




def neighborhoodFractions(df,obs,dfxy):
    tot = False
    if input("count totals instead of fractions? (y)") == "y":
        tot = True
    radii = []
    while True:
        try:
            radii.append(float(input("radius (in um) to consider neighbors: (send blank when done)"))/PXSIZE)
        except:
            if len(radii) < 1:
                print("invalid radius, using 25 as default")
                radii = [25/PXSIZE]
            break
    ch,uch = obMenu(obs,title="category to add neighborhood fractions to dataframe")
    print(uch)
    goodsts = []
    while True:
        inp = input("string to include in celltypes to consider as centers (blank to include all, blank when done. Leaves NAN values if not all): ")
        if inp == "":
            break
        else:
            goodsts.append(inp)
    #much = len(uch)
    obcol = obs.columns[ch]
    if len(goodsts) == 0:
        goodsts = uch
    for radius in radii:
        for uc in uch:
            df[uc+"_"+obcol+"_neighbors_"+str(radius*PXSIZE)] = 0
        for us in obs["slide_scene"].unique():
            key0 = obs["slide_scene"] == us
            tdfxy = dfxy.loc[key0,:]
            tobs = obs.loc[key0,:]
            for i in range(tdfxy.shape[0]):
                ind = tdfxy.index[i]
                if len(goodsts) > 0:
                    check = False
                    for gs in goodsts:
                        if gs in tobs.loc[ind,obcol]:
                            check = True
                if not check:
                    continue

                if i % 1000 == 1:
                    print(i/tdfxy.shape[0]*100,"% done with",us)
                #neighbors = []
                x,y = tdfxy.iloc[i,0],tdfxy.iloc[i,1]
                nx,ny = tdfxy.iloc[:,0],tdfxy.iloc[:,1]
                distanceV = ((x-nx)**2+(y-ny)**2)**.5
                key = distanceV < radius
                neighbors = tobs.loc[key,:]
                neighbors = neighbors.drop(pd.Series(tdfxy.index).iloc[i])
                nnei = neighbors.shape[0]
                if nnei > 1:
                    #nuch = neighbors.loc[:,obcol]
                    for uc in uch:
                        inNei = neighbors.loc[neighbors.loc[:,obcol] == uc,obcol]
                        if tot:
                            df.loc[ind,uc+"_"+obcol+"_neighbors_"+str(radius*PXSIZE)] = inNei.shape[0]
                        else:
                            df.loc[ind,uc+"_"+obcol+"_neighbors_"+str(radius*PXSIZE)] = inNei.shape[0]/nnei
                else:
                    for uc in uch:
                        df.loc[ind,uc+"_"+obcol+"_neighbors_"+str(radius*PXSIZE)] = 0
                        #df.loc[ind,uc+"_"+obcol+"_neighbors_"+str(radius)] = 1/much #add equal fraction of each category
    return(df,obs,dfxy)


def nearestOfType(df,obs,dfxy):
    startRadius = 150/PXSIZE #microns
    maxRad = 2 #multiple of startRadius before quitting search to nearest
    #maybe it would be faster to just search the max from the start? How expensive is searching the whole index vs reindexing...
    #option for above is set startradius high and maxRad = 1
    #yeah this is faster... though possible to optimize further with maxRad = 2 and good start
    chs, uchs,tys = [],[],[]
    while True:
        try:
            ch,uch = obMenu(obs,title="category containing celltype to measure distance from (send non int to escape)")
            types = []
        except:
            if len(types) > 0:
                break
        for i,uc in enumerate(uch):
            print(i,":",uc)
        while True:
            try:
                print("send non-int to escape")
                ich = int(input("number of celltype to measure distance from:"))
                print(ich)
                ty = uch[ich]
                print(ty)
                types.append(ty)
            except:
                print(types)
                if len(types) == 0:
                    print('send integer corresponding to celltype in above list')
                else:
                    break
        chs.append(ch)
        uchs.append(uc)
        tys.append(types)
    for ii in range(len(chs)):
        ch,uch,types = chs[ii],uchs[ii],tys[ii]
        for ty in types:
            tytle = 'nearest '+obs.columns[ch]+': '+ty
            df[tytle] = 9999
            for us in sorted(list(obs["slide_scene"].unique())):
                key0 = obs["slide_scene"] == us
                tdfxy = dfxy.loc[key0,:]
                tobs = obs.loc[key0,:]

                tykey = tobs.iloc[:,ch] == ty
                ydfxy = tdfxy.loc[tykey,:]
                yobs = tobs.loc[tykey,:]

                #print("\n",us)
                for i in tqdm(range(tdfxy.shape[0]),"calculating "+ty+" "+us):
                #for i in range(tdfxy.shape[0]):
                    ind = tdfxy.index[i]
                    #print('    ',i)
                    if tobs.iloc[i,ch] == ty:
                        df.loc[ind,tytle] = 0
                        #print('im')
                        continue
                    nearest = 9999999
                    j = 1
                    while nearest == 9999999:
                        #nearest = getNearest(i,tdfxy,tobs,startRadius * j,ch)
                        x,y = tdfxy.iloc[i,0],tdfxy.iloc[i,1]
                        nx,ny = ydfxy.iloc[:,0],ydfxy.iloc[:,1]
                        distanceV = ((x-nx)**2+(y-ny)**2)**.5
                        key = distanceV < startRadius * j
                        neighbors = yobs.loc[key,:]
                        keyt = neighbors.iloc[:,ch] == ty
                        if keyt.sum() != 0:
                            nearest = distanceV.loc[key & keyt].min()
                            break #redundant because of while statement
                            #print(nearest)
                        if j >= maxRad:
                            nearest = startRadius * j
                        j += 1
                    df.loc[ind,tytle] = nearest * PXSIZE
                    #print("added ", df.iloc[i,:].loc['distance to nearest '+ty],":",nearest * PXSIZE)
    return(df,obs,dfxy)



'''
def getNearest(i,tdfxy,tobs,radius,ch):
    x,y = tdfxy.iloc[i,0],tdfxy.iloc[i,1]
    nx,ny = tdfxy.iloc[:,0],tdfxy.iloc[:,1]
    distanceV = ((x-nx)**2+(y-ny)**2)**.5
    key = distanceV < radius
    neighbors = tobs.loc[key,:]
    keyt = neighbors.iloc[:,ch] == ty
    if keyt.sum == 0:
        return("none")
    else:
        neiTy = neighbors.loc[keyt,:]
'''




def regionAverage(df,obs,dfxy):
    try:
        radius = float(input("radius (um) to consider in average: "))/PXSIZE
    except:
        print("invalid radius, using 25u as default")
        radius = 25/PXSIZE
    ndf = []
    for us in obs["slide_scene"].unique():
        key0 = obs["slide_scene"] == us
        tdfxy = dfxy.loc[key0,:]
        tobs = obs.loc[key0,:]
        tdf = df.loc[key0,:]
        for i in range(tdfxy.shape[0]):
            if i % 1000 == 1:
                print(i/tdfxy.shape[0]*100,"% done with",us)
            neighbors = []
            x,y = tdfxy.iloc[i,0],tdfxy.iloc[i,1]
            nx,ny = tdfxy.iloc[:,0],tdfxy.iloc[:,1]
            distanceV = ((x-nx)**2+(y-ny)**2)**.5
            key = distanceV < radius
            neighbors = tdf.loc[key,:]
            neighbors = neighbors.drop(pd.Series(tdfxy.index).iloc[i])
            if neighbors.shape[0]> 1:
                #print(neighbors)
                #neighbors = pd.DataFrame(neighbors)#,columns = pd.Series(df.columns)+" in range "+str(radius))
                #print(neighbors)
                avg = neighbors.mean(axis=0)
                #print(avg,type(avg))
                #print(avg.values)
                ndf.append(pd.DataFrame(avg.values,index=df.columns,columns = [pd.Series(tdf.index).iloc[i]]).transpose())
            else:
                ndf.append(pd.DataFrame(columns =df.columns ,index=[pd.Series(tdf.index).iloc[i]] ))#
        #print(ndf)
    ndf = pd.concat(ndf,axis=0)
    print("ndf start",ndf,"ndf end")
    avgs = ndf.mean(axis=0)
    for biom in ndf.columns:
        print(biom,"biom")
        ndf.loc[:,biom]=ndf.loc[:,biom].fillna(0)#avgs.loc[biom])
    ndf = ndf.set_axis(pd.Series(ndf.columns)+" in radius "+str(radius*PXSIZE),axis=1)
    df= pd.concat([df,ndf],axis=1)
    if input("sort df? (y)") == "y":
        df = df.loc[:,df.columns.sort_values()]
    return(df,obs,dfxy)


def regionAverage1(df,obs,dfxy): #make only check same slidescene
    radius = float(input("radius to consider in average: "))
    ndf = []
    for i in range(dfxy.shape[0]):
        neighbors = []
        x,y = dfxy.iloc[i,0],dfxy.iloc[i,1]
        for j in range(dfxy.shape[0]):
            if i == j:
                continue
            nx,ny = dfxy.iloc[j,0],dfxy.iloc[j,1]
            distance = ((x-nx)**2+(y-ny)**2)**.5
            if distance < radius:
                neighbors.append(df.iloc[j,:])
        if len(neighbors) > 1:
            print(neighbors)
            neighbors = pd.DataFrame(neighbors)#,columns = pd.Series(df.columns)+" in range "+str(radius))
            print(neighbors)
            avg = neighbors.mean(axis=0)
            print(avg,type(avg))
            ndf.append(avg)
        else:
            ndf.append(pd.DataFrame(columns =df.columns ,index=[pd.Series(df.index).iloc[i]] ))#
    print(ndf)
    ndf = pd.DataFrame(ndf)
    avgs = ndf.mean(axis=0)
    for biom in ndf.columns:
        ndf.loc[:,biom].fillna(avgs.loc[biom])
    ndf = ndf.set_axis(pd.Series(ndf.columns)+" in radius "+str(radius))
    df= pd.concat([df,ndf],axis=1)
    return(df,obs,dfxy)



def regionAverage1(df,obs,dfxy): #make only check same slidescene
    radius = float(input("radius to consider in average: "))
    ndf = []
    for i in range(dfxy.shape[0]):
        neighbors = []
        x,y = dfxy.iloc[i,0],dfxy.iloc[i,1]
        for j in range(dfxy.shape[0]):
            if i == j:
                continue
            nx,ny = dfxy.iloc[i,0],dfxy.iloc[i,1]
            distance = ((x-nx)**2+(y-ny)**2)**.5
            if distance < radius:
                neighbors.append(df.iloc[j,:])
        if len(neighbors) > 1:
            print(neighbors)
            neighbors = pd.DataFrame(neighbors)#,columns = pd.Series(df.columns)+" in range "+str(radius))
            print(neighbors)
            avg = neighbors.mean(axis=0)
            print(avg,type(avg))
            ndf.append(avg)
        else:
            ndf.append(pd.DataFrame(columns =df.columns ,index=[pd.Series(df.index).iloc[i]] ))#
    print(ndf)
    ndf = pd.DataFrame(ndf)
    avgs = ndf.mean(axis=0)
    for biom in ndf.columns:
        ndf.loc[:,biom].fillna(avgs.loc[biom])
    ndf = ndf.set_axis(pd.Series(ndf.columns)+" in radius "+str(radius))
    df= pd.concat([df,ndf],axis=1)
    return(df,obs,dfxy)




def autotype(df,obs,dfxy,chanT=True,name="autoCellType res: ",res=None): #the old version that keeps more information is in cmifAnalysis36
    roundThresh = [1500,1250,1000,750]
    #biomRounds is actually biomChannels
    biomRounds = [['CAV1', 'CK17', 'CK5', 'CK7', 'CK8', 'H3K27', 'MUC1', 'PCNA', 'R0c2', 'R6Qc2', 'Vim', 'aSMA', 'pHH3'],
                  ['AR', 'CCND1', 'CD68', 'CD8', 'CK14', 'CoxIV', 'EGFR', 'H3K4', 'HER2', 'PDPN', 'R0c3', 'R6Qc3', 'pS6RP'],
                  ['BCL2', 'CD31', 'CD4', 'CD45', 'ColIV', 'ER', 'Ki67', 'PD1', 'PgR', 'R0c4', 'R6Qc4', 'gH2AX', 'pRB'],
                  ['CD20', 'CD3', 'CD44', 'CK19', 'CSF1R', 'ColI', 'Ecad', 'FoxP3', 'GRNZB', 'LamAC', 'R0c5', 'R6Qc5', 'RAD51']]

    odf = df.copy()
    if not res:
        res= float(input("number of standard deviations above mean required to count as +"))
    #chanT = False
    if chanT != False:
        if input("check if threshold above channel threshold (can't already be z-scored) (y)") == 'y':
            chanT = True
            key2 = pd.DataFrame(data=np.zeros_like(odf),columns=odf.columns,index=odf.index)
            means = df.mean(axis=0)
            sds = df.std(axis=0)
            zSer = pd.Series(index = df.columns,data=means+sds*res)
            print(zSer)
            for i,roun in enumerate(biomRounds):
                rawThresh = roundThresh[i]
                for bIm in roun:
                    for bim in df.columns:
                        if bIm+'_' in bim:
                            key2.loc[odf.loc[:,bim]>rawThresh,bim] = 1
                            #bimT = zSer.loc[bim]
                            #key2.loc[odf.loc[:,bim]>bimT,bim] = 1
                            #print("ding")
        else:
            chanT = False

    #print(list(key2.iloc[:,0]),"ar key")
    obs[name+str(res)] = " "
    if input("zscore? (required if threshold given in SD)") == "y":
    #print(df)
        df,obs,dfxy = zscorev(df,obs,dfxy)
    #print(df)
    mapp = {}
    toThresh = []
    for biom in df.columns:
        if "neigh" in biom:
            continue
        cType = fillMap(biom)
        if cType != None:
            mapp[biom]=cType
    toThresh = list(mapp.keys())
    #others = ["Ki67", "PCNA", "pHH3","pRB","ER","PgR","AR","HER2","Fox","FOX","GRNZB","SMA","Vim","VIM","ColI","PD1","PD-1","E-cad","Ki-67","CAV"]
    others = list(df.columns)
    for biom in df.columns:
        for o in others:
            if o in biom:
                toThresh.append(biom)
    for biom in toThresh:
        if chanT:
            key3 = key2[biom] == 1
            key = df[biom]>res
            print(biom,any(key),any(key3),any(key & key3))
            obs.loc[key & key3,name+str(res)] += biom + " "
        else:
            key = df[biom]>res
            obs.loc[key,name+str(res)] += biom + " "
    obs = parseTypes(df,obs,dfxy,column=name+str(res))
    print(obs[name+str(res)].unique(),"uobs")
    obs = parseSecondary(df,obs,dfxy,column=name+str(res))
    #if input("keep z-scoring?") == "y":
        #return(df,obs,dfxy)
    return(odf,obs,dfxy)

def manThresh(df,obs,dfxy):
    ch = input("import manual thresholds from csv? (y)")
    if ch == "y":
        obs["Manual Celltype"] = ""
        chh = input("subtract thresholds? (y)")
        path = input('filepath/name:')
        thresh = pd.read_csv(path)
        for biom in thresh.columns:
            th = float(thresh[biom].iloc[0])
            print(biom,th)
            key = df[biom] > th
            obs.loc[key,"Manual Celltype"] += biom
            if chh == "y":
                df[biom] -= th
                df.loc[df[biom]<0,biom] = 0
    else:
        if input("manually add annoations for every combination of positivies for custom biomarker set? (y)") == 'y':
            ch,uch = obMenu(obs,'repeat for each unique annotation in: ')
            for uc in uch:
                key = obs.iloc[:,ch] == uc
                sdf,sobs,sxy = df.loc[key,:],obs.loc[key,:],dfxy.loc[key,:]
                sdf,sobs=ort.tertiary(sdf,sobs,sxy)
                for ob in sobs.columns:
                    if ob not in list(obs.columns):
                        obs[ob] = ''
                obs.loc[key,:] = sobs
            return(df,obs,dfxy)
        chhh = int(input("0:chose obs category or 1:all together"))
        if chhh == 0:
            df,obs = ort.secondary(df,obs,dfxy)
            print(obs.columns)
        else:
            df,obs = ort.primary(df,obs,dfxy)
    obs = parseTypes(df,obs,dfxy,column="Manual Celltype")
    obs = parseSecondary(df,obs,dfxy,column="Manual Celltype")
    return(df,obs,dfxy)


def clort(df,obs,dfxy):
    pass

def clauto(df,obs,dfxy):
    obs = obs.astype(str)
    print(obs.shape)
    oobs = obs.copy()
    chs, uchs = [],[]
    while True:
        try:
            ch,uch=obMenu(obs,"obs category to auto-annotate cell types")
            chs.append(ch)
            uchs.append(uch)
        except:
            break
    for i,ch in enumerate(chs):
        uch = uchs[i]
        adf,aobs,axy = clag(df,obs,dfxy,ch,uch)
        x,aobs,xx = autotype(adf,aobs,axy,chanT=False,name=obs.columns[ch]+"cluster autotype",res=float(input('resolution:')))
        print(obs.shape)
        for col in aobs.columns:
            if obs.columns[ch]+"cluster autotype" in col:
                print(col,aobs.loc[:,col].unique(),"!!")
                obs[col] = ""
                for uc in aobs.index:
                    key = obs.iloc[:,ch] == uc
                    obs.loc[key,col] = aobs.loc[uc,col]
        print(obs.shape,df.shape)
        #print(obs,df)
    return(df,obs,dfxy)




def clag(df,obs,dfxy,ch=None,uch=None,z=False): #switched default to False 4/9/25
    if not ch:
        ch,uch=obMenu(obs,"obs category by which to aggregate")

    if z:
        zdf,zobs,zxy = zscorev(df,obs,dfxy)
    else:
        zdf,zobs,zxy = df,obs,dfxy
    ocol = obs.columns[ch]
    ndf,nobs,nxy = [],[],[]
    for uc in uch:
        key = zobs.loc[:,ocol] == uc
        sdf = zdf.loc[key,:]
        sobs = zobs.loc[key,:]
        sxy = zxy.loc[key,:]
        ndf.append(sdf.mean(axis=0))
        nxy.append(sxy.mean(axis=0))
        #print(sobs.mode(axis=0).iloc[0,:],"/n/n")
        #time.sleep(1)
        nobs.append(sobs.mode(axis=0).iloc[0,:])
    dfs = [ndf,nobs,nxy]
    for i,d in enumerate(dfs):
        dfs[i] =pd.concat(d,axis=1).transpose()
        dfs[i].index = pd.Series(uch).astype(str)
        #print(dfs[i].columns)
        #print(dfs[i].shape)
        #print(dfs[i])
    return(dfs[0],dfs[1],dfs[2])



'''
obs.append(getObModes(sobs))
def getObModes(obs):
    modes = [] #1 per col
    for col in obs.columns:
        mod = obs.loc[:,col].mode(axis=0)
        key = obs.loc[:,col]==mode
        frac = key.sum()/obs.shape[0]
        modes.append(str(mod)+" "+str(frac*100)+"%")
'''

def parseTypes(df,obs,dfxy,column="none"):
    if column == 'none':
        ch,uch = obMenu("column to apply types to")
        column = obs.columns[ch]
    mapp = {}
    for biom in df.columns:
        cType = fillMap(biom)
        if cType != None:
            mapp[biom]=cType
    #print(mapp)
    mapp = {k: v for k, v in sorted(mapp.items(), key=lambda item: item[1])}
    print(mapp)
    obs["Primary Celltype "+column] = "5: stromal"
    for biom in mapp.keys():
        keyCol = obs[column].str.contains(biom)
        #print(list(keyCol))
        unasKey = obs["Primary Celltype "+column] == "5: stromal"
        obs.loc[keyCol & unasKey,"Primary Celltype "+column] = mapp[biom]
    return(obs)


def parseSecondary(df,obs,dfxy,column):
    uch = obs[column].unique()
    #print(uch,"uch")
    #obs["Secondary Celltype"+column] = obs["Primary Celltype "+column] #cmif39 has this version
    obs["proliferating "+column] = "no"
    obs["tumor subtype "+column] = np.nan
    obs["receptors "+column] = np.nan
    obs["immune subtype "+column] = np.nan
    obs["immune checkpoints "+column] = np.nan
    obs["cytotoxic "+column] = np.nan
    obs["fibroblast type "+column] = np.nan


    proL = ["Ki67","PCNA","pHH3","pRB",'pMYC',"Ki-67"]#
    lumL = ["CK19","CK7","CK8"]
    basL = ["CK5","CK14","CK17"]
    mesL = ["Vim","VIM","CD44"] #ANY MES MEANS NOT LUM BAS ETC.
    TL4 = ["CD4_"]
    TL8 = ["CD8"]
    TL3 = ["CD3_"] #LOW PRIORITY
    #if all 3 positive, call CD8, otherwise call CD8 CD4 'other T cell' for CD3_ cd4-cd8-
    BL = ["CD20"]
    macL = ["CD68","CD163"] #ADD CSF1R?
    Hl = ['ER', 'PgR', 'AR']
    HEl = ["HER2"]
    cpL = ["PD1","Fox","FOX","PD-1"]
    cytL = ["GRNZB"]
    acL = ["aSMA","Vim","VIM","ColI_","SMA"]
    for typ in uch:
        key = obs[column] == typ
        typeD = {'pro':0,'Lum':0,'Bas':0,"HR":0,"HER":0,'T-c4':0,'T-c8':0,'T-c3':0,
                 'B-c':0,'Mac':0,"CheckP":0,"CytoT":0,"activeFB":0,"mesen":0}
        if checkL(typ,proL):
            typeD['pro'] = 1
        if checkL(typ,mesL):
            typeD["mesen"] = 1
        if typeD["mesen"] == 0:
            if checkL(typ,lumL):
                typeD["Lum"] = 1
            if checkL(typ,basL):
                typeD["Bas"] = 1
        if checkL(typ,Hl):
            typeD['HR'] = 1
        if checkL(typ,HEl):
            typeD['HER'] = 1
        if checkL(typ,cpL):
            typeD["CheckP"] = 1
        if checkL(typ,cytL):
            typeD["CytoT"] = 1
        if checkL(typ,TL4):
            typeD['T-c4'] = 1
        elif checkL(typ,TL8):
            typeD['T-c8'] = 1
        elif checkL(typ,TL3):
            typeD['T-c3'] = 1
        elif checkL(typ,BL):
            typeD['B-c'] = 1
        elif checkL(typ,macL):
            typeD['Mac'] = 1
        #if checkL(typ,acL):
            #typeD["activeFB"] = 1
        #print(typ,typeD)
        #print(typ,typeD)
        recSwitch = 0
        tuseSwitch = 0
        for sty in typeD.keys():
            if typeD[sty] == 1:
                if sty in "mesen":
                    pKey = obs["Primary Celltype "+column] == "3: tumor"
                    obs.loc[key & pKey,"tumor subtype "+column] = sty
                if sty in "Lum Bas":
                    #print('ding')
                    pKey = obs["Primary Celltype "+column] == "3: tumor"
                    if tuseSwitch == 0:
                        obs.loc[key & pKey,"tumor subtype "+column] = sty + " "
                        tuseSwitch = 1
                    else:
                        obs.loc[key & pKey,"tumor subtype "+column] = obs.loc[key & pKey,"tumor subtype "+column]+ sty + " "
                        print(obs.loc[key & pKey,"tumor subtype "+column])
                if sty in "HR HER":
                    pKey = obs["Primary Celltype "+column] == "3: tumor"
                    if recSwitch != 0:
                        obs.loc[key & pKey,"receptors "+column] =obs.loc[key & pKey,"receptors "+column]+" "+ sty
                        #print(any(key&pKey))
                    else:
                        obs.loc[key & pKey,"receptors "+column] = sty
                        recSwitch = 1
                        #print(any(key&pKey),"!")
                if sty in "pro":
                    obs.loc[key,"proliferating "+column] = "yes"
                if sty in "CheckP":
                    pKey = obs["Primary Celltype "+column] == "2: immune"
                    obs.loc[key & pKey,"immune checkpoints "+column] = "yes"
                if sty in "CytoT":
                    pKey = obs["Primary Celltype "+column] == "2: immune"
                    obs.loc[key & pKey,"cytotoxic "+column] = "yes"
                if sty in "T-c4 T-c8 T-c3 B-c Mac":
                    pKey = obs["Primary Celltype "+column] == "2: immune"
                    obs.loc[key & pKey,"immune subtype "+column] = sty
                #if sty in "activeFB":
                    #pKey = obs["Primary Celltype "+column] == "4 stromal"
                    #obs.loc[key & pKey,"fibroblast type "+column] = "active FB"
                #else:
                    #obs.loc[key,"Secondary Celltype"+column] +=" "+ sty
    pKey = obs["Primary Celltype "+column] == "3: tumor"
    key =pd.isna( obs["tumor subtype " + column])
    #print(key.sum(),"number of np.nan")
    obs.loc[key & pKey,"tumor subtype " + column] = "negative"

    #print(obs.loc[:,"receptors " + column])
    pKey = obs["Primary Celltype "+column] == "3: tumor"
    key = pd.isna(obs["receptors " + column])
    obs.loc[key & pKey,"receptors " + column] = "TN"
    #print(obs.loc[:,"receptors " + column])

    pKey = obs["Primary Celltype "+column] == "2: immune"
    key = pd.isna(obs["cytotoxic " + column])
    obs.loc[key & pKey,"cytotoxic " + column] = "no"

    pKey = obs["Primary Celltype "+column] == "2: immune"
    key = pd.isna(obs["immune subtype " + column])
    obs.loc[key & pKey,"immune subtype " + column] = "Unclassified immune"

    pKey = obs["Primary Celltype "+column] == "2: immune"
    key = pd.isna(obs["immune checkpoints " + column])
    obs.loc[key & pKey,"immune checkpoints " + column] = "no"

    #pKey = obs["Primary Celltype "+column] == "4 stromal"
    #key = pd.isna(obs["fibroblast type "+column])
    #obs.loc[key & pKey,"fibroblast type " + column] = "support FB"

    return(obs)



def checkL(biomsS,lis):
    for ent in lis:
        if ent in biomsS:
            #print("donmg")
            return(True)
    return(False)


def fillMap(biom):
    #print("NOTE: IMMUNE CURRENTLY RESTRICTED TO CD4* CD8")
    bTypes = [["1: endothelial",["CD31"]],
              ["2: immune",["CD",'F480']],
              ["3: tumor",["CK","Ecad","MUC1",'EGFR',"HER",'amy',"E-cad"]],
              ["4: active fibroblast",["aSMA","Vim","VIM","ColI_","CD90","SMA"]]]
    for typeA in bTypes:
        for stem in typeA[-1]:
            if "CD44" in biom or "in radius" in biom or "neighbors" in biom:
                return(None)
            if stem in biom:
                return(typeA[0])






def superBiomDF(df,obs,dfxy):
    ndf = pd.DataFrame(0,index = df.index, columns = ["1: endothelial","2: immune","3: tumor","4: active fibroblast"])
    for biom in df.columns:
        typ = fillMap(biom)
        if typ != None:
            ndf.loc[:,typ] += df.loc[:,biom]
    return(ndf,obs,dfxy)

def onlyPrimaries1(df):
    print("old function?")
    1/0
    primaries = []
    keys = ["CK","CD","Ecad","Vim","aSMA","Ki67"]#,"Col","PDPN","Fox","PD1"]
    for biom in df.columns:
        for k in keys:
            if k in biom and "CD44" not in biom:
                primaries.append(biom)
    df = df.loc[:,sorted(primaries)]
    return(df)

def onlyPrimaries(df,obs,dfxy):
    primaries = []
    df = doPart(df)
    combd = []
    for biom in df.columns:
        #print(biom)
        if "nuclei_" in biom or "cell_" in biom:
            primaries.append(biom)
        elif "comb" in biom:
            #print(biom,"comb")
            combd.append(biom.split("_")[0])
    #print(combd)
    for biom in df.columns:
        if type(fillMap(biom)) == type(None):
            continue
        cond = 0
        for stem in combd:
            if stem in biom and "comb" not in biom:
                cond = 1
        if cond == 0:
            primaries.append(biom)
    print(primaries)
    df = df.loc[:,primaries]
    return(df,obs,dfxy)



def doPart(df):
    dShort=[]
    for col in df.columns:
        shortname = col.split("_")[0]
        if shortname not in dShort:
            dShort.append(shortname)
    for sCol in dShort:
        toComb = []
        for col in df.columns:
            if sCol == col.split("_")[0]:
                toComb.append(col)
        #print("to combine:",sCol,toComb)
        if len(toComb) > 1:
            df=combinePart(df,["_nuc","ellmem","ucadj","ytopla","erinuc","cyto"],toComb,"all",sCol)
            #df=combinePart(df,["uclei","nuc"],toComb,"nuc",sCol)
            #df=combinePart(df,["ellmem"],toComb,"cellmem",sCol)
            #df=combinePart(df,["ucadj","ytopla","erinuc","cyto"],toComb,"cyto",sCol)
    #print(df.columns[df.isna().any()])
    return(df)


def combinePart(df,partitionL,toComb,NName,sCol):
    print(toComb)
    sDF = pd.DataFrame(index=df.index)
    for biomarker in toComb:
        for name in partitionL:
            if name in biomarker:
                sDF[biomarker] = df[biomarker]
                continue
    if sDF.shape[1]>1:
        #print(sDF.columns,"sdf cols\n")
        df.loc[:,sCol+"_"+NName+"_combined"]=sDF.max(axis=1)
    else:
        pass
        #print("sDF only has one entry apparently", sDF.columns)
    return(df)


def celltype(df,obs,dfxy): #manual 1x1
    odf = df.copy()
    if input("keep only primary biomarkers? (y)") == "y":
        df,obs,dfxy = onlyPrimaries(df,obs,dfxy)

    if input("z-score data vertically (y)") == "y":
        df,obs,dfxy = zscorev(df,obs,dfxy)
    if input("z-score data horizontally (y)") == "y":
        df,obs,dfxy = zscoreh(df,obs,dfxy)
    dlen = df.shape[0]
    #dw = df.shape[1]
    try:
        obs.loc[:,"manual cell type"]
        ch = input("manual type exists, start over? (y)")
        if ch == "y" or ch == "Y":
            ch1 = input("keep old cell type? (y)")
            if ch1 == "y" or ch1 == "Y":
                nn = input("name for old cell type?")
                obs[nn] = obs["manual cell type"]
                obs["manual cell type"] = "unidentified"
            else:
                obs["manual cell type"] = "unidentified"
    except:
        obs["manual cell type"] = "unidentified"


    while True:
        print("type 'quit' to exit, send blank to skip")
        ind = random.randint(0,dlen)
        if obs["manual cell type"].iloc[ind] == "unidentified":
            plt.bar(df.columns,df.iloc[ind,:].values)
            plt.xticks(rotation = 90)
            plt.grid(axis='x')
            #plt.xlabel(df.columns)
            plt.show()
            ty = input("cell type:")
            if ty == "quit":
                break
            elif ty == "":
                pass
            else:
                obs["manual cell type"].iloc[ind] = ty
    print(obs["manual cell type"].unique())
    return(odf,obs,dfxy)



def revert(df,obs,dfxy):
    print(obs.columns)
    print(oobs.columns)
    if input('really? Undo all changes?') == 'y':
        return(ODF,OOBS,OXY)
    return(df,obs,dfxy)


def scale1(df,obs,dfxy):
    for col in df.columns:
        mx = max(df[col].max(),-df[col].min())
        df.loc[:,col] = df[col]/mx
    return(df,obs,dfxy)


def multCols(df,obs,dfxy):
    c1,n = obMenu(df,'column 1')
    c2,n = obMenu(df,'column 2')
    print('warning: if z-scored, will also show mutual non-expression')
    df[df.columns[c1]+' x '+df.columns[c2]] = df.iloc[:,c1] * df.iloc[:,c2]
    return(df,obs,dfxy)


def elmarScale(df,obs,dfxy):
    if input("jenny scale instead? div by stdev (y)") == "y":
        for col in df.columns:
            lis = list(df[col])
            std = stat.stdev(lis)
            df[col] = df[col]/std
    else:
        for col in df:
            df.loc[df[col]<1,col] = 1
        A = np.log(df.values)
        df= pd.DataFrame(A,index=df.index,columns=df.columns)
        print(df)
        df,obs,dfxy = zscore(df,obs,dfxy,ax=0)
        print(df)
        df += 3
        df = np.exp(df)
        print(df)
    return(df,obs,dfxy)

def biomRatio(df,obs,dfxy):
    for i,col in enumerate(df.columns):
        print(i,":",col)
    c1 = df.columns[int(input("numerator biomarker (int):"))]
    c2 = df.columns[int(input("denominator biomarker (int):"))]
    df[c1+"/"+c2] = 0
    df[c1+"/"+c2] = df[c1]/df[c2]
    return(df,obs,dfxy)


def log2(df,obs,dfxy):
    try:
        base = int(input('log base? (2 default)'))
    except:
        base = 2
    npones = np.ones(df.values.shape) #/100
    newVals = np.maximum(df.values,npones)
    newVals = np.array(newVals,dtype=float)
    newVals = np.log2(newVals)/np.log2(base)
    df_calx = pd.DataFrame(data = newVals, index = df.index, columns = df.columns)
    return(df_calx,obs,dfxy)


def zscorev(df,obs,dfxy):
    df,obs,dfxy = zscore(df,obs,dfxy,ax=0)
    return(df,obs,dfxy)

def zscoreh(df,obs,dfxy):
    df,obs,dfxy = zscore(df,obs,dfxy,ax=1)
    return(df,obs,dfxy)

def zscore(df,obs,dfxy,ax=None):
    vals = df.values
    shape = vals.shape
    if ax == None:
        print("0 for vertical (by protein), 1 for horizontal")
        ax = int(input("axis (0/1):"))
    newA = np.zeros(shape)
    if ax == 0:
        for i in range(shape[1]):
            col = vals[:,i].tolist()
            try:
                zCol = zScoreL(col)
            except:
                zCol = list(np.zeros(len(col)))
            newA[:,i] = zCol
    if ax == 1:
        for i in range(shape[0]):
            col = vals[i,:].tolist()
            zCol = zScoreL(col)
            newA[i,:] = zCol
    return(pd.DataFrame(data=newA,columns=df.columns,index=df.index),obs,dfxy)

def zScoreL(lis):
    newLis = []
    mean = stat.mean(lis)
    std = stat.stdev(lis)
    if std == 0:
        return(np.zeros(len(lis)))
    for i in lis:
        newLis.append((i-mean)/std)
    return(newLis)

def outliers(df,obs,dfxy):
    if input('subtract QC channel expression instead (y)') == 'y':
        return(subQC(df,obs,dfxy))
    if input('subtract channel thresholds instead? (y)') == 'y':
        return(subChanThresh(df,obs,dfxy))
    df = df.clip(lower=df.quantile(q=0.0013), upper=df.quantile(q=0.9987), axis=1)  # cut by 3sigma = 0.0214 0.9786 or 4sigma = 0.0013 0.9987
    return(df,obs,dfxy)


def subQC(df,obs,dfxy):
    odf = df.copy()
    biomRounds = [["CK5","TFF1","PCNA","MUC1","pHH3","CK7",'Vim','aSMA','CK17','CK8','CAV1','GATA6','p63','FN','LamB1']+['R0c2', 'Vim', 'S100A', 'panCK', 'FN', 'H3K27', 'R1Qc2', 'pHH3', 'aSMA', 'GFAP', 'LamB1', 'NeuN', 'R6Qc2', 'PCNA', 'Myelin', 'CAV1', 'Glut1'],
                  ["HER2","CCND1","EGFR","PDL1","pERK","pS6RP",'CTNNB1','CD68','AR','CD8','CK14','PDPN','CD90','CTNNB','S100A4','p53','CD11c','CD11b','CD133','CD56']+['R0c3', 'HLA-DR', 'CD56', 'p53', 'CD11b', 'H3K4', 'R1Qc3', 'pMYC', 'CD8', 'CTNNB', 'CD68', 'CD90', 'R6Qc3', 'TUBB3', 'PDL1', 'CD11c', 'CD133'],   #555 channel
                  ["KLF4","Ki67","ER","PgR","pAKT","CD31",'pRB','CD45','CD4','BCL2','CD163','PD1','H3K27ac','pMYC','MSH6','gH2AX','pRPA']+['R0c4', 'MSH6', 'BCL2', 'pRPA', 'gH2AX', 'ColIV', 'R1Qc4', 'Ki67', 'CD4', 'IBA1', 'CD45', 'CD31', 'R6Qc4', 'CD163', 'p63', 'PD1', 'pMYCab'],   #680 chan
                  ["CK19","CD44","Ecad","BMP2","ZEB1","YAP1",'FoxP3','CD20','CD3','RAD51','53BP1','GRNZB','LamAC','Rad51','CGA']+['R0c5', 'R0c5', '53BP1', '53BP1', 'Rad51', 'Rad51', 'ZEB1', 'ZEB1', 'CD44', 'CD44', 'ColI', 'ColI', 'R1Qc5', 'R1Qc5', 'CD3', 'CD3', 'FoxP3', 'FoxP3', 'LamAC', 'LamAC', 'CD20', 'CD20', 'YAP1', 'YAP1', 'R6Qc5', 'R6Qc5', 'BMP2', 'BMP2', 'GRNZB', 'GRNZB', 'CGA', 'CGA', 'CSF1R', 'CSF1R'],
                  ]
    skips = ['nuclei_area','nuclei_eccentricity','DAPI']
    for marker in df.columns:
        switch = 0
        print(marker,switch)
        for skip in skips:
            if skip in marker:
                switch = 1
                break
        if switch == 1:
            print(skip,'in',marker,'skipping')
            continue

        QRS = []
        for i,roun in enumerate(biomRounds):
            for mkr in roun:
                if mkr in marker:
                    RounS = ['R0c','R1Qc','R6Qc']
                    #print(RounS,str(i+2))
                    #switch = 0
                    for bm in df.columns:
                        #if switch == 1:
                            #break
                        for RS in RounS:
                            if RS+str(i+2) in bm:
                                QRS.append(bm)
                                #switch = 1
                                break

        if len(QRS) == 0:
            print(marker)
            print('oops something missing')
            continue

        out = []
        for qrn in QRS:
            out.append((df.loc[:,marker] - df.loc[:,qrn]*df.loc[:,marker].mean()/df.loc[:,qrn].mean()))
        print(marker, qrn, df.loc[:,marker].mean()/df.loc[:,qrn].mean(),'ratio marker/qrn')
        print(df.loc[:,marker].mean(),df.loc[:,qrn].mean())
        out = pd.DataFrame(out,columns=df.index).T
        print(out.shape)
        odf.loc[:,marker] = out.min(axis=1)
        #print('subtracting',marker,qrn)
    return(odf,obs,dfxy)


def subQC1(df,obs,dfxy):
    biomRounds = [["CK5","TFF1","PCNA","MUC1","pHH3","CK7",'Vim','aSMA','CK17','CK8','CAV1','GATA6','p63','FN','LamB1']+['R0c2', 'Vim', 'S100A', 'panCK', 'FN', 'H3K27', 'R1Qc2', 'pHH3', 'aSMA', 'GFAP', 'LamB1', 'NeuN', 'R6Qc2', 'PCNA', 'Myelin', 'CAV1', 'Glut1'],
                  ["HER2","CCND1","EGFR","PDL1","pERK","pS6RP",'CTNNB1','CD68','AR','CD8','CK14','PDPN','CD90','CTNNB','S100A4','p53','CD11c','CD11b','CD133','CD56']+['R0c3', 'HLA-DR', 'CD56', 'p53', 'CD11b', 'H3K4', 'R1Qc3', 'pMYC', 'CD8', 'CTNNB', 'CD68', 'CD90', 'R6Qc3', 'TUBB3', 'PDL1', 'CD11c', 'CD133'],   #555 channel
                  ["KLF4","Ki67","ER","PgR","pAKT","CD31",'pRB','CD45','CD4','BCL2','CD163','PD1','H3K27ac','pMYC','MSH6','gH2AX','pRPA']+['R0c4', 'MSH6', 'BCL2', 'pRPA', 'gH2AX', 'ColIV', 'R1Qc4', 'Ki67', 'CD4', 'IBA1', 'CD45', 'CD31', 'R6Qc4', 'CD163', 'p63', 'PD1', 'pMYCab'],   #680 chan
                  ["CK19","CD44","Ecad","BMP2","ZEB1","YAP1",'FoxP3','CD20','CD3','RAD51','53BP1','GRNZB','LamAC','Rad51','CGA']+['R0c5', 'R0c5', '53BP1', '53BP1', 'Rad51', 'Rad51', 'ZEB1', 'ZEB1', 'CD44', 'CD44', 'ColI', 'ColI', 'R1Qc5', 'R1Qc5', 'CD3', 'CD3', 'FoxP3', 'FoxP3', 'LamAC', 'LamAC', 'CD20', 'CD20', 'YAP1', 'YAP1', 'R6Qc5', 'R6Qc5', 'BMP2', 'BMP2', 'GRNZB', 'GRNZB', 'CGA', 'CGA', 'CSF1R', 'CSF1R'],
                  ]
    skips = ['nuclei_area','nuclei_eccentricity','DAPI']
    for marker in df.columns:
        switch = 0
        print(marker,switch)
        for skip in skips:
            if skip in marker:
                switch = 1
                break
        if switch == 1:
            print(skip,'in',marker,'skipping')
            continue
        for i,roun in enumerate(biomRounds):
            for mkr in roun:
                if mkr in marker:
                    qr = 'R0c'+str(i+2)
                    #qr = 'R6Qc'+str(i+2)
                    for bm in df.columns:
                        if qr in bm:
                            qrn = bm
        if not 'qrn' in locals():
            print(df.columns)
            print(qr,'not found in df.columns.')
            qrn = input(qr +' marker name (blank to skip):')
            if qrn == '':
                break
        df.loc[:,marker] = df.loc[:,marker] - df.loc[:,qrn]
        print('subtracting',marker,qrn)
    return(df,obs,dfxy)

def subChanThresh(df,obs,dfxy):
    roundThresh = [1500,1250,1000,750,0]
    '''
    biomRounds = [['CAV1', 'CK17', 'CK5', 'CK7', 'CK8', 'H3K27', 'MUC1', 'PCNA', 'R0c2', 'R6Qc2', 'Vim', 'aSMA', 'pHH3'],
                  ['AR', 'CCND1', 'CD68', 'CD8', 'CK14', 'CoxIV', 'EGFR', 'H3K4', 'HER2', 'PDPN', 'R0c3', 'R6Qc3', 'pS6RP'],
                  ['BCL2', 'CD31', 'CD4', 'CD45', 'ColIV', 'ER', 'Ki67', 'PD1', 'PgR', 'R0c4', 'R6Qc4', 'gH2AX', 'pRB'],
                  ['CD20', 'CD3', 'CD44', 'CK19', 'CSF1R', 'ColI', 'Ecad', 'FoxP3', 'GRNZB', 'LamAC', 'R0c5', 'R6Qc5', 'RAD51']]

    nbr = [['H3K27', 'PCNA', 'R6Qc2', 'LamB1', 'R0c2', 'pHH3', 'FN', 'R1Qc2', 'GFAP', 'Myelin', 'S100A', 'CAV1', 'Glut1', 'Vim', 'NeuN', 'aSMA', 'panCK'],
        ['H3K4', 'TUBB3', 'R6Qc3', 'CD68', 'R0c3', 'pMYC', 'CD11b', 'R1Qc3', 'CTNNB', 'PDL1', 'CD56', 'CD11c', 'CD133', 'HLA-DR', 'CD90', 'CD8', 'p53'],
        ['ColIV', 'CD163', 'R6Qc4', 'CD45', 'R0c4', 'Ki67', 'gH2AX', 'R1Qc4', 'IBA1', 'p63', 'BCL2', 'PD1', 'pMYCab', 'MSH6', 'CD31', 'CD4', 'pRPA'],
        ['ColI', 'BMP2', 'R6Qc5', 'CD20', 'R0c5', 'CD3', 'CD44', 'R1Qc5', 'LamAC', 'GRNZB', 'Rad51', 'CGA', 'CSF1R', '53BP1', 'YAP1', 'FoxP3', 'ZEB1']]
    '''
    biomRounds = [["CK5","TFF1","PCNA","MUC1","pHH3","CK7",'Vim','aSMA','CK17','CK8','CAV1','GATA6','p63','FN','LamB1']+['R0c2', 'Vim', 'S100A', 'panCK', 'FN', 'H3K27', 'R1Qc2', 'pHH3', 'aSMA', 'GFAP', 'LamB1', 'NeuN', 'R6Qc2', 'PCNA', 'Myelin', 'CAV1', 'Glut1'],
                  ["HER2","CCND1","EGFR","PDL1","pERK","pS6RP",'CTNNB1','CD68','AR','CD8','CK14','PDPN','CD90','CTNNB','S100A4','p53','CD11c','CD11b','CD133','CD56']+['R0c3', 'HLA-DR', 'CD56', 'p53', 'CD11b', 'H3K4', 'R1Qc3', 'pMYC', 'CD8', 'CTNNB', 'CD68', 'CD90', 'R6Qc3', 'TUBB3', 'PDL1', 'CD11c', 'CD133'],
                  ["KLF4","Ki67","ER","PgR","pAKT","CD31",'pRB','CD45','CD4','BCL2','CD163','PD1','H3K27ac','pMYC','DAPI','MSH6','gH2AX','pRPA']+['R0c4', 'MSH6', 'BCL2', 'pRPA', 'gH2AX', 'ColIV', 'R1Qc4', 'Ki67', 'CD4', 'IBA1', 'CD45', 'CD31', 'R6Qc4', 'CD163', 'p63', 'PD1', 'pMYCab'],
                  ["CK19","CD44","Ecad","BMP2","ZEB1","YAP1",'FoxP3','CD20','CD3','RAD51','53BP1','GRNZB','LamAC','Rad51','CGA']+['R0c5', 'R0c5', '53BP1', '53BP1', 'Rad51', 'Rad51', 'ZEB1', 'ZEB1', 'CD44', 'CD44', 'ColI', 'ColI', 'R1Qc5', 'R1Qc5', 'CD3', 'CD3', 'FoxP3', 'FoxP3', 'LamAC', 'LamAC', 'CD20', 'CD20', 'YAP1', 'YAP1', 'R6Qc5', 'R6Qc5', 'BMP2', 'BMP2', 'GRNZB', 'GRNZB', 'CGA', 'CGA', 'CSF1R', 'CSF1R'],
                  ['nuclei_area','nuclei_eccentricity']] #RAD51 has high AF and was thresholded at 1000 for ST paper

    for biom in df.columns:
        thresh = 0
        bn = biom.split('_')[0]
        for i,roun in enumerate(biomRounds):
            if bn in roun:
                thresh = roundThresh[i]
        if thresh == 0:
            print(biom)
            try:
                thresh = float(input('threshold:'))
            except:
                thresh = 0
        df.loc[:,biom] = df.loc[:,biom] - thresh
        key = df.loc[:,biom] < 0
        df.loc[key,biom] = 0
    return(df,obs,dfxy)




def equalizeTMA1(df,obs,dfxy):
    try:
        obs.loc[:,"tissue type"]
        tName = "tissue type"
    except:
        tName = "tissue_type"
    utypes = list(obs[tName].unique())
    uSlides = list(obs["slide"].unique())

def equalizeTMA(df,obs,dfxy):
    #0 axis of array will be tissue type, 1 axis will be slide
    try:
        obs.loc[:,"tissue type"]
        tName = "tissue type"
    except:
        tName = "tissue_type"
    key3 = obs["slide_type"] == "JE"
    typesL = list(np.unique(obs.loc[key3,tName]))
    slidesL = list(np.unique(obs.loc[key3,"slide"]))
    print(typesL,slidesL)
    countsA = np.zeros((len(typesL),len(slidesL)))
    for i,ty in enumerate(typesL):
        key1 = obs[tName] == ty
        for j,sl in enumerate(slidesL):
            key2 = obs["slide"] == sl
            countsA[i,j] = obs.loc[key1&key2,:].shape[0]

    shape = countsA.shape
    newdfs = []
    newxys = []
    newobs = []
    print(countsA)
    for i,ty in enumerate(typesL):
        minct = min(countsA[i,:])
        if minct > 0:
            print(ty)
            maxct = max(countsA[i,:])
            key2 = obs[tName] == ty
            for j,sl in enumerate(slidesL):
                key1 = obs["slide"] == sl
                sdf = df.loc[key1&key2,:]
                sxy = dfxy.loc[key1&key2,:]
                sobs = obs.loc[key1&key2,:]
                #print(sdf.index)
                if len(sdf.index)>0:
                    inds = random.choices(sdf.index,k=int(maxct))
                    newdfs.append(sdf.loc[inds,:])
                    newxys.append(sxy.loc[inds,:])
                    newobs.append(sobs.loc[inds,:])
    newdfs.append(df.loc[~key3,:])
    newxys.append(dfxy.loc[~key3,:])
    newobs.append(obs.loc[~key3,:])
    df = pd.concat(newdfs)
    df.index=np.arange(df.shape[0]).astype(str)
    dfxy = pd.concat(newxys)
    dfxy.index=np.arange(df.shape[0]).astype(str)
    obs = pd.concat(newobs)
    obs.index=np.arange(df.shape[0]).astype(str)
    return(df,obs,dfxy)


def housekeeping(df,obs,dfxy):
    keySfolder = r'\\accsmb.ohsu.edu\CEDAR\ChinData\Cyclic_Workflow\cmIF_2023-05-11_KLF4_MCF7+Ms\RegisteredImages\MCF7k-6_scene001'#r'Y:\Cyclic_Workflow\cmIF_2021-11-21_HTAN-A\RegisteredImages\HTA9-14Bx1-7_scene001'
    checkChange(keySfolder,"folder for seeing which bioms are in the round")
    for i,b in enumerate(sorted(list(df.columns))):
        print(i,":",b)
    while True:
        ch = input("number of biomarker to use as housekeeping for its round: ")
        try:
            hb = sorted(list(df.columns))[int(ch)]
            break
        except:
            print("must pick integer in above list, not: ",ch)
    hb = hb.split("_")[0]

    for file in os.listdir(keySfolder):
        if hb+"." in file or hb+"_" in file:
            keyFile = file
            break
    rbl = keyFile.split("_")[1].split(".")
    print(rbl,'bioms to normalize')
    frbl = []
    for biom in df.columns:
        switch = 0
        if "_" not in biom:
            biom = biom+"_"
            switch = 1
        for b in rbl:
            if b+"_" in biom:
                if switch == 1:
                    biom = biom[:-1]
                frbl.append(biom)
    print(frbl)
    ch,uch=obMenu(obs,"obs category to normalize (adjust all means in that biomarkers round to make housekeeping means equal)")
    obcol = obs.columns[ch]
    cdf,cobs,cxy = clag(df,obs,dfxy,ch=ch,uch=uch,z=False)

    if "_" not in hb and "_" in df.columns[0]:
        hb = hb + "_"
    mhk = np.mean(df.loc[:,hb])
    print(mhk)
    for cat in uch:
        print(cat,hb)
        ratio = cdf.loc[cat,hb]/mhk #housekeeping level / mean hk level
        print(ratio,"rat")
        key = obs.loc[:,obcol] == cat
        for biom in frbl:
            #print(df.loc[key,biom].sum())
            ns = df.loc[key,biom].copy() / ratio
            print(ns.sum(),"after",ratio)
            df.loc[key,biom] = ns


    return(df,obs,dfxy)








def combat(df,obs,dfxy):
    ch,uch = obMenu(obs,'which category to use for combat?')
    bayesdata = combat1.combat(df.transpose(),obs.iloc[:,ch])
    df = bayesdata.transpose()
    return(df,obs,dfxy)


def TMAcombat(df,obs,dfxy):
    if input("0: JE-TMA combat  or  1: -KC cell line combat: ") == "0":
        smallDF = df.loc[obs["slide_type"]=="JE",:]
        if smallDF.shape[0] < 10:
            print("ERROR! missing 'JE' 'slide_type' annotation. Must add for JE combat")
        smallOBS = obs.loc[obs["slide_type"]=="JE",:]
    else:
        print("attempting COMBAT with -KC, JEjenum and Tonsil internal cell lines")
        smallDF = df.loc[obs["internal"] == 'True',:]
        if smallDF.shape[0] < 10:
            print("ERROR! missing 'internal' True/False annotation. Must add for KC combat")
        smallOBS = obs.loc[obs["internal"] == 'True',:]

    gamma_star, delta_star, stand_mean, var_pooled = combat1.combat_fit(smallDF.transpose(), smallOBS["batch"])
    bayesdata=combat1.combat_transform(df.transpose(), obs["batch"], gamma_star, delta_star, stand_mean, var_pooled)
    df = bayesdata.transpose()
    return(df,obs,dfxy)

def equalizeBiomLevel(df,obs,dfxy):
    print("only makes sense for vertically z-scored data!")
    for i,b in enumerate(pd.Series(df.columns).sort_values()):
        print(i,":",b)
    housekeep = list(pd.Series(df.columns).sort_values())[int(input('which biom number to use as key:'))]
    print(housekeep,list(pd.Series(df.columns).sort_values()))
    ch,uch = obMenu(obs,title="equalize each (slide/batch/etc.)")
    if input("only consider specific cell type when calculating means? (y)") == 'y':
        ndf,nobs,nxy = pick(df,obs,dfxy)
    else:
        ndf,nobs,nxy = df.copy(),obs.copy(),dfxy.copy()
    means = []
    for bat in uch:
        key = nobs[nobs.columns[ch]] == bat
        ss = ndf.loc[key,housekeep]
        means.append(ss.mean())
    print(means)
    for i,bat in enumerate(uch):
        key = obs[obs.columns[ch]] == bat
        df.loc[key,:] -= means[i]
    return(df,obs,dfxy)





def remNegatives(df,obs,dfxy):
    for col in df.columns:
        if min(df[col]) < 0:
            df[col] -= min(df[col])
    return(df,obs,dfxy)


def phenoToObs(df,obs,dfxy):
    ch,uch = obMenu(obs,"column with phenotypes (list of + biomarkers)")
    sp = input('separator between cols in phenotype? hit enter to use "_":')
    if sp == '':
        sp = '_'
    al = []
    for uc in uch:
        els = uc.split(sp)
        for el in els:
            if el == '' or el == 'nan' or el == 'none' or el == ' ':
                continue
            while el[0] == ' ' and len(el) > 0:
                el = el[1:]
            if el not in al:
                al.append(el)
    ser = obs.iloc[:,ch]
    print(ser)
    input('good?')
    for el in al:
        cn = obs.columns[ch]+': '+el
        obs[cn] = '-'
        key = ser.str.contains(el+sp,regex=False) #evaluated '+' as a regex operator! yay gpt
        print(el+sp,key.sum())
        obs.loc[key,cn] = el
    input()
    return(df,obs,dfxy)



def phenoToObsOld(df,obs,dfxy):
    badS = ["R0","DAPI","R5","R6","R7","nuclei_"]
    ch,uch = obMenu(obs,"column with phenotypes (list of + biomarkers)")
    sp = input('separator between cols in phenotype? hit enter to use "_":')
    if sp == '':
        sp = '_'
    pcol = obs.columns[ch]
    cols = multiObMenu(df,title='columns to evaluate for positivity')
    for col in cols:
        print(col,'processing')
        switch = 0
        for bs in badS:
            if bs in col:
                switch = 1
        if switch == 1:
            continue
        cn = col + '+'
        #bn = col.split('_')[0]+sp
        bn = col
        obs[cn] = 'no'
        for uc in tqdm(uch,col):
            if bn in uc:
                key = obs.loc[:,pcol] == uc
                obs.loc[key,cn] = "yes"
    return(df,obs,dfxy)


def phenoToObs1(df,obs,dfxy):
    print("WARNING: ONLY KLF4 RN")
    badS = ["R0","DAPI","R5","R6","R7","nuclei_"]
    ch,uch = obMenu(obs,"column with phenotypes (list of + biomarkers)")
    sp = input('separator between cols in phenotype? hit enter to use "_":')
    if sp == '':
        sp = '_'
    pcol = obs.columns[ch]
    cols = []
    for col in tqdm(df.columns):
        print(col,'processing')
        if 'KLF' not in col:
            continue
        switch = 0
        for bs in badS:
            if bs in col:
                switch = 1
        if switch == 1:
            continue
        cn = col + '+'
        #bn = col.split('_')[0]+sp
        bn = col
        obs[cn] = 'no'
        cols.append(cn)
        for uc in uch:
            if bn in uc:
                key = obs.loc[:,pcol] == uc
                obs.loc[key,cn] = "yes"
    return(df,obs,dfxy)

def phenopngs(df,obs,dfxy,cdf,savefold = r'T:\Youm\IY script\IY_mesmer_test'):
              #r'\\accsmb.ohsu.edu\CEDAR\ChinData\Cyclic_Analysis\KLF4_Project\figures IY\pTMA1\4076 test case orthotype and 2sd im\pngs'):
    if 'cellid' not in obs.columns:
        obs['cellid'] = 'cell'+pd.Series(np.arange(obs.shape[0])).astype(str)
    savefold = checkChange(savefold,"save folder")
    if input("add each biomarker's positive annotation as obs column? (y)") == 'y':
        df,obs,dfxy = phenoToObs(df,obs,dfxy)
    showR = False
    if input("display over registeredImages in IDE? (y)") == 'y':
        showR = True
        rFolder = r'\\accsmb.ohsu.edu\CEDAR\ChinData\Cyclic_Workflow\cmIF_2023-09-13_MCF7-KLF4\RegisteredImages\MCF7k-7_scene001'
        rFolder = checkChange(rFolder,"registered images folder")
    shA = False
    if input("show all biomarkers? (y)") == 'y':
        shA = True
    cols = []
    for col in df.columns:
        if not shA:
            print(col)
            if input("include? (y)") != "y":
               continue
        cn = col.split('_')[0] + '+'
        if cn in obs.columns:
            cols.append(cn)
    print(obs.columns)
    while True:
        #try:
            usc = sorted(list(obs.loc[:,"slide_scene"].unique()))
            for SlSc in usc:
                imAL,names = zapari(obs,cols,SlSc,nontr=True, skipNo = True)
                for i,n in enumerate(names):
                    print(n)
                    if showR:
                        showReg(cols[i].split("+")[0],rFolder,imAL[i])
                    im = PIL.Image.fromarray(imAL[i])
                    im.save(sphelper(savefold,n+' '+SlSc),"PNG")
            return(df,obs,dfxy)
        #except Exception as e:
            #print(e,"\nError!")
            if input("hit enter to try again or 'exit' to exit function") == 'exit':
                return(df,obs,dfxy)
        #except KeyboardInterrupt:
            return(df,obs,dfxy)

def showReg(biom,fold,annotIm):
    print(biom,fold,annotIm)
    im0 = PIL.Image.fromarray(annotIm)
    im = im0.convert("RGBA")
    datas = im.getdata()
    newData = []
    for item in tqdm(datas,"k=mekn transpern"):
        if item[0] == 0 and item[1] == 0 and item[2] == 0:  # finding black colour by its RGB value
            # storing a transparent value when we find a black colour
            newData.append((255, 255, 255, 0))
        else:
            newData.append((255, 255, 255))
    im.putdata(newData)
    for file in os.listdir(fold):
        try:
            fileL = file.split("_")[1].split(".")
            if biom in fileL:
                chan = int(file.split("_c")[-1][0])
                if chan > 1 and fileL[chan-2] == biom:
                    bim = io.imread(fold+"/"+file)
        except:
            print(file,"no good bioms")
    plt.imshow(bim,cmap='viridis',interpolation='none')
    plt.imshow(im)
    plt.title(biom)
    plt.show()

import os
import numpy as np
import pandas as pd
from skimage import io
import skimage.segmentation
import tifffile
import PIL.Image
from tqdm import tqdm

SEG_SUFFIX = "Ecad_nuc30_cell30_matched_exp5_CellSegmentationBasins.tif"


def _find_seg_file(segpath, slideScene, suffix=SEG_SUFFIX):
    """
    segpath can be a file OR a folder.
    If folder: find file that starts with slideScene + "_" and ends with suffix.
    This avoids matching slideScene like '...sceneB2' inside '...sceneB21...'.
    """
    # If user gave an explicit file path, just use it
    if segpath and os.path.isfile(segpath):
        return segpath

    # If segpath is a directory, search within
    if segpath and os.path.isdir(segpath):
        prefix = slideScene + "_"
        hits = []
        for fn in os.listdir(segpath):
            if fn.startswith(prefix) and fn.endswith(suffix):
                hits.append(fn)
        if len(hits) == 1:
            return os.path.join(segpath, hits[0])
        if len(hits) > 1:
            # prefer the shortest (usually the exact match)
            hits.sort(key=len)
            return os.path.join(segpath, hits[0])

    return None


def savepngs(df, obs, dfxy, cdf, savefold=r'T:\Youm\IY script\IY_mesmer_test'):
    if 'cellid' not in obs.columns:
        obs['cellid'] = 'cell' + pd.Series(np.arange(obs.shape[0]), index=obs.index).astype(str)
        print(obs['cellid'])

    if input("phenotype pngs? (y)") == "y":
        return (phenopngs(df, obs, dfxy, cdf))

    print("!! NOT SAVING OME.TIFF")
    nontr = 'dummy'
    if type(nontr) != bool:
        if input("make non-transparent for zen/tiff? (y)") == 'y':
            nontr = True
        else:
            nontr = False

    cols = []
    for i, col in enumerate(obs.columns):
        print(i,":",col)

    while True:
        try:
            imp = input("col save all unique as pngs (send blank to skip) send 'range(0,3)' for [0,1,2])   ")
            imp = eval(imp)
            try:
                for im in imp:
                    cols.append(obs.columns[im])
            except:
                cols.append(obs.columns[imp])
        except Exception as e:
            i2 = input(str(e) + "\ndone? (y/''):")
            if i2 == "y" or i2 == '':
                break

    savefold = checkChange(savefold, "save folder")

    imAL, names = zapari(obs, cols, str(obs.loc[:, "slide_scene"].values[0]), nontr=nontr)

    # Fast PNG writing
    for i, n in enumerate(names):
        print(n)
        outpath = sphelper(savefold, n)

        chan = imAL[i]
        # normalize to a boolean mask once
        m = (chan != 0)

        if not nontr:
            # White RGB + alpha mask (transparent where background)
            rgb = np.full((chan.shape[0], chan.shape[1], 3), 255, dtype=np.uint8)
            a = (m.astype(np.uint8) * 255)
            rgba = np.dstack([rgb, a])
            im = PIL.Image.fromarray(rgba, mode="RGBA")
            im.save(outpath, "PNG")
        else:
            # Non-transparent: save as 8-bit grayscale (0/255)
            g = (m.astype(np.uint8) * 255)
            im = PIL.Image.fromarray(g, mode="L")
            im.save(outpath, "PNG")

    return (df, obs, dfxy)


def sphelper(foln, filn):
    if not os.path.isdir(foln):
        foln = os.getcwd()

    badchars = [":", "/", "?", ">", "<"]
    for bc in badchars:
        if bc in filn:
            filn = filn.replace(bc, ".")

    if not os.path.isdir(foln):
        print(foln)
        if input("folder does not exist. Try to create folder?") == 'y':
            os.mkdir(foln)
        else:
            savefold = input("path to folder:")
            return (sphelper(savefold, filn))

    return (foln + "/" + filn + '.png')


def zapari(obs, columns, slideScene, cellidn="cellid", nontr=False, skipNo=True, dotted=True):
    OUTLINE = False
    VAL = 1
    if nontr:
        VAL = 1000

    # You can set this to a DIRECTORY now, not just a file:
    segpath = r'T:\Youm\IY script\IY_mesmer_test'
    print(slideScene, ": slide_scene")
    slideScene = checkChange(slideScene, "slide scene")
    cellidn = checkChange(cellidn, "cell ID category")
    print('needs exp5 if matched!! Otherwise, only cyto+ cells included')
    segpath = checkChange(
        segpath,
        "segmentation image folder OR file path (must contain *_Ecad_nuc30_cell30_matched_exp5_CellSegmentationBasins.tif)",
        paths=True
    )

    # Filter obs to this slide_scene (don’t convert the whole dataframe to str)
    keyS = (obs.loc[:, "slide_scene"].astype(str) == str(slideScene))
    obs = obs.loc[keyS, :]

    # Find the correct seg file
    segfile = _find_seg_file(segpath, slideScene, suffix=SEG_SUFFIX)
    if segfile is None:
        print("Could not auto-find seg file in folder for:", slideScene)
        # fall back to asking for an explicit file (keeps your old flow)
        segfile = checkChange(
            segpath,
            "segmentation image FILE path (including file and extension)",
            paths=True
        )
    print("Using seg file:", segfile)

    label = io.imread(segfile)

    if OUTLINE != True:
        if input("show outline only? (y)") == "y":
            OUTLINE = True

    bounds = None
    if OUTLINE:
        bounds = skimage.segmentation.find_boundaries(label, connectivity=1, background=0, mode="thick")

    imAL = []
    names = []

    # Precompute cell integer ids once for this obs slice (fast)
    # expects strings like "cell123"
    cell_str = obs.loc[:, cellidn].astype(str).values
    # vectorized parse: split on "cell" and take last part
    cell_int = np.array([int(s.split("cell")[-1]) for s in cell_str], dtype=np.int32)
    obs = obs.copy()
    obs["_cell_int"] = cell_int

    for column in columns:
        uEnts = sorted(list(pd.unique(obs.loc[:, column].astype(str))))
        for e in uEnts:
            if e == "no" and skipNo:
                continue

            print("cluster:", e, ":  ", column)
            key = (obs.loc[:, column].astype(str) == str(e))
            ids = obs.loc[key, "_cell_int"].values.astype(label.dtype, copy=False)

            print(len(ids))

            # Vectorized mask of pixels belonging to ANY id in this cluster
            m = np.isin(label, ids)

            if OUTLINE:
                chan = np.zeros_like(label)
                chan[m & bounds] = VAL
            else:
                chan = np.zeros_like(label)
                chan[m] = VAL

            imAL.append(chan)
            names.append(column + " " + str(e))

    # clean up temp column (optional; doesn’t change return)
    # (keeping it is harmless since obs is local)
    return (imAL, names)


def checkChange(s,cat,paths=False):
    def _show_example_paths():
        cpaths = [r'\\graylab\Chin_Lab\Cyclic_Workflow\cmIF_2021-11-21_HTAN-A\Segmentation\HTA9-14Bx1-7_CellposeSegmentation\HTA9-14Bx1-7_scene001_nuc30_NucleiSegmentationBasins.tif',
                  r'\\graylab\Chin_Lab\Cyclic_Workflow\cmIF_2021-11-21_HTAN-A\Segmentation\HTA9-14Bx2-5_CellposeSegmentation\HTA9-14Bx2-5_scene001_nuc30_NucleiSegmentationBasins.tif',
                  r'\\graylab\Chin_Lab\Cyclic_Workflow\cmIF_2021-11-21_HTAN-A\Segmentation\HTA9-14Bx2-5_CellposeSegmentation\HTA9-14Bx2-5_scene001_3000-1300-1500-15001.tif',
                  r'\\graylab\Chin_Lab\Cyclic_Workflow\cmIF_2021-11-21_HTAN-A\Segmentation\HTA9-15Bx1-5_CellposeSegmentation\HTA9-15Bx1-5_scene001_3895-12065-1500-15001.tif',
                  r'\\graylab\Chin_Lab\Cyclic_Workflow\cmIF_2021-11-21_HTAN-A\Segmentation\HTA9-15Bx1-5_CellposeSegmentation\HTA9-15Bx1-5_scene002_8450-1300-1500-15001.tif',
                  r'\\graylab\Chin_Lab\Cyclic_Workflow\cmIF_2021-11-21_HTAN-A\Segmentation\HTA9-15Bx2-5_CellposeSegmentation\HTA9-15Bx2-5_scene002_2270-5050-1500-15001.tif',
                  r'\\accsmb.ohsu.edu\CEDAR\ChinData\Cyclic_Workflow\cmIF_2023-04-07_pTMA1\Segmentation\pTMA1-25_CellposeSegmentation\pTMA1-25_sceneB11_nuc30_NucleiSegmentationBasins.tif',
                  r'\\accsmb.ohsu.edu\CEDAR\ChinData\Cyclic_Workflow\cmIF_2023-09-13_MCF7-KLF4\Segmentation\MCF7k-7_CellposeSegmentation\MCF7k-7_scene001_nuc30_NucleiSegmentationBasins.tif',
                  r'\\accsmb.ohsu.edu\CEDAR\ChinData\Cyclic_Workflow\cmIF_2023-09-13_MCF7-KLF4\Segmentation\MCF7k-7_CellposeSegmentation\MCF7k-7_scene001_nuc30_NucleiSegmentationBasins.tif',
                  ]
        for path in cpaths:
            print(path)
    return(shared_check_change(
        s,
        cat,
        input_fn=input,
        before_change=_show_example_paths if paths else None,
        change_prompt=": " if paths else None,
    ))



def save(df,obs,dfxy):
    tag = input("filename?")
    df.to_csv(tag+"_df.csv")
    dfxy.to_csv(tag+"_dfxy.csv")
    obs.to_csv(tag+"_obs.csv")
    return(df,obs,dfxy)

def pick(df,obs,dfxy):
    ch0 = input("0:include,  1:exclude  ")
    for i,ob in enumerate(obs.columns):
        print(i,":",ob)

    print("filter slides by which?")
    ch1 = int(input("enter number:"))
    categories = np.unique(np.array(sorted(obs[obs.columns[ch1]].astype(str))))
    for i,ob in enumerate(categories):
        print(i,":",ob)
    print("Enter one number at a time for entries to include/exclude")
    included = []
    if ch0 == "1":
        name = "exclude"
    else:
        name = "include"
    while True:
        try:
            included.append(categories[int(input(name+" #:"))])
        except:
            break
    print(included)
    key = np.empty((obs.shape[0],1),dtype=object)
    for i in included:
        key=np.append(key,np.array([obs[obs.columns[ch1]]==i]).T,axis=1)
    masterkey = []
    for i in range(key.shape[0]):
        row = key[i,:]
        if any(row):
            masterkey.append(True)
        else:
            masterkey.append(False)
    key=pd.Series(masterkey)
    if ch0 == "1":
        key = -key
    key = list(key)
    df,dfxy,obs = df.loc[key,:],dfxy.loc[key,:],obs.loc[key,:]
    print("df shape",df.shape,obs.shape)

    return(df,obs,dfxy)


def roi(df,obs,dfxy):
    ch,uch = obMenu(obs,'category to color cells by/edit')
    CT_ANNOT = obs.columns[ch]
    defa = False
    if input('use default labels in new category? (y): ') == 'y':
        defa = True
        OutName = input('label to apply to cells in scene but not in ROI (default ""): ')
    else:
        OutCol = CT_ANNOT
        for i,an in enumerate(uch):
            print(i,': ',an)
        try:
            cttc = [uch[int(input('int of celltype to edit (nonint edits all): '))]]
        except:
            cttc = uch
        OutName = input('label to apply to cells in ROI: ')


    while True:
        for i,ss in enumerate(sorted(list(obs.loc[:,'slide_scene'].unique()))):
            print(i,":",ss)
        try:
            ch = int(input('number of slide scene to draw ROI on: '))
            SS = sorted(list(obs.loc[:,'slide_scene'].unique()))[ch]
        except:
            break
        key = obs.loc[:,'slide_scene'] == SS
        sobs = obs.loc[key,:]
        sxy = dfxy.loc[key,:]
        if defa:
            sobs,sxy = roiH(sobs,sxy,SS,CT_ANNOT,OutCol)
        else:
            sobs,sxy = roiH(sobs,sxy,SS,CT_ANNOT,OutCol,OutName,cttc)
        for col in sobs.columns:
            if col not in obs:
                obs[col] = ''
        obs.loc[key,:] = sobs
    return(df,obs,dfxy)

def roiH(obs,dfxy,ss,CT_ANNOT,OutCol,OutName,cttc):
    #input("NO PYGAME PKG...")
    pygame_mod = _require_pygame()
    SCALE = 1
    dfxy['x'] = dfxy.iloc[:,0]
    dfxy['y'] = dfxy.iloc[:,1]
    pygame_mod.init()
    print(pygame_mod.display.get_desktop_sizes())
    DOT_RADIUS = 1
    PADDING = 20  # padding around the display

    while (dfxy['x'].max() / SCALE) > pygame_mod.display.get_desktop_sizes()[0][0]:
        SCALE += 1
    while (dfxy['y'].max() / SCALE) > pygame_mod.display.get_desktop_sizes()[0][1]- 8*PADDING:
        SCALE += 1

    WINDOW_WIDTH = int(dfxy['x'].max() / SCALE + PADDING * 2)
    WINDOW_HEIGHT = int(dfxy['y'].max() / SCALE + PADDING * 2)


    screen = pygame_mod.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame_mod.display.set_caption("ROI Selector")

    # Map annotation (e.g. celltype) to colors
    unique_types = obs[CT_ANNOT].unique()
    print(unique_types)

    colors = {}
    for i, t in enumerate(sorted(list(unique_types))):
        # evenly spaced hues between 0.0–1.0
        #h = i / len(unique_types)
        # full saturation & value
        #r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
        # scale 0–1 floats to 0–255 ints
        #colors[t] = pygame.Color(int(r*255), int(g*255), int(b*255))
        colors[t] = allc.colors[i]
    print(colors)
    # Precompute display positions (scaled + padded)
    positions = list(zip(
        (dfxy['x'] / SCALE + PADDING).astype(float),
        (dfxy['y'] / SCALE + PADDING).astype(float)
    ))
    annotations = obs[CT_ANNOT].tolist()

    # ROI state
    points = []
    font = pygame_mod.font.SysFont(None, 24)
    running = True
    while running:
        screen.fill(pygame_mod.Color('black'))

        #draw_cells() v
        for (x, y), ctype in zip(positions, annotations):
            pygame_mod.draw.circle(screen, colors.get(ctype, pygame_mod.Color('white')), (int(x), int(y)), DOT_RADIUS)
        #draw_polygon(points) v
        if len(points) > 1:
            pygame_mod.draw.lines(screen, pygame_mod.Color('white'), False, points, 2)
        for p in points:
            pygame_mod.draw.circle(screen, pygame_mod.Color('white'), p, 4)

        for event in pygame_mod.event.get():
            if event.type == pygame_mod.QUIT:
                running = False
            elif event.type == pygame_mod.MOUSEBUTTONDOWN and event.button == 1:
                pos = event.pos
                # close polygon if clicking near the first point and at least 3 points
                if len(points) >= 3 and Point(pos).distance(Point(points[0])) < 10:
                    poly = Polygon(points) #this is the shapely polygon of the ROI- I guess it connects the last point to the first point
                    # assign in/out labels to obs
                    roiN = 1
                    while ss+'_ROI_'+str(roiN) in obs.columns:
                        roiN += 1
                    label_col = ss+'_ROI_'+str(roiN)
                    inside = dfxy.apply(
                        lambda row: poly.contains(
                            Point(row['x'] / SCALE + PADDING, row['y'] / SCALE + PADDING)
                        ), axis=1
                    )
                    print(inside)
                    print(inside.shape)
                    print(obs.shape,dfxy.shape)
                    print(obs.index.equals(inside.index))
                    #input()
                    for uc in cttc:
                        key = obs.loc[:,OutCol] == uc
                        obs.loc[key,OutCol] = obs.loc[key,OutCol].where(~inside.loc[key], OutName)
                    running = False

                else:
                    points.append(pos)

        pygame_mod.display.flip()

    pygame_mod.quit()
    #print(allc.colors)
    #print("ROI labels before spatialLite:\n", obs[OutCol].value_counts())
    spatialLite(obs.copy(),ss,allc.colors,dfxy,sorted(list(obs.loc[:,OutCol].unique())),list(obs.columns).index(OutCol))
    #print("ROI labels after spatialLite:\n", obs[OutCol].value_counts())
    return(obs,dfxy)



def roiH1(obs,dfxy,ss,CT_ANNOT,OutName):
    pygame_mod = _require_pygame()
    SCALE = 1
    dfxy['x'] = dfxy.iloc[:,0]
    dfxy['y'] = dfxy.iloc[:,1]
    pygame_mod.init()
    print(pygame_mod.display.get_desktop_sizes())
    DOT_RADIUS = 1
    PADDING = 20  # padding around the display

    while (dfxy['x'].max() / SCALE) > pygame_mod.display.get_desktop_sizes()[0][0]:
        SCALE += 1
    while (dfxy['y'].max() / SCALE) > pygame_mod.display.get_desktop_sizes()[0][1]- 8*PADDING:
        SCALE += 1

    WINDOW_WIDTH = int(dfxy['x'].max() / SCALE + PADDING * 2)
    WINDOW_HEIGHT = int(dfxy['y'].max() / SCALE + PADDING * 2)


    screen = pygame_mod.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame_mod.display.set_caption("ROI Selector")

    # Map annotation (e.g. celltype) to colors
    unique_types = obs[CT_ANNOT].unique()
    print(unique_types)

    colors = {}
    for i, t in enumerate(unique_types):
        # evenly spaced hues between 0.0–1.0
        #h = i / len(unique_types)
        # full saturation & value
        #r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
        # scale 0–1 floats to 0–255 ints
        #colors[t] = pygame.Color(int(r*255), int(g*255), int(b*255))
        colors[t] = allc.colors[i]

    # Precompute display positions (scaled + padded)
    positions = list(zip(
        (dfxy['x'] / SCALE + PADDING).astype(float),
        (dfxy['y'] / SCALE + PADDING).astype(float)
    ))
    annotations = obs[CT_ANNOT].tolist()

    # ROI state
    points = []
    font = pygame_mod.font.SysFont(None, 24)
    running = True
    while running:
        screen.fill(pygame_mod.Color('black'))

        #draw_cells() v
        for (x, y), ctype in zip(positions, annotations):
            pygame_mod.draw.circle(screen, colors.get(ctype, pygame_mod.Color('white')), (int(x), int(y)), DOT_RADIUS)
        #draw_polygon(points) v
        if len(points) > 1:
            pygame_mod.draw.lines(screen, pygame_mod.Color('white'), False, points, 2)
        for p in points:
            pygame_mod.draw.circle(screen, pygame_mod.Color('white'), p, 4)

        for event in pygame_mod.event.get():
            if event.type == pygame_mod.QUIT:
                running = False
            elif event.type == pygame_mod.MOUSEBUTTONDOWN and event.button == 1:
                pos = event.pos
                # close polygon if clicking near the first point and at least 3 points
                if len(points) >= 3 and Point(pos).distance(Point(points[0])) < 10:
                    poly = Polygon(points) #this is the shapely polygon of the ROI- I guess it connects the last point to the first point
                    # assign in/out labels to obs
                    roiN = 1
                    while ss+'_ROI_'+str(roiN) in obs.columns:
                        roiN += 1
                    label_col = ss+'_ROI_'+str(roiN)
                    inside = dfxy.apply(
                        lambda row: poly.contains(
                            Point(row['x'] / SCALE + PADDING, row['y'] / SCALE + PADDING)
                        ), axis=1
                    )
                    obs[label_col] = inside.map({True: 'in ' + label_col, False: OutName})
                    running = False

                else:
                    points.append(pos)

        pygame_mod.display.flip()

    pygame_mod.quit()

    spatialLite(obs,ss,['gray','darkred'],dfxy,sorted(list(obs.loc[:,label_col].unique())),-1)
    return(obs,dfxy)









def old_roi(df,obs,dfxy):
    usc = sorted(list(obs["slide_scene"].unique()))
    ch1,uch = obMenu(obs,"color by?")
    colors = []
    if len(uch) < 8:
        colors = allc.standcolors
    else:
        colors = allc.colors
    while len(uch) > len(colors):
        colors.append(allc.rgb())
    for scene in usc:
        skey = obs.loc[:,"slide_scene"] == scene
        sxy  = dfxy.loc[skey,:]
        maxInY = max(sxy.iloc[:,1])
        spatialLite(obs,scene,colors,dfxy,uch,ch1)
        try:
            xmin = float(eval(input("x min: ")))
            xmax = float(eval(input("x max: ")))
            ymin = float(eval(input("y min: ")))
            ymax = float(eval(input("y max: ")))
        except:
            print("invalid coordinate, skipping")
            continue
        #Y = -y + max(yseries)
        #-Y + max(yseries) = y
        # = max(sxy.iloc[:,-1]) --- need to add min value
        xkey = dfxy.iloc[:,0] >= xmin
        Xkey = dfxy.iloc[:,0] <= xmax
        ykey = dfxy.iloc[:,1] <= -ymin + maxInY
        Ykey = dfxy.iloc[:,1] >= -ymax + maxInY
        newS = obs.loc[skey & xkey & Xkey & ykey & Ykey,:].copy()
        newD = df.loc[newS.index,:].copy()
        newX = dfxy.loc[newS.index,:].copy()
        print(obs.shape)
        obs = obs.loc[~skey,:]
        print(obs.shape)
        df = df.loc[~skey,:]
        dfxy = dfxy.loc[~skey,:]

        obs = pd.concat([obs,newS],axis=0)
        print(obs.shape)
        df = pd.concat([df,newD],axis=0)
        dfxy = pd.concat([dfxy,newX],axis=0)
        spatialLite(obs,scene,colors,dfxy,uch,ch1,ymin=ymin)
    return(df,obs,dfxy)

def spatialLite(nobs,scene,colors,nxy,uch,ch1,ymin=0):
        obs = nobs
        key=nobs["slide_scene"]==scene
        sobs = nobs.loc[key,:]
        #sdf = ndf.loc[key,:]
        sxy = nxy.loc[key,:]
        #ax.set_aspect('equal')
        #ax.legend(uch,colors,bbox_to_anchor=(1.05, 1), loc='upper left')
        try:
            fig,ax = plt.subplots(figsize=((max(sxy.iloc[:,0])-min(sxy.iloc[:,0]))/500,(max(sxy.iloc[:,1])-min(sxy.iloc[:,1]))/500))
            print(max(sxy.iloc[:,1]),"max Y")
        except Exception as e:
            print(e,"error setting fig and ax",scene)
            print(sxy.isna().any(),"isna")
            fig,ax = plt.subplots()
        for i,ty in enumerate(uch):
            co = colors[i]
            #print(sobs.columns[ch1])
            key1 = sobs[sobs.columns[ch1]]==ty
            #print(key1)
            #tobs = sobs.loc[key,:]
            #tdf = sdf.loc[key,:]
            txy = sxy.loc[key1,:]
            x = []
            y = []
            if txy.shape[0] == 0:
                continue
            for j in range(txy.shape[0]):
                pt = list(txy.iloc[j,:])
                #coords.append((pt[0],pt[1]))
                x.append(pt[0])
                y.append(-pt[1])
            Y = pd.Series(y)
            #sxy = list(sxy.astype(float))
            #print(sxy)
            Y += max(sxy.iloc[:,1])+ymin
            ax.scatter(x,Y,color=co,label=ty,s=1.2)
        lg = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')#, scatterpoints=1, fontsize=10)
        try:
            for k in range(len(uch)):
                lg.legend_handles[k]._sizes = [30]
        except Exception as e:
            print(e,"index k:",k,len(uch),lg.legend_handles)
        plt.title(scene+" "+obs.columns[ch1])
        plt.show()


def listAnnots(df,obs,dfxy,Max=50):
    for col in obs.columns.unique():
        uch = sorted(list(obs.loc[:,col].unique()))
        if len(uch) <= Max:
            print('\n'+col+':\n',uch)

    return(df,obs,dfxy)


def fractionTable(df,obs,dfxy):
    toti = input('totals instead of fractions? (y)')
    inclB = input('include biomarker averages and SDs? (y)')
    ch0s = multiObMenu(obs,'number of column to compare across (x axis in separate csv)): ')
    outCols = multiObMenu(obs,'column integer for annotation fractions to add to table, send blank when done: ')
    for ch0 in ch0s:
        print(ch0)
        out = []
        odf = pd.DataFrame(columns = obs.loc[:,ch0].unique())
        for uc1 in odf.columns:
            key1 =  obs.loc[:,ch0] == uc1
            sobs = obs.loc[key1,:]
            for col in outCols:
                for uc in obs.loc[:,col].unique():
                    on = str(col) + ': ' + str(uc)
                    if on not in odf.index:
                        odf.loc[on] = 0
                    #print(sobs.columns)
                    key = sobs.loc[:,col].astype(str) == str(uc)  #str casts are probably redundant in... every case? most cases... in pipeline, broke without str cast
                    #out.append([on,key.sum()/obs.shape[0]])
                    try:
                        if toti == 'y':
                            odf.loc[on,uc1] = key.sum()
                        else:
                            odf.loc[on,uc1] = key.sum()/sobs.shape[0]
                    except:
                        odf.loc[on,uc1] = np.nan
            if inclB == 'y':
                for biom in df.columns:
                    on = biom + ': '
                    osd = 'SD of'+ biom
                    if on not in odf.index:
                        odf.loc[on] = 0
                        odf.loc[osd] = 0
                    odf.loc[on,uc1] = np.mean(df.loc[key1,biom])
                    odf.loc[osd,uc1] = np.std(df.loc[key1,biom])
        ch2 = input(ch0+'filename?')
        if ch2 != '':
            odf.to_csv(ch2+'.csv')
    return(df,obs,dfxy)




def obCluster(df,obs,dfxy):
    obs = obs.astype(str)
    for i,ob in enumerate(obs.columns):
        print(i,":",ob)
    ch = int(input("cluster by?"))
    chob = obs.columns[ch]
    clusters = obs[chob].copy()
    uobs = np.unique(obs[chob].values)
    print(chob,'chob')
    for i,uo in enumerate(uobs):
        print(i,":",uo)
        clusters.loc[obs.loc[:,chob]==uo] = uo
    #obs["Cluster"] = list(clusters)
    obs.name = chob
    return(df,obs,dfxy)

def kmeans(df,obs,dfxy):
    ncl = int(input("n clusters:"))
    km = KMeans(n_clusters=ncl)
    km.fit(df)
    obs["Kmeans_"+str(ncl)] = km.labels_
    return(df,obs.astype(str),dfxy)


def leiden(df,obs,dfxy):
    if input("look for n clusters instead? (y) ") == 'y':
        return(autoleiden(df,obs,dfxy,input("n clusters:")))
    sc, anndata = _load_scanpy_stack("Leiden clustering")
    if sc is None:
        return(df,obs,dfxy)
    print(all(obs.index==df.index),"all index the same")
    res = float(input("recluster with resolution:"))

    adata = anndata.AnnData(df,obs = obs)
    sc.pp.neighbors(adata,use_rep='X')
    sc.tl.leiden(adata, key_added='Cluster', resolution=res)
    cn = "Leiden_"+str(res)
    obs[cn] = adata.obs["Cluster"]
    obs[cn] = obs[cn].astype(str)
    print(len(list(obs.loc[:,cn].unique())))
    return(df,obs,dfxy)

def autoleiden(df,obs,dfxy,target,res=1,incr = .33):
    sc, anndata = _load_scanpy_stack("Auto-Leiden clustering")
    if sc is None:
        return(df,obs,dfxy)
    ncl = 99
    tes = []
    target = int(target)
    while ncl != target:
        print("!! running with res",res)
        adata = anndata.AnnData(df,obs = obs)
        sc.pp.neighbors(adata,use_rep='X')
        sc.tl.leiden(adata, key_added='Cluster', resolution=res)
        obs.loc[:,"Leiden_n" + str(target)] = ""
        obs.loc[:,"Leiden_n" + str(target)] = adata.obs["Cluster"].astype(str)
        ncl = len(list(obs.loc[:,"Leiden_n" + str(target)].unique()))
        print("got",ncl,"clusters!")
        tes.append(res)
        if ncl > target:
            res -= incr
            if res in tes:
                incr = incr/2
                res += incr
        if ncl < target:
            res += incr
            if res in tes:
                incr = incr/2
                res -= incr
    return(df,obs,dfxy)


def gmm(df,obs,dfxy):
    nClusters = int(input("how many different categories? "))
    ctypes = ['full','tied','diag','spherical']
    gmm = GMM(n_components=nClusters).fit(df)
    obs["GMM_"+str(nClusters)] = gmm.predict(df)
    #plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis');
    return(df,obs,dfxy)


'''
VISUALIZATION
'''
def nap(df,obs,dfxy,cdf):
    return(df,obs,dfxy)

'''
def nap(df,obs,dfxy,cdf):
    if 'cellid' not in obs.columns:
        obs['cellid'] = obs.index
    cols = []
    info = []
    for i,col in enumerate(obs.columns):
        print(i,":",col)
    while True:
        try:
            cols.append(obs.columns[int(input("number of column to include in Napari (send blank when done)"))])
        except:
            break
    print("!!! output will not render until kernel stops running !!!")
    for ss in obs.loc[:,"slide_scene"].unique():
        print(ss)
        fold = input("path to folder with segmentation masks (send blank to skip scene): ")
        if fold == "":
            continue
        i1 = input(".png segmentation file name: ")
        i2 = input(".tiff segmentation file name: ")
        info.append([ss,fold,i1,i2])
    for nfo in info:
        NP.main(slideScene=nfo[0],fold=nfo[1],ip=nfo[2], it=nfo[3],columns=cols,obs=obs)
    return(df,obs,dfxy)
'''



def spatial(df,obs,dfxy,cdf,shind=True,SAVE=0):
    dfxy = dfxy.loc[:,dfxy.columns.sort_values()]
    mpl.style.use('default')
    #ch = input("show all? (y)")
    ch = "y"
    if ch != "y":
        ndf,nobs,nxy = pick(df,obs,dfxy)
    else:
        ndf,nobs,nxy = df.copy(),obs.copy(),dfxy.copy()
    obs = obs.astype(str)
    #if  input("use napari? (y)") == "y":
        #napari(ndf,nobs,nxy,cdf)
        #return(df,obs,dfxy)

    #if input("save all? (y)") == "y":
        #SAVE = 1
        #location = input("save location? :").replace("\ "[0],"/")+"/"
    usc = sorted(list(nobs["slide_scene"].unique()))
    ch1,uch = obMenu(obs,"color by?")
    uch = sorted(list(uch))
    try:
        fsm = float(input('figsize modifier?'))
    except:
        fsm = 1
    if input("pick colors?") == "y":
        colors = []
        for uuu in uch:
            if len(uch) == 2:
                if uuu == "yes":
                    colors.append("darkred")
                    continue
                if uuu == "no":
                    colors.append("lightgray")
                    continue
            print(uuu)
            colors.append(input("color:"))

    #elif len(uch) < 8:
        #colors = allc.standcolors
    elif len(uch) == 2:
        colors = []
        for uuu in uch:
            if uuu == "yes":
                colors.append("darkred")
                continue
            if uuu == "no":
                colors.append("lightgray")
                continue
            colors.append(allc.rgb())
    else:
        colors = allc.colors
    while len(uch) > len(colors):
        colors.append(allc.rgb())
    for scene in usc:
        key=nobs["slide_scene"]==scene #was obs not nobs, haven't tested change
        sobs = nobs.loc[key,:]
        #sdf = ndf.loc[key,:]
        sxy = nxy.loc[key,:]
        #ax.set_aspect('equal')
        #ax.legend(uch,colors,bbox_to_anchor=(1.05, 1), loc='upper left')
        try:
            #was 500. Converted to 1500 for QI data (which is in um not px)
            #print("FIGSIZE FOR RNA")
            fig,ax = plt.subplots(figsize=((max(sxy.iloc[:,0])-min(sxy.iloc[:,0]))/500*fsm,(max(sxy.iloc[:,1])-min(sxy.iloc[:,1]))/500*fsm))
        except Exception as e:
            print(e,"error setting fig and ax",scene)
            print(sxy.isna().any(),"isna")
            fig,ax = plt.subplots()
        shd = 0
        for i,ty in enumerate(uch):
            co = colors[i]
            #print(sobs.columns[ch1])
            key1 = sobs[sobs.columns[ch1]]==ty
            if shind:
                p = key1.sum()/sobs.shape[0]
                if p != 0:
                    shd -= p * np.log(p)
            #print(key1)
            #tobs = sobs.loc[key,:]
            #tdf = sdf.loc[key,:]
            txy = sxy.loc[key1,:]
            x = []
            y = []
            if txy.shape[0] == 0:
                continue
            for j in range(txy.shape[0]):
                pt = list(txy.iloc[j,:])
                #coords.append((pt[0],pt[1]))
                x.append(pt[0])
                y.append(-pt[1])
            Y = pd.Series(y)
            #sxy = list(sxy.astype(float))
            #print(sxy)
            Y += max(sxy.iloc[:,-1])
            try:
                ax.scatter(x,Y,color=co,label=ty,s=0.8) #was 1.2 for HTA14
            except ValueError:
                print('invalid color:',co)
                co = allc.colors[i]
                ax.scatter(x,Y,color=co,label=ty,s=0.8)
        lg = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')#, scatterpoints=1, fontsize=10)
        try:
            for k in range(len(uch)):
                lg.legend_handles[k]._sizes = [30]
        except Exception as e:
            pass #this just means that not all points are represented in this scene
            #print(e,"index k:",k,len(uch),lg.legendHandles)
        txt = scene+" "+obs.columns[ch1]
        if shind:
            txt += '\n Shannon Index: ' + str(shd)
        plt.title(txt)

        if SAVE == 1:
            while True:
                try:
                    plt.savefig(location +scene+" "+obs.columns[ch1]+".png")
                    break
                except Exception as e:
                    print("error with save location",e)
                    location = input("save location? :").replace("\\","/")+"/"
        plt.show()
    return(df,obs,dfxy)



def napari(df,obs,dfxy,cdf):
    ch,uch = obMenu(obs,title="Color by:")
    obCol = obs.iloc[:,ch]
    usc = obs["slide_scene"].unique()
    a = pd.Series(obs.index)
    b = a.apply(lambda n: n.split("cell")[-1])
    c = b.astype(int)
    c = pd.Series(data=c.values, index=obs.index)
    print(a,"a",b,"b",c,"c")
    obs["intID"] = ""
    obs["intID"] = c
    print(obs["intID"],"\n\n\n")
    outlabels = []
    for scene in usc:
        print(scene)
        uf = input("segmentation file folder (or 'skip'):")
        if uf == "skip":
            continue

        url = input("png image (CytoProj png): ").replace("\\","/")
        image = io.imread(uf+"/"+url)
        url = input("copy in path to segmentation file (Nuclei segmentation basins tif): ").replace("\\","/")
        segments = io.imread(uf+"/"+url)
        seg=segments.copy()#.astype(object)
        useg = np.unique(seg)



        print(seg,"seg")
        viewer = NAPARI.Viewer()
        viewer.add_image(image,blending='additive')
        labels = obs.iloc[:,ch].loc[obs["slide_scene"] == scene]
        print(labels,"labels")
        #print(obCol,"obCol")


        print(useg,"useg")
        print(c.unique(),"int")
        for se in useg:
            shap = obCol.loc[c==se].shape[0]
            if shap == 0:
                seg[segments==se] = 0
                continue
            uc = obCol.loc[c==se].values[0]
            if uc not in outlabels:
                i = len(labels)
                outlabels.append([uc,i])
            else:
                i = outlabels.index(uc)
            seg[segments==se] = i

        viewer.add_labels(seg,blending='additive')

    outlabels = pd.DataFrame()
    outlabels.to_csv("Napari labels.csv")
    return(df,obs,dfxy)



def obMenu(obs,title="choose category:"):
    for i,col in enumerate(obs.columns):
        print(i,":",col)
    ch = int(input(title)) #multiboxplot needs this to trigger an error if non int sent
    uch = sorted(list(obs[obs.columns[ch]].unique()))
    return(ch,uch)


def multiObMenu(obs,title='columns to include',required=False):
    outCols = []
    for i,ob in enumerate(obs.columns):
        print(i,":",ob)
    while True:
        try:
            ch = eval(input(title))
            if type(ch) == int or type(ch) == str:
                ch = int(ch)
                outCols.append(obs.columns[ch])
            else:
                for c in ch:
                    outCols.append(obs.columns[int(c)])
        except:
            if len(outCols) > 0 or not required:
                ch2 = input('done?')
                if ch2 == 'y' or ch2 == '':
                    break
            else:
                print('add one valid column to continue')
    return(outCols)


def heatmap(df,obs,dfxy,cdf,title=None):
    print(min(10+cdf.shape[0]/5,2**15/100))
    h = min(10+cdf.shape[0]/5,2**15/100)
    f, ax = plt.subplots(figsize=(20, h))
    bbox = ax.get_window_extent().transformed(f.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    height *= f.dpi
    print(height,"HEIGHT")
    sns.heatmap(cdf,xticklabels=cdf.columns,yticklabels=cdf.index,center=np.mean(cdf.values))
    if title != None:
        ax.title.set_text(title)
    plt.show()
    cdf,obs,dfxy = zscore(cdf,obs,dfxy,ax=0)
    f, ax = plt.subplots(figsize=(20, min(10+cdf.shape[0]/5,2**15/100)))
    sns.heatmap(cdf,xticklabels=cdf.columns,yticklabels=cdf.index,center=np.mean(cdf.values))
    if title != None:
        ax.title.set_text(title)
    plt.show()
    return(df,obs,dfxy)

def clusterbar1(df,obs,dfxy,cdf): #New nice version, just needs legends
    if input("single cell heatmap instead?") == "y":
        singHeatmap(df,obs,dfxy)
        return(df,obs,dfxy)
    sns.set(font_scale=1)
    indiit = input("z-score first? (y or '')")
    if indiit == "y" or indiit == '':
        cdf,obs,dfxy = zscorev(cdf,obs,dfxy)
    bars = []
    for i,ob in enumerate(obs.columns):
        print(i,":",ob)
    while True:
        try:
            inp = int(input("add barplot of column number:"))
            bars.append(obs.columns[inp])
        except:
            in2 = input("done?")
            if in2 == "y" or in2 == "":
                break
    #fig = plt.figure(figsize = (cdf.shape[0]/5,cdf.shape[1]/5+len(bars)))
    FSZ = (cdf.shape[1]/5+len(bars),cdf.shape[0]/3)
    fig = sns.clustermap(cdf,yticklabels=True, xticklabels=True,center=0, cmap='bwr',figsize = FSZ)
    #fig = plt.figure()
    #fig.add_subplot(fig0.
    #fig.ax_row_dendrogram.set_visible(False)
    #fig.ax_col_dendrogram.set_visible(False)
    #fig.ax_row_dendrogram.set_xlim([0,0])
    #fig.ax_col_dendrogram.set_xlim([0,0])
    clLabs = fig.ax_heatmap.yaxis.get_majorticklabels()
    print(FSZ,"figsize")
    ncols = 3 + len(bars)
    fig.gs.update(left=0, right=3/ncols)
    #ay = fig.add_subplot(3, ncols, (1, 1))
    #print(ay)
    clN = obs.name
    axs = []
    for i,bar in enumerate(bars):
        gs = matplotlib.gridspec.GridSpec(1,1, left=(5+i)/(ncols+2), right=(6+i)/(ncols+2),top=.7,bottom=0.05)
        #print(dir(fig.fig))
        #print(dir(fig.gs))
        #print(dir(gs))
        print(dir(fig.ax_heatmap))
        print(fig.ax_heatmap.get_ylim(),1)
        print(fig.ax_heatmap.get_yscale(),2)
        print(fig.ax_heatmap.get_ybound(),3)
        print(fig.ax_heatmap.get_yaxis(),4)
        print(dir(gs))
        #print(fig.ax_heatmap.yaxis.axis()
        ax = fig.fig.add_subplot(gs[0]) #,sharey = fig.ax_heatmap.axes )
        #fig.ax_heatmap.sharey(ax)
        #ax = fig.fig.add_subplot(gs[0])
        #fig.fig.align_ylabels()
        ucols = sorted(list(obs.loc[:,bar].unique()))
        a = np.zeros((len(clLabs),len(ucols)))
        bins = []
        #for j,cl in tqdm(enumerate(clLabs),bar):
        for j,cl in enumerate(clLabs):
            label = cl.get_text()
            bins.append(str(label))
            key1 = obs.loc[:,clN].astype(str) == str(label)
            for k,uc in enumerate(ucols):
                key2 = obs.loc[:,bar] == uc

                #bi = obs.loc[key1 & key2,:]
                #a[j,k] = bi.shape[0]
                ks = obs.loc[key1 & key2,:].shape[0]
                print(ks,label,uc)
                a[j,k] = ks
        as1 = a.sum(axis=1,keepdims=True)
        print(a,"a0")
        a = a/as1
        shape = a.shape
        height = np.zeros(shape[0])
        print(a,"a")
        print(as1)
        hands = []
        labs = []
        for l in range(shape[1]):
            try:
                ax.barh(bins,a[:,l],label=ucols[l],left=height,color=allc.colors[l])
                hands.append(ucols[l])
                labs.append(allc.colors[l])
            except:
                clr = allc.rgb()
                ax.barh(bins,a[:,l],label=ucols[l],left=height,color=clr)
                hands.append(ucols[l])
                labs.append(allc.clr)

            height += a[:,l]
        plt.gca().invert_yaxis()
        ax.get_yaxis().set_ticks([])
        ax.get_xaxis().set_ticks([])
        ax.legend(hands,labs)
        #ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left',title=bar)

    plt.show()
    return(df,obs,dfxy)




def clusterbar(df,obs,dfxy,cdf):
    if input("single cell heatmap instead?") == "y":
        singHeatmap(df,obs,dfxy)
        return(df,obs,dfxy)
    #fig,ax = plt.subplots(figsize=(15,min(10+cdf.shape[0]/5,2**15/100)))
    '''
    chrc = input("add row colors? (y)")
    if chrc == "y":
        #ch,uch = obMenu(obs,title="color based on?")
        rowColors = {}
        for op in sorted(list(obs["Cluster"].unique())):
            rc = input(op+" color:")
            if rc in matplotlib.colors.BASE_COLORS or rc in matplotlib.colors.CSS4_COLORS:
                rowColors[op] = rc
            else:
                print("invalid color, using gray")
                rowColors[op] = "gray"
        rowColors = pd.DataFrame(index=rowColors.values(),data=rowColors.keys()).transpose()
        #rowColors = pd.DataFrame(sorted(list(obs["Cluster"].unique())))[0].map(rowColors)
    '''
    indiit = input("z-score first? (y or '')")
    if indiit == "y" or indiit == '':
        cdf,obs,dfxy = zscorev(cdf,obs,dfxy)

    try:
        sns.set(font_scale=1)

        #if chrc == "y":
        if False:

            print(rowColors)
            ay = sns.clustermap(cdf,yticklabels=True, xticklabels=True,center=0, cmap='bwr',figsize=(25,min(10+cdf.shape[0]/5,2**15/100)), row_colors =[rowColors])#vmin=-10, vmax=10,
        else:
            ay = sns.clustermap(cdf,yticklabels=True, xticklabels=True,center=0, cmap='bwr',figsize=(25,min(10+cdf.shape[0]/5,2**15/100)))#,metric="euclidean")#vmin=-10, vmax=10
    except Exception as e:
        print(e,"error")
        return(df,obs,dfxy)
    plt.setp(ay.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
    ayax = ay.ax_heatmap.yaxis.get_majorticklabels()
    for i,op in enumerate(obs.columns):
        print(i,":",op)
    ch = int(input("color by which?"))
    color = list(obs.columns)[ch]
    ucols = sorted(list(obs[color].unique()))
    a = np.zeros((len(ayax),len(ucols)))
    bins = []
    for i,Text in enumerate(ayax):
        Text=Text.get_text()
        key1 = obs[obs.name].astype(str) == str(Text)
        #print(str(Text))
        bins.append(str(Text))
        #print(key1,"1")
        for j,c in enumerate(ucols):
            key2 = obs[color] == c
            #print(key2,"2")
            bi = obs.loc[key1 & key2,:]
            a[i,j] = bi.shape[0]
    fig, ax = plt.subplots(figsize=(10,min(10+cdf.shape[0]/5,2**15/100)))#20))
    shape = a.shape
    height = np.zeros(shape[0])
    for i in range(shape[1]):
        try:
            ax.barh(bins,a[:,i],label=ucols[i],left=height,color=allc.colors[i])
        except:
            ax.barh(bins,a[:,i],label=ucols[i],left=height,color=allc.rgb())
        height += a[:,i]
    plt.gca().invert_yaxis()
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left',title=color)
    plt.show()
    mpl.style.use('default')
    return(df,obs,dfxy)


def biomSortedMap(df,obs,dfxy,nul=9):
    oobs = obs.copy()
    odf = df.copy()
    for i,col in enumerate(df.columns):
        print(i,":",col)

    ch = int(input('column number to sort by:'))
    ch1,uch = obMenu(obs,'column number to color by')
    try:
        ch2,uch2 = obMenu(obs,'make different plots for each category in column:')
    except:
        biomSMH(df,obs,dfxy,ch,ch1,uch)
        return(odf,oobs,dfxy)
    for uc2 in uch2:
        key = obs.iloc[:,ch2] == uc2
        sobs = obs.loc[key,:]
        sdf = df.loc[key,:]
        biomSMH(sdf,sobs,dfxy,ch,ch1,uch,title=uc2)
    return(odf,oobs,dfxy)

def biomSMH(df,obs,dfxy,ch,ch1,uch,title = 'all cells'):

    #obs['toSort'] = df.iloc[:,ch]
    scol = df.columns[ch]
    obs[scol] = df.columns[ch]
    df = df.sort_values(df.columns[ch]).apply(ZSC)
    obs = obs.sort_values(scol)
    vout = 10
    sns.set(font_scale=2.5)

    rcolors = obs.iloc[:,ch1].copy()
    for j,uc in enumerate(uch):
        print(uc)
        key = obs.iloc[:,ch1] == uc
        print(key.sum(),'ks')
        rcolors.loc[key] = allc.colors[j]
    print(rcolors,'rcolors')
    ax=sns.clustermap(df, vmin=-vout, vmax=vout, cmap='bwr',row_colors = rcolors,
                          yticklabels=False, xticklabels=True,center=0,figsize=(25,25),
                          row_cluster=False, col_cluster=False, colors_ratio = 0.01)
    plt.title('                    '+title)
    plt.show()
    return(df,obs,dfxy)

def sortedMap(df,obs,dfxy,
              figtitle = '',dropNA = True,dgram= False):
    if input('sort by biomarker level instead? (and make multiple maps for different categories)') == 'y':
        df,obs,dfxy = biomSortedMap(df,obs,dfxy,9)
        mpl.style.use('default')
        return(df,obs,dfxy)

    #make sort-by-dendrogram function that adds "1- 2-" in front of any label so this has dendrogram option
    #then can run aggregate by leiden and run through here (after sorting highest level label (or any) by dendrogram)

    obs = obs.astype(str)
    odf,oobs,oxy = df.copy(),obs.copy(),dfxy.copy()

    mpl.style.use('default')
    sns.set(font_scale=min(2,2*40/len(list(df.columns)) ))


    allcombs = [[]]
    for i,ob in enumerate(obs.columns):
        print(i,":",ob)
    while True:
        try:
            chi = int(input("int of ob to add to colorbar. nonint to escape"))
            allcombs[0].append(obs.columns[chi])
        except:
            break


    print(allcombs,"allcombs!")
    for cols in allcombs:
        sns.set(font_scale=2)
        col = cols[0]
        print(col,'COL!')
        #input()
        if dropNA:
            nkey = oobs[col].isna()
            nk2 = oobs[col] == "nan"
            nkey = nkey | nk2
            df,obs,dfxy = odf.loc[~nkey,:],oobs.loc[~nkey,:],oxy.loc[~nkey,:]
            df,obs,dfxy = zscorev(df,obs,dfxy)
        allutys = []
        for col in cols:
            utys = sorted(list(obs.loc[:,col].astype(str).unique()))
            allutys.append(utys)
        data,colors = sortMap(df,obs,dfxy,cols,allutys)
        #print("\n\n",colors)
        vout = 10
        ax=sns.clustermap(data, vmin=-vout, vmax=vout, cmap='bwr',row_colors=colors,
                              yticklabels=False, xticklabels=True,center=0,figsize=(25,25),
                              row_cluster=False, col_cluster=False, colors_ratio = 0.01)

        #if SAVEfIGS:
            #plt.savefig(save(0,BX+"_single-heatmap",figtitle+" "+"_".join(cols),typ="png"),bbox_inches='tight')
        plt.show()
        #if dgram != False:
            #singHeatmap(df,obs,dfxy,tcols=tcols,title=figtitle+" d-gram")
        mpl.style.use('default')
        fig = plt.figure(figsize=(4*len(cols),8))
        primts = ["1: endothelial","2: immune","3: tumor","4: active fibroblast","5: stromal"]
        for j,col in enumerate(cols):
            mpl.style.use('default')
            print(j)
            ax = fig.add_subplot(1,len(cols),j+1)
            leg = {}
            utys = sorted(list(obs.loc[:,col].astype(str).unique()))
            #sns.set(font_scale=3/len(utys))

            for i,uo in enumerate(utys):
                if uo == "nan" or uo == "":
                    leg[uo]="lightgray"
                elif uo in primts:
                    uoi = primts.index(uo)
                    leg[uo]=allc.colors[uoi]
                else:
                    leg[uo]=allc.colors[i]

            print(leg)
            kl = leg.keys()
            vl = leg.values()
            x = np.arange(len(kl))
            y = np.ones_like(x)
            ax.bar(x,y,color=vl)
            ax.set_xticks(x)
            ax.set_xticklabels(kl,rotation=90,size=min(20,160/len(utys)))
            #ax.title.set_text(col,rotation = 90)
            ax.set_ylabel(col)
            #ax.set_title(col)
            ax.set_yticks([])
            plt.tight_layout()
        #if SAVEfIGS:
        #    plt.savefig(save(0,BX+"_single-heatmap",figtitle+" legend_"+"_".join(cols),typ="png"),bbox_inches='tight')
        plt.show()
        return(odf,oobs,oxy)


def sortMap(df,obs,dfxy=9,cols=[["no cols included"]],allutys = None):
    data,colors = [],[]
    while len(colors) < len(cols):
        colors.append([])
    col = cols[0]
    utys = allutys[0]
    primts = ["1: endothelial","2: immune","3: tumor","4: active fibroblast","5: stromal"]
    for i,uo in enumerate(utys):
        key = obs.loc[:,col]==uo
        #print(key.sum())
        if key.sum() == 0:
            continue
        sdf = df.loc[key,:]
        sobs = obs.loc[key,:]
        if uo == "nan" or uo == "":
            colors[0].append(pd.Series(np.full((sobs.shape[0]),'lightgray'),index=sobs.index))
        elif uo in primts:
            uoi = primts.index(uo)
            colors[0].append(pd.Series(np.full((sobs.shape[0]),allc.colors[uoi]),index=sobs.index))
        else:
            colors[0].append(pd.Series(np.full((sobs.shape[0]),allc.colors[i]),index=sobs.index))

        if len(cols) > 1:
            d1,c1 = sortMap(sdf,sobs,cols=cols[1:],allutys = allutys[1:])
            data.append(d1)

            for j,cS in enumerate(c1):
                colors[j+1].append(cS)
        else:
            data.append(sdf)
    try:
        data = pd.concat(data,axis=0)
    except:
        print(utys,"no vals to concat in data",obs.loc[:,col])
    c = []
    for ci in range(len(colors)):
        cs = pd.concat(colors[ci],axis=0,ignore_index=False)
        c.append(cs)
    colors = c
    return(data,colors)




def sortedMap1(df,obs,dfxy,vout=10,col_cluster=False):
    mpl.style.use('default')
    sns.set(font_scale=2)
    df,obs,dfxy = zscorev(df,obs,dfxy)
    print(obs.columns)
    #print(df.columns,"df cols")
    #cols = ['Primary Celltype autoCellType res: 1.0','immune checkpoints autoCellType res: 1.0']
    for col in obs.columns:
        lines = [0]
        print(col)
        nkey = obs[col].isna()
        obs.loc[nkey,col] = "na"
        obs.loc[:,col] = obs.loc[:,col].astype(str)
        utys = sorted(list(obs.loc[:,col].astype(str).unique()))
        print(utys)
        if len(utys) < 2 or len(utys) > 20:
            print('no good')
            continue
        title = col
        data = []
        colors = []
        for i,uo in enumerate(utys):
            data.append(df.loc[obs.loc[:,col]==uo,:])
            colors.append(pd.Series(np.full_like(obs.loc[obs.loc[:,col]==uo,col],allc.colors[i])))
            title += "__"+uo+"-"+allc.colors[i]
            lines.append(lines[-1]+data[-1].shape[0])
        print(colors,'colors')
        data = pd.concat(data,axis=0)
        #print(data,data.index)
        #print(data.shape[0],pd.Series(data.index).unique().shape[0])
        colors = pd.concat(colors,axis=0)
        colors.index = data.index

        cN = col
        if ":" in cN:
            cN = ".".join(cN.split(":"))


        ax=sns.clustermap(data, cmap='bwr',row_colors=[colors],
                              yticklabels=False, xticklabels=True,figsize=(25,25),
                              row_cluster=False, col_cluster=col_cluster,vmin=-10,vmax=10) #, vmin=-vout, vmax=vout,,center=0
        #ax=sns.clustermap(data, vmin=-vout, vmax=vout, cmap='bwr',row_colors=[colors],
                              #yticklabels=False, xticklabels=True,center=0,figsize=(25,25),
                              #row_cluster=False, col_cluster=col_cluster)
        ay = ax.ax_heatmap
        for l in lines:
            ay.plot([0,df.shape[1]],[l,l])
        plt.title(title)
        #SD = r"Y:\Cyclic_Analysis\20210413_AMTEC_Analysis\data\figures_iy\2021-11-21\single heatmaps\hta14"+BX
        #if SAVEfIGS:
            #plt.savefig(save(0,BX+"_single-heatmap",cN,typ="png"))
        plt.show()
    mpl.style.use('default')
    return(df,obs,dfxy)

def singHeatmap(df,obs,dfxy,pc=False,vout=10):
    sns.set(font_scale=1)
    if input("z-score first? (y)") == "y":
        df,obs,dfxy = zscorev(df,obs,dfxy)
    if input("pick colors? (y)") == "y":
        pc = True
    for i,col in enumerate(obs.columns):
        print(i,":",col)
    cols = []
    while True:
        try:
            cta = int(input("number of column to include in colors (blank to escape): "))
            uty = obs.iloc[:,cta].unique()
            colCol = obs.iloc[:,cta].copy()
            colCol[:] = "gray"
            if pc:
                for ut in uty:
                    coch = input(ut+' color?')
                    colCol.loc[obs.iloc[:,cta]==ut] = coch
            else:
                for j,ut in enumerate(uty):
                    colCol.loc[obs.iloc[:,cta]==ut] = allc.colors[j]
            cols.append(colCol)
        except Exception as e:
            print(e)
            break
    stt = time.time()
    try:
        ay = sns.clustermap(df,yticklabels=True, xticklabels=True,center=0, cmap='bwr',figsize=(25,25),vmin=-vout, vmax=vout, row_colors =cols)#
    except Exception as e:
        print("plotting failed!\n",e)
        return(df,obs,dfxy)
    #plt.show()
    dgram = ay.dendrogram_row.dendrogram
    #print(dgram)
    #sortD = df.iloc[dgram["leaves"],:]
    distances = []
    for i in range(df.shape[0]-1):
        c1 = df.iloc[dgram["leaves"][i],:]
        c2 = df.iloc[dgram["leaves"][i+1],:]
        dists = (c1-c2)**2
        #print(dists)
        ds = dists.sum()**.5
        distances.append(ds)
    fz,az = plt.subplots()
    az.plot(distances,np.ones(len(distances))*len(distances)-1-np.arange(len(distances)),marker="o",markersize=.5,linestyle="None")
    md = stat.mean(distances)
    sd = stat.stdev(distances)
    lines = []
    res = int(input("minimum cells in cluster: ")) #must be highest distance for this number of distances each direction
    sres = float(input("minimum SD of any cluster border: "))
    i = 0
    lastD = None
    dc = 0
    cN = "dgram cluster "+str(res)+"x"+str(sres)
    obs[cN] = 9999
    while i < len(distances):
        #print(i)
        d = distances[i]
        if d > md+sd*sres:
            nearCells = distances[max(0,i-res):min(i+res,len(distances))]
            m = distances.index(max(nearCells))
            #print(i,m)
            if i == m:
                lines.append(m)
                if lastD != None:
                    lev = dgram['leaves'][lastD+1:i+1]
                    #print(lev)
                    obs.loc[:,cN].iloc[lev] = dc
                    dc += 1
                lastD = i
                i += res+1
            elif m > i:
                i = m
            else:
                i += 1
        else:
            i += 1
    obs.loc[obs.loc[:,cN]== 9999,cN] = dc
    ax = ay.ax_heatmap
    print(lines)
    for l in lines:
        ax.plot([0,df.shape[1]],[l+1,l+1], 'k-', lw = 1)#ax.get_xlim()
    #ay.hlines(lines, *ay.get_xlim())
    if input('save? (y)') == 'y':
        folder = input("folder?")
        fn = input('filename?')
        plt.savefig(folder+'/'+fn,bbox_inches='tight')
    plt.show()
    #heatmap(df,obs,dfxy,df)

    print((time.time()-stt)/60," minute runtime")
    mpl.style.use('default')
    return(df,obs,dfxy)

def neighPie(df,obs,dfxy,cdf):
    print("standard pie chart requires consistent radii at 25,50,70,100u")
    if input("custom pie chart? (y)")=="y":
        for Col in df.columns:
           print(Col)
        neighbors = []
        keySL = []
        while True:
            keySt = input("string to be found in all compared entities (any, send blank when done): ")
            if keySt == "":
                break
            else:
                keySL.append(keySt)
        goodSL = []
        while True:
            inp = input("substring (radius) to include as layer (blank to exit): ")
            if inp == "":
                break
            goodSL.append(inp)
            neighbors.append([])

        for col in df.columns:
            for keyS in keySL:
                if keyS in col:
                    for i,st in enumerate(goodSL):
                        if st in col:
                            neighbors[i].append(col)
        #neighPieH1(df,obs,dfxy,neighbors)
        for slide in obs["slide"].unique():
            key = obs["slide"] == slide
            sdf,sxy,sobs = df.loc[key,:],dfxy.loc[key,:],obs.loc[key,:]
            neighPieH1(sdf,sobs,sxy,neighbors)
            cats = ["Primary Celltype autoCellType res: 1.0",
            'tumor subtype autoCellType res: 1.0',
            'receptors autoCellType res: 1.0',
            'immune subtype autoCellType res: 1.0',
            'immune checkpoints autoCellType res: 1.0',
            'cytotoxic autoCellType res: 1.0',]
            for cat in cats:
                for uty in obs[cat].unique():
                    print(cat,uty)
                    key = obs[cat] == uty
                    tdf,txy,tobs = sdf.loc[key,:],sxy.loc[key,:],sobs.loc[key,:]
                    neighPieH1(tdf,tobs,txy,neighbors,cat=cat)
    else:
        odf,oobs,oxy = df.copy(),obs.copy(),dfxy.copy()
        neighPieH2(df,obs,dfxy)
        for slide in obs["slide"].unique():
            key = obs["slide"] == slide
            sdf,sxy,sobs = df.loc[key,:],dfxy.loc[key,:],obs.loc[key,:]
            neighPieH2(sdf,sobs,sxy)
            cats = ["Primary Celltype autoCellType res: 1.0",
            'tumor subtype autoCellType res: 1.0',
            'receptors autoCellType res: 1.0',
            'immune subtype autoCellType res: 1.0',
            'immune checkpoints autoCellType res: 1.0',
            'cytotoxic autoCellType res: 1.0',]
            for cat in cats:
                for uty in obs[cat].unique():

                    print(cat,uty)
                    key = obs[cat] == uty
                    tdf,txy,tobs = sdf.loc[key,:],sxy.loc[key,:],sobs.loc[key,:]
                    neighPieH2(tdf,tobs,txy,cat=cat)
    return(df,obs,dfxy)



'''
        try:
            df,obs,dfxy = pick(odf.copy(),oobs.copy(),oxy.copy())
            neighPieH1(df,obs,dfxy)
        except ValueError:
            return(odf,oobs,oxy)
'''

def neighPieH1(df,obs,dfxy,neighbors,cat=None):
    n = neighbors

    rds = []
    for i in range(len(neighbors)):
        rds.append(df.loc[:,n[i]])
    rds.reverse()
    #print(rds)
    size = 0
    fig, ax = plt.subplots()
    #fig1, ax1 = plt.subplots()
    allinf = []
    sizeinc = .9/len(rds)
    for rd in rds:
        #print(rd.sum(axis=0))
        lab = round(rd.sum(axis=0) *100/rd.sum(axis=0).sum(),1)
        #allinf.append(list(lab)+[mixingScore(rd)])
        lab = rd.columns+"_"+lab.astype(str)+"%"
        try:
            ax.pie(rd.sum(axis=0), radius=1-size, colors=plt.colormaps["tab20c"]([0,4,6,8,16]),
                   wedgeprops=dict(width=sizeinc, edgecolor='w'),labels=lab)

        except:
            print("could not print to plot",cat)
            return(df,obs,dfxy)

        size += sizeinc
    #print(allinf)
    try:
        CL = []
        for LAb in rd.columns:
            CL.append(LAb.split("_")[0])
        tab = ax.table(cellText=allinf,colLabels=CL)#+["tum_N/imm_N"])
        tab.auto_set_font_size(False)
        tab.set_fontsize(10)
        plt.tight_layout()
    except Exception as e:
        print(e,"can't make table")
    if cat == None:
        ax.set(title=" ".join(list(obs['slide'].unique())+list(obs["Primary Celltype autoCellType res: 1.0"].unique())))
    else:
        ax.set(title=" ".join(list(obs['slide'].unique())+list(obs[cat].unique()))+" "+cat+"  total cells="+str(df.shape[0]))


    plt.show()
    return(df,obs,dfxy)

def neighPieH2(df,obs,dfxy,cat=None):
    #plt.rcParams["figure.autolayout"] = True
    neighbors = []
    while len(neighbors) < 4:
        neighbors.append([])

    for i in range(4):
        keyS = "ors_"+str(25*(i+1))
        for col in df.columns:
            if keyS in col:
                neighbors[i].append(col)
    n = neighbors
    #print(n)
    d25,d50,d75,d100 = df.loc[:,n[0]],df.loc[:,n[1]],df.loc[:,n[2]],df.loc[:,n[3]]
    rds = [d25,d50,d75,d100]
    rds.reverse()
    #print(rds)
    size = 0
    fig, ax = plt.subplots(figsize=(8,8))
    #fig1, ax1 = plt.subplots()
    allinf = []
    for rd in rds:
        #print(rd.sum(axis=0))
        lab = round(rd.sum(axis=0) *100/rd.sum(axis=0).sum(),1)
        allinf.append(list(lab)+[mixingScore(rd)])
        lab = rd.columns+"_"+lab.astype(str)+"%"
        try:
            ax.pie(rd.sum(axis=0), radius=1-size, colors=plt.colormaps["tab20c"]([0,4,6,8,16]),
                   wedgeprops=dict(width=.2, edgecolor='w'),labels=lab)

        except:
            print("could not print to plot",cat)
            return(df,obs,dfxy)

        size += .2
    #print(allinf)
    CL = []
    for LAb in rd.columns:
        CL.append(LAb.split("_")[0])
    tab = ax.table(cellText=allinf,colLabels=CL+["tum_N/imm_N"])
    tab.auto_set_font_size(False)
    tab.set_fontsize(10)
    ax.axis('tight')
    #ax.axis('off')
    if cat == None:
        ax.set(title=" ".join(list(obs['slide'].unique())+list(obs["Primary Celltype autoCellType res: 1.0"].unique())))
    else:
        ax.set(title=" ".join(list(obs['slide'].unique())+list(obs[cat].unique()))+" "+cat+"  total cells="+str(df.shape[0]))

    plt.show()
    return(df,obs,dfxy)

def mixingScore(rd):
    for col in rd.columns:
        if "umor_Primary" in col:
            tcol = col
            continue
        if "mmune_Primary" in col:
            icol = col
            continue
    try:
        score = rd[tcol].sum()/rd[icol].sum()
        print(score,"tumor interactions / immune interactions")
        return(round(score,2))
    except:
        return(None)

def graphST(df,obs,dfxy,cdf):
    #adata = sc.read_visium(file_fold, count_file='filtered_feature_bc_matrix.h5', load_images=True)
    adata = anndata.AnnData(df,obs = obs)
    model = GraphST.GraphST(adata, device='cpu') #device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    radius = 50
    tool = 'mclust' # mclust, leiden, and louvain
    # clustering
    if tool == 'mclust':
       clustering(adata, n_clusters, radius=radius, method=tool, refinement=True) # For DLPFC dataset, we use optional refinement step.
    elif tool in ['leiden', 'louvain']:
       clustering(adata, n_clusters, radius=radius, method=tool, start=0.1, end=2.0, increment=0.01, refinement=False)
    #Unfinished, for Dmytro?

def ripleyl(df,obs,dfxy,cdf=None):
    input("NO SQUIDPY PKG...")
    return(df,obs,dfxy)
    ch,uch = obMenu(obs,'sort by: ')
    adata = anndata.AnnData(df,obs = obs)
    adata.obsm["spatial"] = dfxy.values
    ck = obs.columns[ch]
    adata.obs[ck] = adata.obs[ck].astype('category')
    mode = "L"
    sq.gr.ripley(adata, cluster_key=ck, mode=mode)

    fig = sq.pl.ripley(adata, cluster_key=ck, mode=mode,    show=False,return_fig=True,)
    fig.savefig(r'\\graylab.ohsu.edu\Chin_Lab\Youm\IFA4 for ripley plot dec 2025\riply_allcells_hm.png', dpi = 150)    #'\\accsmb.ohsu.edu\CEDAR\ChinData\Youm\temp-ripley-for-HM\ripley.png', dpi=150)
    return(df,obs,dfxy)


def volcanoPlot(df,obs,dfxy,cdf,mannw = 0):
    print('find cells with high variation pca- negative binomial normalization')
    odf = df.copy()
    ch,uch = obMenu(obs,title='category to split for plot')
    for i,uc in enumerate(uch):
        print(i,":",uc)
    ch1 = int(input('label to compare against all other labels?'))
    if input(' ttest instead? (direct relation to differential exp) (y)') == 'y':
        mannw = 1
    if input(' simple StDev pca? (y)') == 'y':
        print('will display 100x stdev for consistent Y range. Y axis label incorrect')
        mannw = 2

    if mannw < 2:
        df = ZSC(df).fillna(0) # zscore fills with nan if all 0s
    key = obs.iloc[:,ch] == uch[ch1]
    #print(key.sum(),df.shape[0])

    pop1 = df.loc[key,:]
    pop2 = df.loc[~key,:]

    #pvs = []
    #diffs = []
    fig,ax = plt.subplots(figsize=(12,5))
    pvals = []
    print(df.isna().any().any(),'any nan')
    for marker in sorted(list(df.columns)):
        #print(df.loc[df[marker].isna(),marker])
        if mannw == 0:
            mstat, opval = mannwhitneyu(pop1.loc[:,marker],pop2.loc[:,marker], nan_policy='omit')
        elif mannw == 1:
            tstat, opval = ttest_ind(pop1.loc[:,marker],pop2.loc[:,marker], nan_policy='omit')
        else:
            std = np.std(df.loc[:,marker])
            pval = std
            opval = pval+1
            print(marker,std)
            #opval = 10**(-std)
            #print(opval,'\n')


        nans = []
        if opval == 0:
            pval = 500

        elif mannw < 2:
            try:
                pval = -math.log(opval,10)
                if np.isnan(pval):
                    print(marker,'nan pval',opval)
                    nans.append(marker)
                    pval = -1
            except Exception as e:
                print(marker, ' has raw  pval of',opval,'  .. ',e)
                pval = -1


        diff = np.mean(pop1.loc[:,marker])-np.mean(pop2.loc[:,marker])
        pvals.append(pval)#+(diff**2)**.5/10000) #diff**2 as tiebreaker
        plt.scatter([diff],[pval],c = 'blue')
        plt.text(diff,pval,marker)

    nam1 = str(uch[ch1])
    if len(uch) == 2:
        nam2 = str(uch[1-ch1])

    else:
        nam2= 'other cells'

    titl = obs.columns[ch]+': '+nam1+' vs. ' +nam2
    plt.title(titl)
    plt.xlabel('average difference between populations '+nam1+' - '+nam2)
    if mannw == 2:
        plt.ylabel('standard deviation')
    else:
        plt.ylabel('-log10(p)')
    plt.show()
    top = input('top variable biomarkers to save as separate df? non int to skip')
    try:
        top = int(top)
        print(sorted(list(df.columns)),'\n',pvals)

        ds = pd.DataFrame(columns = sorted(list(df.columns)),data = [sorted(list(df.columns)),pvals])
        print(ds)
        ds = ds.transpose()
        print(ds,'after transpose')
        ds.index=df.columns
        ds.columns=['genes','pvalues']
        ds = ds.sort_values('pvalues',axis=0,ascending=False)
        ds = ds.iloc[:top,:]
        print(ds)
        ds.index = ds['genes']
        print(ds.shape)
        ndf = odf.loc[:,ds.index]
        print(ndf,'ndf')
        save(ndf,obs,dfxy)
    except Exception as e:
        print(e)

    print('the following markers had "nan" as output from ttest or mann-witney:',nans,'total of',len(nans))
    return(odf,obs,dfxy)


def BGtest(df,obs,dfxy):
    odf,oobs,oxy = df.copy(),obs.copy(),dfxy.copy()
    df,obs,dfxy = subQC(df,obs,dfxy)
    #df = df.mask(df < 0, 0)
    df = df - df.min()
    shp = int(df.shape[0]/2)
    out = {}
    for col in df.columns:
        s = df.loc[:,col].nlargest(shp).iloc[:shp]
        omax = odf.loc[:,col].quantile(.9)
        out[col] = s.std()/omax
    print(out)
    out = pd.DataFrame(out.values(),index=out.keys())
    out.to_csv('BGtest.csv')
    return(odf,oobs,oxy)





    '''
    df = df.apply(ZSC)
    sums = df.sum(axis=1)
    print(sums.shape)
    df['sums'] = (sums - sums.mean())/sums.std() #inconsistent zscore method but .apply(ZSC) raises error.. specify axis?
    cdf = corMat(df)
    cdf = cdf.sort_values(by ='sums',ascending=False)
    print(cdf)
    with pd.option_context('display.max_rows', None):
        print(cdf.loc[:,'sums'])
    return(odf,oobs,oxy)
    '''

def corMat(df):
    cm = []
    for b1 in df.columns:
        cl = []
        for b2 in df.columns:
            cl.append((df.loc[:,b1] * df.loc[:,b2]).sum()/df.shape[0])
        cm.append(cl)
    return(pd.DataFrame(cm,columns=df.columns,index = df.columns))


def statTest(df,obs,dfxy):
    ch,uch = obMenu(obs,'column with annotations to test between')
    print(uch)
    out = {}
    if len(uch) == 2:
        k1 = obs.iloc[:,ch]==uch[0]
        k2 = obs.iloc[:,ch]==uch[1]
        p1,p2 = df.loc[k1,:],df.loc[k2,:]
        for bm in df.columns:
            pop1,pop2 = p1.loc[:,bm].dropna(),p2.loc[:,bm].dropna()
            t_stat, p_val = ttest_ind(pop1,pop2)
            mstat,pval = mannwhitneyu(pop1,pop2)
            out[bm] = [p_val, pval, pop1.mean(), pop2.mean(), pop1.shape[0], pop2.shape[0]]
        print(out)
        out = pd.DataFrame.from_dict(out,orient='index',columns = ['t-test p-val','Mann-Whitney p-val',
                                                                   uch[0]+' mean',uch[1]+' mean',uch[0]+' N datapoints',
                                                                   uch[1]+' N datapoints'])
        print(out)
        out.to_csv(input('filename: ')+'.csv')
        print(os.getcwd(), 'saving here')
        return(df,obs,dfxy)
    else:
        input('multiple categories stats are not implemented in this path')
        return(df,obs,dfxy)


def barplot(df,obs,dfxy,cdf,piechart = False,biomchart=False):
    oobs = obs.copy()
    obs = obs.astype(str)
    #rch = int(input("show actual number (0) or percentages (1)?"))
    for i,he in enumerate(obs.columns):
        print(i,":",he)
    ch = int(input("sort x axis by:"))
    binCol = obs.columns[ch]
    try:
        obs[binCol+' temp'] = obs[binCol].apply(lambda n: n.split('_')[0])
        obs[binCol+' temp']=obs[binCol+' temp'].astype(float).astype(int)
        print('int type bins')
    except:
        print('str type bins')
        obs[binCol+' temp'] = obs[binCol]+'!'
    if input("apply custom sorting by keystring? (y)") == "y":
        print(obs[binCol+' temp'].unique())
        print('notice that "!" has been added to the end of each bin')
        keyS = []
        i = 1
        while True:
            print("send blank when done")
            inp = input("keystring to put in position "+str(i)+": ")
            if inp == "":
                if len(keyS) == 0:
                    print("error, must include at least one keystring")
                else:
                    break
            else:
                keyS.append(inp)
                i += 1
        nbc = []
        done = []
        i = 0
        for kS in keyS:
            for cat in obs[binCol].unique():
                if kS in cat and cat not in done:
                    key = obs.loc[:,binCol] == cat
                    obs.loc[key,binCol] = str(i) + obs.loc[key,binCol]
                    nbc.append(obs.loc[key,binCol])
                    done.append(cat)
                    print("adding",cat)
                    i += 1
        try:
            obs[binCol+' temp'] = pd.concat(nbc,axis=0)
            print(obs[binCol+' temp'].unique(),"ordered")
        except:
            print('error, no values to concat')




        obs = obs.sort_values(binCol+' temp',axis=0) #sorting changes index order
    #obs[binCol]=obs[binCol].astype(str)
    bins = sorted(list(obs.iloc[:,ch].unique()))
    print(bins,'sorted bins')

    for i,he in enumerate(obs.columns):
        print(i,":",he)
    ch2 = int(input("color bars by:"))
    pc = input("pick colors for barplot? (y)")
    colors = sorted(list(obs.iloc[:,ch2].unique()))
    colCol = obs.columns[ch2]
    a = np.zeros((len(bins),len(colors)))
    b = np.zeros((len(bins),len(colors)))
    c = b.copy()
    d = c.copy()
    #table = np.zeroes(len(bins),len(colors))
    nClrs = []
    for clr in colors:
        nClrs.append(obs.loc[obs.loc[:,colCol] == clr,:].shape[0])

    for i in range(len(bins)):
        Bin = obs.loc[obs[binCol]==bins[i],:] #sobs
        nBin = Bin.shape[0]
        for j in range(len(colors)):
            n = Bin.loc[Bin[colCol]==colors[j],:].shape[0] #tobs

            if nBin == 0:
                nBin = 1
            a[i,j] = n
            b[i,j] = n/nBin
            c[i,j] = n/nClrs[j]
    ctots = np.sum(c,axis=1)
    print(ctots,ctots.shape)
    for i in range(c.shape[0]):
        d[i,:] = c[i,:]/ctots[i]*100
    #print(bins,len(bins),len(bins[0]))
    if biomchart:
        in2 = False
        in3 = False
        if input("use z-scored (by biomarker) data? (y) for errorbarplot") == "y":
            in2 = True
        if input("pick colors for errorbarplot? (y)") == "y":
            in3 = True
        zdf,ttt,tttt = zscorev(df,obs,dfxy)
        #barx = 0
        for binName in bins:
            kEY1 = obs.iloc[:,ch] == binName
            if kEY1.sum() < 2:
                continue
            for coloName in sorted(list(obs.loc[:,colCol].unique())):
                kEY2 = obs.loc[:,colCol] == coloName
                if in2:
                    bdf = zdf.loc[kEY1 & kEY2,:]
                else:
                    bdf = df.loc[kEY1 & kEY2,:]
                if bdf.shape[0] > 2:
                    fig1,ax1=plt.subplots(figsize=(bdf.shape[1]/10,6))
                    mean = bdf.mean(axis=0)
                    stdev = bdf.std(axis=0)
                    if in3:
                        keyss = []
                        colorss = []
                        while True:
                            keyS = input("key string for color assignment (sent blank to exit)")
                            if keyS == "":
                                break
                            else:
                                keyss.append(keyS)
                                colorss.append(input("color for bars containing this string"))
                        for i in range(bdf.shape[1]):
                            cn = bdf.columns[i]
                            for j,kS in enumerate(keyss):
                                if kS in cn:
                                    clr = colorss[j]
                                    ax1.errorbar(i,mean[i],yerr=stdev[i],
                                                 marker="o",ls='none',ecolor=clr,c="black")
                                    break
                            else:
                                ax1.errorbar(i,mean[i],yerr=stdev[i],
                                             marker="o",ls='none',ecolor="black",c="black")




                    else:
                        ax1.errorbar(np.arange(mean.shape[0]),mean,yerr=stdev,
                                     marker="o",ls='none')
                    ax1.set(title=binName+" "+coloName)
                    ax1.grid(axis='x')
                    plt.xticks(np.arange(zdf.shape[1]),labels=zdf.columns, rotation='vertical', fontsize=7)
                    plt.show()

                #barx += mean.shape[0]
    if piechart:
        for k in range(a.shape[0]):
            f,ax = plt.subplots()
            ax.pie(a[k,:],autopct='%.0f%%', labels=colors)
            plt.title(bins[k])
            ax.axis('equal')
            plt.show()

    colorDict = {}
    if pc == "y":
        for i in range(a.shape[1]):
            print(colors[i])
            colorDict[colors[i]] = input("color:")
    else:
        for i in range(len(colors)):
            if colors[i] == 'yes':
                colorDict[colors[i]] = 'darkred'
            elif colors[i] == 'no':
                colorDict[colors[i]] = 'darkgray'
            elif str(colors[i]) == 'nan':
                colorDict[colors[i]] = 'lightgray'
            colorDict[colors[i]] = allc.colors[i]

    for array in [a,b,]: #c,d
        a = array.copy()
        fig, ax = plt.subplots(figsize=(15+len(bins)/5,10))
        shape = a.shape
        #print("a",a)
        height = np.zeros(shape[0])

        for i in range(shape[1]):

            try:
                AX = ax.bar(bins,a[:,i],label=colors[i],bottom=height,color=colorDict[colors[i]])
            except Exception as e:
                AX = ax.bar(bins,a[:,i],label=colors[i],bottom=height)
            for j,rect in enumerate(AX):
                nh = a[j,i]#rect.get_height()
                plt.text(rect.get_x()+rect.get_width()/2,height[j]+nh/2,round(nh,2))
            #cont = ax.BarContainer()
            #labs = np.round(a[:,i],3)
            #print(labs)
            #ax.bar_label(cont, labels=labs, label_type='center')
            height += a[:,i]
        plt.xticks(rotation = 85)
        plt.xlabel(binCol)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left',title=colCol)
        plt.show()



    daf = pd.DataFrame(data=b,index=bins,columns=colors).transpose()
    f, ax = plt.subplots(figsize=(10+daf.shape[1]/5, 10+daf.shape[0]/5))
    sns.heatmap(daf,xticklabels=daf.columns,yticklabels=daf.index,center=np.mean(daf.values),annot=True)
    plt.show()
    return(df,oobs.astype(str),dfxy)


def boxplot(df,obs,dfxy,cdf):
    oobs = obs.copy()
    print('multibox does not work with only one category')
    if input("stack multiple categories vertically?") == 'y':
        multiBoxplot(df,obs,dfxy)
        return(df,oobs,dfxy)
    if input("show all markers in one plot first?") == "y":
        fig,ax = plt.subplots(figsize=(len(list(df.columns)),8))
        ax=sns.boxplot(data=df)
        plt.xticks(rotation = 85)
        plt.show()
    vich = input("violin plot instead? (y): ")
    fich = input("show outliers? (y)")
    for i,he in enumerate(obs.columns):
        print(i,":",he)

    ch = int(input("sort x axis by:"))

    try:
        binCol = obs.columns[ch].astype(int)
    except:
        binCol = obs.columns[ch]
    try:
        obs[binCol+' temp'] = obs[binCol].apply(lambda n: n.split('_')[0])
        try:
            obs[binCol+' temp']=obs[binCol+' temp'].astype(int)
        except:
            obs[binCol+' temp']=obs[binCol+' temp'].astype(float)
        print('int type bins')
    except Exception as e:
        print('str type bins',e)
        obs[binCol+' temp'] = obs[binCol]+'!'
    if input("apply custom sorting by keystring? (y)") == "y":
        print(obs[binCol+' temp'].unique())
        print('notice that "!" has been added to the end of each bin')
        keyS = []
        i = 1
        while True:
            print("send blank when done")
            inp = input("keystring to put in position "+str(i)+": ")
            if inp == "":
                if len(keyS) == 0:
                    print("error, must include at least one keystring")
                else:
                    break
            else:
                keyS.append(inp)
                i += 1
        nbc = []
        done = []
        i = 0
        for kS in keyS:
            for cat in sorted(list(obs[binCol+' temp'].unique())):
                if kS in cat and cat not in done:
                    key = obs.loc[:,binCol+' temp'] == cat
                    obs.loc[key,binCol+' temp'] = str(i) + obs.loc[key,binCol]
                    nbc.append(obs.loc[key,binCol+' temp'])
                    done.append(cat)
                    print("adding",cat)
                    i += 1




    obs = obs.sort_values(binCol+' temp',axis=0) #sorting changes index order
    obs[binCol]=obs[binCol].astype(str)
    bins = list(obs.iloc[:,ch].unique())


    for i,he in enumerate(obs.columns):
        print(i,":",he)
    try:
        corr = False
        print("send non-int to have same color")
        ch = int(input("color boxes by:"))
        colCol = obs[obs.columns[ch]].sort_values() #new change 23/10/13
        ucols = sorted(list(colCol.unique()))
    except Exception as e:
        print(e)
        colCol = None
        ucols = [1]
        uBinCol = list(obs.loc[:,binCol].unique())
        if len(uBinCol) == 2: #23 dec 14
            corr = True
    dfo = df.merge(obs,left_index=True,right_index=True).sort_values(binCol+' temp')
    if len(ucols) == 1:
        colCol = None
    #print(dfo)
    for marker in df.columns:
        fig,ax = plt.subplots(figsize=(max(8,int(len(bins)*len(ucols)**.7/2)),8))
        if vich == "y":
            if colCol is None:
                ax=sns.violinplot(data=dfo, x=binCol, y=marker, cut=0)
            else:
                ax=sns.violinplot(hue=colCol, data=dfo, x=binCol, y=marker, cut=0)
        else:
            if fich == "y":
                ax=sns.boxplot(hue=colCol, data=dfo, x=binCol, y=marker,showfliers=True)
            else:
                ax=sns.boxplot(hue=colCol, data=dfo, x=binCol, y=marker,showfliers=False)
        legend = ax.get_legend()
        if legend is not None:
            try:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left',title=colCol)
            except:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')


        if corr:
            key = dfo.loc[:,binCol] == uBinCol[0]
            pop1 = dfo.loc[key,marker]
            pop2 = dfo.loc[~key,marker]
            t_stat, p_val = ttest_ind(pop1,pop2)
            mstat,pval = mannwhitneyu(pop1,pop2)
            print('t-statistic:', t_stat)
            print('p-value:', p_val)
            print(mstat,"mstat")
            plt.title('t-test p-value: '+str(p_val)+'\nMann-Whitney p-value:'+str(pval))
        plt.grid(visible=True)
        plt.xticks(rotation = 85)
        plt.show()
    return(df,oobs,dfxy)

def multiBoxplot(df,obs,dfxy,cdf=None):
    cor = False
    oobs = obs.copy()
    ch0,uch0 = obMenu(obs,'divide data into separate plots by: ')
    uch0 = sorted(list(uch0))

    ch1,uch1 = obMenu(obs,'sort x axis by: ')
    binCol = obs.columns[ch1]
    try:
        obs[binCol+' temp'] = obs[binCol].apply(lambda n: n.split('_')[0])
        obs[binCol+' temp']=obs[binCol+' temp'].astype(int)
        print('int type bins')
    except:
        print('str type bins')
        obs[binCol+' temp'] = obs[binCol]+'!'

    obs = obs.sort_values(binCol+' temp',axis=0)

    try:
        ch2,uch2 = obMenu(obs,'color boxes by (send non-int to skip): ')
        if len(uch2) == 2:
            cor = True
            key3 = obs.iloc[:,ch2] == uch2[0]
        try:
            colCol = obs[obs.columns[ch2]].astype(int)
        except:
            colCol = obs[obs.columns[ch2]]
        ucols = sorted(list(colCol.unique()))
    except Exception as e:
        print(e)
        colCol = None
        ucols = [1]

    dfo = df.merge(obs,left_index=True,right_index=True)
    dfo = dfo.sort_values(binCol+' temp',axis=0)
    corrs = []
    for marker in df.columns:
        fig,axA = plt.subplots(len(uch0),1,sharex=True,
                               figsize=(len(uch1)*len(ucols)**.75/2+5,len(uch0)*1.5+5))
        #fig.suptitle(marker)

        for i,uc in enumerate(uch0):
            axA[i].title.set_text(uc)

            key = obs.iloc[:,ch0] == uc

            if cor:
                for buc in obs.loc[:,binCol+' temp'].unique():
                    key4 = obs.loc[:,binCol+' temp'] == buc
                    pop1 = df.loc[key & key3 & key4,marker]
                    pop2 = df.loc[key & ~key3 & key4,marker]
                    try:
                        mstat,pval = mannwhitneyu(pop1,pop2)
                    except:
                        pval = np.nan
                    diff = pop1.mean() - pop2.mean()
                    info = [marker,uc,buc,diff,pval]
                    corrs.append(info)



            #ax = axA[i]

            try:
                axA[i]=sns.boxplot(hue=colCol, data=dfo.loc[key,:], x=binCol, y=marker ,showfliers=False,ax=axA[i])
                #hue=colCol.loc[key]
            except:
                axA[i]=sns.boxplot(data=dfo.loc[key,:], x=binCol, y=marker ,showfliers=False,ax=axA[i])
            if type(colCol) != type(None):
                if i == 0:
                    axA[i].legend(loc=(1.01,1.01),title=obs.columns[ch2])
                else:
                    axA[i].get_legend().remove()
            if i != len(uch0)-1:
                axA[i].set_xlabel('',size=10**-5)
            axA[i] = axA[i]
        plt.xticks(rotation = 90)
        fig.tight_layout()
        plt.show()
    if cor:
        ccol =obs.columns[ch2]
        ccu = obs.loc[:,ccol].unique()

        fn1 = 'correlations for '+ccu[0]+' vs. '+ccu[1]+'.csv'
        print(fn1)
        if input('use default filename to save file? (y)') == 'y':
            fn = fn1

        else:
            fn = input('filename for correlations file:') + '.csv'
        if fn != '.csv':
            #ccu = obs.loc[:,ccol].unique()
            cordf = pd.DataFrame(corrs,columns = ['marker',obs.columns[ch0],obs.columns[ch1],'difference'+ccu[0]+' - '+ccu[1],'pscore: '+ccu[0]+' vs. '+ccu[1]])
            cordf.to_csv(fn)
    return(df,oobs,dfxy)

def boxplot1(df,obs,dfxy,cdf):
    switch = 0
    if input("violin plot instead? (y)") == "y":
        switch = 1
    for i,he in enumerate(obs.columns):
        print(i,":",he)
    ch = int(input("sort x axis by:"))
    binCol = obs.columns[ch]
    bins = list(obs.iloc[:,ch].unique())
    for i,he in enumerate(obs.columns):
        print(i,":",he)
    ch = int(input("color boxes by:"))
    colCol = obs.columns[ch]
    dfo = df.merge(obs,left_index=True,right_index=True)
    #print(dfo)
    for marker in df.columns:
        fig,ax = plt.subplots(figsize=(24,8))
        if switch == 0:
            ax=sns.boxplot(hue=colCol, data=dfo, x=binCol, y=marker,showfliers=False)
        else:
            sns.violinplot(hue=colCol, data=dfo, x=binCol, y=marker,showfliers=False)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation = 85)
        plt.show()
    return(df,obs,dfxy)


'''
'''
def hist(df,obs,dfxy,nn): #['Primary Celltype autoCellType res: 1.0','Primary Celltype Leiden_30_primariescluster autotype0.75']
    ch,uch = obMenu(obs,"category to show in histogram")
    colNames = [obs.columns[ch]]
    for col in df.columns:
        for CN in colNames:
        #continue                                  # f!!!!!!!!!! !!! !!!!!!!!!!!!!!!!
            xs,ys = [],[]
            sCN = sorted(list(obs[CN].unique()))
            for ty in sCN:
                if "n_50" in col:
                    tdf = df.loc[obs[CN]==ty,:]
                    tcol = trimExtremes(tdf[col])
                    x,y = makeHist(tcol,tdf.shape[0]/2)
                    xs.append(x)
                    ys.append(y)
                else:
                    tdf = df.loc[obs[CN]==ty,:]
                    tcol = trimExtremes(tdf[col])
                    try:
                        x,y = makeHist(tcol,tdf.shape[0]/2)
                    except:
                        continue
                    xs.append(x)
                    ys.append(y)
                    #ys.append(rollingAve(rollingAve(y,n=int(tdf.shape[0]/20)),n=int(tdf.shape[0]/50)))
            fig,ax=plt.subplots()
            for i in range(len(xs)):
                try:
                    x,y,c,ty = xs[i],ys[i],allc.amcolors[i],sCN[i]
                    ax.plot(x,y,color=c,label=ty, alpha=0.5)
                except:
                    print("hist can't use amcolors")
                    x,y,c,ty = xs[i],ys[i],allc.colors[i],sCN[i]
                    ax.plot(x,y,color=c,label=ty, alpha=0.5)
            ax.legend()
            ax.set_title(col+" "+CN)
            plt.ylabel("log2 cell counts")
            '''
            if SAVEfIGS:
                fig = ax.get_figure()
                fig.savefig(save(0,BX+"_histograms",col+CN,typ="png"),bbox_inches='tight')
            '''
            plt.show()
    return(df,obs,dfxy)

def rollingAve1(l,n=4): #n in each direction
    newL = []
    for i in range(len(l)):
        nears = []
        for j in range(n):
            left = i - j - 1
            right = i + j + 1
            if  left >= 0:
                nears.append(l[left])
            if right < len(l):
                nears.append(l[right])
        #print(nears)
        newL.append(stat.mean(nears))
    return(newL)



def makeHist(x,nb,orientation="vertical"):
    mx,Mx = min(x),max(x)
    rx = Mx-mx
    if rx == 0:
        print("no cells in histogram!!!")
        return(x,x)
    sx = rx/nb
    binCts=[]
    bins = np.arange(mx,Mx+sx,sx)
    for i in range(1,len(bins)):
        key = x>=bins[i-1]
        key1 = x<bins[i]
        ss = x.loc[key&key1]
        if ss.shape[0] > 1:
            binCts.append(np.log10(ss.shape[0])/np.log10(2))
        else:
            binCts.append(0)
    y = np.arange(len(binCts))*sx+mx
    X = binCts
    if orientation == "vertical":
        return(y,X)
    else:
        return(X,y)


def trimExtremes(ser,quantile=.97):
    sS = ser.sort_values()
    nInd = sS.shape[0]
    cutoff = sS.iloc[int(nInd * quantile)]
    #print(cutoff,"!")
    key = ser < cutoff
    ser = ser.loc[key]
    return(ser)

'''
'''


def correlationMatrix(df,obs,dfxy,cdf,clean=True):
    ordf,oobs,oxy = df.copy(),obs.copy(),dfxy.copy()
    zsw = False
    if input("z-score data vertically?") == "y":
        zsw = True
    ch,uch = obMenu(obs,"sort correlation matrices by:")
    cordfs = []
    for uc in uch:
        key = obs.iloc[:,ch] == uc
        sdf = df.loc[key,:]
        if zsw:
            sdf = scipy.stats.zscore(sdf)
        if clean:
            sdf = sdf.dropna(axis=1, how='all')
        cordfs.append(corMat(sdf))
    for i,uc in enumerate(uch):
        odf = cordfs[i]
        heatmap(df,obs,dfxy,odf,title=uc)
        ay = sns.clustermap(odf,yticklabels=True, xticklabels=True,center=np.mean(odf.values), cmap='bwr',figsize=(20,20)).fig.suptitle(uc)
        plt.show()

    for i in range(len(cordfs)):
        for j in range(len(cordfs)):
            if j > i:
                nodf = cordfs[j] - cordfs[i]
                heatmap(df,obs,dfxy,nodf,title=uch[j]+' - '+uch[i])

                #heatmap(df,obs,dfxy,nodf,title=uch[j]+" - "+uch[i])
    '''
    for i in range(len(uch)):
        for j in range(len(uch)):
            if j < i:
                odf = cordfs[j] - cordfs[i]
                heatmap(df,obs,dfxy,odf,title=uch[j]+" - "+uch[i])
    print('I think it still keeps zscore')
    '''
    return(ordf,oobs,oxy)

def corMat(df):
    return(df.corr())
    cm = []
    for b1 in df.columns:
        cl = []
        for b2 in df.columns:
            cl.append((df.loc[:,b1] * df.loc[:,b2]).sum()/(df.shape[0]-1))
        cm.append(cl)
    return(pd.DataFrame(cm,columns=df.columns,index = df.columns))


def correlationMatrix1(df,obs,dfxy,cdf):
    ordf,oobs,oxy = df.copy(),obs.copy(),dfxy.copy()
    switch = False
    if input("save correlations csv?") == "y":
        switch = True
    if input("z-score data vertically?") == "y":
        df,obs,dfxy=zscorev(df,obs,dfxy)
    col = df.columns

    ch,uch = obMenu(obs,"sort correlation matrices by:")
    odfs = []
    df1 = df
    a = df1.values
    shape = a.shape
    '''
    out = np.zeros((shape[1],shape[1]))
    for e in range(shape[0]):
        for i in range(shape[1]):
            for j in range(shape[1]):
                out[i,j] += a[e,i] * a[e,j] * 1/shape[0]
    '''
    out = []
    for b1 in df.columns:
        cl = []
        for b2 in df.columns:
            cl.append((df.loc[:,b1] * df.loc[:,b2]).sum()/df.shape[0])
        out.append(cl)
    '''
    '''
    odf = pd.DataFrame(data=out,columns = col,index=col)
    if switch:
        fn = input("filename?")
        if ".csv" not in fn:
            fn = fn+".csv"
        odf.to_csv(fn)
    heatmap(df,obs,dfxy,odf,title="all data")
    ay = sns.clustermap(odf,yticklabels=True, xticklabels=True,center=0,cmap='bwr',figsize=(25,min(10+cdf.shape[0]/5,2**15/100)), )
    #center=np.mean(odf.values)
    plt.show()
    odfs.append(odf)
    for uc in uch:
        df1 = df.loc[obs.iloc[:,ch]==uc,:]
        a = df1.values
        shape = a.shape

        out = []
        for b1 in df.columns:
            cl = []
            for b2 in df.columns:
                cl.append((df.loc[:,b1] * df.loc[:,b2]).sum()/df.shape[0])
            out.append(cl)

        '''
        out = np.zeros((shape[1],shape[1]))
        for e in range(shape[0]):
            for i in range(shape[1]):
                for j in range(shape[1]):
                    out[i,j] += a[e,i] * a[e,j] * 1/shape[0]
        '''
        odf = pd.DataFrame(data=out,columns = col,index=col)
        heatmap(df,obs,dfxy,odf,title=uc)
        odfs.append(odf)
    for i in range(len(odfs)-1):
        for j in range(len(odfs)-1):
            if j > i:
                nodf = odfs[j] - odfs[i]
                heatmap(df,obs,dfxy,nodf,title=uch[j]+" - "+uch[i])

    return(ordf,oobs,oxy)

def scatterplot(df,obs,dfxy,cdf,zsc=False,LCOR=True):
    if input('old scatter plot with histogram and broken line of best fit? (y)') == 'y':
        return(oldScatterplot(df,obs,dfxy))
    zdf = df.apply(ZSC)
    ch,uch = obMenu(obs,title='color different categories by?')
    for i,col in enumerate(df.columns):
        print(i,":",col)
    ch1=int(input("x axis number:"))
    x = df.iloc[:,ch1]
    zx = zdf.iloc[:,ch1]
    try:
        ch2 = eval(input('ys to show? use int or range[) - range support is unreliable'))
        if type(ch2) != list:
            ch2 = [ch2]
    except:
        print('error, showing all bioms')
        ch2 = range(0,df.shape[1])
    for yc in ch2:
        if yc == ch1:
            continue
        fig,ax = plt.subplots(figsize = (10,10))
        y = df.iloc[:,yc]
        zy = zdf.iloc[:,yc]
        cor = np.dot(zx,zy)/df.shape[0]
        #print(cor)
        if zsc:
            x = zx
            y = zy
        for i,uc in enumerate(uch):
            key = obs.iloc[:,ch] == uc
            if LCOR == True:
                if i == 0:
                    print("RUNTIME WARNING, calculating local correlation!")
                localCor = round(np.dot(zScoreL(zx.loc[key]),zScoreL(zy.loc[key]))/key.sum(),3)
            else:
                localCor = ''
            ax.scatter(x.loc[key],y.loc[key],edgecolors=allc.colors[i],facecolors='none',
                       marker='o',s=15,label=uc+'   r='+str(localCor), alpha = .2)
            #print(i,uc)
            #'to move back from single-cat plots, unindent everything below'
        ax.set_title('r = ' + str(cor))
        ax.set_xlabel(df.columns[ch1])
        ax.set_ylabel(df.columns[yc])
        lg = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        try:
            for k in range(len(uch)):
                lg.legend_handles[k]._sizes = [30]
                lg.legend_handles[k].set_alpha(1)
        except Exception as e:
            print(e,"legend handle edits failed",k,len(uch),lg.legend_handles)

        plt.show()

    return(df,obs,dfxy)



def oldScatterplot(df,obs,dfxy,cdf):
    df["celln"]=np.arange(df.shape[0])
    for i,col in enumerate(df.columns):
        print(i,":",col)
    ch1=int(input("x axis number:"))
    ch2=int(input("y axis number:"))
    x,y =df.iloc[:,ch1],df.iloc[:,ch2]
    #y = x*h
    #x-*y = x- * x * h
    #(xT*x)- * xt*y) = h
    X = pd.concat((x,pd.Series(np.ones_like(x),index=x.index)),axis=1).values
    print(X)
    theta = np.linalg.inv(X.T.dot(X))#.dot(X.T.dot(y))
    print(theta)
    theta = theta.dot(X.T.dot(y))
    print(theta,"theta")
            #np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    #print(theta)
    #y = theta*x
    dumx = np.arange(int(max(x/10)))
    print(dumx,"dx")
    dumy = np.matmul(theta,np.append([np.ones_like(dumx)],[dumx],axis=0))
    print(dumy,"dy")
    xn,yn = df.columns[ch1],df.columns[ch2]
    fig,((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, sharex='col', sharey='row',figsize=(10, 10))
    ax3.scatter(x,y, color='red', marker='x',s=.1)
    nb = int(200)#df.shape[0]/100)
    X,Y=makeHist(y,nb,orientation="horizontal")
    ax4.plot(X,Y)
    X,Y=makeHist(x,nb)
    ax1.plot(X,Y)
    if input('include line of best fit? (y)') == 'y':
        ax3.plot(dumy)
    ax3.set_xlabel(xn+" level")
    ax3.set_ylabel(yn+" level")
    ax4.set_xlabel(yn+" number of cells (log10)")
    ax1.set_ylabel(xn+" number of cells (log10)")
    plt.show()
    return(df,obs,dfxy)

def rollingAve(l,n=8): #n bins in each direction
    newL = []
    for i in range(len(l)):
        nears = []
        for j in range(n):
            left = i - j - 1
            right = i + j + 1
            if  left >= 0:
                nears.append(l[left])
            if right < len(l):
                nears.append(l[right])
        #print(nears)
        newL.append(stat.mean(nears))
    return(newL)

def makeHist(x,nb,orientation="vertical"):#),log2=False):
    mx,Mx = min(x),max(x)
    rx = Mx-mx
    sx = rx/nb
    binCts=[]
    bins = np.arange(mx,Mx+sx,sx)
    for i in range(1,len(bins)):
        key = x>=bins[i-1]
        key1 = x<bins[i]
        ss = x.loc[key&key1]
        if ss.shape[0] > 1:
            binCts.append(np.log10(ss.shape[0])/np.log10(2))
        else:
            binCts.append(0)
    #y = rollingAve(np.arange(len(binCts))*sx)
    y = np.arange(len(binCts))*sx
    X = rollingAve(binCts)
    #print(X,y)
    if orientation == "vertical":
        return(y,X)
    else:
        return(X,y)

def scanpyv(df,obs,dfxy,cdf=None):
    sc, anndata = _load_scanpy_stack("Scanpy visuals")
    if sc is None:
        return(df,obs,dfxy)
    if input("z-score data vertically? (y)") == "y":
        df,obs,dfxy = zscorev(df,obs,dfxy)
    ch,uch = obMenu(obs,"sort x axis by:")
    gb = obs.columns[ch]
    adata = anndata.AnnData(df,obs = obs)
    sc.pl.dotplot(adata, adata.var_names, groupby=gb, show=True)
    sc.pl.stacked_violin(adata, adata.var_names, groupby=gb, show=True)
    sc.pl.matrixplot(adata, adata.var_names, groupby=gb, show=True,dendrogram=True )
    return(df,obs,dfxy)

def showUmap(df,obs,dfxy,cdf=None): #make it so there's an option to re-color without re-genning whole map
    sc, anndata = _load_scanpy_stack("UMAP visualization")
    if sc is None:
        return(df,obs,dfxy)
    mpl.style.use('default')
    #26obs = obs.astype(str)
    df = df.astype(float)
    obs = obs.astype(str)
    print(df.index)
    print(obs.index)
    if input("replace 'no' with np.nan in columns with no nan entry (to color gray) (y)") == 'y':
        for color in obs.columns:
            naStrs = ["no","No","Na","NA","NAN"]
            keys = []
            for ns in naStrs:
                keys.append(obs.loc[:,color] == ns)
            keys = pd.concat(keys,axis=1)
            key0 = keys.sum(axis=1)
            #print(key0,"key0",key0.max())
            if key0.max() == 1:
                print('replacing no with nan',color)
                key1 = key0 != 0
                obs.loc[key1,color] = np.nan
                #print(obs.loc[:,color])
    adata = anndata.AnnData(df,obs = obs)
    sc.pp.neighbors(adata,use_rep='X')
    sc.tl.umap(adata)
    plt.rcParams['figure.figsize'] = 8, 8
    while True:
        try:
            for i,clr in enumerate(obs.columns):
                print(i,":",clr)
            color = obs.columns[int(input("color by (send non-int to escape)"))]
            #obs[color] = obs[color].astype(str)
            print("0:show all or 1:select whic or 2:each unique value in own map?")
            ch = input("0, 1, or 2:")
            itm = []
            clr = []
            #shapes = []
            if ch == "1":
                uColors = sorted(list(obs[color].unique()))
                for c in uColors:
                    print("show? ",c)
                    try:
                        cho = input("0 for no, 1 for yes:")
                    except:
                        cho = "0"
                    if cho == "0":
                        #obs[color].loc[obs[color]==c] = float('nan')
                        itm.append(c)
                        clr.append('lightgray')
                        #shapes.append("s")
                    else:
                        itm.append(c)
                        clr.append(input("which color?"))
                        #shapes.append(input("which marker? . v p s * ><^ "))

            print("umap color:",color)
            cd = dict(zip(itm,clr))
            if ch=="1":
                sc.pl.umap(adata,color = color,palette=cd)#,na_color='lightgray'
            elif ch == "2":
                obs[color] = obs.loc[:,color].astype(str)
                uColors = sorted(list(obs[color].unique()))
                for i,clr in enumerate(uColors):
                    itm = []
                    clr = []
                    for j in range(len(uColors)):
                        itm.append(uColors[j])
                        if i == j:
                            clr.append("blue")
                        else:
                            clr.append("lightgray")
                    cd = dict(zip(itm,clr))
                    sc.pl.umap(adata,color = color,palette=cd)

            else:
               sc.pl.umap(adata,color = color,na_color='lightgray',palette="Set1")
            plt.show()
        except Exception as e:
            print(type(e),e)
            break
    ch = input("show umap of biomarkers? (y/n)")
    if ch == "":
        pass
    elif ch[0] == "y" or ch[0] == "Y":
        for biom in df.columns:
            sc.pl.umap(adata,color=biom,vmin=np.mean(df[biom])-np.std(df[biom]),vmax=np.mean(df[biom])+2*np.std(df[biom]),color_map='viridis')
            plt.show()
    return(df,obs,dfxy)


if __name__ == "__main__":
    #'''#"zzz_hta14_tumorneighborhoodcts1"
    folder = ''#r'C:\Users\youm\Desktop\src\BR MFC7 GL data pre 230808 pre vietnam'#r"C:\Users\youm\Desktop\src\zzzzzzzzzzz_current/"
    #stem = 'zz_test'#'196_MCF7'#'95_GL'#'95_GL'#'z3_GL631'#'zzz_hta14'#'hta14bx1_ck7'#'recent june 3 23/PIPELINE_hta14_bx1_with_svm'#'PIPELINE_94'#'PIPELINE_hta14_bx1_99'#'93_hta14_no_neigh'#'93_hta14'#87_LC-4'##'89_LC-4_withN'#''96_LC'#cl56_depth_study_H12'#'96_LC'#'97_mtma2'#'93_hta14'###'96_hta14_primary'#'97_hta14bx1_primary_celltype'#'99_hta14'#"temp"#"zzz_hta1499"#"zzz14bx1_97"#"hta14bx1 dgram"#folder+"14_both"##"tempHta14_200"#"HTA14f"#"zzzz_hta1498_neighborhoodsOnly"#"hta1415Baf1"#"HTA15f"#"0086 HTA14+15"#"99HTA14"#"z99_ROIs_5bx_HTA1415"#"temp"#"z99_ROIs_5bx_HTA1415"#<-this one has old celltyping no TN #"0084 HTA14+15" #"HTA9-14Bx1-7 only"#"0.93 TNP-TMA-28"#"0.94.2 TNP-TMA-28 primaries"#"1111 96 TNP-28" #'0093 HTA14+15'#"0094.7 manthreshsub primaries HTA14+15"#"0094 HTA14+15" #"096 2021-11-21 px only" #'095.08 primaries only manthreshsub 2021-11-21 px only'#"094 manthreshsub 2021-11-21 px only" #  '095.1 primaries only manthreshsub 2021-11-21 px only'#
    #stem= r'C:\Users\youm\Desktop\src\old data Feb 2 24/manual_7_B_1' #'z93_MCF7-7_long_obs'
    #stem = 'z03_W'#'2223_b_mw_1000_cbt'
    #stem = r'C:\Users\youm\Desktop\src\old data Feb 22 24/W22_98'
    #stem = 'MIT_01'
    #stem = r'D:\SMT_101_Hisham\other processed data (typed together and cosmx)/cosmx_00'
    stem = r'D:\WOO\W_5_tumonly_agg' #r'D:\WOO\WOO_CK19CK17'
    print("axis labels %s on barplots more ticks/lines")
    #stem = folder + "/" + stem
    print(stem)
    df = pd.read_csv(stem+"_df.csv",index_col=0).dropna(axis=1)
    print(df)
    #input()
    obs = pd.read_csv(stem+"_obs.csv",index_col=0).astype(str)
    obs.name = obs.columns[4]
    dfxy = pd.read_csv(stem+"_dfxy.csv",index_col=0)
    print(df.shape[0],"cells")
    #print(manual_type_tma-35_dfxy"\n\n THIS DATA IS ALREADY Z-SCORED \n\n")
    '''
    df = pd.read_csv("zz_C_L.1_AMTEC_df.csv",index_col=0)
    obs = pd.read_csv("zz_C_L.1_AMTEC_obs.csv",index_col=0)
    dfxy = pd.read_csv("zz_C_L.1_AMTEC_dfxy.csv",index_col=0)#zL.1_final_amtec_dfxy
    '''
    #sortedMap(df,obs,dfxy)
    #nearestOfType(df,obs,dfxy)
    #phenoToObs(df,obs,dfxy)
    #subChanThresh(df,obs,dfxy)
    #biomSortedMap(df,obs,dfxy)
    #boxplot(df,obs,dfxy,9)
    #graphST(df,obs,dfxy,9)
    #fractionTable(df,obs,dfxy)
    #volcanoPlot(df,obs,dfxy,9)
    main(df,obs,dfxy)
    #ort.spatial(df,obs,dfxy,1600,'Ki67_nuclei')

'''
tissue-to-tissue biomarker accessability varies
standard procedure for RNA-seq analysis (GAPDH gene)
normalization method
    housekeeping proteins      - every cell should have baseline level for basic cell homeostasis
        lamAC c750 (expression varies between tumor/nontumor)
        CD4 c647 (only immune cells, should be similar in all immune cells)
        CD31 647 (endothelial specific)

    continuous antigen   - expression level changes between cell activities/age
        CK7 c555 (PDAC)
        CK7 488  (Br)
        CK8 488
        CK5 488
        CK19 750
        aSMA 488
        Vim 488


cd31+ aSMA- CollIV- imature vessle/angeogenesis
all+ mature blood vessle


...
ck7 ck5 ki67 cd45
'''


'''
def a-utotype(df,obs,dfxy):
    odf = df.copy()
    if input("zscore?") == "y":
        df,obs,dfxy = zscorev(df,obs,dfxy)
    df = onlyPrimaries(df)
    df["tumor"] = 0
    df["stromal"] = 0
    df["immune"] = 0
    df["Ki67+"] = 0
    df["CD31+"] = 0
    df[" fibroblast"] = 0
    tums = 0
    stroms = 0
    ims = 0
    for biom in df.columns:
        ser = df[biom]
        #ser.loc[df[biom]<0] = 0
        if "CK" in biom or "Ecad" in biom or "Muc1" in biom:
            df["tumor"] += ser
            tums += .5
        if "CD" in biom and "31" not in biom and "44" not in biom:
            df["immune"] += ser
            ims += .5
        if "aSMA" in biom or "Vim" in biom or "Coll" in biom or "CAV1" in biom:
            df["stromal"] += ser
            stroms += .5
        if "CD31" in biom:
            df["CD31+"] += ser
        if "Ki67" in biom:
            df["Ki67+"] += ser
    df["tumor"] = df["tumor"]/tums
    df["immune"] = df["immune"]/ims
    df["stromal"] = df["stromal"]/stroms

    res = float(input("number of standard deviations (or raw intensity if not zscored) above mean required to count as +"))
    obs["autoCellType res: "+str(res)] = " "
    superB = ["CD31+","immune"," fibroblast","tumor","Ki67+"]#,"CD44+"]
    for sb in superB:
        key = df[sb]>res
        obs.loc[key,"autoCellType res: "+str(res)] += sb + " "
    if input("keep superbioms and z-scoring?") == "y":
        return(df,obs,dfxy)
    return(odf,obs,dfxy)



Hello,

I've made a few minor changes to improve readability, but I'm not sure how to fix a couple of the sentences.

Polo-like kinases (PLKs) regulate multiple intracellular processes, including DNA replication, mitosis, and stress response. PLK1, the most studied, is upregulated in many cancer types. High PLK1 correlates with unfavorable outcomes. While PLK inhibitors have had success in preclinical trials, their efficacy as monotherapy has been limited. Here, we analyzed the spatial response to a PLK1-inhibitor, Onvansertib, in two PDAC cases to address biological advantages with PLK1-inhibitor. The rate of tumor cell proliferation was greatly reduced in one patient and remained unchanged in the other. Comparing information from high-plex immunofluorescence assay, we measured dozens of protein expression every single cell at subcellular resolution and spatially analyzed changes therein between patients and biopsies. Unsupervised clustering on spatial information revealed several neighborhoods with distinct tumor and immune characteristics. Analyses of patient baselines and changes during treatment revealed distinctive patterns that provide insight into response and resistance mechanisms.

'Here, we analyzed the spatial response to a PLK1-inhibitor, Onvansertib, in two PDAC cases to address biological advantages with PLK1-inhibitor.' -> Here, we analyzed spatial characteristics of tissue biopsies taken before and during treatment in order to identify factors that help explain the differential response.  ?

Comparing information from high-plex immunofluorescence assay, we measured dozens of protein expression every single cell at subcellular resolution and spatially analyzed changes therein between patients and biopsies. -> We measured each cell's expression of 15 proteins in their appropriate sub-cellular location and analyzed changes in tissue composition and spatial dynamics. (but then the nest sentence begins talking about the specific spatial analysis, making the last clause of this sentence redundant).

I think the version I sent at 9 is easier to read and conveys mostly the same information. The points you want included are:

PLKs are involved in DNA replication and stress response
Here's the start to my introduction:
Polo-like kinases are proteins that facilitate mitotic cell division and are most highly expressed at the beginning of the M phase.  They activate regulators of mitotic entry, chromosome and centrosome separation, and the formation of the contractile ring which causes the cytoplasm to split.  PLKs also regulate DNA damage checkpoints, interrupting DNA-damage repair pathways during cell division.  The N-terminal of a PLK is responsible for activating proteins by transferring phosphate groups from ATP.  The C-terminal’s role is to recognize proteins.
I removed some detail from the abstract (to save words) because anyone who wants to know more will see more in the introduction.
(references: Zheng Zhang, Rosie Elizabeth Ann Gutteridge, Gjertsen)

PLK-1 is the most highly studied

PLK1 regulates mitosis

The IF assay is high-plex with subcellular resolution

I can include this information with less modification to scentence structure:

PLK-1, the most studied Polo-Like Kinase, .is seen in many cancers. and while PLK inhibitors have had success in preclinical trials, their efficacy as monotherapy has been limited. Here, we analyze the response to the PLK1-inhibitor drug Onvansertib in two PDAC cases. In one patient, the rate of cell proliferation was greatly reduced, while in the other, proliferation was not altered. We refer to these patients as the responding patient (RP) and non-responding patient (NRP) respectively, though this refers only to rates of cell proliferation and not to patient outcome. Comparing information from multiple assays with detailed immunofluorescence information, we measure levels of protein expression and analyze changes therein between patients and biopsies. Unsupervised clustering on spatial information reveals several neighborhoods with distinct tumor and immune characteristics. Analyses of patient baselines and changes during treatment reveal patterns that provide insight into response and resistance mechanisms.




'''
