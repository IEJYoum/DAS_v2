# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 03:25:53 2021

@author: youm
"""

import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
import copy
import math
import seaborn as sns
#import phenograph
import scipy
from sklearn.metrics import adjusted_rand_score
#import re
import statistics as stat



FOLDER = r"C:\Users\youm\Desktop\WO data\714WO.csv"#"C:/Users/youm/Desktop/WO data/excel/394_allCols.csv"

FILE = "" #features_WO-394_FilteredMeanIntensity_DAPI7_DAPI11.csv"


manualThresh = {'53BP1_nuclei': 2750,'CAV1_perinuc5': 2000, 'CC3_nucadj2': 2250,'CD20_perinuc5': 2000, 'CD31_perinuc5': 2000, 'CD3_perinuc5': 1500,
                'CD44_perinuc5': 1500, 'CD45_perinuc5': 1000, 'CD4_perinuc5': 2000, 'CD68_perinuc5': 3500, 'CD8_perinuc5': 5000,'CK14_cytoplasm': 1000,
                'CK17_cytoplasm': 1500, 'CK19_cytoplasm': 2000,'CSF1R_perinuc5': 3000, 'ColIV_perinuc5': 4000, 'ColI_perinuc5': 6000, 'CoxIV_perinuc5': 1500,
                'EGFR_perinuc5': 2500, 'Ecad_exp5nucmembrane25': 2000, 'FoxP3_nuclei': 3000, 'GRNZB_nuclei': 1500,'H3K27_nuclei': 1250, 'H3K4_nuclei': 2250,
                'Ki67_nuclei': 1500,'LamAC_nuclei': 1250,'MSH6_nuclei': 1500, 'MUC1_cytoplasm': 1750, 'PCNA_nuclei': 2000, 'PD1_perinuc5': 3250,
                'PDL1_perinuc5': 3000, 'RAD51_nuclei': 2500, 'SYP_perinuc5': 2000, 'TFF1_perinuc5': 1750,'CK8_cytoplasm': 1000,'aSMA_perinuc5': 2000,
                'cPARP_nuclei': 1500,'gH2AX_nuclei': 1500,'p63_nuclei': 1750, 'pAKT_perinuc5': 2000, 'pERK_nuclei': 2600,
                'pHH3_nuclei': 2000, 'pS6RP_perinuc5': 2500 }

NAMES = []#"tumor","immune","stromal","endothelial","cycling","signalling","DNA damage","apoptosis"]

SHOWPLOTS = True

def primary(df,obs,dfxy,name="all", allBioms = "ask"):
    #main function in cmif50 is tertiary
    global NAMES
    NAMES.clear()
    #df = maxPrimaries(df)
    if type(allBioms) != bool:
        if input("threshold only primary biomarkers? (y)") == "y":
            allBioms = False
        else:
            allBioms = True

    for biom in df.columns:
        if "Ki67" in biom:
            NAMES.append(biom)
    for biom in df.columns:
        if "CD31" in biom:
            NAMES.append(biom)
    for biom in df.columns:
        if allBioms or "CD" in biom or "vim" in biom or "Vim" in biom or "VIM"  in biom or "aSMA" in biom or"CK" in biom in biom or "Ecad" in biom or "MUC1" in biom or "CAV1" in biom or "EGFR" in biom or "HER2" in biom:
            if biom not in NAMES:
                if allBioms or "44" not in biom:
                    NAMES.append(biom)
    #print(NAMES,"NAMES0")
    df,obs = scatterThresh(df,obs,dfxy,name=name)
    if input("keep only primary biomarkers? (y)") == "y":
        df = df.loc[:,NAMES]
    return(df,obs)


def secondary(df,obs,dfxy):
    NAMES.clear()
    for i,ob in enumerate(obs.columns):
        print(i,ob)
    obs['Manual Celltype'] = ''
    print("Find orthogonals based on each (slide is recommended)")
    ch = int(input("number:"))
    clr = obs.columns[ch]
    ucolors = obs[clr].unique()
    for name in NAMES:
        df[name] = ""
    for ucol in ucolors:
        print(ucol)
        key = obs.loc[:,clr] == ucol
        smalldf = df.loc[key,:]
        smalldf,smallObs = primary(smalldf,obs.loc[key,:],name=ucol)
        df.loc[key,:] = smalldf
        obs.loc[key,:] = smallObs
    return(df,obs)


def tertiary(df,obs,dfxy):
    global SHOWPLOTS
    NAMES.clear()
    for bm in sorted(list(df.columns)):
        #print('WARNING: TEMPORARILY DEFAULTING TO CD31 Ki67 FIND ORTHOGONAL7.py def tir to fix')
        goodSts = []#['CK5','Vim','CD44']

        if len(goodSts) == 0:
            if input("include "+bm+" ? (y)") == 'y':
                NAMES.append(bm)
        else:
            if any(st in bm for st in goodSts):
                NAMES.append(bm)

    #ndf = df.loc[:,incl]
    sp = input('show plots? (y)')
    SHOWPLOTS = False
    if sp == 'y':
        SHOWPLOTS = True
    df,obs=scatterThresh(df,obs,dfxy, SHOWPLOTS = SHOWPLOTS)
    name = input("name of celltype annotations")
    if name == '':
        return(df,obs)
    obs[name] = ""
    for uo in obs["Manual Celltype"].unique():
        print(uo)
        typ = input("celltype?")
        obs.loc[obs["Manual Celltype"]==uo,name] = typ
    return(df,obs)



def scatterThresh(df,obs,dfxy,name="all", SD = False, SHOWPLOTS = False):
    odf = df.copy()
    names = NAMES#names are markers  old: ["tumor","immune","stromal","endothelial","cycling","signalling","DNA damage","apoptosis"]
    print(names,"NAMES1")
    thresholds = []
    while len(names) < 3:
        names.append(names[0])
    for i,n1 in enumerate(names):
        if i == 0:
            n2 = names[1]
            n3 = names[2]
        elif i == 1:
            n2 = names[0]
            n3 = names[2]
        else:
            n2 = names[1]
            n3 = names[0]
        stdev=stat.stdev(df.loc[:,n1])+stat.mean(df.loc[:,n1])
        iles = [.75,.33,.66,.5,.95,.96,.97,.98,.99]
        print(iles)
        quants = np.quantile(df.loc[:,n1],iles)
        print(quants,"\n quantiles corresponding to the above list")
        quart = quants[0]
        x,y = trimExtremes(df.loc[:,n1],df.loc[:,n2])
        x3,y3 = trimExtremes(df.loc[:,n1],df.loc[:,n3])
        xn,yn = n1,n2
        xn3,yn3 = n1,n3
        if SHOWPLOTS:
            showPlot(x3,y3,xn3,yn3,name=name,stdev=stdev,quart=quart)
            showPlot(x,y,xn,yn,name=name,stdev=stdev,quart=quart)
        while True:
            try:
                if SD:
                    thresh = stdev
                else:
                    print(stdev,"mean+stdev\n",quart,".75th quantile\n",stat.mean(df.loc[:,n1]),"mean")
                    thresh = float(eval(input("threshold for "+n1+": ")))
            except:
                thresh = 0
            if not SD:
                showPlot(x,y,xn,yn,vline=thresh,name=name,stdev=stdev,quart=quart)
            try:
                spatial(df,obs,dfxy,thresh,n1)
            except Exception as e:
                print(e,'\n could not show spatial')
            try:
                if SD:
                    1/0
                an=int(input("done? 0:yes, 1:no :"))
                1/an
            except:
                break

        thresholds.append(thresh)
        #thresholds.append(np.mean(thresh))
    switch = 0

    obs["Manual Celltype"] = ""
    #"CD31+","immune","active fibroblast","tumor","Ki67+"
    for i,biom in enumerate(names):
        key = df[biom] > thresholds[i]
        obs.loc[key,"Manual Celltype"] += biom #+ "5: Support Fibroblast"
    if input("subtract thresholds? (y)") == "y":
        for i,n in enumerate(names):
            df[n] -= thresholds[i]
        switch = 1

    print(obs.columns)

    if switch == 1:
        df.values[df.values<0] = 0
        return(df,obs)
    else:
        return(odf,obs)


def spatial(df,obs,dfxy,thresh,marker):
    for scene in obs.loc[:,'slide_scene'].unique():
        sdf = df.loc[obs.loc[:,'slide_scene'] == scene,:]
        sxy = dfxy.loc[obs.loc[:,'slide_scene'] == scene,:]

        fig,ax = plt.subplots()
        key = sdf.loc[:,marker] < thresh
        color = 'lightgray'
        for i in range(2):
            ax.scatter(sxy.iloc[:,0].loc[key],-sxy.iloc[:,1].loc[key],color = color,s=1.2)
            key = ~key
            color = 'darkred'
        plt.title(scene+' '+marker)
        plt.show()



def showPlot(x,y,xn,yn,vline=None,name="all",stdev=None,quart=None):
    fig,((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, sharex='col', sharey='row',figsize=(10, 10))
    #ax3.scatter(x,y)
    ax3.plot(x, y, color='red', marker='x', linestyle='none', markersize=1.5)
    nb = int(200)#df.shape[0]/100)
    X,Y=makeHist(y,nb,orientation="horizontal")
    ax4.plot(X,Y)
    X,Y=makeHist(x,nb)
    ax1.plot(X,Y)
    if vline:
        ax3.vlines(x=vline,ymin=min(y),ymax=max(y),color="b")
        ax1.vlines(x=vline,ymin=0,ymax=max(Y),color="r")
    if stdev:
        ax3.vlines(x=stdev,ymin=min(y),ymax=max(y),color="cyan")
        ax1.vlines(x=stdev,ymin=0,ymax=max(Y),color="cyan")
    if quart:
        ax3.vlines(x=quart,ymin=min(y),ymax=max(y),color="green")
        ax1.vlines(x=quart,ymin=0,ymax=max(Y),color="green")
    ax3.set_xlabel(xn+" level")
    ax3.set_ylabel(yn+" level")
    ax4.set_xlabel(yn+" number of cells")
    ax1.set_ylabel(xn+" number of cells")
    fig.suptitle(name)
    plt.show()



def trimExtremes(s1,s2,quantile=.99):
    for ser in [s1,s2]:
        sS = ser.sort_values()
        nInd = sS.shape[0]
        cutoff = sS.iloc[int(nInd * quantile)]
        print(cutoff,"!")
        key = ser < cutoff
        s1 = s1.loc[key]
        s2 = s2.loc[key]
    return(s1,s2)



def makeHist(x,nb,orientation="vertical",log2=False):
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
            binCts.append(np.log10(ss.shape[0]))
        else:
            binCts.append(0)
    y = np.arange(len(binCts))*sx+mx
    X = binCts
    if orientation == "vertical":
        return(y,X)
    else:
        return(X,y)



def maxPrimaries(df):
    tumors = []
    immunes = []
    stromals = []
    cds = []
    kis = []
    sigs = []
    dnas = []
    apos = []
    for biom in df.columns:
        if "CK" in biom or "Ecad" in biom:
            tumors.append(biom)
        elif "CD31" in biom:
            cds.append(biom)
        elif "CD" in biom and "31" not in biom:
            immunes.append(biom)
        elif "Vim" in biom or "aSMA" in biom:
            stromals.append(biom)
        elif "Ki67" in biom or "PCNA" in biom or "pHH3" in biom:
            kis.append(biom)
        elif "CC3" in biom or "cPARP" in biom:
            apos.append(biom)
        elif "53BP1" in biom or "MSH6" in biom or "RAD51" in biom or "gH2AX" in biom:
            dnas.append(biom)
        elif "H3K4" in biom or "pAKT" in biom or "pERK" in biom or "pS6RP" in biom:
            sigs.append(biom)
        alls = [tumors,immunes,stromals,cds,kis,sigs,dnas,apos]
        names = NAMES#["tumor","immune","stromal","endothelial","cycling","signalling","DNA damage","apoptosis"]
    for i,typ in enumerate(alls):
        smallA = df.loc[:,typ].values
        maxes = smallA.max(axis=1)
        df[names[i]] = maxes
    return(df)
'''
COLS = ["CK1","CK2","CD","CD31","Vim","aSMA","d"]
values = [[1,2,3,4,5,6,7],[7,6,5,4,3,2,1]]
DF = pd.DataFrame(data=values,columns = COLS)
primary(DF)
'''

def main():
    df = pd.read_csv(FOLDER+FILE,index_col=0)
    df = df.fillna(0)
    df = dropOthers(df)
    columns = df.columns.values
    values = df.values[:,:-1]
    values = np.array(values,dtype = float)

    orthogonals = findOrthogonals(values,columns,show=False)
    levels = []
    names = []
    for pair in orthogonals:
        t0,t1 = histPlot(values[:,pair[0]],values[:,pair[1]],[columns[pair[0]],columns[pair[1]]])
        levels.append(t0)
        levels.append(t1)
        names.append(columns[pair[0]])
        names.append(columns[pair[1]])
    return()
    thresholds = findThresholds(values,columns,orthogonals)



def histPlot(x,y,names,t0=0,t1=0):
    fig,((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, sharex='col', sharey='row')
    ax3.scatter(x,y)
    #x,y=np.log2(x),np.log2(y) -- bad, need to log2 the height of the bars isntead
    ax4.hist(y,400,orientation="horizontal")
    ax1.hist(x,400)
    ax3.set_xlabel(names[0]+" level")
    ax3.set_ylabel(names[1]+" level")
    ax4.set_xlabel(names[1]+" number of cells")
    ax1.set_ylabel(names[0]+" number of cells")
    ax3.hlines(y=t1,xmin=min(x),xmax=max(x),color="r")
    ax3.vlines(x=t0,ymin=min(y),ymax=max(y),color="r")
    ax1.vlines(x=t0,ymin=0,ymax=ax1.get_ylim()[1],color="r")
    ax4.hlines(y=t1,xmin=0,xmax=ax4.get_xlim()[1],color="r")
    plt.show()
    while True:
        try:
            print("\nhit enter twice once thresholds are final")
            t0 = float(input("threshold for "+names[0]+":"))
            t1 = float(input("threshold for "+names[1]+":"))
            histPlot(x,y,names,t0,t1)
        except:
            return(t0,t1)



def main2(DF,obs,show=True): #manual, always shows
    for i,ob in enumerate(obs.columns):
        print(i,ob)
    print("Find orthogonals based on each (slide is recommended)")
    ch = int(input("number:"))
    clr = obs.columns[ch]
    ucols = obs[clr].unique()
    DF = tryDrop(DF,["nuclei_eccentricity"])
    for uc in ucols:
        key = obs[clr] == uc
        df = DF.loc[key,:]
        #df = tryDrop(df,["nuclei_eccentricity"]) causes bug
        columns = df.columns.values
        values = np.array(df.values,dtype = float)
        #print(values)
        orthogonals = findOrthogonals(values,columns,show=show)
        levels = []
        names = []
        for pair in orthogonals:
            t0,t1 = histPlot(values[:,pair[0]],values[:,pair[1]],[columns[pair[0]],columns[pair[1]]])
            levels.append(t0)
            levels.append(t1)
            names.append(columns[pair[0]])
            names.append(columns[pair[1]])
        df = pd.DataFrame(data=values,columns=columns,index=df.index)
        done = []
        for i,name in enumerate(names):
            if name not in done:
                df[name] -= levels[i]
                done.append(name)
        print(done)
        df.values[df.values<0] = 0
        DF.loc[key,:] = df
    return(DF)



def main1(DF,obs,show=False): #automatic
    for i,ob in enumerate(obs.columns):
        print(i,ob)
    print("Find orthogonals based on each (slide is recommended)")
    ch = int(input("number:"))
    clr = obs.columns[ch]
    ucols = obs[clr].unique()
    DF = tryDrop(DF,["nuclei_eccentricity"])
    for uc in ucols:
        key = obs[clr] == uc
        df = DF.loc[key,:]
        #df = tryDrop(df,["nuclei_eccentricity"]) causes bug
        columns = df.columns.values
        values = np.array(df.values,dtype = float)
        #print(values)
        orthogonals = findOrthogonals(values,columns,show=show)
        thresholds = findThresholds(values,columns,orthogonals,show=show)
        print(thresholds)
        for i,lvl in enumerate(thresholds):
            values[:,i] -= lvl
        df = pd.DataFrame(data=values,columns=columns,index=df.index)
        df.values[df.values<0] = 0
        DF.loc[key,:] = df
    return(DF)



def findThresholds(values,columns,orthogonals,show=True):
    thresholds = []
    for pair in orthogonals:
        toThreshold = values[:,pair[0]]
        orth = values[:,pair[1]]
        rang = max(orth) - min(orth)
        space = 9999999999999
        threshLvl = max(toThreshold)
        while space > rang * .4 and threshLvl > 10:
            threshLvl = threshLvl * .98
            higher = []
            i = 0
            for point in toThreshold:
                if point > threshLvl:
                    higher.append(orth[i])
                i += 1
            space = max(orth) - max(higher)
        print(columns[pair[0]],threshLvl)
        thresholds.append(threshLvl)
        if show:
            plt.plot(values[:,pair[0]],'rx')
            plt.plot(np.ones(len(values[:,pair[0]]))*threshLvl)
            plt.ylabel(columns[pair[0]])
            plt.show()
    return(thresholds)





def findOrthogonals(values,columns,show=True):
    shape = values.shape
    orthogonals = []
    for i in range(shape[1]):
        scores = []
        for j in range(shape[1]):
            cov = getCov(values[:,i],values[:,j])
            scores.append(cov)
        orthInd = scores.index(min(scores))
        print("orthogonal genes:",columns[i],columns[orthInd])
        if show:
            plt.scatter(values[:,i],values[:,orthInd])
            plt.xlabel(columns[i])
            plt.ylabel(columns[orthInd])
            plt.show()
        orthogonals.append([i,orthInd])
    return(orthogonals)


def getCov(a,b):
    a = scipy.stats.zscore(a)
    b = scipy.stats.zscore(b)
    cov = np.dot(a,b)
    return(cov)

def dropOthers(df):
    dropped = []
    for col in df.columns:
        if col not in manualThresh.keys():
            dropped.append(col)
    for d in dropped:
        df = df.drop(d,axis=1)
    print(dropped,"dropped")
    return(df)


def tryDrop(df,dropList):
    for colName in dropList:
        try:
            df = df.drop([colName],axis = 1)
        except:
            print(colName,'not in dataframe')
    return(df)

if __name__ == "__main__":
    #main()
    pass



'''
def scatterThresh1(df):
    names = NAMES#["tumor","immune","stromal","endothelial","cycling","signalling","DNA damage","apoptosis"]
    thresholds = []
    for i,n1 in enumerate(names):
        thresh = []
        for j,n2 in enumerate(names):
            if n1 != n2:
                x,y = df.loc[:,n1],df.loc[:,n2]

                xn,yn = n1,n2
                fig,((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, sharex='col', sharey='row',figsize=(10, 10))
                ax3.scatter(x,y)
                nb = 1000
                X,Y=makeHist(y,nb,orientation="horizontal")
                ax4.plot(X,Y)
                X,Y=makeHist(x,nb)
                ax1.plot(X,Y)
                ax3.set_xlabel(xn+" level")
                ax3.set_ylabel(yn+" level")
                ax4.set_xlabel(yn+" number of cells")
                ax1.set_ylabel(xn+" number of cells")
                plt.show()
                while True:
                    try:
                        thresh.append(float(input("threshold for "+n1+": ")))
                        break
                    except:
                        print("invalid numrical input")
        thresholds.append(np.mean(thresh))
    for i,n in enumerate(names):
        df[n] -= thresholds[i]
        df.values[df.values<0] = 0
    return(df)
'''









