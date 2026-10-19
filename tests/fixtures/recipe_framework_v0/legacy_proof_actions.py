"""Frozen pre-port proof functions; test-only numerical and renderer-input baseline."""

import numpy as np
import pandas as pd

def zscore(dfs, com=[], cat=''):
    if len(com) == 0:
        ow = input('overwrite? (y)')
        return ([], [ow])

    ow = com[1]
    dfs[0] = dfs[0].apply(pd.to_numeric, errors='coerce')

    if ow == 'y':
        dfs[0] = zsc2(dfs[0])
    else:
        z = zsc2(dfs[0])
        z = z.rename(columns={c: c + '_zscored' for c in z.columns})
        dfs[0] = pd.concat([dfs[0], z], axis=1)

    return (dfs, [])


def zsc2(num):
    mu = num.mean(axis=0, skipna=True)
    sd = num.std(axis=0, skipna=True, ddof=0)  # ddof=0 matches common zscore convention

    # all-NaN columns => sd will be NaN; constant columns => sd==0
    z = (num - mu) / sd

    # constant columns: set zscore to 0 (instead of inf/NaN)
    z.loc[:, sd == 0] = 0

    # all-NaN columns remain all NaN (fine)
    return(z)



'''
selection and annotation
'''


def boxplot(dfs,com=[],cat=''):
    df,obs = dfs[0].copy(),dfs[1].copy()
    if len(com) == 0:
        vich = logInput("violin plot instead? (y): ")
        fich = logInput("show outliers? (y)")
        print(r'C:\Users\youm\Desktop\src\maxey matrices\markers_to_show.csv\nonly works with matrix type- for autotype, change column name in boxh')
        mich = logInput('show only relevant celltypes, as outlined in the file specified above? (y)')
        colors = getCats(obs,title='category to color by')
        try:
            ncols = int(logInput('boxplots per row in fig'))
        except:
            ncols = 6


        return([],[colors,vich,fich,ncols,mich])


    colors,vich,fich,ncols,mich = com[1],com[2],com[3],com[4],com[5]
    ch = list(obs.columns).index(cat)
    for color in sorted(colors):
        if color not in obs.columns:
            print('skipping',color)
            continue
        try:
            binCol = obs.columns[ch].astype(int)
        except:
            binCol = obs.columns[ch]
        try:
            obs[binCol+' temp'] = obs[binCol].apply(lambda n: n.split('_')[0])
            flt = False
            if '.' in obs[binCol+' temp'].iloc[0]:
                flt = True
            if not flt:
                obs[binCol+' temp']=obs[binCol+' temp'].astype(int)
                print('int type bins')
            else:
                obs[binCol+' temp']=obs[binCol+' temp'].astype(float)
                print('float type bins')

        except Exception as e:
            print('str type bins',e)
            obs[binCol+' temp'] = obs[binCol]+'!'


        obs = obs.sort_values(binCol+' temp',axis=0) #sorting changes index order
        obs[binCol]=obs[binCol].astype(str)
        colCol = obs[color].sort_values()
        title = cat+' x '+color

        boxH(df,obs,binCol,colCol,vich,fich,mich,title=title,corr=False,ncols=ncols)
    return()#9,[dfs]) #also doesn't need to return anything... temps are getting returned somehow, maybe axis messed up..


def getCats(obs,title='',required = True, typ = 'abc'):
    cols = []
    for i,ob in enumerate(obs.columns):
        print(i,":",ob)
    print('\n'+title+'\n')
    while True:
        imp = logInput("category number or range() to add (x when done): ")
        if not required and imp.strip().lower() in {'', 'x', 'q', 'done', 'quit'}:
            print(cols)
            break
        try:
            imp = eval(imp)
            try:
                for im in imp:
                    cols.append(obs.columns[im])
            except:
                cols.append(obs.columns[imp])
        except Exception as e:
            print(cols)
            if required:
                if len(cols) > 0:
                    break
            else:
                i2 = logInput(str(e)+"\ndone? (y/''):")
                if i2=="y" or i2 == '':
                    break
    nc = []
    print(type(typ))
    if type(typ) == int:
        print('returning col inds')
        for col in cols:
            nc.append(list(obs.columns).index(col))
        print(nc)
        return(nc)
    return(cols)

