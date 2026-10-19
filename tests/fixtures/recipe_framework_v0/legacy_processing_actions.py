"""Frozen pre-port actions, TEST ONLY; never used for runtime compatibility."""

def regionAverage(dfs,com=[],cat=''): #make only check same slidescene
    if len(com) == 0:
        while True:
            try:
                radius = float(input("radius (um) to consider in average: "))/PXSIZE
                return([],[radius])
            except Exception:
                print("invalid radius, send number")
    dfs,n = ifv.autoClean(dfs,['n'])
    df,obs,dfxy = dfs[0],dfs[1],dfs[2]
    radius = com[1]
    n_cols = df.shape[1]
    col_names = df.columns
    all_indices = []
    all_avgs = []
    for us in tqdm(list(obs["slide_scene"].unique()), desc="regionAverage"):
        key0 = obs["slide_scene"] == us
        tdfxy = dfxy.loc[key0,:]
        tdf = df.loc[key0,:]
        coords = tdfxy.values.astype(float)
        tdf_vals = tdf.values.astype(float)
        tree = cKDTree(coords)
        neighbor_lists = tree.query_ball_point(coords, radius)
        indices = tdfxy.index
        for i in range(len(neighbor_lists)):
            nlist = [j for j in neighbor_lists[i] if j != i]
            if len(nlist) > 1:
                all_avgs.append(tdf_vals[nlist].mean(axis=0))
            else:
                all_avgs.append(np.full(n_cols, np.nan))
            all_indices.append(indices[i])
    ndf = pd.DataFrame(np.array(all_avgs), index=all_indices, columns=col_names)
    print("ndf start",ndf,"ndf end")
    for biom in ndf.columns:
        print(biom,"biom")
        ndf.loc[:,biom]=ndf.loc[:,biom].fillna(0)
    ndf = ndf.set_axis(pd.Series(ndf.columns)+" in radius "+str(radius*PXSIZE),axis=1)
    df= pd.concat([df,ndf],axis=1)
    return([df,obs,dfxy],[])

def neighborhoodCount(dfs,com=[],cat=''):
    df,obs,dfxy = dfs[0],dfs[1],dfs[2]
    if len(com) == 0:
        tot = input("count totals instead of fractions? (y)")
        radii = []
        while True:
            try:
                radii.append(float(input("radius (in um) to consider neighbors: (send blank when done)"))/PXSIZE)
            except Exception:
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
            goodsts.append(inp)
        min_neighbors = 0
        if tot != 'y':
            try:
                mn = input("minimum neighbors to compute fractions (blank for 0): ").strip()
                if mn != '':
                    min_neighbors = int(mn)
            except Exception:
                pass
        return([], [tot, radii, ch, goodsts, min_neighbors])
    obs = obs.astype(str)
    tot = com[1]
    radii = list(com[2])
    ch = int(com[3])
    goodsts = list(com[4])
    min_neighbors = int(com[5]) if len(com) > 5 else 0
    uch = list(obs.iloc[:,ch].unique())
    obcol = obs.columns[ch]
    if len(goodsts) == 0:
        goodsts = uch
    for radius in radii:
        col_names = [uc+"_"+obcol+"_neighbors_"+str(radius*PXSIZE) for uc in uch]
        for cn in col_names:
            df[cn] = 0
        for us in tqdm(list(obs["slide_scene"].unique()), desc="neighborhoodCount"):
            key0 = obs["slide_scene"] == us
            tdfxy = dfxy.loc[key0,:]
            tobs = obs.loc[key0,:]
            coords = tdfxy.values.astype(float)
            # vectorized centroid mask
            center_mask = pd.Series(False, index=tobs.index)
            for gs in goodsts:
                center_mask |= tobs[obcol].str.contains(gs, regex=False)
            center_positions = np.where(center_mask.values)[0]
            if len(center_positions) == 0:
                continue
            tree = cKDTree(coords)
            neighbor_lists = tree.query_ball_point(coords[center_positions], radius)
            type_codes = np.array(pd.Categorical(tobs[obcol], categories=uch).codes)
            n_types = len(uch)
            for ci, pos_i in enumerate(center_positions):
                ind = tobs.index[pos_i]
                nlist = [j for j in neighbor_lists[ci] if j != pos_i]
                nnei = len(nlist)
                if nnei > 1:
                    counts = np.bincount(type_codes[nlist], minlength=n_types)
                    if tot == 'y':
                        for ti, cn in enumerate(col_names):
                            df.loc[ind, cn] = int(counts[ti])
                    elif nnei < min_neighbors:
                        pass  # leave as 0
                    else:
                        for ti, cn in enumerate(col_names):
                            df.loc[ind, cn] = counts[ti] / nnei
    return([df,obs,dfxy],[])

def neighborhoodEntropy(dfs,com=[],cat=''):
    df,obs,dfxy = dfs[0],dfs[1],dfs[2]
    if len(com) == 0:
        radii = []
        while True:
            try:
                radii.append(float(input("radius (in um) to consider neighbors: (send blank when done)"))/PXSIZE)
            except Exception:
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
            goodsts.append(inp)
        return([], [radii, ch, goodsts])
    obs = obs.astype(str)
    radii = list(com[1])
    ch = int(com[2])
    goodsts = list(com[3])
    uch = list(obs.iloc[:,ch].unique())
    obcol = obs.columns[ch]
    if len(goodsts) == 0:
        goodsts = uch
    for radius in radii:
        ecol = obcol+"_entropy_"+str(radius*PXSIZE)
        df[ecol] = 0
        for us in tqdm(sorted(list(obs["slide_scene"].unique())), desc="entropy"):
            key0 = obs["slide_scene"] == us
            tdfxy = dfxy.loc[key0,:]
            tobs = obs.loc[key0,:]
            coords = tdfxy.values.astype(float)
            # vectorized centroid mask
            center_mask = pd.Series(False, index=tobs.index)
            for gs in goodsts:
                center_mask |= tobs[obcol].str.contains(gs, regex=False)
            center_positions = np.where(center_mask.values)[0]
            if len(center_positions) == 0:
                continue
            tree = cKDTree(coords)
            neighbor_lists = tree.query_ball_point(coords[center_positions], radius)
            type_codes = np.array(pd.Categorical(tobs[obcol], categories=uch).codes)
            n_types = len(uch)
            for ci, pos_i in enumerate(center_positions):
                ind = tobs.index[pos_i]
                nlist = [j for j in neighbor_lists[ci] if j != pos_i]
                nnei = len(nlist)
                entropy = 0
                if nnei > 1:
                    counts = np.bincount(type_codes[nlist], minlength=n_types)
                    fracs = counts / nnei
                    fracs = fracs[fracs > 0]
                    entropy = -np.sum(fracs * np.log2(fracs))
                df.loc[ind,ecol] = entropy
    return([df,obs,dfxy],[])

def nearestOfType(dfs,com=[],cat=''):
    df,obs,dfxy = dfs[0],dfs[1],dfs[2]
    if len(com) == 0:
        chs = []
        tys = []
        while True:
            try:
                ch,uch = obMenu(obs,title="category containing celltype to measure distance from (send non int to escape)")
            except Exception:
                break
            types = []
            for i,uc in enumerate(uch):
                print(i,":",uc)
            while True:
                try:
                    print("send non-int to escape")
                    ich = int(input("number of celltype to measure distance from:"))
                    ty = uch[ich]
                    types.append(str(ty))
                except Exception:
                    if len(types) == 0:
                        print('send integer corresponding to celltype in above list')
                    else:
                        break
            if len(types) > 0:
                chs.append(ch)
                tys.append(types)
        return([], [chs, tys])
    obs = obs.astype(str)
    chs = list(com[1])
    tys = list(com[2])
    for ii in range(len(chs)):
        ch = int(chs[ii])
        types = list(tys[ii])
        for ty in types:
            tytle = 'nearest '+obs.columns[ch]+': '+ty
            df[tytle] = 9999
            for us in tqdm(sorted(list(obs["slide_scene"].unique())), desc="nearest "+ty):
                key0 = obs["slide_scene"] == us
                tdfxy = dfxy.loc[key0,:]
                tobs = obs.loc[key0,:]
                key1 = tobs.iloc[:,ch] == ty
                tbxy = tdfxy.loc[key1,:]
                if tbxy.shape[0] == 0:
                    continue
                target_coords = tbxy.values.astype(float)
                all_coords = tdfxy.values.astype(float)
                tree = cKDTree(target_coords)
                dists, _ = tree.query(all_coords, k=1)
                df.loc[tdfxy.index, tytle] = dists * PXSIZE
    return([df,obs,dfxy],[])

'''
scaling
'''

def stretch(dfs,com=[],cat=''):
    if len(com) == 0:
        return([],[])
    df = dfs[0]
    for col in df.columns:
        nin = np.quantile(df.loc[:,col],.999)
        df.loc[:,col] = df.loc[:,col]/nin
    df = np.clip(df,0,1)
    return([df,dfs[1],dfs[2]],[])

def multCols(dfs,com=[],cat=''):
    df,obs,dfxy = dfs[0],dfs[1],dfs[2]
    if len(com) == 0:
        for i,col in enumerate(df.columns):
            print(i,":",col)
        c1 = int(logInput("number: "))
        c2 = int(logInput("number: "))
        return([], [c1, c2])
    c1 = int(com[1])
    c2 = int(com[2])
    cn1 = df.columns[c1]
    cn2 = df.columns[c2]
    print('warning: if z-scored, will also show mutual non-expression')
    df[cn1+' x '+cn2] = df.iloc[:,c1] * df.iloc[:,c2]
    return([df,obs,dfxy],[])

def biomRatio(dfs,com=[],cat=''):
    df,obs,dfxy = dfs[0],dfs[1],dfs[2]
    if len(com) == 0:
        for i,col in enumerate(df.columns):
            print(i,":",col)
        c1 = int(logInput("number: "))
        c2 = int(logInput("number: "))
        return([], [c1, c2])
    c1 = int(com[1])
    c2 = int(com[2])
    cn1 = df.columns[c1]
    cn2 = df.columns[c2]
    df[cn1+"/"+cn2] = 0
    df[cn1+"/"+cn2] = df[cn1]/df[cn2]
    return([df,obs,dfxy],[])

def remNegatives(dfs,com=[],cat=''):
    if len(com) == 0:
        return([], [])
    df = dfs[0].apply(pd.to_numeric, errors='coerce').copy()
    for col in df.columns:
        try:
            mn = float(df[col].min())
        except Exception:
            continue
        if mn < 0:
            df[col] -= mn
    return([df, dfs[1], dfs[2]], [])

