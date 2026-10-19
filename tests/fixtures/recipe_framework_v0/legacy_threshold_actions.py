"""Frozen pre-port scaling and matrix typing; TEST ONLY, not a runtime path."""

def log2(dfs,com=[],cat=''):
    if len(com) == 0:
        base = input('log base? (2 default)')
        return([], [base])
    df = dfs[0].apply(pd.to_numeric, errors='coerce')

    base = 2
    try:
        if str(com[1]).strip() != '':
            base = int(com[1])
    except Exception:
        base = 2
        print('using base 2')
    for c in df.columns: 
        while 0 < df[c].max() < base*2: df[c] *= 100
    npones = np.ones(df.values.shape)
    newVals = np.maximum(df.values, npones)
    newVals = np.array(newVals, dtype=float)
    newVals = np.log2(newVals) / np.log2(base)
    df = pd.DataFrame(data=newVals, index=df.index, columns=df.columns)
    return([df, dfs[1], dfs[2]], [])

def outliers(dfs,com=[],cat=''):
    if len(com) == 0:
        subth = input('subtract channel thresholds instead? (y)')
        return([], [subth])
    df = dfs[0].apply(pd.to_numeric, errors='coerce').copy()
    if com[1] == 'y':
        for biom in df.columns:
            bn = biom.split('_')[0]
            thresh = MANUALtHRESHOLDS.get(bn, 0)
            df.loc[:,biom] = df.loc[:,biom] - thresh
            key = df.loc[:,biom] < 0
            df.loc[key,biom] = 0
    else:
        df = df.clip(lower=df.quantile(q=0.0013), upper=df.quantile(q=0.9987), axis=1)
    return([df, dfs[1], dfs[2]], [])

def scaleByMean(dfs,com=[],cat=''):
    if len(com) == 0:
        ch0,uch = obMenu(dfs[1],'column with annotation to scale by')
        col1 = dfs[1].columns[ch0]

        for i,uc in enumerate(uch):
             print(i,":",uc)
        ch = ''
        while not ch.isdigit():
            ch = input('annotation whose mean to scale by')
        ch = int(ch)
        anno = uch[int(ch)]

        #print(col1,anno,'col anno')
        return([],[col1,anno])

    dfs,n = ifv.autoClean(dfs,['n'])
    col,anno = com[1],com[2]

    df,obs,dfxy = dfs[0],dfs[1],dfs[2]

    key = obs.loc[:,col] == anno
    #print(key.sum())
    sdf = df.loc[key,:]
    means = sdf.mean(axis=0)
    #print(means)
    df = df.div(means)
    return([df,obs,dfxy],[])

def maxeyType(dfs,com=[],cat='', clean = True): #jessica maxey
    global MAXEY_MATRIX_DIR
    project_matrix_dir = Path.cwd() / "resources" / "maxey matrices"
    if project_matrix_dir.is_dir() and any(project_matrix_dir.glob("*.csv")):
        MAXEY_MATRIX_DIR = project_matrix_dir
        print("using project maxey matrices from:",MAXEY_MATRIX_DIR)
    else:
        MAXEY_MATRIX_DIR = _IF_ANALYSIS_DIR / "maxey matrices"
    #send chan thresh to 0 after zscore (and rank? do tied cells all get the lowest score or if all but 1 are tied they all get 2..) and before calculating type scores
    if len(com) == 0:
        chanT = input('compare to chanel threshold? (y)')
        log = input('log2 transform data (before scoring step- does not impact thresholds) (y)')
        return([],[chanT,log])
    # Matrix typing may rename/clean its working dataframe.  Keep that work
    # isolated so auxiliary neighborhood/radius features remain available to
    # downstream analysis after the new obs labels are added.
    original_df,original_obs,original_dfxy = dfs[0],dfs[1],dfs[2]
    drop_cols = [col for col in dfs[0].columns if 'neighbors' in str(col).lower() or 'radius' in str(col).lower()]
    scoring_df = original_df.drop(columns=drop_cols).copy()
    if len(drop_cols) > 0:
        print('ignoring neighborhood/radius columns for matrix celltyping:',drop_cols)
    dfs = [scoring_df,original_obs.copy(),original_dfxy.copy()]
    if clean:
        print(dfs[0].shape,dfs[0].columns,'... cleaning')
        dfs,n = ifv.autoClean(dfs,['n'])
        print(dfs[0].shape,dfs[0].columns)
    for col in [col for col in dfs[0].columns if '_' not in str(col)]:
        dfs[0].rename(columns={col: str(col)+'_'}, inplace=True)
    chanT,log = com[1],com[2]
    if chanT == 'y':
        btkey = chanThresh(dfs[0].copy())
    else:
        btkey = pd.DataFrame(np.full_like(dfs[0],False),columns=dfs[0].columns,index=dfs[0].index,dtype=bool)



    typeName = 'Primary Celltype: Matrix'
    #dfs[1][typeName] = '' #this should not be necessary but secondary tumor types were being assigned to non-tumor cells IFF maxeytype was being run on multiple categories (not all data) AND there was already a maxeytype category in the data.
    #ah, the above line didn't work, let's try the one below..
    #if typeName in dfs[1].columns:
    #    dfs[1].drop(typeName,axis=1,inplace=True)
    #nope this doesn't work either. No time today, Koei wants to see primary celltypes, can avoid glitch by deleting matrixtype columns before typing
    #Haven't been able to figure out why, key.sum() stays the same (whether there's a glitch or not, whether the glitch is from running on multiple cats or from the label already existingin the data- both conditions must be met for glitch)
    if not (MAXEY_MATRIX_DIR / 'primary_celltype.csv').exists():
        print('WARNING: primary_celltype.csv not found in',MAXEY_MATRIX_DIR)
        print('WARNING: primary celltyping is required; maxeyType cannot run without it.')
    typed_dfs = maxeyTypeH(dfs, below_thresh_key = btkey, bias = BIAS, threshold = PRIMARY_MT_THRESH, method = PRIMARY_MT_METHOD,log=log)
    if typed_dfs is None:
        print('celltyping failed')
        return([original_df,original_obs,original_dfxy],[])
    dfs = typed_dfs

    # Subtypes are valid only within their current primary-celltype branch.
    # Reset old results before recombining so a cell reclassified on a later
    # Maxey run cannot retain a stale tumor or immune subtype.
    dfs[1]['Tumor Subtype: Matrix'] = 'nan'
    dfs[1]['Immune Subtype: Matrix'] = 'nan'

    #each of the 4 pieces below is independent: subsets fresh from dfs, and only recombines
    #if its own matrix file was found, so a missing tumor/immune matrix just leaves that
    #piece's obs column at its 'nan' default instead of aborting the other pieces too
    idfs,key = subset(dfs,typeName,['3: epithelial'])
    print(idfs[0].shape[0],'idf shape')
    idfs = maxeyTypeH(idfs, typeName = 'Tumor Subtype: Matrix', default = ' ', fileName = 'tumor_celltype.csv', singleType = True, below_thresh_key = btkey.loc[key,:],log=log)
    if idfs is not None:
        dfs = recombine(dfs,idfs,key)
    else:
        print('skipping Tumor Subtype: Matrix (no matrix file found)')

    #tumor functional annots added to/vs all celltypes
    idfs,key = subset(dfs,typeName,list(dfs[1].loc[:,typeName].unique())) #
    idfs = maxeyTypeH(idfs, typeName = 'Tumor Functional: Matrix', default = ' ', fileName = 'tumor_functional.csv',
                      singleType = False, below_thresh_key = btkey.loc[key,:], threshold = .5,log=log)
    if idfs is not None:
        dfs = recombine(dfs,idfs,key)
    else:
        print('skipping Tumor Functional: Matrix (no matrix file found)')

    idfs,key = subset(dfs,typeName,['2: immune'])
    idfs = maxeyTypeH(idfs, typeName = 'Immune Subtype: Matrix', default = 'unclassified immune', fileName = 'immune_celltype.csv',method='rank',
                      threshold = .2, below_thresh_key = btkey.loc[key,:],log=log, singleType = True) #SINGLETYPE should be True for general use and default = 'unclassified'
    if idfs is not None:
        dfs = recombine(dfs,idfs,key)
    else:
        print('skipping Immune Subtype: Matrix (no matrix file found)')

    idfs,key = subset(dfs,typeName,['2: immune'])
    idfs = maxeyTypeH(idfs,typeName = 'Immune Functional: Matrix', default = ' ', fileName = 'immune_functional.csv', singleType = False, below_thresh_key = btkey.loc[key,:],log=log)
    if idfs is not None:
        dfs = recombine(dfs,idfs,key)
    else:
        print('skipping Immune Functional: Matrix (no matrix file found)')


    type_cols = [col for col in dfs[1].columns if str(col).endswith(': Matrix')]
    for col in type_cols:
        # Rows removed only from the temporary scoring view have no valid
        # matrix result.  Mark those explicitly rather than dropping their
        # measurements or retaining a stale label from an earlier run.
        original_obs[col] = 'not evaluated'
        original_obs.loc[dfs[1].index,col] = dfs[1].loc[:,col]
    if dfs[0].shape[0] != original_df.shape[0]:
        print('matrix celltyping retained all original data; ',original_df.shape[0] - dfs[0].shape[0],
              'cells were not evaluated after temporary cleaning')
    return([original_df,original_obs,original_dfxy],[])

def chanThresh(df):
    global MANUALtHRESHOLDS
    override = MANUALtHRESHOLDS
    print('\n\n',override,'MANUAL THRESHOLDS USED!!!')

    roundThresh = [1500,1250,1000,750]

    biomRounds = [['CAV1', 'CK17', 'CK5', 'CK7', 'CK8', 'H3K27', 'MUC1', 'PCNA', 'R0c2', 'R6Qc2', 'Vim', 'aSMA', 'pHH3'],
                  ['AR', 'CCND1', 'CD68', 'CD8', 'CK14', 'CoxIV', 'EGFR', 'H3K4', 'HER2', 'PDPN', 'R0c3', 'R6Qc3', 'pS6RP', 'GATA6'],
                  ['BCL2', 'CD31', 'CD4', 'CD45', 'ColIV', 'ER', 'Ki67', 'PD1', 'PgR', 'R0c4', 'R6Qc4', 'gH2AX', 'pRB','pERK'],
                  ['CD20', 'CD3', 'CD44', 'CK19', 'CSF1R', 'ColI', 'Ecad', 'FoxP3', 'GRNZB', 'LamAC', 'R0c5', 'R6Qc5', 'RAD51']]

    nbr = [['H3K27', 'PCNA', 'R6Qc2', 'LamB1', 'R0c2', 'pHH3', 'FN', 'R1Qc2', 'GFAP', 'Myelin', 'S100A', 'CAV1', 'Glut1', 'NeuN', 'aSMA', 'panCK'],
        ['H3K4', 'TUBB3', 'R6Qc3', 'CD68', 'R0c3', 'pMYC', 'CD11b', 'R1Qc3', 'CTNNB', 'PDL1', 'CD56', 'CD11c', 'CD133', 'HLA-DR', 'CD90', 'CD8', 'p53'],
        ['ColIV', 'CD163', 'R6Qc4', 'CD45', 'R0c4', 'Ki67', 'gH2AX', 'R1Qc4', 'IBA1', 'p63', 'BCL2', 'PD1', 'pMYCab', 'MSH6', 'CD31', 'CD4', 'pRPA'],
        ['ColI', 'BMP2', 'R6Qc5', 'CD20', 'R0c5', 'CD3', 'CD44', 'R1Qc5', 'LamAC', 'GRNZB', 'Rad51', 'CGA', 'CSF1R', '53BP1', 'YAP1', 'FoxP3', 'ZEB1']]
    for i in range(len(nbr)):
        biomRounds[i] += nbr[i]


    biomRounds = [[o] for o in override.keys()] + biomRounds
    roundThresh = list(override.values()) + roundThresh
    #print(biomRounds,roundThresh)
    key = []
    for i,col in enumerate(df.columns):
        coln = col.split('_')[0]
        thresh = -1
        switch = 0
        for j,roun in enumerate(biomRounds):
            if switch == 1:
                break
            for biom in roun:
                #print(coln,biom,coln==biom)
                if coln == biom:
                    thresh = roundThresh[j]
                    print('threshold found',biom,thresh)
                    switch = 1
                    break
        if thresh < 0:
            print('\n',coln,' not found.\nmean:',df.loc[:,col].mean())
            try:
                if DEVMODE:
                    1/0
                thr = input('threshold for marker: ')
                thresh = float(thr)
            except Exception as e:
                print('\nusing 1000 for threshold') #no need to print e
                thresh = 1000
                MANUALtHRESHOLDS[coln] = thresh
                print(MANUALtHRESHOLDS)

        lk = df.loc[:,col] < thresh
        key.append(lk)
    key = pd.concat(key,axis=1)
    #key.to_csv('test key.csv')

    print(key,'threshold key: cells < threshold = True')


    return(key)

