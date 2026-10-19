"""Frozen clustering collectors and math, TEST ONLY; not imported by DAS."""

def kmeans(dfs,com=[],cat=''):
    if len(com) == 0:
        ncls = _collect_multi_int("n clusters (blank when done): ")
        normalize = _prompt_normalize_before_clustering()
        return([],[ncls,normalize])
    df,obs = dfs[0],dfs[1]
    ncls = _as_int_list(com[1])
    cdf = _zscore_for_clustering(df,com[2] if len(com) > 2 else 'n')
    for ncl in ncls:
        km = KMeans(n_clusters=ncl)
        km.fit(cdf)
        cn = _scoped_cluster_output_name("Kmeans_"+str(ncl), cat)
        obs[cn] = _scoped_cluster_labels(km.labels_, cat)
    return([dfs[0],obs,dfs[2]],[])

def gmm(dfs,com=[],cat=''):
    if len(com) == 0:
        ncls = _collect_multi_int("n clusters (blank when done): ")
        normalize = _prompt_normalize_before_clustering()
        return([],[ncls,normalize])
    ncls = _as_int_list(com[1])
    df,obs,dfxy = dfs[0],dfs[1],dfs[2]
    cdf = _zscore_for_clustering(df,com[2] if len(com) > 2 else 'n')
    #ctypes = ['full','tied','diag','spherical']
    for ncl in ncls:
        gmmModel = GMM(n_components=ncl).fit(cdf)
        cn = _scoped_cluster_output_name("GMM_"+str(ncl), cat)
        obs[cn] = _scoped_cluster_labels(gmmModel.predict(cdf), cat)
    return([df,obs,dfxy],[])

def leiden(dfs,com=[],cat=''):
    if len(com) == 0:
        res = float(input("recluster with resolution:"))
        primo = input('only use primary markers? (y)')
        normalize = _prompt_normalize_before_clustering()
        return([],[res,primo,normalize])
    sc, anndata = _load_scanpy_stack("Leiden clustering")
    if sc is None:
        return([dfs[0],dfs[1],dfs[2]],[])
    res,primo = com[1],com[2]
    df,obs,dfxy = dfs[0],dfs[1],dfs[2]
    odf = df.copy()
    if primo == 'y':
        df,obs,dfxy = cm.onlyPrimaries(df,obs,dfxy)
    print(all(obs.index==df.index),"all index the same")
    cdf = _zscore_for_clustering(df,com[3] if len(com) > 3 else 'n')
    adata = anndata.AnnData(cdf,obs = obs)
    sc.pp.neighbors(adata,use_rep='X')
    sc.tl.leiden(adata, key_added='Cluster', resolution=res)
    cn = _scoped_cluster_output_name("Leiden_"+str(res), cat)
    obs[cn] = _scoped_cluster_labels(adata.obs["Cluster"].astype(str), cat)
    obs[cn] = obs[cn].astype(str)
    return([odf,obs,dfxy],[])

def autoleiden(dfs,com=[],cat=''):
    if len(com) == 0:
        target = int(input("get n clusters:"))
        res = float(input("starting Leiden resolution:"))
        primo = input('only use primary markers? (y)')
        normalize = _prompt_normalize_before_clustering()

        return([],[res,target,primo,normalize])
    sc, anndata = _load_scanpy_stack("Auto-Leiden clustering")
    if sc is None:
        return([dfs[0],dfs[1],dfs[2]],[])
    df,obs,dfxy = dfs[0],dfs[1],dfs[2]
    res,target,primo = com[1],com[2],com[3]
    odf = df.copy()
    if primo == 'y':
        df,obs,dfxy = cm.onlyPrimaries(df,obs,dfxy)
    cdf = _zscore_for_clustering(df,com[4] if len(com) > 4 else 'n')
    incr = res/4
    ncl = 99
    tes = []
    target = int(target)
    ret = 0
    rthre = 10
    while ncl != target:
        if ret > 50:
            target -= 1
            ret = 0
        ret += 1
        print(res,incr)

        #print("!! running with res",res)
        adata = anndata.AnnData(cdf,obs = obs)
        sc.pp.neighbors(adata,use_rep='X')
        sc.tl.leiden(adata, key_added='Cluster', resolution=res)
        cn = _scoped_cluster_output_name("Leiden_n" + str(target), cat)
        obs.loc[:,cn] = ""
        obs.loc[:,cn] = _scoped_cluster_labels(adata.obs["Cluster"].astype(str), cat)
        ncl = len(list(obs.loc[:,cn].unique()))
        if ret > rthre:
            rthre += 10
            print("got",ncl,"clusters!")
        tes.append(res)
        if incr < 10**-15:
            incr *= 5
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
    return([odf,obs,dfxy],[])

def torchCluster(dfs,com=[],cat=''):
    if len(com) == 0:
        ncl = int(input("n clusters:"))
        nit = int(input("number of iterations:"))
        lra = float(input("convergence tolerance (blank not allowed, 1e-4 is a reasonable start):"))
        output_col = input("output name (blank for TorchCluster_<n>): ")
        normalize = _prompt_normalize_before_clustering()
        return([],[ncl,nit,lra,output_col,normalize])
    df,obs,dfxy = dfs[0],dfs[1],dfs[2]
    ncl = int(com[1])
    nit = int(com[2])
    lra = float(com[3])
    output_col = str(com[4]).strip() if len(com) > 4 else ""
    cdf = _zscore_for_clustering(df,com[5] if len(com) > 5 else 'n')
    labels,centroids,meta = ml_tc.cluster_dataframe(
        cdf,
        n_clusters=ncl,
        max_iter=nit,
        learning_rate=lra,
    )
    if output_col == "":
        output_col = "TorchCluster_"+str(ncl)
    cn = _scoped_cluster_output_name(output_col, cat)
    obs[cn] = _scoped_cluster_labels(labels.astype(str), cat)
    meta["output_col"] = cn
    meta["output_stem"] = _safe_token(cn)
    meta["cluster_axis"] = "rows"
    summary_path = Path(os.getcwd()) / (_safe_token(cn) + ".summary.txt")
    meta["summary_path"] = str(summary_path)
    ml_tc.write_summary_file(summary_path, ml_tc.summary_lines(meta))
    print("Torch centroid clustering added obs column:", cn)
    print("cluster counts:", meta.get("cluster_counts"))
    print("Torch clustering summary saved:", str(summary_path))
    return([df,obs,dfxy],[])

def _as_int_list(com_val):
    #old saved lastrun commands recorded a single int here; new ones record a list
    if isinstance(com_val,(list,tuple)):
        return(list(com_val))
    return([com_val])

