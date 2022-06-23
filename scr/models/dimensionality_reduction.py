from sklearn.decomposition import PCA
from sklearn import cluster


def set_dim_reduction(method:str):
    if method == 'PCA':
        return PCA(n_components=0.99)
    elif method == 'FeatureAgg':
        return cluster.FeatureAgglomeration(n_clusters=6, compute_distances=True)
    else:
        print('Wrong Dimentionality Reduction Method')
        return 0