'''
This module contains functions for feature importance ranking, along with several helper functions.

The feature ranking methods fall into several categories:
    - Correlation based methods - these will be better for regression problems. They can be used for binary classification, but will be limited as to what trends they can detect. 
    - Distribution based methods - these currently can only be used for binary classification problems. 
    - Tree based methods - these are the most generalizable, but also the most computationally expensive.
    - Aggregation methods - these combine the results of multiple methods to produce a final ranking.

Each method is implemented with the following imputs/outputs:
    - Inputs:
        - df : pd.DataFrame
            Dataframe containing features and target
        - target_label : str
            Name of target column
        - n_features : int, optional
            Number of features to return, by default None
            If None, all features will be returned
        - **kwargs : optional
            Additional arguments used by individual methods
    - Outputs:
        - top_feats : list
            List of top features in order of importance
            
Some methods will be able to rank all available features, while others, like MRMR, are "minimal optimal" methods 
with the goal of reducing redundancy in selected features.


'''
import numpy as np
from numpy.polynomial import Polynomial
from scipy.stats import wasserstein_distance
import rdu_utils

#### FEATURE RANKING METHODS ####


## Correlation based methods ##

def simple_correlation(df, target_label : str, n_features : int = None):
    '''
    Ranks features by their correlation with target
    '''
    
    feat_list = df.columns.tolist()
    feat_list.remove(target_label)
    if n_features == None:
        n_features = len(feat_list)
        
    correlations = []
    for feat in feat_list:
        correlations.append(np.abs(np.corrcoef(df[target_label], df[feat])[0,1]))

    corr_sorted = np.array(correlations)[np.argsort(correlations)[::-1]]
    corr_sorted_feats = np.array(feat_list)[np.argsort(correlations)[::-1]]
    #find nan indexes
    nan_idxs = np.where(np.isnan(corr_sorted))[0]
    #remove nan indexes
    corr_sorted = np.delete(corr_sorted, nan_idxs)
    corr_sorted_feats = np.delete(corr_sorted_feats, nan_idxs)
    top_feats = corr_sorted_feats[0:n_features]
    return top_feats
    
    
def MRMR(df, target_label : str, n_features : int = None, colinearity_cutoff : float = 0.8):
    '''
    Select feature with highest correlation to target, then remove all other features with correlation 
    higher than colinearity_cutoff with that selected feature, then repeat. 
    '''
    corr = df.corr()
    ordered_correlators =  corr[target_label].drop(target_label).abs().sort_values(ascending=False)
    select_features = list(ordered_correlators.index)
    if n_features is None:
        n_features = len(select_features) 
        
    for feature_1 in ordered_correlators.index:
        if feature_1 not in select_features:
            continue
        for feature_2 in ordered_correlators.drop(feature_1).index:
            if feature_2 not in select_features:
                continue
            if abs(corr[feature_1][feature_2]) > colinearity_cutoff:
                if feature_2 in select_features:
                    select_features.remove(feature_2)


    if len(select_features) > n_features:
        select_features = select_features[0:n_features]
    return select_features

def decorrelated_maxcor(df, target_label : str, n_features : int = None):
    '''
    Find feature with highest correlation to target
    Decorrelate all other features from that one feature (using rdu_utils.decorrelate)
    Repeat
    
    For linear regression problems, this is basically equivalent to forward search.
    
    '''
    best_feats = []
    df = df.copy()
    feat_list = df.columns.tolist()
    feat_list.remove(target_label)
    #replace nans with mean
    for feat in feat_list:
        df[feat] = df[feat].fillna(df[feat].mean())
    
    if n_features is None:
        n_features = len(feat_list) 
        
    while len(best_feats) < n_features:
        try: 
            # print(f"% complete: {len(best_feats)/n_features*100}", end='\r')
            correlations = []
            for feat in feat_list:
                correlations.append(np.abs(np.corrcoef(df[target_label], df[feat])[0,1]))
            max_corr_idx = np.argmax(correlations)
            top_feat = feat_list[max_corr_idx]
            best_feats.append(top_feat)
            feat_list.remove(top_feat)
            for feat in feat_list:
                df[feat] = rdu_utils.decorrelate(df[feat], df[top_feat])
        except:
            break
    return best_feats


def decorrelated_maxcor2(df, target_label : str, n_features : int = None, decor_rate : float = 1):
    '''
    Similar to decorrelated_maxcor, but instead of decorrelating all features from the selected feature,
    it decorrelates the target from each selected feature before continuing.
    
    optional decor_rate parameter controls how much of the target is removed for each feature-fit.
    
    This should be equivalent to Orthogonal Matching Pursuit: https://scikit-learn.org/stable/modules/linear_model.html#omp
    '''
    best_feats = []
    df = df.copy()
    feat_list = df.columns.tolist()
    feat_list.remove(target_label)
    target = df[target_label].values.copy()
    #replace nans with mean
    for feat in feat_list:
        df[feat] = df[feat].fillna(df[feat].mean())
    
    if n_features is None:
        n_features = len(feat_list) 
        
    while len(best_feats) < n_features:
        try: 
            # print(f"% complete: {len(best_feats)/n_features*100}", end='\r')
            correlations = []
            for feat in feat_list:
                correlations.append(np.abs(np.corrcoef(target, df[feat])[0,1]))

            max_corr_idx = np.argmax(correlations)
            top_feat = feat_list[max_corr_idx]
            best_feats.append(top_feat)
            feat_list.remove(top_feat)
            fit = Polynomial.fit(df[top_feat], target, deg=1)
            pred = fit(df[top_feat])
            target = target - pred*decor_rate
        except:
            break
            
    return best_feats 



## Distribution based methods ##

def max_split_distance(df, target_label : str, n_features : int = None, metric = wasserstein_distance, normalize = True):
    '''
    find the feature, x, that results in the maximal distance metric between distributions when split by values of y
    if normalize, each feature will be normalized as a whole before split and metric is applied
    This only works for binary [0,1] targets
    Metric can be any function that takes two arrays and returns a distance between them. See helper methods at bottom of file for examples.
    '''
    y =  df[target_label].values
    feat_select = []
    remaining_feats = df.columns.tolist()
    
    if n_features is None:
        n_features = len(remaining_feats)
    
    distances = []
    feat_list = df.columns.tolist()
    feat_list.remove(target_label)
    for feat in feat_list:
        if feat == target_label:
            continue
        x = df[feat].values
        if normalize:
            x = (x - x.mean())/x.std()
        dist = metric(x[y==0], x[y==1])
        distances.append(dist)
    order = np.argsort(distances)[::-1]      
    feat_select = [feat_list[i] for i in order]
    return feat_select[:n_features]
    
def mutual_information(df, target_label : str, n_features : int = None, n_neighbors : int = 7, norm : str = None):
    '''
    Find feature with highest mutual information with target
    Drops all nan rows from features
    '''
    
    import sklearn.feature_selection as fs

    X = df.drop([target_label], axis=1).values
    y = df[target_label].values

    if n_features is None:
        n_features = X.shape[1]
        
    mask = ~np.isnan(X).any(axis=1)
    X_cleaned = X[mask,:]
    y_cleaned = y[mask]
    # print(X_cleaned.shape)

    if norm is not None:
        if norm == 'standardize':
            X_cleaned = rdu_utils.standardize(X_cleaned.transpose()).transpose()
        elif norm == 'normalize':
            X_cleaned = rdu_utils.normalize(X_cleaned.transpose()).transpose()
        mask = ~np.isnan(X_cleaned).any(axis=1)
        X_cleaned = X_cleaned[mask,:]
        y_cleaned = y_cleaned[mask]
    
    scores = fs.mutual_info_classif(X_cleaned, y_cleaned,n_neighbors=n_neighbors)
    cols = list(df.columns)
    order = np.argsort(scores)[::-1]
    top_feats = [cols[i] for i in order][0:n_features]
    return top_feats



## Tree based methods ##

def xgb_importance(df, target_label : str, n_features : int = None):
    '''
    Uses xgboost to rank features by importance
    '''
    import xgboost as xgb
    feat_list = df.columns.tolist()
    feat_list.remove(target_label)
    model = xgb.XGBClassifier()
    model.fit(df.drop(target_label, axis=1).values, df[target_label].values)
    top_feats = np.array(feat_list)[np.argsort(model.feature_importances_)[::-1]][0:n_features]
    return top_feats
    


## Aggregation methods ##

#sets
all_regr_methods = [ 
    simple_correlation,
    MRMR,
    decorrelated_maxcor
]
all_binary_methods = [
    max_split_distance,
    mutual_information,
    xgb_importance,
]
all_split_distance_methods = [
    max_split_distance, #this uses the default metric, wasserstein_distance
    lambda df, target_label, n_features: max_split_distance(df, target_label, n_features, metric = distribution_overlap_ratio),
    lambda df, target_label, n_features: max_split_distance(df, target_label, n_features, metric = hist_correlation),
    lambda df, target_label, n_features: max_split_distance(df, target_label, n_features, metric = total_varitional_distance),
    lambda df, target_label, n_features: max_split_distance(df, target_label, n_features, metric = kl_divergence),
    lambda df, target_label, n_features: max_split_distance(df, target_label, n_features, metric = max_cdf_diff),
]
ak_select = [
    simple_correlation,
    mutual_information,
    xgb_importance,
    lambda df, target_label, n_features: aggregation_of_methods(df, target_label, n_features, methods = all_split_distance_methods)
]

def aggregation_of_methods(df, target_label : str, n_features : int = None, methods = ak_select, unbias = False):
    '''
    Aggregates the results of multiple methods to produce a final ranking by summing the rank of each feature across methods.
    unbias = True will ensure that the number of features returned is equal to the minimum returned length of features, 
    since some methods might return fewer features than others. This ensures that the final ranking is not biased towards methods that return fewer features.
    '''
    agg = {}
    results = {}
    min_len = np.inf
    for method in methods:
        results[method] = method(df, target_label, n_features)
        if len(results[method]) < min_len:
            min_len = len(results[method])

    for method in methods:
        if unbias:
            num_feats = min_len
        else:
            num_feats = len(results[method])
        for i in range(num_feats):
            feat = results[method][i]
            feat_base = feat #feat.split(',seg_n')[0]+'),'
            feat_score = num_feats - i
            if feat_base not in agg.keys():
                agg[feat_base] = feat_score
            else:
                agg[feat_base] += feat_score
            
    sorted_feats = [feat for feat, score in sorted(agg.items(), key=lambda item: item[1], reverse=True)]
    return sorted_feats[0:n_features]



#### HELPER FUNCITONS ####

def print_base_feats(top_feats):
    '''
    removes individual channels from feature names and prints only the base for each top feature, 
    in order of first appearance in list
    '''
    n = len(top_feats)
    used_feats = []
    i = 0
    for feat in top_feats:
        feat_base = feat.split('seg_n')[0]+'),'
        if feat_base not in used_feats:
            print(feat_base)
            used_feats.append(feat_base)
            i+=1
        if i == n:
            break
    return used_feats
    
def compute_optimal_split(x, y, n_bins = 100):
    '''
    Compute the optimal split point between two distributions, x and y, by finding the point of maximal difference between CDFs
    '''
    #remove outliers
    x = rdu_utils.remove_outliers(x, threshold=2)
    y = rdu_utils.remove_outliers(y, threshold=2)
    bins = np.linspace(np.min(np.hstack((x,y))), np.max(np.hstack((x,y))), n_bins)
    x_hist = np.histogram(x, bins=bins, density=True)[0]
    y_hist = np.histogram(y, bins=bins, density=True)[0]
    x_cdf = np.cumsum(x_hist)
    y_cdf = np.cumsum(y_hist)
    cdf_diff = np.abs(x_cdf - y_cdf)
    optimal_split = bins[np.argmax(cdf_diff)]
    return optimal_split

def plot_feat_by_target(df, feats_to_plot, target_label, n_bins = 100, outlier_cutoff = 95, drop_zeros = True):
    '''
    Plots the distributions of each feature split by target
    '''
    #create subplots for each feature
    import matplotlib.pyplot as plt
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    target_vector = df[target_label]
    for i, feat in enumerate(feats_to_plot):
        size = int(np.ceil(len(feats_to_plot)**.5)) # size of subplot grid
        subplot = fig.add_subplot(size, size, i + 1)
        outlier_cutoff_high = np.percentile(df[feat], outlier_cutoff)
        outlier_cutoff_low = np.percentile(df[feat], 100 - outlier_cutoff)
        non_sz_vals = df[target_vector==0][feat]    #non_sz_df[feat]
        # non_sz_vals = non_sz_vals[non_sz_vals < outlier_cutoff_high]
        # non_sz_vals = non_sz_vals[non_sz_vals > outlier_cutoff_low]
        sz_vals = df[target_vector==1][feat] 
        # sz_vals = sz_vals[sz_vals < outlier_cutoff_high]
        # sz_vals = sz_vals[sz_vals > outlier_cutoff_low]
        
        if drop_zeros:
            non_sz_vals = non_sz_vals[non_sz_vals != 0]
            sz_vals = sz_vals[sz_vals != 0]

        bins = np.linspace(outlier_cutoff_low, outlier_cutoff_high, n_bins)
        hist_non_sz = plt.hist(non_sz_vals, bins=bins, alpha=0.5, label='non-sz', density=True)
        hist_sz = plt.hist(sz_vals, bins=bins, alpha=0.5, label='sz', density=True)
        _max = np.max(np.hstack((hist_non_sz[0], hist_sz[0])))
        optimal_split = compute_optimal_split(non_sz_vals, sz_vals, n_bins = 200)
        plt.vlines(optimal_split, 0, _max, label='optimal split', color='r', alpha=0.5)
        subplot.legend()
        subplot.set_title(i)
    fig.show()
    
    
# the following are metrics to be used in max_split_distance

def distribution_overlap_ratio(x, y):
    x = x[x != 0] #drop zeros because often that is just a fallback number and not representative of real data
    y = y[y != 0]
    all = np.concatenate([x,y])
    outlier_cutoff_high = np.percentile(all, 90)
    outlier_cutoff_low = np.percentile(all, 10)
    bins = np.linspace(outlier_cutoff_low, outlier_cutoff_high, 100)
    x_dist = np.histogram(x, bins=bins)[0]
    y_dist = np.histogram(y, bins=bins)[0]
    return 1 - np.sum(np.minimum(x_dist, y_dist))/np.sum(np.maximum(x_dist, y_dist))

def kl_divergence(x, y):
    x = x[x != 0] #drop zeros because often that is just a fallback number and not representative of real data
    y = y[y != 0]
    all = np.concatenate([x,y])
    outlier_cutoff_high = np.percentile(all, 90)
    outlier_cutoff_low = np.percentile(all, 10)
    bins = np.linspace(outlier_cutoff_low, outlier_cutoff_high, 100)
    x_dist = np.histogram(x, bins=bins)[0]
    y_dist = np.histogram(y, bins=bins)[0]
    return np.sum(np.where(x_dist != 0, x_dist * np.log(x_dist / y_dist), 0))

def hist_correlation(x, y):
    x = x[x != 0] #drop zeros because often that is just a fallback number and not representative of real data
    y = y[y != 0]
    all = np.concatenate([x,y])
    outlier_cutoff_high = np.percentile(all, 90)
    outlier_cutoff_low = np.percentile(all, 10)
    bins = np.linspace(outlier_cutoff_low, outlier_cutoff_high, 100)
    x_dist = np.histogram(x, bins=bins)[0]
    y_dist = np.histogram(y, bins=bins)[0]
    return 1-np.corrcoef(x_dist, y_dist)[0,1]

def total_varitional_distance(x, y):
    x = x[x != 0] #drop zeros because often that is just a fallback number and not representative of real data
    y = y[y != 0]
    all = np.concatenate([x,y])
    outlier_cutoff_high = np.percentile(all, 90)
    outlier_cutoff_low = np.percentile(all, 10)
    bins = np.linspace(outlier_cutoff_low, outlier_cutoff_high, 100)
    x_dist = np.histogram(x, bins=bins)[0]
    y_dist = np.histogram(y, bins=bins)[0]
    return np.sum(np.abs(x_dist - y_dist))

def max_cdf_diff(x, y):
    x = x[x != 0] #drop zeros because often that is just a fallback number and not representative of real data
    y = y[y != 0]
    all = np.concatenate([x,y])
    outlier_cutoff_high = np.percentile(all, 90) #trim outliers
    outlier_cutoff_low = np.percentile(all, 10)
    bins = np.linspace(outlier_cutoff_low, outlier_cutoff_high, 100)
    x_dist = np.histogram(x, bins=bins, density=False)[0]
    y_dist = np.histogram(y, bins=bins, density=False)[0]
    x_cdf = np.cumsum(x_dist)
    y_cdf = np.cumsum(y_dist)
    cdf_diff = x_cdf - y_cdf
    return np.sum(np.abs(cdf_diff))

