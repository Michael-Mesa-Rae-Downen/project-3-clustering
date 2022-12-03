#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
import env
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split


import wrangle as w
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = w.wrangle_zillow()


# In[3]:


train, validate, test = w.split_data(df)


# In[4]:


train_scaled, validate_scaled, test_scaled = w.scale_data(train, validate, test)


# In[5]:


X_train_scaled = train_scaled[['bedrooms','bathrooms', 'sq_feet']]


# In[6]:


cluster_vars = ['bedrooms','bathrooms', 'sq_feet']
cluster_name = 'interior_cluster_k7'
k_range = range(2,20)


# In[7]:


def find_k(X_train_scaled, cluster_vars, k_range):
    sse = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k)

        # X[0] is our X_train dataframe..the first dataframe in the list of dataframes stored in X. 
        kmeans.fit(X_train_scaled[cluster_vars])

        # inertia: Sum of squared distances of samples to their closest cluster center.
        sse.append(kmeans.inertia_) 

    # compute the difference from one k to the next
    delta = [round(sse[i] - sse[i+1],0) for i in range(len(sse)-1)]

    # compute the percent difference from one k to the next
    pct_delta = [round(((sse[i] - sse[i+1])/sse[i])*100, 1) for i in range(len(sse)-1)]

    # create a dataframe with all of our metrics to compare them across values of k: SSE, delta, pct_delta
    k_comparisons_df = pd.DataFrame(dict(k=k_range[0:-1], 
                             sse=sse[0:-1], 
                             delta=delta, 
                             pct_delta=pct_delta))

    # plot k with inertia
    plt.plot(k_comparisons_df.k, k_comparisons_df.sse, 'bx-')
    plt.xlabel('k')
    plt.ylabel('SSE')
    plt.title('The Elbow Method to find the optimal k\nFor which k values do we see large decreases in SSE?')
    plt.show()

    # plot k with pct_delta
    plt.plot(k_comparisons_df.k, k_comparisons_df.pct_delta, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Percent Change')
    plt.title('For which k values are we seeing increased changes (%) in SSE?')
    plt.show()

    # plot k with delta
    plt.plot(k_comparisons_df.k, k_comparisons_df.delta, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Absolute Change in SSE')
    plt.title('For which k values are we seeing increased changes (absolute) in SSE?')
    plt.show()

    return k_comparisons_df


# In[8]:


def create_clusters(X_train_scaled, cluster_vars, k=7):
    # create kmean object
    kmeans = KMeans(n_clusters=k, random_state = 123)

    # fit to train and assign cluster ids to observations
    kmeans.fit(X_train_scaled[cluster_vars])

    return kmeans


# In[9]:


# get the centroids for each distinct cluster...

def get_centroids(kmeans, cluster_vars, cluster_name):
    # get the centroids for each distinct cluster...

    centroid_col_names = ['centroid_' + i for i in cluster_vars]

    centroid_df = pd.DataFrame(kmeans.cluster_centers_, 
                               columns=centroid_col_names).reset_index().rename(columns={'index': cluster_name})

    return centroid_df


# In[10]:


# label cluster for each observation in X_train (X[0] in our X list of dataframes), 
# X_validate (X[1]), & X_test (X[2])

def assign_clusters(df, kmeans, cluster_vars, cluster_name, centroid_df):
    #for i in range(len(df)):
        clusters = pd.DataFrame(kmeans.predict(df[cluster_vars]), 
                            columns=[cluster_name], index=df.index)

        clusters_centroids = clusters.merge(centroid_df, on=cluster_name, copy=False).set_index(clusters.index.values)

        df = pd.concat([df, clusters_centroids], axis=1)
        return df


# In[11]:


def get_train_clusters():
    
    X_train_scaled = train_scaled[['bedrooms','bathrooms', 'sq_feet']]
    X_train = train[['bedrooms','bathrooms', 'sq_feet']]
    
    cluster_vars = ['bedrooms','bathrooms', 'sq_feet']
    cluster_name = 'interior_cluster_k7'
    k_range = range(2,20)
    
    kmeans = create_clusters(X_train_scaled, cluster_vars, k=7)
    
    centroid_df = get_centroids(kmeans, cluster_vars, cluster_name)
    
    X = assign_clusters(X_train_scaled, kmeans, cluster_vars, cluster_name, centroid_df)
    
    dummy_df = pd.get_dummies(X['interior_cluster_k7'], dummy_na=False, drop_first=False)
    
    dummy_df = dummy_df.rename(columns={3:'is_cluster_3_k7'})
    
    X_train_scaled = pd.concat([X_train_scaled, dummy_df['is_cluster_3_k7']], axis=1)
    X_train = pd.concat([X_train, dummy_df['is_cluster_3_k7']], axis=1)
    
    return X_train, X_train_scaled


# In[12]:


def get_validate_pred():
    
    X_validate_scaled = validate_scaled[['bedrooms','bathrooms','sq_feet']]
    #X_train = train[['bedrooms','bathrooms', 'sq_feet']]
    
    #cluster_vars = ['bedrooms','bathrooms', 'sq_feet']
    #cluster_name = 'interior_cluster_k7'
    #k_range = range(2,20)
    
    #kmeans = create_clusters(X_train_scaled, 7, cluster_vars)
    
    #centroid_df = get_centroids(kmeans, cluster_vars, cluster_name)
    
    #X_train = assign_clusters(X_train, kmeans, cluster_vars, cluster_name, centroid_df)
    
    val_pred = pd.DataFrame(kmeans.predict(X_validate_scaled), index=X_validate_scaled.index)
    
    dummy_df_val = pd.get_dummies(val_pred[0], dummy_na=False, drop_first=False)
    
    #val_pred_dummies = pd.get_dummies(val_pred[0], dummy_na=False, drop_first=False)
    
    dummy_df_val = dummy_df_val.rename(columns={3:'is_cluster_3_k7'})
    
    X_validate_scaled = pd.concat([X_validate_scaled, dummy_df_val['is_cluster_3_k7']], axis=1)
    #X_validate_scaled = X_validate_scaled.dropna()
    #X_train = pd.concat([X_train, dummy_df['is_cluster_3_k7']], axis=1)
    
    return X_validate_scaled


# In[13]:


def get_test_pred():
    
    X_test_scaled = test_scaled[['bedrooms','bathrooms','sq_feet']]
    #X_train = train[['bedrooms','bathrooms', 'sq_feet']]
    
    #cluster_vars = ['bedrooms','bathrooms', 'sq_feet']
    #cluster_name = 'interior_cluster_k7'
    #k_range = range(2,20)
    
    #kmeans = create_clusters(X_train_scaled, 7, cluster_vars)
    
    #centroid_df = get_centroids(kmeans, cluster_vars, cluster_name)
    
    #X_train = assign_clusters(X_train, kmeans, cluster_vars, cluster_name, centroid_df)
    
    test_pred = pd.DataFrame(kmeans.predict(X_test_scaled), index=X_test_scaled.index)
    
    dummy_df_test = pd.get_dummies(test_pred[0], dummy_na=False, drop_first=False)
    
    #val_pred_dummies = pd.get_dummies(val_pred[0], dummy_na=False, drop_first=False)
    
    dummy_df_test = dummy_df_test.rename(columns={3:'is_cluster_3_k7'})
    
    X_test_scaled = pd.concat([X_test_scaled, dummy_df_test['is_cluster_3_k7']], axis=1)
    #X_validate_scaled = X_validate_scaled.dropna()
    #X_train = pd.concat([X_train, dummy_df['is_cluster_3_k7']], axis=1)
    
    return X_test_scaled


# In[47]:


def get_cluster_3_ttest(X_train_scaled):
    
    ttest_df = pd.concat([train.logerror, X_train_scaled.is_cluster_3_k7], axis=1)
    
    t, p = stats.ttest_1samp(ttest_df[ttest_df['is_cluster_3_k7'] == 1].logerror.abs(), ttest_df.logerror.abs().mean())
    print(f't     = {t:.4f}')
    print(f'p     = {p:.4f}')
    #return t, pval


# In[48]:


def get_bed_log_corr():
    
    y = train.logerror
    
    corr, p = stats.pearsonr(X_train_scaled.bedrooms, y)
    print(f'corr  = {corr:.4f}')
    print(f'p     = {p:.4f}')


# In[49]:


def get_bath_log_corr():
    
    y = train.logerror  
    
    corr, p = stats.pearsonr(X_train_scaled.bathrooms, y)
    print(f'corr  = {corr:.4f}')
    print(f'p     = {p:.4f}')


# In[50]:


def get_sq_feet_log_corr():
    
    y = train.logerror
    
    corr, p = stats.pearsonr(X_train_scaled.sq_feet, y)
    print(f'corr  = {corr:.4f}')
    print(f'p     = {p:.4f}')

