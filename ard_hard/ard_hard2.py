#!/usr/bin/env python
# coding: utf-8
## Created by: Alexander Delgado for Fortuna Silver Mines and Subsidiaries
#pip install pandas
#pip install matplotlib
#pip install -U scikit-learn

#from distutils.sysconfig import get_python_inc

#from distutils.sysconfig import get_python_inc
#from sysconfig import get_python_version
from IPython import get_ipython
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn import linear_model, datasets

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
plt.rcParams['figure.figsize'] = (6, 6)

# ## Parametros

def readdata(rutaarhvivo):
    data_dup = pd.read_csv(rutaarhvivo)
    return data_dup


# ### Filter by QC, Campaign and Element

def filter_calcARD(dataf, QC, org, dup, Elem, ldl, ARD):
    #df = readdata('./xxArizaro/Datos de Muestras/Results_Field duplicate_ALL_Au.csv')
    df = dataf.copy()
    dfz = df.dropna(subset=[org])
    dfx = dfz.dropna(subset=[dup])
    
    xx = dfx.loc[:,[ 'DataSet', 'Orig_SampleID', org, 'SampleID', dup]]
    LD = (ldl)*10
    xx1 = (xx
           .assign(a=lambda xx: xx[org] <= LD, 
                   b=lambda xx: xx[dup] <= LD)
           .copy()
           .replace({'a': {(False): 'Ok', (True): 'Fail'}})
           .replace({'b': {(False): 'Ok', (True): 'Fail'}})
           .assign(union=lambda xx: xx[['a', 'b']].apply(lambda x: '-'.join(x), axis=1))
           )
    dfz= xx1[~xx1.union.isin(['Fail-Fail'])]
    x = dfz[org]
    y = dfz[dup]
    dfz1 = (dfz
            .assign(difABS=abs(x - y) / ((x + y) / 2), 
                    difREL=(x - y) / ((x + y) / 2) * 100)
            .copy()
            .assign(Success='n')
            .assign(Success=lambda dfz: dfz.Success.mask(dfz.difABS<ARD, 'y'))
            )
    n_totalData = len(dfx)
    n_total = len(dfz1)
    n_warnings = len(dfz1[dfz1['Success'] == 'y'])
    pctDups = (n_warnings/n_total)*100   
    print (QC)
    print ('Element: '+Elem)
    print ('Count Total Data: '+str(n_totalData))
    print ('Count Dup ARD: '+str(n_total))
    print ('Total ARD pct: '+str(round(pctDups,1)))
    return dfz1

def compute_filters_stats(df): 
    """Compute statistics regarding the relative quanties of ... """
    n_total = len(df)
    n_warnings = len(df[df['Success'] == 'y'])
    return(pd.Series(data = {
        'No. of duplicates analyzed#': n_total,        
        #'warnings': n_warnings,       
        'Percent of samples meeting ARD* acceptance criteria': round ((n_warnings/n_total*100),1)
    }))
    
def highlight_col(column, bool_masks, color_list, default="white"):        
    cond_bg_style = [f'background-color: {color}' for color in color_list]
    default_bg_style = f'background-color:{default}'
    return np.select(bool_masks, cond_bg_style, default=default_bg_style)

def filter_calcHARD(datafh, QC, org, dup, Elem, ldl):
    df = datafh.copy()
    dfz = df.dropna(subset=[org])
    dfx = dfz.dropna(subset=[dup])
    xx = dfx.loc[:,[ 'DataSet', 'Orig_SampleID', org, 'SampleID', dup]]
    #LD = (ldl/2)*10
    xx1 = (xx
           .assign(a=lambda xx: xx[org] < ldl, 
                   b=lambda xx: xx[dup] < ldl)
           .copy()
           .replace({'a': {(False): 'Ok', (True): 'Fail'}})
           .replace({'b': {(False): 'Ok', (True): 'Fail'}})
           .assign(union=lambda xx: xx[['a', 'b']].apply(lambda x: '-'.join(x), axis=1))
           )
    dfz= xx1[~xx1.union.isin(['Fail-Fail'])]
    x = dfz[org]
    y = dfz[dup]
    dfz2 = (dfz
            .assign(difABS=0.5*(abs(x - y) / ((x + y) / 2)))
            .copy()
            )
    dfz1 = dfz2.copy()
    dfz1['succs'] = QC
    dfz1['percentile'] = dfz1.groupby('succs')['difABS'].rank(pct=True)
    xxx = dfz1.groupby('succs').agg(min_val = ('difABS','min'), percentile_90 = ('difABS',lambda x: x.quantile(0.9)))
    per = xxx['percentile_90'].mean()
    dfz2 = dfz1.copy()
    #n_totalData = len(dfx)
    n_total = len(dfz1)
    #print ('T. Data: '+str(n_totalData))
    print ('Count Dup HARD: '+str(n_total)) 
    print ('HARD Percentil 90%: ', str(round(per*100,1)))
    print ('LimitLower:', ldl)#'Type:', QC, ' Element:', Elem, 
    return dfz2

def resumen(dataf, datafh, maxx,QC, org, dup, Elem, ldl, ARD):
    df = filter_calcARD(dataf, QC, org, dup, Elem, ldl, ARD)
    df1x = df[df['Success'].str.startswith('y')]
    pearson  = df[org].corr(df[dup])
    p = round(pearson**2,2)
    X = df[org].values[:,np.newaxis]
    y = df[dup].values
    ##
    n_total = len(df1x)
    n_warnings = len(df1x[df1x['Success'] == 'y'])
    pctDups = (n_warnings/n_total)*100
    ##
    # Visualización
    plt.rcParams['axes.grid'] = True # fijar grillas en On para cada subfigura
    fig = plt.figure(tight_layout=True, figsize=(10, 7))
    figGrid = gridspec.GridSpec(2, 2)
    skatter = fig.add_subplot(figGrid[0, 0])
    mapS = fig.add_subplot(figGrid[0, 1])
    MapX = fig.add_subplot(figGrid[1, :])

    # skatter plot
    skatter.plot(df1x[org], df1x[dup], 'ob', markersize=4, alpha=0.2)
    model = LinearRegression()
    model.fit(X, y)
    skatter.plot(X, model.predict(X), lw=1, color='k')
    #
    line = mlines.Line2D([0, 1], [0, 1], lw=1, color='red', label='Y=X $r^2$: '+str(p*100)+'%')
    transform = skatter.transAxes
    line.set_transform(transform)
    skatter.add_line(line)
    #
    line1 = mlines.Line2D([0, 1], [0, 1-ARD], lw=0.75, color='g', label='+/-'+str(ARD*100))
    transform1 = skatter.transAxes
    line1.set_transform(transform1)
    skatter.add_line(line1)
    #
    line2 = mlines.Line2D([0, 1], [0, 1+ARD], lw=0.75, color='g')#, label='-'+str(ARD*100)
    transform2 = skatter.transAxes
    line2.set_transform(transform2)
    skatter.add_line(line2)
    #
    skatter.legend(loc='lower right')
    skatter.set_ylabel(Elem+' Duplicate')
    skatter.set_xlabel(Elem+' Original')
    skatter.set_ylim(ymin=0, ymax=maxx )
    skatter.set_xlim(xmin=0, xmax=maxx )
    #plt.title(items+': \n as of June 30, 2018')
    # mapS plot
    df = df.sort_values("difREL")
    df1 = df.loc[:,['difREL']]
    xs = df1.count()
    x1s = xs.values
    df1.loc[:,'Order'] = np.arange(1,x1s+1)
    df1.loc[:,['FrexAcum']] = (df1['Order']/x1s)*100
    mapS.plot(df1['difREL'], df1['FrexAcum'], 'ob', markersize=4, alpha=0.2)
    mapS.set_ylim(ymin=0, ymax=101)
    mapS.set_xlim(xmin=-70, xmax=70)
    mapS.axvline(x=(ARD*100), color='g', linestyle='-', linewidth=1)
    mapS.axvline(x=-(ARD*100), color='g', linestyle='-', linewidth=1)
    mapS.axvline(x=-(ARD*100/3), color='r', linestyle='-', linewidth=1)
    mapS.axvline(x=(ARD*100/3), color='r', linestyle='-', linewidth=1)
    mapS.set_ylabel('Cumulative Frequency (%)')
    mapS.set_xlabel('Relative Difference (%)')
    
    # MAPD plot
    df1 = filter_calcHARD(datafh, QC, org, dup, Elem, ldl)
    xxx = df1.groupby('succs').agg(min_val = ('difABS','min'), percentile_90 = ('difABS',lambda x: x.quantile(0.9)))
    per = xxx['percentile_90'].mean()
    maxs = df1['percentile'].max()
    MapX.plot(df1['percentile'],df1['difABS'], 'ob', markersize=4, alpha=0.2)  
    MapX.annotate('90th Percentile = '+ str(round(per*100,1))+'%', xy=(0.15,maxs-.2), xycoords="data",
                  va="center", ha="center",
                  bbox=dict(boxstyle="round", fc="w"))
    ax = [0,0.9]
    ay = [per,per]
    MapX.plot(ax, ay, linewidth=1, color="r")
    ax1 = [0.9,0.9]
    ay1 = [0,per]
    MapX.plot(ax1, ay1, linewidth=1, color="r")
    MapX.set_xlim(xmin=0, xmax=1)
    MapX.set_ylim(ymin=0, ymax=maxs)
    MapX.set_ylabel('MAPD')
    MapX.set_xlabel('Percentile')
    fig.align_labels()
    plt.savefig(QC+Elem+'.jpg', dpi=600,
            bbox_inches='tight', transparent=True)
    plt.show()
# # Reports


#data_dup = readdata('./xxArizaro/Datos de Muestras/CheckAssay/Results_CheckAssay_ARD09-29_Au_ACME_ALS.csv')
#data_dup.info()
#data_dup

#df=filter_calcARD(data_dup, 'CHECK ASSAY', 'OriginalResult', 'RepeatResult', 'Au', 0.01, 0.15)

#df=filter_calcHARD(data_dup, 'CHECK ASSAY', 'OriginalResult', 'RepeatResult', 'Au', 0.01)

#resumen(dataf, QC, org, dup, Elem, ldl, ARD)
#resumen(data_dup, 2,'CHECK ASSAY', 'OriginalResult', 'RepeatResult', 'Au', 0.01, 0.15)
def plot_r(df, orgs, dups, Elem):   
    X = df[orgs].values[:,np.newaxis]
    y = df[dups].values
    # Fit line using all data
    lr = linear_model.LinearRegression()
    lr.fit(X, y)

    # Robustly fit linear model with RANSAC algorithm
    ransac = linear_model.RANSACRegressor(estimator=LinearRegression(),
                         min_samples=50, max_trials=100,
                         random_state=42,
                         residual_threshold=5)
    ransac.fit(X, y)
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)

    # Predict data of estimated models
    line_X = np.arange(X.min(), X.max())[:, np.newaxis]
    line_y = lr.predict(line_X)
    line_y_ransac = ransac.predict(line_X)

    # Compare estimated coefficients
    print(Elem+" Estimated coefficients (true, linear regression, RANSAC):")
    print(lr.coef_, ransac.estimator_.coef_)#coef, 
    print(len(inlier_mask))
    print(len(outlier_mask))
    #print(np.count(outlier_mask))
    
    lw = 1
    plt.scatter(X[inlier_mask], y[inlier_mask], color='b', marker='o',s=10,
            label='Inliers')
    plt.scatter(X[outlier_mask], y[outlier_mask], color='r', marker='x',s=5,
            label='Outliers')
    plt.plot(line_X, line_y, color='navy', linewidth=lw, label='Linear regressor')
    plt.plot(line_X, line_y_ransac, color='cornflowerblue', linewidth=lw,
             label='RANSAC regressor')
    plt.xlim(0, 120 )
    plt.ylim(0, 120 )
    plt.legend(loc='upper left')
    plt.xlabel("Input")
    plt.ylabel("Response")
    plt.show()
# End

def resumen2(dataf, datafh, maxx,QC, org, dup, Elem, ldl, ARD):
    df = filter_calcARD(dataf, QC, org, dup, Elem, ldl, ARD)
    df1x = df[df['Success'].str.startswith('y')]
    pearson  = df[org].corr(df[dup])
    p = round(pearson**2,2)
    X = df[org].values[:,np.newaxis]
    y = df[dup].values
    ##
    n_total = len(df1x)
    n_warnings = len(df1x[df1x['Success'] == 'y'])
    pctDups = (n_warnings/n_total)*100
    ##
    # Visualización
    plt.rcParams['axes.grid'] = True # fijar grillas en On para cada subfigura
    fig = plt.figure(tight_layout=True, figsize=(10, 7))
    figGrid = gridspec.GridSpec(2, 2)
    skatter = fig.add_subplot(figGrid[0, 0])
    mapS = fig.add_subplot(figGrid[0, 1])
    MapX = fig.add_subplot(figGrid[1, :])

    # skatter plot
    # Fit line using all data
    lr = linear_model.LinearRegression()
    lr.fit(X, y)

    # Robustly fit linear model with RANSAC algorithm
    ransac = linear_model.RANSACRegressor(estimator=LinearRegression(),
                         min_samples=50, max_trials=100,
                         random_state=42,
                         residual_threshold=5)
    ransac.fit(X, y)
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)

    outliers_x = X[outlier_mask]
    outliers_y = y[outlier_mask]

    # Predict data of estimated models
    line_X = np.arange(X.min(), X.max())[:, np.newaxis]
    line_y = lr.predict(line_X)
    line_y_ransac = ransac.predict(line_X)

    # Compare estimated coefficients
    #print(Elem+" Estimated coefficients (true, linear regression, RANSAC):")
    #print(lr.coef_, ransac.estimator_.coef_)#coef, 
    print(len(outliers_x))
    print(len(outliers_y))
    #print(np.count(outlier_mask))
    
    lw = 1
    skatter.scatter(X[inlier_mask], y[inlier_mask], color='b', marker='o',s=10,
            label='Inliers')
    skatter.scatter(X[outlier_mask], y[outlier_mask], color='r', marker='x',s=5,
            label='Outliers')
    skatter.plot(line_X, line_y, color='navy', linewidth=lw, label='Linear regressor')
    skatter.plot(line_X, line_y_ransac, color='cornflowerblue', linewidth=lw,
             label='RANSAC regressor')
    skatter.set_ylim(ymin=0, ymax=maxx )
    skatter.set_xlim(xmin=0, xmax=maxx )
    skatter.legend()
    skatter.set_ylabel(Elem+' Duplicate')
    skatter.set_xlabel(Elem+' Original')


    #skatter.plot(df1x[org], df1x[dup], 'ob', markersize=4, alpha=0.2)
#    model = LinearRegression()
#    model.fit(X, y)
#    skatter.plot(X, model.predict(X), lw=1, color='k')
    #
    line = mlines.Line2D([0, 1], [0, 1], lw=1, color='red', label='Y=X $r^2$: '+str(p*100)+'%')
    transform = skatter.transAxes
    line.set_transform(transform)
    skatter.add_line(line)
    #
    line1 = mlines.Line2D([0, 1], [0, 1-ARD], lw=0.75, color='g', label='+/-'+str(ARD*100))
    transform1 = skatter.transAxes
    line1.set_transform(transform1)
    skatter.add_line(line1)
    #
    line2 = mlines.Line2D([0, 1], [0, 1+ARD], lw=0.75, color='g')#, label='-'+str(ARD*100)
    transform2 = skatter.transAxes
    line2.set_transform(transform2)
    skatter.add_line(line2)
    #
    skatter.legend()#loc='lower right'
    skatter.set_ylabel(Elem+' Duplicate')
    skatter.set_xlabel(Elem+' Original')
    skatter.set_ylim(ymin=0, ymax=maxx )
    skatter.set_xlim(xmin=0, xmax=maxx )
    #plt.title(items+': \n as of June 30, 2018')
    # mapS plot
    df = df.sort_values("difREL")
    df1 = df.loc[:,['difREL']]
    xs = df1.count()
    x1s = xs.values
    df1.loc[:,'Order'] = np.arange(1,x1s+1)
    df1.loc[:,['FrexAcum']] = (df1['Order']/x1s)*100
    mapS.plot(df1['difREL'], df1['FrexAcum'], 'ob', markersize=4, alpha=0.2)
    mapS.set_ylim(ymin=0, ymax=101)
    mapS.set_xlim(xmin=-70, xmax=70)
    mapS.axvline(x=(ARD*100), color='g', linestyle='-', linewidth=1)
    mapS.axvline(x=-(ARD*100), color='g', linestyle='-', linewidth=1)
    mapS.axvline(x=-(ARD*100/3), color='r', linestyle='-', linewidth=1)
    mapS.axvline(x=(ARD*100/3), color='r', linestyle='-', linewidth=1)
    mapS.set_ylabel('Cumulative Frequency (%)')
    mapS.set_xlabel('Relative Difference (%)')
    
    # MapX plot
    df1 = filter_calcHARD(datafh, QC, org, dup, Elem, ldl)
    xxx = df1.groupby('DataSet').agg(min_val = ('difABS','min'), percentile_90 = ('difABS',lambda x: x.quantile(0.9)))
    per = xxx['percentile_90'].mean()
    maxs = df1['percentile'].max()
    MapX.plot(df1['percentile'],df1['difABS'], 'ob', markersize=4, alpha=0.2)  
    MapX.annotate('90th Percentile = '+ str(round(per*100,1))+'%', xy=(0.15,maxs-.2), xycoords="data",
                  va="center", ha="center",
                  bbox=dict(boxstyle="round", fc="w"))
    ax = [0,0.9]
    ay = [per,per]
    MapX.plot(ax, ay, linewidth=1, color="r")
    ax1 = [0.9,0.9]
    ay1 = [0,per]
    MapX.plot(ax1, ay1, linewidth=1, color="r")
    MapX.set_xlim(xmin=0, xmax=1)
    MapX.set_ylim(ymin=0, ymax=maxs)
    MapX.set_ylabel('MAPD')
    MapX.set_xlabel('Percentile')
    fig.align_labels()
    plt.savefig(QC+Elem+'.jpg', dpi=600,
            bbox_inches='tight', transparent=True)
    plt.show()
