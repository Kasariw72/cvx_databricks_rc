# Databricks notebook source
# MAGIC %md 
# MAGIC # Compare the effect of different scalers on data with outliers
# MAGIC 
# MAGIC 
# MAGIC Feature 0 (median income in a block) and feature 5 (number of households) of
# MAGIC the `California housing dataset
# MAGIC <http://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html>`_ have very
# MAGIC different scales and contain some very large outliers. These two
# MAGIC characteristics lead to difficulties to visualize the data and, more
# MAGIC importantly, they can degrade the predictive performance of many machine
# MAGIC learning algorithms. Unscaled data can also slow down or even prevent the
# MAGIC convergence of many gradient-based estimators.
# MAGIC 
# MAGIC Indeed many estimators are designed with the assumption that each feature takes
# MAGIC values close to zero or more importantly that all features vary on comparable
# MAGIC scales. In particular, metric-based and gradient-based estimators often assume
# MAGIC approximately standardized data (centered features with unit variances). A
# MAGIC notable exception are decision tree-based estimators that are robust to
# MAGIC arbitrary scaling of the data.
# MAGIC 
# MAGIC This example uses different scalers, transformers, and normalizers to bring the
# MAGIC data within a pre-defined range.
# MAGIC 
# MAGIC Scalers are linear (or more precisely affine) transformers and differ from each
# MAGIC other in the way to estimate the parameters used to shift and scale each
# MAGIC feature.
# MAGIC 
# MAGIC ``QuantileTransformer`` provides non-linear transformations in which distances
# MAGIC between marginal outliers and inliers are shrunk. ``PowerTransformer`` provides
# MAGIC non-linear transformations in which data is mapped to a normal distribution to
# MAGIC stabilize variance and minimize skewness.
# MAGIC 
# MAGIC Unlike the previous transformations, normalization refers to a per sample
# MAGIC transformation instead of a per feature transformation.
# MAGIC 
# MAGIC The following code is a bit verbose, feel free to jump directly to the analysis
# MAGIC of the results_.

# COMMAND ----------

#%load_ext autoreload
#%autoreload 2
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

# basic functionalities
import re
import os
import sys
import datetime
import itertools
import math

# data transforamtion and manipulation
import pandas as pd
#import pandas_datareader.data as web
import numpy as np
# prevent crazy long pandas prints
pd.options.display.max_columns = 16
pd.options.display.max_rows = 16
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(precision=5, suppress=True)

# COMMAND ----------

#1 - import module
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.functions import when
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from pyspark.sql.types import DoubleType
from pyspark.sql.types import IntegerType
# import numpy
# import pandas

# COMMAND ----------

## File Part in HDP Server to be stored images of charts or graph or temp files
path = "/databricks/driver/RC_T_MAIN_DATA_01/"
sparkAppName = "RC_T_MAIN_DATA_01"
hdfsPathCSV = "dbfs:/RC/MSATER_DATA_FILES/RC_T_MAIN_DATA0*.csv"

# COMMAND ----------

#4 - Read file to spark DataFrame
from pyspark.sql import SQLContext
sqlContext = SQLContext(spark)
df = (sqlContext
        .read
        .option("header","true")
        .option("inferSchema", "true")
        .csv(hdfsPathCSV)
        #.parquet(hdfsPathParquet)
     )
# If the path don't have file:/// -> it will call hdfs instead of local file system
df.cache()
print("finish caching data in Spark!")

# COMMAND ----------

#6 - change column name
renamed_df = df.selectExpr(
"OBJECT_CODE as CODE",
"TIME as DAYTIME",
"N_YEAR as YEAR",
"N_MONTH as MONTH",
"N_DAY as DAY",
"N_HOUR as HOUR",
"N_MM as MM",
"CYCLE",
"ROUND(45 - LUBE_OIL_PRESS, 7) AS RESI_LL_LUBE_OIL_PRESS", 
"ROUND(LUBE_OIL_TEMP - 190,7) AS RESI_HH_LUBE_OIL_TEMP", 
"ROUND(820 - SPEED,7) AS RESI_LL_SPEED", 
"ROUND(SPEED - 1180,7) AS RESI_HH_SPEED", 
"ROUND(0.33 - VIBRA_ENGINE,7) AS RESI_LL_VIBRATION", 
"ROUND(VIBRA_ENGINE - 0.475,7) AS RESI_HH_VIBRATION", 
"ROUND(293.33 - THROW_1_DISC_TEMP,7) AS RESI_HH_THROW1_DIS_TEMP", 
"ROUND(THROW_1_SUC_TEMP - 190,7) AS RESI_HH_THROW1_SUC_TEMP", 
'LUBE_OIL_PRESS',
'LUBE_OIL_TEMP',
'THROW_1_DISC_PRESS',
'THROW_1_DISC_TEMP',
'THROW_1_SUC_PRESS',
'THROW_1_SUC_TEMP',
'THROW_2_DISC_PRESS',
'THROW_2_DISC_TEMP',
'THROW_2_SUC_PRESS',
'THROW_2_SUC_TEMP',
'THROW_3_DISC_PRESS',
'THROW_3_DISC_TEMP',
'THROW_3_SUC_PRESS',
'THROW_3_SUC_TEMP',
'THROW_4_DISC_PRESS',
'THROW_4_DISC_TEMP',
'VIBRATION',
'CYL_1_TEMP',
'CYL_2_TEMP',
'CYL_3_TEMP',
'CYL_4_TEMP',
'CYL_5_TEMP',
'CYL_6_TEMP',
'CYL_7_TEMP',
'CYL_8_TEMP',
'CYL_9_TEMP',
'CYL_10_TEMP',
'CYL_11_TEMP',
'CYL_12_TEMP',
'FUEL_GAS_TEMP',
'LUBE_OIL_PRESS_ENGINE',
'MANI_PRESS',
'RIGHT_BANK_EXH_TEMP',
'RUNNING_STATUS',
'SPEED',
'VIBRA_ENGINE',
# 'S1_RECY_VALVE',
# 'S1_SUCT_PRESS',
# 'S1_SUCT_TEMPE',
# 'S2_STAGE_DISC_PRESS',
# 'S2_SCRU_SCRUB_LEVEL',
# 'GAS_LIFT_HEAD_PRESS',
# 'IN_CONT_CONT_VALVE',
# 'IN_SEP_PRESS',
# 'WH_PRESSURE',
"RUL as RUL",
"RUL*-1 as IN_RUL",
"EVENT_ID as EVENT_ID",
"LABEL1 as LABEL1",
"LABEL2 as LABEL2",
"BF_EVENT_TYPE as BF_SD_TYPE")

# COMMAND ----------

renamed_df.columns

# COMMAND ----------

renamed_df.printSchema()

# COMMAND ----------

columnList = ['RESI_LL_LUBE_OIL_PRESS',
 'RESI_HH_LUBE_OIL_TEMP',
 'RESI_LL_SPEED',
 'RESI_HH_SPEED',
 'RESI_LL_VIBRATION',
 'RESI_HH_VIBRATION',
 'RESI_HH_THROW1_DIS_TEMP',
 'RESI_HH_THROW1_SUC_TEMP',
'LUBE_OIL_PRESS',
'LUBE_OIL_TEMP',
'THROW_1_DISC_PRESS',
'THROW_1_DISC_TEMP',
'THROW_1_SUC_PRESS',
'THROW_1_SUC_TEMP',
'THROW_2_DISC_PRESS',
'THROW_2_DISC_TEMP',
'THROW_2_SUC_PRESS',
'THROW_2_SUC_TEMP',
'THROW_3_DISC_PRESS',
'THROW_3_DISC_TEMP',
'THROW_3_SUC_PRESS',
'THROW_3_SUC_TEMP',
'THROW_4_DISC_PRESS',
'THROW_4_DISC_TEMP',
'VIBRATION',
'CYL_1_TEMP',
'CYL_2_TEMP',
'CYL_3_TEMP',
'CYL_4_TEMP',
'CYL_5_TEMP',
'CYL_6_TEMP',
'CYL_7_TEMP',
'CYL_8_TEMP',
'CYL_9_TEMP',
'CYL_10_TEMP',
'CYL_11_TEMP',
'CYL_12_TEMP',
#'FUEL_GAS_TEMP',
'LUBE_OIL_PRESS_ENGINE',
'MANI_PRESS',
'RIGHT_BANK_EXH_TEMP',
'RUNNING_STATUS',
'SPEED',
'VIBRA_ENGINE']
#,
# 'S1_RECY_VALVE',
# 'S1_SUCT_PRESS',
# 'S1_SUCT_TEMPE',
# 'S2_STAGE_DISC_PRESS',
# 'S2_SCRU_SCRUB_LEVEL',
# 'GAS_LIFT_HEAD_PRESS',
# 'IN_CONT_CONT_VALVE',
# 'IN_SEP_PRESS']

columnListDem = ['YEAR',
 'MONTH',
 'DAY',
 'HOUR',
 'MM',
 'CYCLE',
 'RUL',
 'EVENT_ID',
 'LABEL1',
 'LABEL2']

# COMMAND ----------

### This step is for cleaning data using simple median value of each colum to the missing or unknown value of sensors.
def replaceByMedian(pySparkDF, columnList):
    for colName in columnList:
      #med = pySparkDF.approxQuantile(colName, [0.5], 0.25)
      RULFrom = 1
      RULTo = 1440 
      bfEventType = "UCM"
      med = getDataByCol(colName,RULFrom,RULTo,bfEventType)
      print(med)
      
      pySparkDF = pySparkDF.withColumn(colName,pySparkDF[colName].cast(DoubleType()))
      pySparkDF = pySparkDF.withColumn(colName, when(pySparkDF[colName].isNotNull(),pySparkDF[colName]).otherwise(med))
      
      print("Relaced [",colName,"] by ", med)
    
    return pySparkDF

### This step is for cleaning data using simple median value of each colum to the missing or unknown value of sensors.
def convertStr2Int(pySparkDF, columnList):
    for colName in columnList:
      pySparkDF = pySparkDF.withColumn(colName,pySparkDF[colName].cast(IntegerType()))    
    return pySparkDF

def getDataByCol(colName,RULFrom,RULTo,bfEventType):
  #sql = "select "+colName+" from rc_data where RUL between "+str(RULFrom)+" and "+str(RULTo)+" and BF_SD_TYPE='"+bfEventType+"' and "+colName+"<>'NULL' and "+colName+" is not null"
  sql = "select "+colName+" from rc_data where RUL between "+str(RULFrom)+" and "+str(RULTo)+" and BF_SD_TYPE='"+bfEventType+"' "
  rawdata = spark.sql(sql)
  data = rawdata.withColumn(colName,rawdata[colName].cast(DoubleType()))
  med = data.approxQuantile(colName, [0.5], 0.25)
  return med[0]

# COMMAND ----------

#12 - compute dataframe using sql command via string
renamed_df.createOrReplaceTempView("rc_data")

# COMMAND ----------

renamed_df = convertStr2Int(renamed_df,columnListDem)
renamed_df = replaceByMedian(renamed_df,columnList)

# COMMAND ----------

#12 - compute dataframe using sql command via string
renamed_df.createOrReplaceTempView("rc_data")
## Filter out the data during RC S/D
renamed_df = spark.sql("select * from rc_data where EVENT_ID <> 9999999 and RUL <> 0 ")
renamed_df3 = spark.sql("select EVENT_ID, CODE , COUNT(*) as num_steps from rc_data where EVENT_ID <> 9999999 and RUL <> 9999999 and RUL <> 0 GROUP BY  CODE, EVENT_ID")
renamed_df3.createOrReplaceTempView("tmp_step_event")
renamed_df3 = spark.sql("select EVENT_ID, CODE, COUNT(*) as num_steps from rc_data where EVENT_ID <> 9999999 and RUL <> 9999999 and RUL <> 0 GROUP BY  CODE, EVENT_ID")

# COMMAND ----------

renamed_df.printSchema()

# COMMAND ----------

display(renamed_df.describe())

# COMMAND ----------

# sampleEvent = spark.sql("select * from rc_data where EVENT_ID=3568 and RUL between 1 and 1440 and YEAR=2016 order by IN_RUL ")
# # pick a large window size of 50 cycles
# sequence_length = 1440
# sampleEvent = sampleEvent.toPandas()
# sampleEvent = sampleEvent.sort_values(['IN_RUL'],ascending=True)

# COMMAND ----------

def getPairData(colName,RULFrom,RULTo,eventId, bfEventType):
  
  if eventId=="":
    sql = "select CODE,YEAR,MONTH,RUL, "+colName[0]+","+colName[1]+" from rc_data where RUL between "+str(RULFrom)+" and "+str(RULTo)+" and BF_SD_TYPE='"+bfEventType+"' AND CODE NOT IN ('BEWP-SK1060') order by IN_RUL "
  else:
    sql = "select CODE,YEAR,MONTH,RUL, "+colName[0]+","+colName[1]+" from rc_data where EVENT_ID="+str(eventId)+" and RUL between "+str(RULFrom)+" and "+str(RULTo)+" order by IN_RUL "
  
  #and CODE='LAWB-SK1060'
  sampleEvent = spark.sql(sql)
  
  # pick a large window size of 50 cycles
  ##sequence_length = 1440
  #sampleEvent = sampleEvent.toPandas()
  ##sampleEvent = sampleEvent.sort_values(['IN_RUL'],ascending=True)
  #pariDataSet = sampleEvent[sampleEvent['RUL'] <= sampleEvent['RUL'].min() + sequence_length]
  #sampleEvent = sampleEvent[colName]
  
  return sampleEvent

def getDataCount(colName,RULFrom,RULTo,eventId, bfEventType):
  
#   sql = "select LUBE_OIL_TEMP, LUBE_OIL_PRESS as LUBE_OIL_PRESS from rc_data where RUL between "+str(RULFrom)+" and "+str(RULTo)+" and LUBE_OIL_TEMP>0 and LUBE_OIL_PRESS >0 and LUBE_OIL_TEMP<=60 and THROW_1_SUC_PRESS <2000 "
  
  sql = "select CODE,YEAR,MONTH,RUL, "+colName[0]+","+colName[1]+" from rc_data where RUL between "+str(RULFrom)+" and "+str(RULTo)+" "
  #and THROW_1_SUC_PRESS >0 and THROW_1_SUC_PRESS <2000
  
  
  #and CODE='LAWB-SK1060'
  sampleEvent = spark.sql(sql)
  sampleEvent = sampleEvent.toPandas()
  return sampleEvent

# COMMAND ----------

#allCols = ['LUBE_OIL_PRESS','LUBE_OIL_TEMP','RUL','CODE','MONTH','YEAR']
allCols = ['THROW_1_DISC_PRESS','THROW_1_DISC_TEMP','RUL','CODE','MONTH','YEAR']
pairCols = ['THROW_1_DISC_PRESS','THROW_1_DISC_TEMP']
#pairCols = ['THROW_1_DISC_PRESS','THROW_1_DISC_TEMP']
eventId = ""
#3568
bfEventType = "UCM"
dataset = getPairData(allCols,1,720,eventId, bfEventType)
#dataset = getDataCount(pairCols,1,420,eventId, bfEventType)

# COMMAND ----------

#display(dataset.where("CODE = 'LAWE-ME-E7400'"))
display(dataset)

# COMMAND ----------

#display(dataset.where("CODE = 'LAWE-ME-E7400'"))
display(dataset)

# COMMAND ----------

display(dataset)

# COMMAND ----------

display(dataset)

# COMMAND ----------

pdDataDF = dataset.toPandas()

# COMMAND ----------

#data_pivots = dataset.pivot_table(index='CODE', columns='origin', values='arr_delay')

data_pivots = pdDataDF.pivot_table(index=['YEAR'], columns='MONTH',values=pairCols[0] , aggfunc='mean')
data_pivots.plot(kind='box', figsize=[16,8])
display()

# COMMAND ----------

#data_pivots = dataset.pivot_table(index='CODE', columns='origin', values='arr_delay')

data_pivots = pdDataDF.pivot_table(index=['YEAR'], columns='MONTH',values=pairCols[1] , aggfunc='mean')
data_pivots.plot(kind='box', figsize=[16,8])
display()

# COMMAND ----------

# Import library and dataset
import seaborn as sns
import matplotlib.pyplot as plt
##df = sns.load_dataset('iris')

# Cut the window in 2 parts
f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
 
# Add a graph in each part
sns.boxplot(pdDataDF[pairCols[0]], ax=ax_box)
sns.distplot(pdDataDF[pairCols[0]], ax=ax_hist)
 
# Remove x axis name for the boxplot
ax_box.set(xlabel=pairCols[0])
display()

# COMMAND ----------

# Cut the window in 2 parts
f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
 
# Add a graph in each part
sns.boxplot(pdDataDF[pairCols[1]], ax=ax_box)
sns.distplot(pdDataDF[pairCols[1]], ax=ax_hist)
 
# Remove x axis name for the boxplot
ax_box.set(xlabel=pairCols[1])
display()

# COMMAND ----------

# MAGIC %matplotlib inline

# COMMAND ----------

# Author:  Raghav RV <rvraghav93@gmail.com>
#          Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Thomas Unterthiner
# License: BSD 3 clause

from __future__ import print_function

import numpy as np

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import cm

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn import preprocessing

#from sklearn.preprocessing import PowerTransformer
#from sklearn.datasets import fetch_california_housing

def prepareScaler(pairCols, pdDataDF):
  print(__doc__)
  ##datasetTest = fetch_california_housing()
  ##X_full, y_full = datasetTest.data, datasetTest.target
  dataSampleCol = pairCols
  dataSampleColTarget =  pairCols[1]

  dataSet1 = pdDataDF[dataSampleCol]
  dataTarget = pdDataDF[dataSampleColTarget]

  X_full = dataSet1.as_matrix()
  y_full =dataTarget.as_matrix()

  y_full = y_full.reshape(1,len(dataTarget))
  
  N_MinMaxScaler = preprocessing.MinMaxScaler()
  #N_minmax_scale = preprocessing.minmax_scale()
  N_MaxAbsScaler = preprocessing.MinMaxScaler()
  N_StandardScaler = preprocessing.StandardScaler()
  N_RobustScaler = preprocessing.RobustScaler(quantile_range=(25, 75))
  N_QuantileTransformerGaussian = preprocessing.QuantileTransformer(output_distribution='normal')
  N_QuantileTransformer = preprocessing.QuantileTransformer(output_distribution='uniform')
  N_Normalizer= preprocessing.Normalizer()
  
  #sampleEvent['RUL'].min() 
  #y_full = np.arange(1440)
  # Take only 2 features to make visualization easier

  # Feature of 0 has a long tail distribution.
  # Feature 5 has a few but very large outliers.
  X = X_full[:, [0, 1]]
  
  print("len X_Full", len(X_full))
  print("len y_full", len(y_full[0]))
  print("y_full", y_full)
  
  distributions = [
      ('Unscaled data', X),
      ('Data after standard scaling',
          StandardScaler().fit_transform(X)),
      ('Data after min-max scaling',
         N_MinMaxScaler.fit_transform(X)),
      ('Data after max-abs scaling',
          N_MaxAbsScaler.fit_transform(X)),
      ('Data after robust scaling',
          N_RobustScaler.fit_transform(X)),
  #     ('Data after power transformation (Yeo-Johnson)',
  #      PowerTransformer(method='yeo-johnson').fit_transform(X)),
  #     ('Data after power transformation (Box-Cox)',
  #      PowerTransformer(method='box-cox').fit_transform(X)),
      ('Data after quantile transformation (gaussian pdf)',
          N_QuantileTransformerGaussian.fit_transform(X)),
      ('Data after quantile transformation (uniform pdf)',
          N_QuantileTransformer.fit_transform(X)),
      ('Data after sample-wise L2 normalizing',
          N_Normalizer.fit_transform(X)),
      ('Data after robust scaling & Quantile transformation (uniform pdf)',
          N_QuantileTransformer.fit_transform(N_RobustScaler.fit_transform(X)))
          ,
      ('Data after robust scaling & Data after quantile transformation (gaussian pdf)',
          N_QuantileTransformerGaussian.fit_transform(N_RobustScaler.fit_transform(X))),
  ]

  # scale the output between 0 and 1 for the colorbar
  #y = minmax_scale(y_full)
  y = minmax_scale(y_full[0])

  # plasma does not exist in matplotlib < 1.5
  cmap = getattr(cm, 'plasma_r', cm.hot_r)
  return distributions, X, y,y_full,cmap

# COMMAND ----------

def create_axes(title, figsize=(16, 6)):
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title)

    # define the axis for the first plot
    left, width = 0.1, 0.22
    bottom, height = 0.1, 0.7
    bottom_h = height + 0.15
    left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.1]
    rect_histy = [left_h, bottom, 0.05, height]

    ax_scatter = plt.axes(rect_scatter)
    ax_histx = plt.axes(rect_histx)
    ax_histy = plt.axes(rect_histy)

    # define the axis for the zoomed-in plot
    left = width + left + 0.2
    left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.1]
    rect_histy = [left_h, bottom, 0.05, height]

    ax_scatter_zoom = plt.axes(rect_scatter)
    ax_histx_zoom = plt.axes(rect_histx)
    ax_histy_zoom = plt.axes(rect_histy)

    # define the axis for the colorbar
    left, width = width + left + 0.13, 0.01

    rect_colorbar = [left, bottom, width, height]
    ax_colorbar = plt.axes(rect_colorbar)

    return ((ax_scatter, ax_histy, ax_histx),
            (ax_scatter_zoom, ax_histy_zoom, ax_histx_zoom),
            ax_colorbar)


def plot_distribution(axes, X, y, hist_nbins=50, title="",
                      x0_label="", x1_label="", cmap=[]):
    ax, hist_X1, hist_X0 = axes

    ax.set_title(title)
    ax.set_xlabel(x0_label)
    ax.set_ylabel(x1_label)

    # The scatter plot
    colors = cmap(y)
    #colors=np.array([[120,130,140]])/256.0
    #colors = plt.cm.Spectral
    
    ax.scatter(X[:, 0], X[:, 1], alpha=0.5, marker='o', s=5, lw=0, c=colors)

    # Removing the top and the right spine for aesthetics
    # make nice axis layout
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))

    # Histogram for axis X1 (feature 5)
    hist_X1.set_ylim(ax.get_ylim())
    hist_X1.hist(X[:, 1], bins=hist_nbins, orientation='horizontal',
                 color='grey', ec='grey')
    hist_X1.axis('off')

    # Histogram for axis X0 (feature 0)
    hist_X0.set_xlim(ax.get_xlim())
    hist_X0.hist(X[:, 0], bins=hist_nbins, orientation='vertical',
                 color='grey', ec='grey')
    hist_X0.axis('off')

# COMMAND ----------

# MAGIC %md Two plots will be shown for each scaler/normalizer/transformer. The left
# MAGIC figure will show a scatter plot of the full data set while the right figure
# MAGIC will exclude the extreme values considering only 99 % of the data set,
# MAGIC excluding marginal outliers. In addition, the marginal distributions for each
# MAGIC feature will be shown on the side of the scatter plot.

# COMMAND ----------

def make_plot(item_idx,pairCols, pdDataDF):
    
    distributions, X, y,y_full,cmap = prepareScaler(pairCols, pdDataDF)
    
    title, X = distributions[item_idx]
    ax_zoom_out, ax_zoom_in, ax_colorbar = create_axes(title)
    axarr = (ax_zoom_out, ax_zoom_in)
    plot_distribution(axarr[0], X, y, hist_nbins=200,
                      x0_label=pairCols[0],
                      x1_label=pairCols[1],
                      title="Full data",cmap=cmap)

    # zoom-in
    zoom_in_percentile_range = (0, 99)
    cutoffs_X0 = np.percentile(X[:, 0], zoom_in_percentile_range)
    cutoffs_X1 = np.percentile(X[:, 1], zoom_in_percentile_range)

    non_outliers_mask = (
        np.all(X > [cutoffs_X0[0], cutoffs_X1[0]], axis=1) &
        np.all(X < [cutoffs_X0[1], cutoffs_X1[1]], axis=1))
    plot_distribution(axarr[1], X[non_outliers_mask], y[non_outliers_mask],
                      hist_nbins=200,
                      x0_label=pairCols[0],
                      x1_label=pairCols[1],
                      title="Zoom-in",cmap=cmap)

    norm = mpl.colors.Normalize(y_full.min(), y_full.max())
    mpl.colorbar.ColorbarBase(ax_colorbar, cmap=cmap,
                              norm=norm, orientation='vertical',
                              label='Color mapping for values of y')
    
#     imgPath = "file:/databricks/driver/"+pairCols[0]+"_"+pairCols[1]+"_"+title+".png"
#     mpl.savefig(imgPath)

# COMMAND ----------

# MAGIC %md 
# MAGIC Original data
# MAGIC -------------
# MAGIC 
# MAGIC Each transformation is plotted showing two transformed features, with the
# MAGIC left plot showing the entire dataset, and the right zoomed-in to show the
# MAGIC dataset without the marginal outliers. A large majority of the samples are
# MAGIC compacted to a specific range, [0, 10] for the median income and [0, 6] for
# MAGIC the number of households. Note that there are some marginal outliers (some
# MAGIC blocks have more than 1200 households). Therefore, a specific pre-processing
# MAGIC can be very beneficial depending of the application. In the following, we
# MAGIC present some insights and behaviors of those pre-processing methods in the
# MAGIC presence of marginal outliers.

# COMMAND ----------

print(pairCols)

# COMMAND ----------

make_plot(0,pairCols,pdDataDF)
display()

# COMMAND ----------

# MAGIC %md StandardScaler
# MAGIC --------------
# MAGIC 
# MAGIC ``StandardScaler`` removes the mean and scales the data to unit variance.
# MAGIC However, the outliers have an influence when computing the empirical mean and
# MAGIC standard deviation which shrink the range of the feature values as shown in
# MAGIC the left figure below. Note in particular that because the outliers on each
# MAGIC feature have different magnitudes, the spread of the transformed data on
# MAGIC each feature is very different: most of the data lie in the [-2, 4] range for
# MAGIC the transformed median income feature while the same data is squeezed in the
# MAGIC smaller [-0.2, 0.2] range for the transformed number of households.
# MAGIC 
# MAGIC ``StandardScaler`` therefore cannot guarantee balanced feature scales in the
# MAGIC presence of outliers.

# COMMAND ----------

make_plot(1,pairCols,pdDataDF)
display()

# COMMAND ----------

# MAGIC %md MinMaxScaler
# MAGIC ------------
# MAGIC 
# MAGIC ``MinMaxScaler`` rescales the data set such that all feature values are in
# MAGIC the range [0, 1] as shown in the right panel below. However, this scaling
# MAGIC compress all inliers in the narrow range [0, 0.005] for the transformed
# MAGIC number of households.
# MAGIC 
# MAGIC As ``StandardScaler``, ``MinMaxScaler`` is very sensitive to the presence of
# MAGIC outliers.

# COMMAND ----------

make_plot(2,pairCols,pdDataDF)
display()

# COMMAND ----------

# MAGIC %md MaxAbsScaler
# MAGIC ------------
# MAGIC 
# MAGIC ``MaxAbsScaler`` differs from the previous scaler such that the absolute
# MAGIC values are mapped in the range [0, 1]. On positive only data, this scaler
# MAGIC behaves similarly to ``MinMaxScaler`` and therefore also suffers from the
# MAGIC presence of large outliers.

# COMMAND ----------

make_plot(3,pairCols,pdDataDF)
display()

# COMMAND ----------

# MAGIC %md RobustScaler
# MAGIC ------------
# MAGIC 
# MAGIC Unlike the previous scalers, the centering and scaling statistics of this
# MAGIC scaler are based on percentiles and are therefore not influenced by a few
# MAGIC number of very large marginal outliers. Consequently, the resulting range of
# MAGIC the transformed feature values is larger than for the previous scalers and,
# MAGIC more importantly, are approximately similar: for both features most of the
# MAGIC transformed values lie in a [-2, 3] range as seen in the zoomed-in figure.
# MAGIC Note that the outliers themselves are still present in the transformed data.
# MAGIC If a separate outlier clipping is desirable, a non-linear transformation is
# MAGIC required (see below).

# COMMAND ----------

make_plot(4,pairCols,pdDataDF)
display()

# COMMAND ----------

# MAGIC %md PowerTransformer
# MAGIC ----------------
# MAGIC 
# MAGIC ``PowerTransformer`` applies a power transformation to each feature to make
# MAGIC the data more Gaussian-like. Currently, ``PowerTransformer`` implements the
# MAGIC Yeo-Johnson and Box-Cox transforms. The power transform finds the optimal
# MAGIC scaling factor to stabilize variance and mimimize skewness through maximum
# MAGIC likelihood estimation. By default, ``PowerTransformer`` also applies
# MAGIC zero-mean, unit variance normalization to the transformed output. Note that
# MAGIC Box-Cox can only be applied to strictly positive data. Income and number of
# MAGIC households happen to be strictly positive, but if negative values are present
# MAGIC the Yeo-Johnson transformed is to be preferred.

# COMMAND ----------

# make_plot(5)
# make_plot(6)

# COMMAND ----------

# MAGIC %md QuantileTransformer (Gaussian output)
# MAGIC -------------------------------------
# MAGIC 
# MAGIC ``QuantileTransformer`` has an additional ``output_distribution`` parameter
# MAGIC allowing to match a Gaussian distribution instead of a uniform distribution.
# MAGIC Note that this non-parametetric transformer introduces saturation artifacts
# MAGIC for extreme values.

# COMMAND ----------

make_plot(5,pairCols,pdDataDF)
display()

# COMMAND ----------

# MAGIC %md QuantileTransformer (uniform output)
# MAGIC ------------------------------------
# MAGIC 
# MAGIC ``QuantileTransformer`` applies a non-linear transformation such that the
# MAGIC probability density function of each feature will be mapped to a uniform
# MAGIC distribution. In this case, all the data will be mapped in the range [0, 1],
# MAGIC even the outliers which cannot be distinguished anymore from the inliers.
# MAGIC 
# MAGIC As ``RobustScaler``, ``QuantileTransformer`` is robust to outliers in the
# MAGIC sense that adding or removing outliers in the training set will yield
# MAGIC approximately the same transformation on held out data. But contrary to
# MAGIC ``RobustScaler``, ``QuantileTransformer`` will also automatically collapse
# MAGIC any outlier by setting them to the a priori defined range boundaries (0 and
# MAGIC 1).

# COMMAND ----------

make_plot(6,pairCols,pdDataDF)
display()

# COMMAND ----------

# MAGIC %md Normalizer
# MAGIC ----------
# MAGIC 
# MAGIC The ``Normalizer`` rescales the vector for each sample to have unit norm,
# MAGIC independently of the distribution of the samples. It can be seen on both
# MAGIC figures below where all samples are mapped onto the unit circle. In our
# MAGIC example the two selected features have only positive values; therefore the
# MAGIC transformed data only lie in the positive quadrant. This would not be the
# MAGIC case if some original features had a mix of positive and negative values.

# COMMAND ----------

make_plot(7,pairCols,pdDataDF)
display()
# show()
# #plt.show()

# COMMAND ----------

make_plot(8,pairCols,pdDataDF)
display()
# show()
# #plt.show()

# COMMAND ----------

make_plot(9,pairCols,pdDataDF)
display()
# show()
# #plt.show()

# COMMAND ----------

# from sklearn.pipeline import Pipeline
# from sklearn.ensemble import RandomForestClassifier
# from tsfresh.examples import load_robot_execution_failures
# from tsfresh.transformers import RelevantFeatureAugmenter

# pipeline = Pipeline([('augmenter', RelevantFeatureAugmenter(column_id='id', column_sort='time')),
#             ('classifier', RandomForestClassifier())])

# df_ts, y = load_robot_execution_failures()
# X = pd.DataFrame(index=y.index)

# pipeline.set_params(augmenter__timeseries_container=df_ts)
# pipeline.fit(X, y)

# COMMAND ----------

# importing some libraries
# import numpy as np
from pyspark.sql import functions as F
# from pyspark.sql import SQLContext
# sqlContext = SQLContext(sc)
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import VectorAssembler
# checking if spark context is already created
print(sc.version)
# reading your data as a dataframe
df = spark.sql(" SELECT * FROM rc_data where EVENT_ID IN (SELECT EVENT_ID FROM tmp_step_event where num_steps > 720) AND BF_SD_TYPE IN ('UCM','NMD') ")
#df2 = spark.sql(" SELECT * FROM rc_data")

# COMMAND ----------

df.count()

# COMMAND ----------

#display(df2.select("CODE"))
display(df.select('CODE').groupby('CODE').count())

# COMMAND ----------

df2 = spark.sql(" SELECT YEAR, CODE, BF_SD_TYPE, COUNT(DISTINCT EVENT_ID) as NUM FROM rc_data where EVENT_ID IN (SELECT EVENT_ID FROM tmp_step_event where num_steps > 720) AND BF_SD_TYPE IN ('UCM','NMD') GROUP BY YEAR, CODE, BF_SD_TYPE")

# COMMAND ----------

display(df2.where("BF_SD_TYPE='UCM'").orderBy(["YEAR","CODE"],ascending=[1,0]))

# COMMAND ----------

display(df2.where("BF_SD_TYPE='NMD'").orderBy(["YEAR","CODE"],ascending=[1,0]))

# COMMAND ----------

df3 = spark.sql(" SELECT YEAR, CODE, BF_SD_TYPE, LABEL2, count(*) num FROM rc_data where EVENT_ID IN (SELECT EVENT_ID FROM tmp_step_event where num_steps > 720) AND BF_SD_TYPE IN ('UCM','NMD') GROUP BY YEAR, CODE, BF_SD_TYPE, LABEL2 ")

# COMMAND ----------

display(df3.where("BF_SD_TYPE='UCM'"))

# COMMAND ----------

df3 = spark.sql(" SELECT YEAR, CODE, BF_SD_TYPE, LABEL1, count(*) num FROM rc_data where EVENT_ID IN (SELECT EVENT_ID FROM tmp_step_event where num_steps > 720) AND BF_SD_TYPE IN ('UCM','NMD') GROUP BY YEAR, CODE, BF_SD_TYPE, LABEL2 ")

# COMMAND ----------

df.columns

# COMMAND ----------

featurs = [
  'RESI_LL_LUBE_OIL_PRESS',
 'RESI_HH_LUBE_OIL_TEMP',
 'RESI_LL_SPEED',
 'RESI_HH_SPEED',
 'RESI_LL_VIBRATION',
 'RESI_HH_VIBRATION',
 'RESI_HH_THROW1_DIS_TEMP',
 'RESI_HH_THROW1_SUC_TEMP',
 'LUBE_OIL_PRESS',
 'LUBE_OIL_TEMP',
 'THROW_1_DISC_PRESS',
 'THROW_1_DISC_TEMP',
 'THROW_1_SUC_PRESS',
 'THROW_1_SUC_TEMP',
 'THROW_2_DISC_PRESS',
 'THROW_2_DISC_TEMP',
 'THROW_2_SUC_PRESS',
 'THROW_2_SUC_TEMP',
 'THROW_3_DISC_PRESS',
 'THROW_3_DISC_TEMP',
 'THROW_3_SUC_PRESS',
 'THROW_3_SUC_TEMP',
 'THROW_4_DISC_PRESS',
 'THROW_4_DISC_TEMP',
 'VIBRATION',
 'CYL_1_TEMP',
 'CYL_2_TEMP',
 'CYL_3_TEMP',
 'CYL_4_TEMP',
 'CYL_5_TEMP',
 'CYL_6_TEMP',
 'CYL_7_TEMP',
 'CYL_8_TEMP',
 'CYL_9_TEMP',
 'CYL_10_TEMP',
 'CYL_11_TEMP',
 'CYL_12_TEMP',
 #'FUEL_GAS_TEMP',
 'LUBE_OIL_PRESS_ENGINE',
 'MANI_PRESS',
 'RIGHT_BANK_EXH_TEMP',
 'RUNNING_STATUS',
 'SPEED',
 'VIBRA_ENGINE']

avaliableNormalRangeCols = ['LUBE_OIL_PRESS',
'LUBE_OIL_TEMP',
'THROW_1_DISC_PRESS',
'THROW_1_DISC_TEMP',
'THROW_2_DISC_PRESS',
'THROW_2_DISC_TEMP',
'THROW_3_DISC_PRESS',
'THROW_3_DISC_TEMP',
'THROW_4_DISC_PRESS',
'THROW_4_DISC_TEMP',
'CYL_1_TEMP',
'CYL_2_TEMP',
'CYL_3_TEMP',
'CYL_4_TEMP',
'CYL_5_TEMP',
'CYL_6_TEMP',
'CYL_7_TEMP',
'CYL_8_TEMP',
'CYL_9_TEMP',
'CYL_10_TEMP',
'CYL_11_TEMP',
'CYL_12_TEMP',
'LUBE_OIL_PRESS_ENGINE',
'RIGHT_BANK_EXH_TEMP',
'SPEED',
'VIBRA_ENGINE']

#,'GAS_LIFT_HEAD_PRESS'

normalSensorRange = {'LUBE_OIL_PRESS' :[65,50],
'LUBE_OIL_TEMP' :[180,150],
'THROW_1_DISC_PRESS' :[470,320],
'THROW_1_DISC_TEMP' :[300,200],
'THROW_2_DISC_PRESS' :[1290,1000],
'THROW_2_DISC_TEMP' :[310,200],
'THROW_3_DISC_PRESS' :[470,320],
'THROW_3_DISC_TEMP' :[300,200],
'THROW_4_DISC_PRESS' :[1290,1000],
'THROW_4_DISC_TEMP' :[310,200],
'CYL_1_TEMP' :[1325,800],
'CYL_2_TEMP' :[1325,800],
'CYL_3_TEMP' :[1325,800],
'CYL_4_TEMP' :[1325,800],
'CYL_5_TEMP' :[1325,800],
'CYL_6_TEMP' :[1325,800],
'CYL_7_TEMP' :[1325,800],
'CYL_8_TEMP' :[1325,800],
'CYL_9_TEMP' :[1325,800],
'CYL_10_TEMP' :[1325,800],
'CYL_11_TEMP' :[1325,800],
'CYL_12_TEMP' :[1325,800],
'LUBE_OIL_PRESS_ENGINE' :[65,30],
'RIGHT_BANK_EXH_TEMP' :[1325,800],
'SPEED' :[1200,750],
'VIBRA_ENGINE' :[0.25,0.1]}

#'GAS_LIFT_HEAD_PRESS' :[1290,1000]


# COMMAND ----------

for colName in avaliableNormalRangeCols:
    print("Soomth Extream Noise by Actual Range of Sensor 10%["+colName, "]")
    print(" validating < ", normalSensorRange[colName][0])
    #df_new = df.withColumn(colName, when(df[colName]<=normalSensorRange[colName][0], df[colName]).otherwise(normalSensorRange[colName][0]))
    
    f_new = df.withColumn(colName, when(df[colName]>normalSensorRange[colName][0], normalSensorRange[colName][0] ).otherwise(df[colName]))
    #print(" validating > ", normalSensorRange[colName][1])
    #df_new = df_new.withColumn(colName, when(df_new[colName]<normalSensorRange[colName][1], normalSensorRange[colName][1]).otherwise(df_new[colName]))
    
    #df_new = df.withColumn(colName, F.log(F.when(df[colName] < normalSensorRange[colName][1],normalSensorRange[colName][1]).when(df[colName] > normalSensorRange[colName][0], normalSensorRange[colName][0]).otherwise(df[colName]) +1).alias(colName))

# COMMAND ----------

df_new.select('CYL_1_TEMP').where("CYL_1_TEMP< 800 ").count()

# COMMAND ----------

display(df_new.select('CYL_1_TEMP'))

# COMMAND ----------

display(df_new.select('CYL_1_TEMP').describe())

# COMMAND ----------

pdF = df_new.select('THROW_1_DISC_PRESS','THROW_1_DISC_TEMP').toPandas()

# COMMAND ----------

display(pdF)

# COMMAND ----------

# Cut the window in 2 parts
f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
 
# Add a graph in each part
sns.boxplot(pdF['THROW_1_DISC_PRESS'], ax=ax_box)
sns.distplot(pdF['THROW_1_DISC_TEMP'], ax=ax_hist)
 
# Remove x axis name for the boxplot
ax_box.set(xlabel='THROW_1_DISC_PRESS')
display()

# COMMAND ----------

# empty dictionary d
d = {}
# Fill in the entries one by one
for col in featurs:
  appQuantile = df_new.approxQuantile(col,[0.01,0.99],0.25)
  d[col] = appQuantile
  print(col + ":",appQuantile)

# COMMAND ----------

# settingFeatures = ['RESI_LL_LUBE_OIL_PRESS',
#  'RESI_HH_LUBE_OIL_TEMP',
#  'RESI_LL_SPEED',
#  'RESI_HH_SPEED',
#  'RESI_LL_VIBRATION',
#  'RESI_HH_VIBRATION',
#  'RESI_HH_THROW1_DIS_TEMP',
#  'RESI_HH_THROW1_SUC_TEMP']

# for col in settingFeatures:
#   appQuantile = df.approxQuantile(col,[0.01,0.99],0.25)
#   d[col] = appQuantile
#   print(col + ":",appQuantile)

# COMMAND ----------

print(d)

appQuantile ={'RESI_LL_LUBE_OIL_PRESS': [-55.0, 45.005832], 'RESI_HH_LUBE_OIL_TEMP': [-190.0, -7.6543], 'RESI_LL_SPEED': [-387.7206, 820.0], 'RESI_HH_SPEED': [-1180.0, 27.7206], 'RESI_LL_VIBRATION': [-0.0896955, 0.3301184], 'RESI_HH_VIBRATION': [-0.4751184, -0.0553045], 'RESI_HH_THROW1_DIS_TEMP': [8.96931, 293.33], 'RESI_HH_THROW1_SUC_TEMP': [-190.0, -36.02103], 'LUBE_OIL_PRESS': [-0.0058319517, 100.0], 'LUBE_OIL_TEMP': [0.0, 182.3457], 'THROW_1_DISC_PRESS': [0.0, 520.79382], 'THROW_1_DISC_TEMP': [0.0, 284.36069], 'THROW_1_SUC_PRESS': [0.0, 352.8252], 'THROW_1_SUC_TEMP': [0.0, 153.97897], 'THROW_2_DISC_PRESS': [0.0, 1274.5869], 'THROW_2_DISC_TEMP': [0.0, 302.7619], 'THROW_2_SUC_PRESS': [0.0, 592.4502], 'THROW_2_SUC_TEMP': [0.0, 161.96288], 'THROW_3_DISC_PRESS': [0.0, 545.07574], 'THROW_3_DISC_TEMP': [0.0, 281.87323], 'THROW_3_SUC_PRESS': [0.0, 352.8252], 'THROW_3_SUC_TEMP': [0.0, 153.97897], 'THROW_4_DISC_PRESS': [0.0, 1274.5869], 'THROW_4_DISC_TEMP': [0.0, 299.61646], 'VIBRATION': [-0.0099999998, 0.33007166], 'CYL_1_TEMP': [0.0, 1315.9589], 'CYL_2_TEMP': [0.0, 1352.5356], 'CYL_3_TEMP': [0.0, 1320.5012], 'CYL_4_TEMP': [0.0, 1312.4865], 'CYL_5_TEMP': [0.0, 1349.6497], 'CYL_6_TEMP': [0.0, 1377.9078], 'CYL_7_TEMP': [0.0, 1295.5079], 'CYL_8_TEMP': [0.0, 1319.0], 'CYL_9_TEMP': [0.0, 1318.7515], 'CYL_10_TEMP': [0.0, 1292.4983], 'CYL_11_TEMP': [0.0, 1292.8647], 'CYL_12_TEMP': [0.0, 1313.5933], 'LUBE_OIL_PRESS_ENGINE': [0.0, 100.0], 'MANI_PRESS': [0.0, 149.0], 'RIGHT_BANK_EXH_TEMP': [0.0, 1317.0784], 'RUNNING_STATUS': [0.0, 1.0], 'SPEED': [0.0, 1207.7206], 'VIBRA_ENGINE': [-0.00011842105, 0.4196955]}

# COMMAND ----------



# COMMAND ----------

# looping through the columns, doing log(x+1) transformations
for col in featurs:
  df_new = df.withColumn(col, F.log(F.when(df[col] < d[col][0],d[col][0]).when(df[col] > d[col][1], d[col][1]).otherwise(df[col] ) +1).alias(col))

# COMMAND ----------

display(df_new.describe())

# COMMAND ----------

display(df_new)

# COMMAND ----------

pairCols = ['THROW_1_DISC_PRESS','THROW_1_DISC_TEMP']

spPD = df_new.select(pairCols).toPandas()

# COMMAND ----------

pdDataDF = spPD

# COMMAND ----------

make_plot(6,pairCols, pdDataDF)

# COMMAND ----------

assembler = VectorAssembler().setInputCols(df_new.columns).setOutputCol("features")
transformed = assembler.transform(df_new)
scaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures")
scalerModel =  scaler.fit(transformed.select("features"))
scaledData = scalerModel.transform(transformed)

def extract(row):
  return (row.pmid, )+tuple(row.scaledFeatures.toArray().tolist())

 final_data = scaledData.select("pmid","scaledFeatures").rdd.map(extract).toDF(df.columns)

# COMMAND ----------

# saving the file
final_data.repartition(1).write.csv("file_name.csv")

# COMMAND ----------

import numpy as np

def outliers_iqr(ys):
    quartile_1, quartile_3 = np.percentile(ys, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((ys > upper_bound) | (ys < lower_bound))

# COMMAND ----------

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
plt.style.use('ggplot')
%matplotlib inline  

from pandas.tools.plotting import scatter_matrix

#df_tr_lbl = df.toPandas()

# COMMAND ----------

df_tr_lbl[featurs].std().plot(kind='bar', figsize=(8,6), title="Features Standard Deviation")
display()

# COMMAND ----------

df_tr_lbl[featurs].std().plot(kind='bar', figsize=(8,6), logy=True,title="Features Standard Deviation (log)")

# COMMAND ----------

# get ordered list of top variance features:
featurs_top_var = df_tr_lbl[featurs].std().sort_values(ascending=False)
featurs_top_var

# COMMAND ----------

# get ordered list features correlation with regression label ttf

df_tr_lbl[featurs].corrwith(df_tr_lbl.ttf).sort_values(ascending=False)

# COMMAND ----------

# MAGIC %md
# MAGIC # list of features having low or no correlation with regression label ttf and very low or no variance
# MAGIC # These features will be target for removal in feature selection

# COMMAND ----------

low_cor_featrs = ['setting3', 's1', 's10', 's18','s19','s16','s5', 'setting2', 'setting1']
df_tr_lbl[low_cor_featrs].describe()

# COMMAND ----------

# list of features having high correlation with regression label ttf

correl_featurs = ['s12', 's7', 's21', 's20', 's6', 's14', 's9', 's13', 's8', 's3', 's17', 's2', 's15', 's4', 's11']

df_tr_lbl[correl_featurs].describe()

# COMMAND ----------

# plot a heatmap to display +ve and -ve correlation among features and regression label:
correl_featurs_lbl = featurs 
import seaborn as sns
cm = np.corrcoef(df_tr_lbl[correl_featurs_lbl].values.T)
sns.set(font_scale=1.0)
fig = plt.figure(figsize=(10, 8))
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 8}, yticklabels=correl_featurs_lbl, xticklabels=correl_featurs_lbl)
plt.title('Features Correlation Heatmap')
#plt.show()
display()

# COMMAND ----------



# COMMAND ----------

#reset matplotlib original theme
sns.reset_orig()
#create scatter matrix to disply relatiohships and distribution among features and regression label
scatter_matrix(df_tr_lbl[correl_featurs_lbl], alpha=0.2, figsize=(20, 20), diagonal='kde')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC There is a very high correlation (> 0.8) between some features: (s14, s9), (s11, s4), (s11, s7), (s11, s12), (s4, s12), (s8,s13), (s7, s12)
# MAGIC This may hurt the performance of some ML algorithms.
# MAGIC 
# MAGIC So, some of the above features will be target for removal in feature selection

# COMMAND ----------

# MAGIC %md
# MAGIC Most of the features have normal distribution which has positve effect on machine learning algorithms.
# MAGIC 
# MAGIC Most of the features have non-linear relationship with the regression label ttf, so using polynomial models may lead to better results.
# MAGIC 
# MAGIC Let us create a helper function to ease exploration of each feature invidually:

# COMMAND ----------

def explore_col(s, e):
    
    """Plot 4 main graphs for a single feature.
    
        plot1: histogram 
        plot2: boxplot 
        plot3: line plot (time series over cycle)
        plot4: scatter plot vs. regression label ttf
        
    Args:
        s (str): The column name of the feature to be plotted.
        e (int): The number of random engines to be plotted for plot 3. Range from 1 -100, 0:all engines, >100: all engines.

    Returns:
        plots
    
    """
    
    fig = plt.figure(figsize=(10, 8))


    sub1 = fig.add_subplot(221) 
    sub1.set_title(s +' histogram') 
    sub1.hist(df_tr_lbl[s])

    sub2 = fig.add_subplot(222)
    sub2.set_title(s +' boxplot')
    sub2.boxplot(df_tr_lbl[s])
    
    #np.random.seed(12345)
    
    if e > 100 or e <= 0:
        select_engines = list(pd.unique(df_tr_lbl.id))
    else:
        select_engines = np.random.choice(range(1,101), e, replace=False)
        
    sub3 = fig.add_subplot(223)
    sub3.set_title('time series: ' + s +' / cycle')
    sub3.set_xlabel('cycle')
    for i in select_engines:
        df = df_tr_lbl[['cycle', s]][df_tr_lbl.id == i]
        sub3.plot(df['cycle'],df[s])
    
    sub4 = fig.add_subplot(224)
    sub4.set_title("scatter: "+ s + " / ttf (regr label)")
    sub4.set_xlabel('ttf')
    sub4.scatter(df_tr_lbl['ttf'],df_tr_lbl[s])


    plt.tight_layout()
    plt.show()

# COMMAND ----------

explore_col("s12", 10)

# COMMAND ----------

