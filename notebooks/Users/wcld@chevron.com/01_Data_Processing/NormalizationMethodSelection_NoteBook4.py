# Databricks notebook source
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

spark.conf.set("spark.sql.execution.arrow.enabled", "true")

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
from pyspark.ml.feature import StandardScaler, PCA, RFormula
from pyspark.ml.feature import MinMaxScaler

# import numpy
# import pandas

# COMMAND ----------

## File Part in HDP Server to be stored images of charts or graph or temp files
path = "/databricks/driver/RC_T_MAIN_DATA_01/"
sparkAppName = "RC_T_MAIN_DATA_01"
hdfsPathCSV = "dbfs:/RC/MSATER_DATA_FILES/RC_T_MAIN_DATA*.csv"

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
#'FUEL_GAS_TEMP',
'LUBE_OIL_PRESS_ENGINE',
'MANI_PRESS',
'RIGHT_BANK_EXH_TEMP',
#'RUNNING_STATUS',
'SPEED',
'VIBRA_ENGINE',
'S1_RECY_VALVE',
'S1_SUCT_PRESS',
'S1_SUCT_TEMPE',
'S2_STAGE_DISC_PRESS',
'S2_SCRU_SCRUB_LEVEL',
'GAS_LIFT_HEAD_PRESS',
'IN_CONT_CONT_VALVE',
'IN_SEP_PRESS',
'WH_PRESSURE',
"RUL",
#"RUL*-1 as IN_RUL",
"EVENT_ID",
"LABEL1",
"LABEL2",
"BF_EVENT_TYPE as BF_SD_TYPE")

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
#'RUNNING_STATUS',
'SPEED',
'VIBRA_ENGINE',
'S1_RECY_VALVE',
'S1_SUCT_PRESS',
'S1_SUCT_TEMPE',
'S2_STAGE_DISC_PRESS',
'S2_SCRU_SCRUB_LEVEL',
'GAS_LIFT_HEAD_PRESS',
'IN_CONT_CONT_VALVE',
'IN_SEP_PRESS',
'WH_PRESSURE']

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
      RULFrom = 60
      RULTo = 10080
      
      bfEventType = "UCM"
      med = getDataByCol(colName,RULFrom,RULTo,bfEventType)
      #print(:med)
      
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

# make_plot(0,pairCols,pdDataDF)
# display()

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

# make_plot(1,pairCols,pdDataDF)
# display()

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

# make_plot(2,pairCols,pdDataDF)
# display()

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

# make_plot(6,pairCols,pdDataDF)
# display()

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
filledSelectedDF = spark.sql(" SELECT * FROM rc_data where EVENT_ID IN (SELECT EVENT_ID FROM tmp_step_event where num_steps >= 320) AND BF_SD_TYPE IN ('UCM','NMD') AND EVENT_ID <>0 AND RUL <>0 and CODE NOT IN ('BEWK-ZZZ-K0110A') ")
filledSelectedDF.cache()
#df2 = spark.sql(" SELECT * FROM rc_data")

# COMMAND ----------

# featurs = [
#  'RESI_LL_LUBE_OIL_PRESS',
#  'RESI_HH_LUBE_OIL_TEMP',
#  'RESI_LL_SPEED',
#  'RESI_HH_SPEED',
#  'RESI_LL_VIBRATION',
#  'RESI_HH_VIBRATION',
#  'RESI_HH_THROW1_DIS_TEMP',
#  'RESI_HH_THROW1_SUC_TEMP',
#  'LUBE_OIL_PRESS',
#  'LUBE_OIL_TEMP',
#  'THROW_1_DISC_PRESS',
#  'THROW_1_DISC_TEMP',
#  'THROW_1_SUC_PRESS',
#  'THROW_1_SUC_TEMP',
#  'THROW_2_DISC_PRESS',
#  'THROW_2_DISC_TEMP',
#  'THROW_2_SUC_PRESS',
#  'THROW_2_SUC_TEMP',
#  'THROW_3_DISC_PRESS',
#  'THROW_3_DISC_TEMP',
#  'THROW_3_SUC_PRESS',
#  'THROW_3_SUC_TEMP',
#  'THROW_4_DISC_PRESS',
#  'THROW_4_DISC_TEMP',
#  'VIBRATION',
#  'CYL_1_TEMP',
#  'CYL_2_TEMP',
#  'CYL_3_TEMP',
#  'CYL_4_TEMP',
#  'CYL_5_TEMP',
#  'CYL_6_TEMP',
#  'CYL_7_TEMP',
#  'CYL_8_TEMP',
#  'CYL_9_TEMP',
#  'CYL_10_TEMP',
#  'CYL_11_TEMP',
#  'CYL_12_TEMP',
#  #'FUEL_GAS_TEMP',
#  'LUBE_OIL_PRESS_ENGINE',
#  'MANI_PRESS',
#  'RIGHT_BANK_EXH_TEMP',
#  #'RUNNING_STATUS',
#  'SPEED',
#  'VIBRA_ENGINE',
# 'S1_RECY_VALVE',
# 'S1_SUCT_PRESS',
# 'S1_SUCT_TEMPE',
# 'S2_STAGE_DISC_PRESS',
# 'S2_SCRU_SCRUB_LEVEL',
# 'GAS_LIFT_HEAD_PRESS',
# 'IN_CONT_CONT_VALVE',
# 'IN_SEP_PRESS']

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

# # empty dictionary d
# d = {}
# # Fill in the entries one by one
# for col in columnList:
#   appQuantile = filledSelectedDF.approxQuantile(col,[0.01,0.99],0.25)
#   d[col] = appQuantile
#   print(col + ":",appQuantile)
d={'RESI_LL_LUBE_OIL_PRESS': [-39.951653, 45.005832], 'RESI_HH_LUBE_OIL_TEMP': [-190.0, -7.6543], 'RESI_LL_SPEED': [-387.7206, 820.0], 'RESI_HH_SPEED': [-1180.0, 27.7206], 'RESI_LL_VIBRATION': [-0.0896955, 0.3300074], 'RESI_HH_VIBRATION': [-0.4750074, -0.0553045], 'RESI_HH_THROW1_DIS_TEMP': [9.13457, 293.33], 'RESI_HH_THROW1_SUC_TEMP': [-190.0, -47.21338], 'LUBE_OIL_PRESS': [-0.0058319517, 84.951653], 'LUBE_OIL_TEMP': [0.0, 182.3457], 'THROW_1_DISC_PRESS': [0.0, 520.79382], 'THROW_1_DISC_TEMP': [0.0, 284.19543], 'THROW_1_SUC_PRESS': [0.0, 384.00885], 'THROW_1_SUC_TEMP': [0.0, 137.4287], 'THROW_2_DISC_PRESS': [0.0, 1274.5869], 'THROW_2_DISC_TEMP': [0.0, 302.7619], 'THROW_2_SUC_PRESS': [0.0, 646.75], 'THROW_2_SUC_TEMP': [0.0, 161.96288], 'THROW_3_DISC_PRESS': [0.0, 545.07574], 'THROW_3_DISC_TEMP': [0.0, 280.6301], 'THROW_3_SUC_PRESS': [0.0, 384.00885], 'THROW_3_SUC_TEMP': [0.0, 137.4287], 'THROW_4_DISC_PRESS': [0.0, 1274.5869], 'THROW_4_DISC_TEMP': [0.0, 299.61646], 'VIBRATION': [-0.0099999998, 0.33007166], 'CYL_1_TEMP': [0.0, 1352.5356], 'CYL_2_TEMP': [0.0, 1322.2463], 'CYL_3_TEMP': [0.0, 1320.5012], 'CYL_4_TEMP': [0.0, 1312.4865], 'CYL_5_TEMP': [0.0, 1349.6497], 'CYL_6_TEMP': [0.0, 1377.9078], 'CYL_7_TEMP': [0.0, 1293.186], 'CYL_8_TEMP': [0.0, 1319.0], 'CYL_9_TEMP': [0.0, 1318.7515], 'CYL_10_TEMP': [0.0, 1291.7678], 'CYL_11_TEMP': [0.0, 1324.1879], 'CYL_12_TEMP': [0.0, 1266.2074], 'LUBE_OIL_PRESS_ENGINE': [0.0, 85.976784], 'MANI_PRESS': [0.0, 130.98006], 'RIGHT_BANK_EXH_TEMP': [0.0, 1322.7722], 'SPEED': [0.0, 1207.7206], 'VIBRA_ENGINE': [-7.3744154e-06, 0.4196955], 'S1_RECY_VALVE': [0.0, 100.0], 'S1_SUCT_PRESS': [0.0, 301.21982], 'S1_SUCT_TEMPE': [0.0, 135.27954], 'S2_STAGE_DISC_PRESS': [0.0, 1274.5869], 'S2_SCRU_SCRUB_LEVEL': [0.0, 99.967957], 'GAS_LIFT_HEAD_PRESS': [0.0, 1282.1912], 'IN_CONT_CONT_VALVE': [0.0, 100.0], 'IN_SEP_PRESS': [0.0, 302.10098], 'WH_PRESSURE': [0.0, 602.50574]}

# COMMAND ----------

# colAdd = 'WH_PRESSURE'
# appQuantile = filledSelectedDF.approxQuantile(colAdd,[0.01,0.99],0.25)
# d[colAdd] = appQuantile
print(d)

# COMMAND ----------

# looping through the columns, doing log(x+1) transformations
for col in columnList:
  df_new = filledSelectedDF.withColumn(col, \
  F.log(F.when(filledSelectedDF[col] < d[col][0],d[col][0])\
  .when(filledSelectedDF[col] > d[col][1], d[col][1])\
  .otherwise(filledSelectedDF[col] ) +1).alias(col))
  print(col+" done")

# COMMAND ----------

q = {'RESI_LL_LUBE_OIL_PRESS': [-16.0, -12.134922], 'RESI_HH_LUBE_OIL_TEMP': [-25.0, -18.47659], 'RESI_LL_SPEED': [-136.51276, -113.69519], 'RESI_HH_SPEED': [-239.70312, -216.5979], 'RESI_LL_VIBRATION': [0.1520122, 0.2853944], 'RESI_HH_VIBRATION': [-0.3199616, -0.2799316], 'RESI_HH_THROW1_DIS_TEMP': [73.17158, 117.42123], 'RESI_HH_THROW1_SUC_TEMP': [-96.409996, -47.21338], 'LUBE_OIL_PRESS': [58.0, 61.028156], 'LUBE_OIL_TEMP': [165.0, 171.52341], 'THROW_1_DISC_PRESS': [270.05258, 437.74664], 'THROW_1_DISC_TEMP': [192.13684, 220.15842], 'THROW_1_SUC_PRESS': [131.0782, 143.29826], 'THROW_1_SUC_TEMP': [93.590004, 118.75208], 'THROW_2_DISC_PRESS': [1159.647, 1194.0951], 'THROW_2_DISC_TEMP': [244.0, 270.71844], 'THROW_2_SUC_PRESS': [460.37476, 520.28021], 'THROW_2_SUC_TEMP': [101.41174, 138.57855], 'THROW_3_DISC_PRESS': [263.0, 420.48215], 'THROW_3_DISC_TEMP': [191.79619, 235.76436], 'THROW_3_SUC_PRESS': [120.0, 135.49545], 'THROW_3_SUC_TEMP': [76.643799, 114.99268], 'THROW_4_DISC_PRESS': [489.49811, 1190.9818], 'THROW_4_DISC_TEMP': [196.0, 266.81018], 'VIBRATION': [0.1002849, 0.15000001], 'CYL_1_TEMP': [1138.0527, 1215.1787], 'CYL_2_TEMP': [1139.4019, 1228.1166], 'CYL_3_TEMP': [1136.2756, 1225.0497], 'CYL_4_TEMP': [1113.7179, 1196.1139], 'CYL_5_TEMP': [1083.9575, 1234.2267], 'CYL_6_TEMP': [1102.1942, 1197.9279], 'CYL_7_TEMP': [1158.1827, 1220.5721], 'CYL_8_TEMP': [1140.4087, 1223.7587], 'CYL_9_TEMP': [1144.2157, 1218.1738], 'CYL_10_TEMP': [1099.0, 1208.3251], 'CYL_11_TEMP': [1082.009, 1194.5598], 'CYL_12_TEMP': [1100.2706, 1167.0231], 'LUBE_OIL_PRESS_ENGINE': [49.903023, 65.864731], 'MANI_PRESS': [29.0, 36.677399], 'RIGHT_BANK_EXH_TEMP': [1178.969, 1257.0023], 'SPEED': [940.29688, 963.4021], 'VIBRA_ENGINE': [0.15503845, 0.23002379], 'S1_RECY_VALVE': [3.5812631, 38.0], 'S1_SUCT_PRESS': [120.51469, 136.01794], 'S1_SUCT_TEMPE': [77.066277, 114.99268], 'S2_STAGE_DISC_PRESS': [998.69843, 1195.4448], 'S2_SCRU_SCRUB_LEVEL': [15.383576, 26.0], 'GAS_LIFT_HEAD_PRESS': [1159.5414, 1184.2358], 'IN_CONT_CONT_VALVE': [100.0, 100.0], 'IN_SEP_PRESS': [122.0, 138.8398], 'WH_PRESSURE': [142.52789, 483.06967]}

# Fill in the entries one by one
# for col in columnList:
#   appQuantile = filledSelectedDF.approxQuantile(col,[0.25,0.75],0.00)
#   ##df.stat.approxQuantile("value", Array(0.25,0.75),0.0)
#   q[col] = appQuantile
#   print(col + "Q1 : Q3 ",appQuantile," IQR = Q3 - Q1>> ",appQuantile[1]-appQuantile[0])

# COMMAND ----------

print(q)
percentile ={'RESI_LL_LUBE_OIL_PRESS': [-16.0, -12.134922], 'RESI_HH_LUBE_OIL_TEMP': [-25.0, -18.47659], 'RESI_LL_SPEED': [-136.51276, -113.69519], 'RESI_HH_SPEED': [-239.70312, -216.5979], 'RESI_LL_VIBRATION': [0.1520122, 0.2853944], 'RESI_HH_VIBRATION': [-0.3199616, -0.2799316], 'RESI_HH_THROW1_DIS_TEMP': [73.17158, 117.42123], 'RESI_HH_THROW1_SUC_TEMP': [-96.409996, -47.21338], 'LUBE_OIL_PRESS': [58.0, 61.028156], 'LUBE_OIL_TEMP': [165.0, 171.52341], 'THROW_1_DISC_PRESS': [270.05258, 437.74664], 'THROW_1_DISC_TEMP': [192.13684, 220.15842], 'THROW_1_SUC_PRESS': [131.0782, 143.29826], 'THROW_1_SUC_TEMP': [93.590004, 118.75208], 'THROW_2_DISC_PRESS': [1159.647, 1194.0951], 'THROW_2_DISC_TEMP': [244.0, 270.71844], 'THROW_2_SUC_PRESS': [460.37476, 520.28021], 'THROW_2_SUC_TEMP': [101.41174, 138.57855], 'THROW_3_DISC_PRESS': [263.0, 420.48215], 'THROW_3_DISC_TEMP': [191.79619, 235.76436], 'THROW_3_SUC_PRESS': [120.0, 135.49545], 'THROW_3_SUC_TEMP': [76.643799, 114.99268], 'THROW_4_DISC_PRESS': [489.49811, 1190.9818], 'THROW_4_DISC_TEMP': [196.0, 266.81018], 'VIBRATION': [0.1002849, 0.15000001], 'CYL_1_TEMP': [1138.0527, 1215.1787], 'CYL_2_TEMP': [1139.4019, 1228.1166], 'CYL_3_TEMP': [1136.2756, 1225.0497], 'CYL_4_TEMP': [1113.7179, 1196.1139], 'CYL_5_TEMP': [1083.9575, 1234.2267], 'CYL_6_TEMP': [1102.1942, 1197.9279], 'CYL_7_TEMP': [1158.1827, 1220.5721], 'CYL_8_TEMP': [1140.4087, 1223.7587], 'CYL_9_TEMP': [1144.2157, 1218.1738], 'CYL_10_TEMP': [1099.0, 1208.3251], 'CYL_11_TEMP': [1082.009, 1194.5598], 'CYL_12_TEMP': [1100.2706, 1167.0231], 'LUBE_OIL_PRESS_ENGINE': [49.903023, 65.864731], 'MANI_PRESS': [29.0, 36.677399], 'RIGHT_BANK_EXH_TEMP': [1178.969, 1257.0023], 'SPEED': [940.29688, 963.4021], 'VIBRA_ENGINE': [0.15503845, 0.23002379], 'S1_RECY_VALVE': [3.5812631, 38.0], 'S1_SUCT_PRESS': [120.51469, 136.01794], 'S1_SUCT_TEMPE': [77.066277, 114.99268], 'S2_STAGE_DISC_PRESS': [998.69843, 1195.4448], 'S2_SCRU_SCRUB_LEVEL': [15.383576, 26.0], 'GAS_LIFT_HEAD_PRESS': [1159.5414, 1184.2358], 'IN_CONT_CONT_VALVE': [100.0, 100.0], 'IN_SEP_PRESS': [122.0, 138.8398], 'WH_PRESSURE': [142.52789, 483.06967]}

# COMMAND ----------

# def outliers_iqr (colName, percentile, value):
#     quartile_1, quartile_3 = percentile[colName][0], percentile[colName][1]
#     print("quartile_1:",quartile_1, ",quartile_3:",quartile_3)
#     #np.percentile(ys, [25, 75])
#     iqr = quartile_3 - quartile_1
    
#     lower_bound = quartile_1 - (iqr * 1.5)
#     upper_bound = quartile_3 + (iqr * 1.5)
    
#     if value>=upper_bound:
#       value = upper_bound
#       print("Update Uper>",upper_bound)
#     elif value<=lower_bound:
#       value = lower_bound
#       print("Update Lower>",lower_bound)
#     else:
#       print("Value>",value)
    
#     return value

# boundary ={}

# def gen_boundary_iqr(colList, percentile):
#   for colName in colList:
#     quartile_1, quartile_3 = percentile[colName][0], percentile[colName][1]
#     print("quartile_1:",quartile_1, ",quartile_3:",quartile_3)
#     #np.percentile(ys, [25, 75])
#     iqr = quartile_3 - quartile_1
#     lower_bound = quartile_1 - (iqr * 1.5)
#     upper_bound = quartile_3 + (iqr * 1.5)
#     boundary[colName] = [upper_bound,lower_bound]
#   return boundary

# boundary = gen_boundary_iqr(featurs, percentile)

# #newvalue = outliers_iqr("THROW_1_DISC_TEMP", percentile, 500)
# #print("newvalue:",newvalue)
# # empty dictionary d

# COMMAND ----------

# print(boundary)

# COMMAND ----------

#  for colName in featurs:
#     #print("Soomth Extream Noise by Actual Range of Sensor 10% ["+colName, "]")
#     #print(" validating < ", normalSensorRange[colName][0])
#     #df_new = df.withColumn(colName, when(df[colName]<=normalSensorRange[colName][0], df[colName]).otherwise(normalSensorRange[colName][0]))
#     #f_new = df.withColumn(colName, when(df[colName]>normalSensorRange[colName][0], normalSensorRange[colName][0] ).otherwise(df[colName]))
#     #print(" validating > ", normalSensorRange[colName][1])
#     #df_new = df_new.withColumn(colName, when(df_new[colName]<normalSensorRange[colName][1], normalSensorRange[colName][1]).otherwise(df_new[colName]))
#     df_new = df.withColumn(colName, F.log(F.when(df[colName] < boundary[colName][1],boundary[colName][1]).when(df[colName] > boundary[colName][0], boundary[colName][0]).otherwise(df[colName]) +1).alias(colName))
    
# #     df_new = df.withColumn(colName, F.log(F.when(df[colName] < normalSensorRange[colName][1],normalSensorRange[colName][1]).when(df[colName] > normalSensorRange[colName][0], normalSensorRange[colName][0]).otherwise(df[colName]) +1).alias(colName))

# COMMAND ----------

columnListDem = ["CODE","DAYTIME","YEAR","MONTH","DAY","HOUR","MM","CYCLE","RUL","IN_RUL","EVENT_ID","LABEL1","LABEL2","BF_SD_TYPE"]

# COMMAND ----------

sparkDF01 = df_new.selectExpr("CODE",
"DAYTIME",
"YEAR",
"MONTH",
"DAY",
"HOUR",
"MM",
"CYCLE",
"RUL",
"EVENT_ID",
"LABEL1",
"LABEL2",
"BF_SD_TYPE",
"CYCLE AS NOM_CYCLE",
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
'LUBE_OIL_PRESS_ENGINE',
'MANI_PRESS',
'RIGHT_BANK_EXH_TEMP',
'SPEED',
'VIBRA_ENGINE',
'S1_RECY_VALVE',
'S1_SUCT_PRESS',
'S1_SUCT_TEMPE',
'S2_STAGE_DISC_PRESS',
'S2_SCRU_SCRUB_LEVEL',
'GAS_LIFT_HEAD_PRESS',
'IN_CONT_CONT_VALVE',
'IN_SEP_PRESS',
'WH_PRESSURE')

sparkDF01.cache()

# COMMAND ----------

pandasDF = df_new.toPandas()

# COMMAND ----------

def normalizeQuantileTransformer(test_df,quantile_scaler):
  ######
  # TEST
  ######
  #Data after robust scaling & Data after quantile transformation (uniform pdf)
  test_df['cycle_norm'] = test_df['CYCLE']
  cols_normalize = test_df.columns.difference(['CODE','DAYTIME','YEAR','MONTH','DAY','HOUR','MM','CYCLE','RUL','EVENT_ID','LABEL1','LABEL2','BF_SD_TYPE'])
  print(cols_normalize)
  
  norm_test_df = pd.DataFrame(quantile_scaler.fit_transform(test_df[cols_normalize]), 
                              columns=cols_normalize, 
                              index=test_df.index)
  
  test_join_df = test_df[test_df.columns.difference(cols_normalize)].join(norm_test_df)
  test_df = test_join_df.reindex(columns = test_df.columns)
  #test_df = test_df.reset_index(drop=True)
  print("Finish normalization test dataset!")
  return test_df

# COMMAND ----------

def normalizeMaxMinTrain(sampleDataSet,min_max_scaler):
# MinMax normalization (from 0 to 1)
  sampleDataSet['cycle_norm'] = sampleDataSet['CYCLE']
  cols_normalize = sampleDataSet.columns.difference(['CODE','DAYTIME','YEAR','MONTH','DAY','HOUR','MM','CYCLE','RUL','EVENT_ID','LABEL1','LABEL2','BF_SD_TYPE'])
  norm_train_df = pd.DataFrame(min_max_scaler.fit_transform(sampleDataSet[cols_normalize]), 
                               columns=cols_normalize, 
                               index=sampleDataSet.index)
  join_df = sampleDataSet[sampleDataSet.columns.difference(cols_normalize)].join(norm_train_df)
  sampleDataSet = join_df.reindex(columns = sampleDataSet.columns)
  print("Finish normalization train dataset!")
  return sampleDataSet


# COMMAND ----------

quantile_scaler = preprocessing.QuantileTransformer(output_distribution='uniform')
scaledDatSetQT = normalizeQuantileTransformer(pandasDF,quantile_scaler)

# COMMAND ----------

# # sccaledDatSet['LUBE_OIL_PRESS'].hist()
# # display()
# pairCols = ['LUBE_OIL_PRESS','LUBE_OIL_TEMP']
# make_plot(0,pairCols,scaledDatSetQT[pairCols])
# display()

# COMMAND ----------

# min_max_scaler = preprocessing.MinMaxScaler()
# scaledDatSetMinMax = normalizeMaxMinTrain(pandasDF,min_max_scaler)

# COMMAND ----------

# # sccaledDatSet['LUBE_OIL_PRESS'].hist()
# # display()
# pairCols = ['LUBE_OIL_PRESS','LUBE_OIL_TEMP']
# make_plot(0,pairCols,scaledDatSetMinMax[pairCols])
# display()

# COMMAND ----------

normalizedQTSDF = spark.createDataFrame(scaledDatSetQT)
# normalizedMinMaxSDF = spark.createDataFrame(scaledDatSetMinMax)

normalizedQTSDF.createOrReplaceTempView("rc_normalized_quantile_transfer")
# normalizedMinMaxSDF.createOrReplaceTempView("rc_normalized_minmax")

# COMMAND ----------

display(normalizedQTSDF.take(10))

# COMMAND ----------

# %sh
# mkdir datafile3
# %sh
# mkdir datafile
localFolder = "datafileNMD"
try:
  dbutils.fs.mkdirs("file:/databricks/driver/"+localFolder)
  print("Created folder datafileNMD!.")
except:
  print("Error creating folder datafileNMD!")

# COMMAND ----------

def copyData(fromPath, toPath,recurseFlag=True):
  dbutils.fs.cp(fromPath, toPath,recurse=recurseFlag)
  print("Completed copying files to "+toPath)

# COMMAND ----------

import math
def round_down_float(n, decimals=0):
    multiplier = 10 ** decimals
    return float(math.floor(n * multiplier) / multiplier)

# COMMAND ----------

def exportCSV(p_sd_type,normalizedQTSDF,sqlNotInId, localFolder="datafile"):
  
  if sqlNotInId=="":
    normalized_QTDF = spark.sql("select EVENT_ID, CODE, COUNT(*) as num_steps from rc_normalized_quantile_transfer where BF_SD_TYPE = '"+p_sd_type+"' GROUP BY  CODE, EVENT_ID")
  else:
    normalized_QTDF = spark.sql("select EVENT_ID, CODE, COUNT(*) as num_steps from rc_normalized_quantile_transfer where BF_SD_TYPE = '"+p_sd_type+"'"+ sqlNotInId + "GROUP BY  CODE, EVENT_ID")
  
  totalCount = normalized_QTDF.count()
  
  print("Total Events :",str(totalCount))
  events = normalized_QTDF.toPandas()
  toPath = "dbfs:/RC/RC_EVENTS/ALL_FILES_RV2"
  fromPath = "file:/databricks/driver/"
  localpath = "/databricks/driver/"+localFolder+"/"
  
  completedCount = 0
  
  for event_id in events.EVENT_ID:
    #tdf = df.where(" RUL <>0 AND EVENT_ID="+str(event_id))
    tdf = normalizedQTSDF.where("EVENT_ID="+str(event_id)).orderBy(["CYCLE"],ascending=[1])
    numrows = tdf.count()
    whpCode = tdf.first().CODE[:4]
    
    fullPath = localpath+whpCode+"_"+str(numrows)+"_ID_"+str(event_id)+".csv"
    tdf.toPandas().to_csv(fullPath)
    csvLocalPath = fromPath+localFolder+"/"+whpCode+"_"+str(numrows)+"_ID_"+str(event_id)+".csv"
    copyData(csvLocalPath,toPath,False)
    #tdf.repartition(1).write.csv(fullPath)
    completedCount = completedCount+1
    print(event_id, ":", whpCode , " >> ", numrows, " Completed ", str(round_down_float(((completedCount/totalCount)*100),2)),"%")
  print("****** End Exporting Data Files at ", localpath)

# COMMAND ----------

# indexDF = normalizedQTSDF.select('BF_SD_TYPE','YEAR','EVENT_ID','MONTH','DAY','HOUR','MM','RUL','CODE')
# indexDF.toPandas().to_csv("INDEX_RC_DATA_3.csv")

# COMMAND ----------

# MAGIC %sh
# MAGIC ls -al datafileNMD

# COMMAND ----------

# dbutils.fs.cp("file:/databricks/driver/INDEX_RC_DATA_3.csv", "dbfs:/RC/RC_EVENTS/ALL_FILES_RV2/")

# COMMAND ----------

exportCSV("NMD",normalizedQTSDF,"",localFolder)

# COMMAND ----------

# MAGIC %sh
# MAGIC zip rc_event_data03_NMD.zip datafile3/*.csv

# COMMAND ----------

# MAGIC %fs
# MAGIC cp file:/databricks/driver/rc_event_data03_NMD.zip dbfs:/mnt/Exploratory/WCLD

# COMMAND ----------

