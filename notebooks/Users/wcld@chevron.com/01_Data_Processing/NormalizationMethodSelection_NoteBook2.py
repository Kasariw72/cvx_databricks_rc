# Databricks notebook source
# MAGIC %md 
# MAGIC # Exploratory Data Analysis

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

spark.conf.set("spark.sql.execution.arrow.enabled", "true")

# COMMAND ----------

import matplotlib.pyplot as plt
plt.style.use('ggplot')
%matplotlib inline  

from pandas.tools.plotting import scatter_matrix

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
'RUNNING_STATUS',
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
"RUL as RUL",
"RUL*-1 as IN_RUL",
"EVENT_ID as EVENT_ID",
"LABEL1 as LABEL1",
"LABEL2 as LABEL2",
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
'IN_SEP_PRESS']

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

meanDic = {'RESI_LL_LUBE_OIL_PRESS': -13.34679, 'RESI_HH_LUBE_OIL_TEMP': -15.00072, 'RESI_LL_SPEED': 819.3537983, 'RESI_HH_SPEED': -182.47516, 'RESI_LL_VIBRATION': 0.3297389, 'RESI_HH_VIBRATION': -0.2719401, 'RESI_HH_THROW1_DIS_TEMP': 281.231386, 'RESI_HH_THROW1_SUC_TEMP': -79.90926, 'LUBE_OIL_PRESS': 68.853477, 'LUBE_OIL_TEMP': 170.73135, 'THROW_1_DISC_PRESS': 493.11111, 'THROW_1_DISC_TEMP': 252.18327, 'THROW_1_SUC_PRESS': 166.40938, 'THROW_1_SUC_TEMP': 124.16805, 'THROW_2_DISC_PRESS': 1213.7482, 'THROW_2_DISC_TEMP': 268.50851, 'THROW_2_SUC_PRESS': 490.68488, 'THROW_2_SUC_TEMP': 133.93256, 'THROW_3_DISC_PRESS': 470.68506, 'THROW_3_DISC_TEMP': 243.81787, 'THROW_3_SUC_PRESS': 165.38683, 'THROW_3_SUC_TEMP': 109.06955, 'THROW_4_DISC_PRESS': 1210.5085, 'THROW_4_DISC_TEMP': 286.57486, 'VIBRATION': 0.17999946, 'CYL_1_TEMP': 1242.7988, 'CYL_2_TEMP': 1241.1833, 'CYL_3_TEMP': 1314.7606, 'CYL_4_TEMP': 1213.8179, 'CYL_5_TEMP': 1234.2267, 'CYL_6_TEMP': 1247.9746, 'CYL_7_TEMP': 1224.9188, 'CYL_8_TEMP': 1231.4215, 'CYL_9_TEMP': 1222.579, 'CYL_10_TEMP': 1203.325, 'CYL_11_TEMP': 1192.7275, 'CYL_12_TEMP': 1173.8546, 'LUBE_OIL_PRESS_ENGINE': 69.623924, 'MANI_PRESS': 102.42593, 'RIGHT_BANK_EXH_TEMP': 1246.0, 'SPEED': 1057.7806, 'VIBRA_ENGINE': 0.2, 'S1_RECY_VALVE': 20.171038, 'S1_SUCT_PRESS': 168.81329, 'S1_SUCT_TEMPE': 120.17609, 'S2_STAGE_DISC_PRESS': 1198.1665, 'S2_SCRU_SCRUB_LEVEL': 40.054184, 'GAS_LIFT_HEAD_PRESS': 1211.9084, 'IN_CONT_CONT_VALVE': 100.0, 'IN_SEP_PRESS': 175.5134, 'WH_PRESSURE': 370.74503}


### This step is for cleaning data using simple median value of each colum to the missing or unknown value of sensors.
def replaceByMedian(pySparkDF, columnList):
    for colName in columnList:
      #med = pySparkDF.approxQuantile(colName, [0.5], 0.25)
      RULFrom = 60
      RULTo = 10080
      
      bfEventType = "UCM"
      
      if meanDic=="":
        med = getDataByCol(colName,RULFrom,RULTo,bfEventType)
      else:
        med = meanDic[colName]
      
      #med = getDataByCol(colName,RULFrom,RULTo,bfEventType)
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

display(renamed_df.where("YEAR=2018").describe())

# COMMAND ----------

def getPairData(colName,RULFrom,RULTo,eventId, bfEventType):
  
  if eventId=="":
    sql = "select CODE,YEAR,MONTH,RUL, "+colName[0]+","+colName[1]+" from rc_data where RUL between "+str(RULFrom)+" and "+str(RULTo)+" and BF_SD_TYPE='"+bfEventType+"' AND CODE NOT IN ('BEWP-SK1060') order by IN_RUL "
  else:
    sql = "select CODE,YEAR,MONTH,RUL, "+colName[0]+","+colName[1]+" from rc_data where EVENT_ID="+str(eventId)+" and RUL between "+str(RULFrom)+" and "+str(RULTo)+" order by IN_RUL "
  
  sampleEvent = spark.sql(sql)
  
  return sampleEvent

def getAllData(RULFrom,RULTo,eventId, bfEventType):
  
  if eventId=="":
    sql = "select * from rc_data where RUL between "+str(RULFrom)+" and "+str(RULTo)+" and BF_SD_TYPE='"+bfEventType+"' AND CODE NOT IN ('BEWP-SK1060') order by IN_RUL "
  else:
    sql = "select * from rc_data where EVENT_ID="+str(eventId)+" and RUL between "+str(RULFrom)+" and "+str(RULTo)+" order by IN_RUL "
  
  sampleEvent = spark.sql(sql)
  return sampleEvent

def getDataCount(colName,RULFrom,RULTo,eventId, bfEventType):
  sql = "select CODE,YEAR,MONTH,RUL, "+colName[0]+","+colName[1]+" from rc_data where RUL between "+str(RULFrom)+" and "+str(RULTo)+" "
  sampleEvent = spark.sql(sql)
  sampleEvent = sampleEvent.toPandas()
  return sampleEvent

# COMMAND ----------

#allCols = ['LUBE_OIL_PRESS','LUBE_OIL_TEMP','RUL','CODE','MONTH','YEAR']
allCols = ['LUBE_OIL_PRESS','LUBE_OIL_TEMP','RUL','CODE','MONTH','YEAR','EVENT_ID']
pairCols = ['LUBE_OIL_PRESS','LUBE_OIL_TEMP']
#pairCols = ['THROW_1_DISC_PRESS','THROW_1_DISC_TEMP']
eventId = ""
#3568
bfEventType = "UCM"
#dataset = getPairData(allCols,60,1440,eventId, bfEventType)
#dataset = getDataCount(pairCols,1,420,eventId, bfEventType)

dataset = getAllData(60,1440,eventId, bfEventType)

# COMMAND ----------

#plot and compare the standard deviation of input features:
sampleSetPD = dataset.where("YEAR=2018").toPandas()


# COMMAND ----------

sampleSetPD[columnList].std().plot(kind='bar', figsize=(16,8), title="Features Standard Deviation")
display()

# COMMAND ----------

# plot and compare the log standard deviation of input features:

sampleSetPD[columnList].std().plot(kind='bar', figsize=(16,8), logy=True,title="Features Standard Deviation (log)")
display()

# COMMAND ----------

# get ordered list of top variance features:
featurs_top_var = sampleSetPD[columnList].std().sort_values(ascending=False)

# COMMAND ----------

print(featurs_top_var)

# COMMAND ----------

sampleSetPD[columnList].corrwith(sampleSetPD.RUL).sort_values(ascending=False)

# COMMAND ----------

correl_featurs = ['THROW_2_DISC_TEMP','THROW_1_DISC_TEMP','THROW_3_DISC_TEMP','LUBE_OIL_PRESS','IN_CONT_CONT_VALVE','RIGHT_BANK_EXH_TEMP','RESI_HH_THROW1_SUC_TEMP','THROW_1_SUC_TEMP']
low_cor_featrs = ['RESI_HH_THROW1_SUC_TEMP','THROW_1_SUC_TEMP','RIGHT_BANK_EXH_TEMP','LUBE_OIL_PRESS','IN_CONT_CONT_VALVE','THROW_3_DISC_TEMP','THROW_1_DISC_TEMP','THROW_2_DISC_TEMP']

# COMMAND ----------

# get ordered list features correlation with regression label2
pdSet = pd.DataFrame(sampleSetPD[columnList].corrwith(sampleSetPD.RUL).sort_values(ascending=False))
#display(pdSet)

display(sampleSetPD[low_cor_featrs].describe())

# COMMAND ----------

display(sampleSetPD[correl_featurs].describe())

# COMMAND ----------

# add the regression label 'ttf' to the list of high corr features 
correl_featurs_lbl = correl_featurs + ['RUL']

# COMMAND ----------

# plot a heatmap to display +ve and -ve correlation among features and regression label:
import seaborn as sns
cm = np.corrcoef(sampleSetPD[correl_featurs_lbl].values.T)
sns.set(font_scale=1.0)
fig = plt.figure(figsize=(16, 12))
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 8}, yticklabels=correl_featurs_lbl, xticklabels=correl_featurs_lbl)
plt.title('Features Correlation Heatmap')
display()
#plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC There is a very high correlation (> 0.8) between some features: (s14, s9), (s11, s4), (s11, s7), (s11, s12), (s4, s12), (s8,s13), (s7, s12)
# MAGIC This may hurt the performance of some ML algorithms.
# MAGIC 
# MAGIC So, some of the above features will be target for removal in feature selection

# COMMAND ----------

print(featurs_top_var)

# COMMAND ----------



# COMMAND ----------

# plot a heatmap to display +ve and -ve correlation among features and regression label:
#import seaborn as sns
low_cor_featrs = low_cor_featrs + ['LABEL2']
cm = np.corrcoef(sampleSetPD[low_cor_featrs].values.T)
sns.set(font_scale=1.0)
fig = plt.figure(figsize=(16, 12))
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 8}, yticklabels=low_cor_featrs, xticklabels=low_cor_featrs)
plt.title('Low Correlation Feature Heatmap')
display()
#plt.show()

# COMMAND ----------

allFeatureCorr = columnList+['LABEL2']

cm = np.corrcoef(sampleSetPD[allFeatureCorr].values.T)
sns.set(font_scale=1.0)
fig = plt.figure(figsize=(16, 12))
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 6}, yticklabels=allFeatureCorr, xticklabels=allFeatureCorr)
plt.title('Features Correlation Heatmap')
display()
#plt.show()

# COMMAND ----------

#reset matplotlib original theme
sns.reset_orig()

# COMMAND ----------

#create scatter matrix to disply relatiohships and distribution among features and regression label
scatter_matrix(sampleSetPD[correl_featurs_lbl], alpha=0.2, figsize=(20, 20), diagonal='kde')
display()

# COMMAND ----------

# MAGIC %md
# MAGIC Most of the features haven't normal distribution which has negative effect on machine learning algorithms
# MAGIC Most of the features have non-linear relationship with the Label 2, so using polynomial models may not lead to better results.
# MAGIC Let us create a helper function to ease exploration of each feature invidually:

# COMMAND ----------

#display(dataset.where("CODE = 'LAWE-ME-E7400'"))
display(sampleSetPD)

# COMMAND ----------

def explore_col(s, e,dataFrame):
    
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
    
    fig = plt.figure(figsize=(16, 12))


    sub1 = fig.add_subplot(221) 
    sub1.set_title(s +' histogram') 
    sub1.hist(dataFrame[s])

    sub2 = fig.add_subplot(222)
    sub2.set_title(s +' boxplot')
    sub2.boxplot(dataFrame[s])
    
    #np.random.seed(12345)
    
    if e > 100 or e <= 0:
        select_engines = list(pd.unique(dataFrame.EVENT_ID))
    else:
        select_engines = np.random.choice(range(1,101), e, replace=False)
        
    sub3 = fig.add_subplot(223)
    sub3.set_title('time series: ' + s +' / cycle')
    sub3.set_xlabel('IN_RUL')
    for i in select_engines:
        df = dataFrame[['IN_RUL', s]][dataFrame.EVENT_ID == i]
        sub3.plot(dataFrame['IN_RUL'],dataFrame[s])
    
    sub4 = fig.add_subplot(224)
    sub4.set_title("scatter: "+ s + " / RUL (Classification Label)")
    sub4.set_xlabel('IN_RUL')
    sub4.scatter(dataFrame['RUL'],dataFrame[s])


    plt.tight_layout()
    display()

# COMMAND ----------

# display(dataset.select("CODE","EVENT_ID").where("LUBE_OIL_TEMP<100").groupby("CODE","EVENT_ID").COUNT())
#display(dataset.where("LUBE_OIL_PRESS>52 and LUBE_OIL_PRESS <66 and LUBE_OIL_TEMP >= 138 "))
correl_featurs = ['RESI_LL_LUBE_OIL_PRESS','S2_SCRU_SCRUB_LEVEL','S1_RECY_VALVE','RESI_HH_THROW1_DIS_TEMP','THROW_2_SUC_PRESS','VIBRA_ENGINE']
low_cor_featrs = ['RESI_HH_THROW1_SUC_TEMP','THROW_1_SUC_TEMP','RIGHT_BANK_EXH_TEMP','LUBE_OIL_PRESS','IN_CONT_CONT_VALVE','THROW_3_DISC_TEMP','THROW_1_DISC_TEMP','THROW_2_DISC_TEMP']

explore_col(correl_featurs[0], 10,sampleSetPD)

# COMMAND ----------

#pdDataDF = dataset.where("LUBE_OIL_PRESS>56 and LUBE_OIL_PRESS <66 and LUBE_OIL_TEMP >= 138 ").toPandas()
explore_col(correl_featurs[1], 10,sampleSetPD)

# COMMAND ----------

# Create a function to explore the time series plot each sensor selecting random sample engines
def plot_time_series(s,dataFrame):
    
    """Plot time series of a single sensor for 10 random sample engines.
    
        Args:
        s (str): The column name of the sensor to be plotted.

    Returns:
        plots
        
    """
    
    fig, axes = plt.subplots(10, 1, sharex=True, figsize = (15, 15))
    fig.suptitle(s + ' time series / cycle', fontsize=15)
    
    #np.random.seed(12345)
    select_engines = np.random.choice(range(1,1000), 10, replace=False).tolist()
    
    for e_id in select_engines:
        df = dataFrame[['IN_RUL', s]][dataFrame.EVENT_ID == e_id]
        i = select_engines.index(e_id)
        axes[i].plot(df['IN_RUL'],df[s])
        axes[i].set_ylabel('engine ' + str(e_id))
        axes[i].set_xlabel('IN_RUL')
        #axes[i].set_title('engine ' + str(e_id), loc='right')

    #plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    #plt.show()
    display()

# COMMAND ----------

#plot_time_series(correl_featurs[1],sampleSetPD)

# COMMAND ----------

#Let us check some stat on the classifcation labels:
# print stat for binary classification label
print(sampleSetPD['LABEL2'].value_counts())
print('\nNegaitve samples =  {0:.0%}'.format(sampleSetPD['LABEL2'].value_counts()[0]/sampleSetPD['LABEL2'].count()))
print('\nPosiitve samples =  {0:.0%}'.format(sampleSetPD['LABEL2'].value_counts()[0]/sampleSetPD['LABEL2"'].count()))

# COMMAND ----------

# print stat for multiclass classification label

print(sampleSetPD['LABEL2'].value_counts())
print('\nClass 0 samples =  {0:.0%}'.format(sampleSetPD['LABEL2'].value_counts()[0]/sampleSetPD['LABEL2'].count()))
print('\nClass 1 samples =  {0:.0%}'.format(sampleSetPD['LABEL2'].value_counts()[1]/sampleSetPD['LABEL2'].count()))
print('\nClass 2 samples =  {0:.0%}'.format(sampleSetPD['LABEL2'].value_counts()[2]/sampleSetPD['LABEL2'].count()))

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

#dataset.select('YEAR','CODE').where("LUBE_OIL_PRESS>56 and LUBE_OIL_PRESS <66 and LUBE_OIL_TEMP >= 138 ").count().groupby('YEAR','CODE')
pk = spark.sql("SELECT YEAR, CODE, COUNT(DISTINCT EVENT_ID) n FROM rc_data where LUBE_OIL_PRESS>56 and LUBE_OIL_PRESS <66 and LUBE_OIL_TEMP >= 138 group by YEAR, CODE ")
display(pk)

# COMMAND ----------

# MAGIC %matplotlib inline
# MAGIC # Author:  Raghav RV <rvraghav93@gmail.com>
# MAGIC #          Guillaume Lemaitre <g.lemaitre58@gmail.com>
# MAGIC #          Thomas Unterthiner
# MAGIC # License: BSD 3 clause
# MAGIC 
# MAGIC from __future__ import print_function
# MAGIC 
# MAGIC import numpy as np
# MAGIC 
# MAGIC import matplotlib as mpl
# MAGIC from matplotlib import pyplot as plt
# MAGIC from matplotlib import cm
# MAGIC 
# MAGIC from sklearn.preprocessing import MinMaxScaler
# MAGIC from sklearn.preprocessing import minmax_scale
# MAGIC from sklearn.preprocessing import MaxAbsScaler
# MAGIC from sklearn.preprocessing import StandardScaler
# MAGIC from sklearn.preprocessing import RobustScaler
# MAGIC from sklearn.preprocessing import Normalizer
# MAGIC from sklearn.preprocessing import QuantileTransformer
# MAGIC from sklearn import preprocessing
# MAGIC 
# MAGIC #from sklearn.preprocessing import PowerTransformer
# MAGIC #from sklearn.datasets import fetch_california_housing
# MAGIC 
# MAGIC def prepareScaler(pairCols, pdDataDF):
# MAGIC   print(__doc__)
# MAGIC   ##datasetTest = fetch_california_housing()
# MAGIC   ##X_full, y_full = datasetTest.data, datasetTest.target
# MAGIC   dataSampleCol = pairCols
# MAGIC   dataSampleColTarget =  pairCols[1]
# MAGIC 
# MAGIC   dataSet1 = pdDataDF[dataSampleCol]
# MAGIC   dataTarget = pdDataDF[dataSampleColTarget]
# MAGIC 
# MAGIC   X_full = dataSet1.as_matrix()
# MAGIC   y_full =dataTarget.as_matrix()
# MAGIC 
# MAGIC   y_full = y_full.reshape(1,len(dataTarget))
# MAGIC   
# MAGIC   N_MinMaxScaler = preprocessing.MinMaxScaler()
# MAGIC   #N_minmax_scale = preprocessing.minmax_scale()
# MAGIC   N_MaxAbsScaler = preprocessing.MinMaxScaler()
# MAGIC   N_StandardScaler = preprocessing.StandardScaler()
# MAGIC   N_RobustScaler = preprocessing.RobustScaler(quantile_range=(25, 75))
# MAGIC   N_QuantileTransformerGaussian = preprocessing.QuantileTransformer(output_distribution='normal')
# MAGIC   N_QuantileTransformer = preprocessing.QuantileTransformer(output_distribution='uniform')
# MAGIC   N_Normalizer= preprocessing.Normalizer()
# MAGIC   
# MAGIC   #sampleEvent['RUL'].min() 
# MAGIC   #y_full = np.arange(1440)
# MAGIC   # Take only 2 features to make visualization easier
# MAGIC 
# MAGIC   # Feature of 0 has a long tail distribution.
# MAGIC   # Feature 5 has a few but very large outliers.
# MAGIC   X = X_full[:, [0, 1]]
# MAGIC   
# MAGIC   print("len X_Full", len(X_full))
# MAGIC   print("len y_full", len(y_full[0]))
# MAGIC   print("y_full", y_full)
# MAGIC   
# MAGIC   distributions = [
# MAGIC       ('Unscaled data', X),
# MAGIC       ('Data after standard scaling',
# MAGIC           StandardScaler().fit_transform(X)),
# MAGIC       ('Data after min-max scaling',
# MAGIC          N_MinMaxScaler.fit_transform(X)),
# MAGIC       ('Data after max-abs scaling',
# MAGIC           N_MaxAbsScaler.fit_transform(X)),
# MAGIC       ('Data after robust scaling',
# MAGIC           N_RobustScaler.fit_transform(X)),
# MAGIC   #     ('Data after power transformation (Yeo-Johnson)',
# MAGIC   #      PowerTransformer(method='yeo-johnson').fit_transform(X)),
# MAGIC   #     ('Data after power transformation (Box-Cox)',
# MAGIC   #      PowerTransformer(method='box-cox').fit_transform(X)),
# MAGIC       ('Data after quantile transformation (gaussian pdf)',
# MAGIC           N_QuantileTransformerGaussian.fit_transform(X)),
# MAGIC       ('Data after quantile transformation (uniform pdf)',
# MAGIC           N_QuantileTransformer.fit_transform(X)),
# MAGIC       ('Data after sample-wise L2 normalizing',
# MAGIC           N_Normalizer.fit_transform(X)),
# MAGIC       ('Data after robust scaling & Quantile transformation (uniform pdf)',
# MAGIC           N_QuantileTransformer.fit_transform(N_RobustScaler.fit_transform(X)))
# MAGIC           ,
# MAGIC       ('Data after robust scaling & Data after quantile transformation (gaussian pdf)',
# MAGIC           N_QuantileTransformerGaussian.fit_transform(N_RobustScaler.fit_transform(X))),
# MAGIC   ]
# MAGIC 
# MAGIC   # scale the output between 0 and 1 for the colorbar
# MAGIC   #y = minmax_scale(y_full)
# MAGIC   y = minmax_scale(y_full[0])
# MAGIC 
# MAGIC   # plasma does not exist in matplotlib < 1.5
# MAGIC   cmap = getattr(cm, 'plasma_r', cm.hot_r)
# MAGIC   return distributions, X, y,y_full,cmap

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
df = spark.sql(" SELECT * FROM rc_data where EVENT_ID IN (SELECT EVENT_ID FROM tmp_step_event where num_steps >= 320) AND BF_SD_TYPE IN ('UCM','NMD') AND EVENT_ID <>0 AND RUL <>0 and CODE NOT IN ('BEWK-ZZZ-K0110A') ")
#df2 = spark.sql(" SELECT * FROM rc_data")

# COMMAND ----------

df.count()

# COMMAND ----------

df2 = spark.sql(" SELECT YEAR, CODE, BF_SD_TYPE, COUNT(DISTINCT EVENT_ID) as NUM FROM rc_data where EVENT_ID IN (SELECT EVENT_ID FROM tmp_step_event where num_steps >= 320) AND BF_SD_TYPE IN ('UCM','NMD') GROUP BY YEAR, CODE, BF_SD_TYPE")

# COMMAND ----------

display(df2.where("BF_SD_TYPE='UCM'").orderBy(["YEAR","CODE"],ascending=[1,0]))

# COMMAND ----------

display(df2.where("BF_SD_TYPE='NMD'").orderBy(["YEAR","CODE"],ascending=[1,0]))

# COMMAND ----------

df3 = spark.sql(" SELECT YEAR, CODE, BF_SD_TYPE, LABEL2, count(*) num FROM rc_data where EVENT_ID IN (SELECT EVENT_ID FROM tmp_step_event where num_steps > 720) AND BF_SD_TYPE IN ('UCM','NMD') GROUP BY YEAR, CODE, BF_SD_TYPE, LABEL2 ")

# COMMAND ----------

df3 = df3.withColumn("LABEL2",when(df3['LABEL2'].isNotNull(), df3['LABEL2']).otherwise(0))
display(df3.where("BF_SD_TYPE='NMD'").orderBy(["YEAR","CODE"],ascending=[1,0]))

# COMMAND ----------

df4 = spark.sql(" SELECT YEAR, CODE, BF_SD_TYPE, LABEL2, count(*) num FROM rc_data where EVENT_ID IN (SELECT EVENT_ID FROM tmp_step_event where num_steps > 720) AND BF_SD_TYPE IN ('UCM','NMD') AND RUL BETWEEN 1 AND 7200 GROUP BY YEAR, CODE, BF_SD_TYPE, LABEL2 ")

# COMMAND ----------

display(df4.where("BF_SD_TYPE='UCM'").orderBy(["YEAR","CODE"],ascending=[1,0]))

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
 'VIBRA_ENGINE',
'S1_RECY_VALVE',
'S1_SUCT_PRESS',
'S1_SUCT_TEMPE',
'S2_STAGE_DISC_PRESS',
'S2_SCRU_SCRUB_LEVEL',
'GAS_LIFT_HEAD_PRESS',
'IN_CONT_CONT_VALVE',
'IN_SEP_PRESS']

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

df5 = spark.sql("select DAYTIME, EVENT_ID,YEAR, MONTH, RUL,THROW_1_DISC_TEMP, THROW_1_DISC_PRESS,CYL_1_TEMP,CYL_2_TEMP FROM rc_data where EVENT_ID IN (SELECT EVENT_ID FROM tmp_step_event where num_steps > 720) AND BF_SD_TYPE IN ('UCM') and CODE='LAWA-ZZZ-A0110A' and MONTH BETWEEN 1 and 3 AND YEAR=2018 ORDER BY DAYTIME, EVENT_ID,YEAR, MONTH, DAY, CYCLE ")

# COMMAND ----------

#display(df.where("THROW_1_DISC_TEMP<200").orderBy(["DAYTIME","EVENT_ID","CYCLE","RUL"],ascending=[1,0,0,1]))
display(df5)

# COMMAND ----------

d = {}
# Fill in the entries one by one
for col in featurs:
  appQuantile = df.approxQuantile(col,[0.25,0.75],0.00)
  ##df.stat.approxQuantile("value", Array(0.25,0.75),0.0)
  
  d[col] = appQuantile
  print(col + "Q1 : Q3 ",appQuantile," IQR = Q3 - Q1>> ",appQuantile[1]-appQuantile[0])

# COMMAND ----------

print(d)
percentile ={'RESI_LL_LUBE_OIL_PRESS': [-16.0, -12.12846], 'RESI_HH_LUBE_OIL_TEMP': [-25.0, -18.47009], 'RESI_LL_SPEED': [-136.45178, -113.71814], 'RESI_HH_SPEED': [-239.70038, -216.61414], 'RESI_LL_VIBRATION': [0.1520522, 0.2856199], 'RESI_HH_VIBRATION': [-0.3199616, -0.2621263], 'RESI_HH_THROW1_DIS_TEMP': [73.09018, 117.33], 'RESI_HH_THROW1_SUC_TEMP': [-96.287819, -56.00027], 'LUBE_OIL_PRESS': [58.0, 61.010353], 'LUBE_OIL_TEMP': [165.0, 171.52991], 'THROW_1_DISC_PRESS': [270.75699, 437.81003], 'THROW_1_DISC_TEMP': [192.30486, 242.93376], 'THROW_1_SUC_PRESS': [131.16534, 143.33781], 'THROW_1_SUC_TEMP': [93.712181, 114.57061], 'THROW_2_DISC_PRESS': [1159.8107, 1194.1128], 'THROW_2_DISC_TEMP': [244.0, 270.74588], 'THROW_2_SUC_PRESS': [460.38181, 529.30219], 'THROW_2_SUC_TEMP': [101.4219, 131.97696], 'THROW_3_DISC_PRESS': [263.11688, 420.52292], 'THROW_3_DISC_TEMP': [191.95029, 235.84973], 'THROW_3_SUC_PRESS': [120.0, 135.49648], 'THROW_3_SUC_TEMP': [76.695137, 115.82826], 'THROW_4_DISC_PRESS': [489.81097, 1191.0], 'THROW_4_DISC_TEMP': [196.0, 266.85648], 'VIBRATION': [0.10036462, 0.15000001], 'CYL_1_TEMP': [1138.1047, 1207.7437], 'CYL_2_TEMP': [1139.5728, 1228.1698], 'CYL_3_TEMP': [1136.4255, 1229.8782], 'CYL_4_TEMP': [1113.8582, 1196.1135], 'CYL_5_TEMP': [1083.9126, 1218.7815], 'CYL_6_TEMP': [1102.3951, 1202.2028], 'CYL_7_TEMP': [1158.213, 1220.5302], 'CYL_8_TEMP': [1140.5035, 1223.7732], 'CYL_9_TEMP': [1144.4458, 1218.1936], 'CYL_10_TEMP': [1099.0552, 1197.9183], 'CYL_11_TEMP': [1081.7288, 1204.0948], 'CYL_12_TEMP': [1100.3782, 1168.6615], 'LUBE_OIL_PRESS_ENGINE': [49.878487, 65.936882], 'MANI_PRESS': [29.0, 38.071968], 'RIGHT_BANK_EXH_TEMP': [1178.5801, 1289.3641], 'RUNNING_STATUS': [0.0, 0.0], 'SPEED': [940.29962, 963.38586], 'VIBRA_ENGINE': [0.15503845, 0.28999999], 'S1_RECY_VALVE': [3.5043261, 37.605804], 'S1_SUCT_PRESS': [120.51157, 136.01047], 'S1_SUCT_TEMPE': [77.0886, 115.99894], 'S2_STAGE_DISC_PRESS': [998.94324, 1195.424], 'S2_SCRU_SCRUB_LEVEL': [15.382514, 26.0], 'GAS_LIFT_HEAD_PRESS': [1159.5924, 1184.2358], 'IN_CONT_CONT_VALVE': [100.0, 100.0], 'IN_SEP_PRESS': [122.0, 138.82828]}

# COMMAND ----------

# lowerRange = Q1 - 1.5*IQR
# upperRange = Q3+ 1.5*IQR

# outliers = df.filter(s"value < $lowerRange or value > $upperRange")

# COMMAND ----------

def outliers_iqr (colName, percentile, value):
    quartile_1, quartile_3 = percentile[colName][0], percentile[colName][1]
    print("quartile_1:",quartile_1, ",quartile_3:",quartile_3)
    #np.percentile(ys, [25, 75])
    iqr = quartile_3 - quartile_1
    
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    
    if value>=upper_bound:
      value = upper_bound
      print("Update Uper>",upper_bound)
    elif value<=lower_bound:
      value = lower_bound
      print("Update Lower>",lower_bound)
    else:
      print("Value>",value)
    
    return value

boundary ={}

def gen_boundary_iqr(colList, percentile):
  for colName in colList:
    quartile_1, quartile_3 = percentile[colName][0], percentile[colName][1]
    print("quartile_1:",quartile_1, ",quartile_3:",quartile_3)
    #np.percentile(ys, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    boundary[colName] = [upper_bound,lower_bound]
  return boundary

boundary = gen_boundary_iqr(featurs, percentile)

#newvalue = outliers_iqr("THROW_1_DISC_TEMP", percentile, 500)
#print("newvalue:",newvalue)
# empty dictionary d

# COMMAND ----------

print(boundary)

# COMMAND ----------

 for colName in featurs:
    #print("Soomth Extream Noise by Actual Range of Sensor 10% ["+colName, "]")
    #print(" validating < ", normalSensorRange[colName][0])
    #df_new = df.withColumn(colName, when(df[colName]<=normalSensorRange[colName][0], df[colName]).otherwise(normalSensorRange[colName][0]))
    #f_new = df.withColumn(colName, when(df[colName]>normalSensorRange[colName][0], normalSensorRange[colName][0] ).otherwise(df[colName]))
    #print(" validating > ", normalSensorRange[colName][1])
    #df_new = df_new.withColumn(colName, when(df_new[colName]<normalSensorRange[colName][1], normalSensorRange[colName][1]).otherwise(df_new[colName]))
    df_new = df.withColumn(colName, F.log(F.when(df[colName] < boundary[colName][1],boundary[colName][1]).when(df[colName] > boundary[colName][0], boundary[colName][0]).otherwise(df[colName]) +1).alias(colName))
    
#     df_new = df.withColumn(colName, F.log(F.when(df[colName] < normalSensorRange[colName][1],normalSensorRange[colName][1]).when(df[colName] > normalSensorRange[colName][0], normalSensorRange[colName][0]).otherwise(df[colName]) +1).alias(colName))

# COMMAND ----------

display(df_new.select('EVENT_ID','CODE').where("CYL_1_TEMP < 800 and CODE NOT IN ('BEWK-ZZZ-K0110A') ").groupby("EVENT_ID",'CODE').count())

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

# # empty dictionary d
# d = {}
# # Fill in the entries one by one
# for col in featurs:
#   appQuantile = df_new.approxQuantile(col,[0.01,0.99],0.25)
#   d[col] = appQuantile
#   print(col + ":",appQuantile)

# COMMAND ----------

# # looping through the columns, doing log(x+1) transformations
# for col in featurs:
#   df_new = df.withColumn(col, F.log(F.when(df[col] < d[col][0],d[col][0]).when(df[col] > d[col][1], d[col][1]).otherwise(df[col] ) +1).alias(col))

# COMMAND ----------

pairCols = ['THROW_1_DISC_PRESS','THROW_1_DISC_TEMP']

spPD = df_new.select(pairCols).toPandas()

# COMMAND ----------

pdDataDF = spPD

# COMMAND ----------

make_plot(6,pairCols, pdDataDF)

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
"IN_RUL",
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
'RUNNING_STATUS',
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

# COMMAND ----------

# pandasDF = df_new.toPandas()

# COMMAND ----------

# def normalizeQuantileTransformer(test_df,quantile_scaler,columnListDem):
#   ######
#   # TEST
#   ######
#   #Data after robust scaling & Data after quantile transformation (uniform pdf)
#   #test_df['cycle_norm'] = test_df['CYCLE']
#   cols_normalize = test_df.columns.difference(['CODE','DAYTIME','YEAR','MONTH','DAY','HOUR','MM','CYCLE','RUL','IN_RUL','EVENT_ID','LABEL1','LABEL2','BF_SD_TYPE'])
#   print(cols_normalize)
  
#   norm_test_df = pd.DataFrame(quantile_scaler.fit_transform(RobustScaler(quantile_range=(25, 75)).fit_transform(test_df[cols_normalize])), 
#                               columns=cols_normalize, 
#                               index=test_df.index)
#   test_join_df = test_df[test_df.columns.difference(cols_normalize)].join(norm_test_df)
#   test_df = test_join_df.reindex(columns = test_df.columns)
#   #test_df = test_df.reset_index(drop=True)
#   print("Finish normalization test dataset!")
#   return test_df

# COMMAND ----------

# display(pandasDF.describe())

# COMMAND ----------

# #min_max_scaler = preprocessing.MinMaxScaler()
# quantile_scaler = preprocessing.QuantileTransformer(output_distribution='uniform')

# sccaledDatSet = normalizeQuantileTransformer(pandasDF,quantile_scaler,columnListDem)

# COMMAND ----------

print(sparkDF02.columns)

# COMMAND ----------

# MAGIC %%time
# MAGIC 
# MAGIC #----------------------------- PCA feature grouping on warning related features --------------------------#
# MAGIC sparkDF02 = sparkDF01.withColumn("KEY", concat(sparkDF01.CODE,lit("_"),sparkDF01.EVENT_ID,lit("_"),sparkDF01.CYCLE,lit("_"),sparkDF01.BF_SD_TYPE))
# MAGIC # step 1
# MAGIC # Use RFormula to create the feature vector
# MAGIC #formula = RFormula(formula = "~" + "+".join(warning_all))
# MAGIC # output = formula.fit(df_new_2).transform(df_new_2).select("key","features") 

# COMMAND ----------

# Import `DenseVector`
from pyspark.ml.linalg import DenseVector
from pyspark.ml.feature import StandardScaler

# Define the `input_data` 
input_data = sparkDF02.rdd.map(lambda x: (x[0:14], DenseVector(x[15:66]),x[67]))

# Replace `df` with the new DataFrame
sparkRDDNew = spark.createDataFrame(input_data, ["label", "features","key"])

# COMMAND ----------

# display(sparkDDR.take(5))

# COMMAND ----------

scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=False)
# scaler = MaxAbsScaler(inputCol="features", outputCol="scaledFeatures")
# Compute summary statistics and generate MaxAbsScalerModel
scalerModel = scaler.fit(sparkRDDNew)
# rescale each feature to range [-1, 1].
scaledData = scalerModel.transform(sparkRDDNew)
scaledData.select("features", "scaledFeatures").show()
## Compute summary statistics by fitting the StandardScaler
#scalerModel = scaler.fit(sparkRDDNew)
## Normalize each feature to have unit standard deviation.
#scaledData = scalerModel.transform(sparkRDDNew)

# COMMAND ----------

# step 3
pca = PCA(k=20, inputCol="scaledFeatures", outputCol="pcaFeatures")
model = pca.fit(scaledData)
result = model.transform(scaledData).select("key","pcaFeatures")

# to check how much variance explained by each component
print(model.explainedVariance)

# COMMAND ----------

# step 4
# convert pca result, a vector column, to mulitple columns
# The reason why we did this was because later on we need to use those columns to generate more features (rolling compute) 
def extract(row):
    return (row.key, ) + tuple(float(x) for x in row.pcaFeatures.values)

pca_outcome = result.rdd.map(extract).toDF(["key"])

# rename columns of pca_outcome
oldColumns = pca_outcome.schema.names

# COMMAND ----------

print(oldColumns)

# COMMAND ----------

display(scaledFeatureOut)

# COMMAND ----------

# step 4
# convert pca result, a vector column, to mulitple columns
# The reason why we did this was because later on we need to use those columns to generate more features (rolling compute) 
def extract(row):
    return (row.key, ) + tuple(float(x) for x in row.scaledFeatures.values)

scaledFeatureOut = scaledData.rdd.map(extract).toDF(["key"])

# rename columns of pca_outcome
oldColumns = scaledFeatureOut.schema.names
print(oldColumns)

# COMMAND ----------

from pyspark.ml.feature import StandardScaler, PCA, RFormula

# COMMAND ----------

assembler = VectorAssembler().setInputCols(sparkDDR.columns).setOutputCol("features")
transformed = assembler.transform(sparkDDR)
scaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures")
scalerModel =  scaler.fit(transformed.select("features"))
scaledData = scalerModel.transform(transformed)

def extract(row):
  return (row.pmid, )+tuple(row.scaledFeatures.toArray().tolist())

final_data = scaledData.select("pmid","scaledFeatures").rdd.map(extract).toDF(sparkDDR.columns)

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



# COMMAND ----------

# from pyspark.sql import SparkSession
# from pypandas.outlier import KMeansOutlierRemover

# # Create Spark Session
# # #spark = SparkSession.builder.getOrCreate()

# # Load your dataframe here
# # data = load_data()

# COMMAND ----------

# # Instantiate Outlier Remover from the factory
# # Available choices are "kmeans", "bisectingkmeans", and "gaussian"
# km = OutlierRemover.factory("kmeans")

# # Check (default) parameters
# km.k
# # 3

# # Set parameters based on user's knowledge on the dataset
# km.set_param(k=5)
# km.k
# # 5

# # Perform KMeans clustering
# km.fit(df, ["Initial Cost", "Total Est Fee"])

# # Get clustering summary
# s = km.summary()
# s.show()