# Databricks notebook source
# MAGIC %md # Random Sample from Large Datasets

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


# remove warnings
import warnings
warnings.filterwarnings('ignore')


# plotting and plot stying
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# jupyter wdgets
#from ipywidgets import interactive, widgets, RadioButtons, ToggleButtons, Select, FloatSlider, FloatProgress
from IPython.display import set_matplotlib_formats, Image

# COMMAND ----------

plt.style.use('seaborn')
#sns.set_style("whitegrid", {'axes.grid' : False})
#set_matplotlib_formats('pdf', 'png')
plt.rcParams['savefig.dpi'] = 80
plt.rcParams['figure.autolayout'] = False
plt.rcParams['figure.figsize'] = (16, 8)
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['font.size'] = 16
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['lines.markersize'] = 8
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['text.usetex'] = False
#plt.rcParams['font.family'] = "serif"
plt.rcParams['font.serif'] = "cm"
plt.rcParams['text.latex.preamble'] = b"\usepackage{subdepth}, \usepackage{type1cm}"

# COMMAND ----------

#1 - import module
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
# import numpy
# import pandas

# COMMAND ----------

#2 - Create SparkContext
# sc = SparkContext()
# print(sc)

# COMMAND ----------

# MAGIC %sh
# MAGIC mkdir RCPredictiveMA
# MAGIC mkdir RCPredictiveMA/sample

# COMMAND ----------

# dbutils.fs.mv("dbfs:/FileStore/tables/tmp_rc/data07_20190126_1440T_CLEAN.csv", "dbfs:/RC/datasets")

# COMMAND ----------

## File Part in HDP Server to be stored images of charts or graph or temp files
path = "/databricks/driver/RCPredictiveMA/"
sparkAppName = "RCDataAnlaysisSample"
hdfsPathCSV = "/RC/MSATER_DATA_FILES/data08_20190131_ALL_CLEAN.csv"

# COMMAND ----------

#3 - Setup SparkSession(SparkSQL)
spark = (SparkSession
         .builder
         .appName(sparkAppName)
         .getOrCreate())
print(spark)

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
# renamed_df = df.selectExpr("object_id as OBJECT_ID","OBJECT_CODE as CODE",
# "time as DAYTIME","n_year as YEAR","n_month as MONTH",
# "n_day as DAY","n_hour as HOUR",
# "n_mm as MM","CURRENT_CYCLE as CYCLE",
# "SD_TYPE as sd_type","DIGIT_SD_TYPE as DIGIT_SD_TYPE",
# "ROUND(45 - S0_LUBE_OIL_PRESS, 7) AS RESI_LL_LUBE_OIL_PRESS", 
# "ROUND(S2_LUBE_OIL_TEMP - 190,7) AS RESI_HH_LUBE_OIL_TEMP", 
# "ROUND(820 - SB18_SPEED,7) AS RESI_LL_SPEED", 
# "ROUND(SB18_SPEED - 1180,7) AS RESI_HH_SPEED", 
# "ROUND(0.33 - SB19_VIBRA_ENGINE,7) AS RESI_LL_VIBRATION", 
# "ROUND(SB19_VIBRA_ENGINE - 0.475,7) AS RESI_HH_VIBRATION", 
# "ROUND(293.33 - S6_THROW_1_DISC_TEMP,7) AS RESI_HH_THROW1_DIS_TEMP", 
# "ROUND(S8_THROW_1_SUC_TEMP - 190,7) AS RESI_HH_THROW1_SUC_TEMP", 
# "S0_LUBE_OIL_PRESS AS LUBE_OIL_PRESS",
# "S2_LUBE_OIL_TEMP AS LUBE_OIL_TEMP",
# "S5_THROW_1_DISC_PRESS AS THROW_1_DISC_PRESS",
# "S6_THROW_1_DISC_TEMP AS THROW_1_DISC_TEMP",
# "S7_THROW_1_SUC_PRESS AS THROW_1_SUC_PRESS",
# "S8_THROW_1_SUC_TEMP AS THROW_1_SUC_TEMP",
# "S9_THROW_2_DISC_PRESS AS THROW_2_DISC_PRESS",
# "S10_THROW_2_DISC_TEMP AS THROW_2_DISC_TEMP",
# "S11_THROW_2_SUC_PRESS as THROW_2_SUC_PRESS",
# "S12_THROW_2_SUC_TEMP AS THROW_2_SUC_TEMP",
# "S13_THROW_3_DISC_PRESS AS  THROW_3_DISC_PRESS",
# "S14_THROW_3_DISC_TEMP AS THROW_3_DISC_TEMP",
# "S15_THROW_3_SUC_PRESS AS THROW_3_SUC_PRESS",
# "S16_THROW_3_SUC_TEMP AS THROW_3_SUC_TEMP",
# "S17_THROW_4_DISC_PRESS AS THROW_4_DISC_PRESS",
# "S18_THROW_4_DISC_TEMP AS THROW_4_DISC_TEMP",
# "S19_VIBRATION AS VIBRATION",
# "SB1_CYL_1_TEMP AS CYL_1_TEMP",
# "SB2_CYL_2_TEMP AS CYL_2_TEMP",
# "SB3_CYL_3_TEMP AS CYL_3_TEMP",
# "SB4_CYL_4_TEMP AS CYL_4_TEMP",
# "SB5_CYL_5_TEMP AS CYL_5_TEMP",
# "SB6_CYL_6_TEMP AS CYL_6_TEMP",
# "SB7_CYL_7_TEMP AS CYL_7_TEMP",
# "SB8_CYL_8_TEMP AS CYL_8_TEMP",
# "SB9_CYL_9_TEMP AS CYL_9_TEMP",
# "SB10_CYL_10_TEMP AS CYL_10_TEMP",
# "SB11_CYL_11_TEMP AS CYL_11_TEMP",
# "SB12_FUEL_GAS_PRESS AS FUEL_GAS_PRESS",
# "SB14_LUBE_OIL_PRESS_ENGINE AS LUBE_OIL_PRESS_ENGINE",
# "SB18_SPEED AS SPEED",
# "SB19_VIBRA_ENGINE AS VIBRA_ENGINE",
# "RUL as RUL",
# "RUL*-1 as IN_RUL",
# "EVENT_ID as EVENT_ID",
# "LABEL1 as LABEL1",
# "LABEL2 as LABEL2",
# "BE_SD_TYPE as BE_SD_TYPE")

#Creating Index Table

renamed_df = df.selectExpr("object_id as OBJECT_ID",
"OBJECT_CODE as CODE",
"time as DAYTIME","n_year as YEAR",
"n_month as MONTH",
"n_day as DAY","n_hour as HOUR",
"n_mm as MM",
"CURRENT_CYCLE as CYCLE",
"RUL as RUL",
"EVENT_ID as EVENT_ID",
"BE_SD_TYPE as BE_SD_TYPE")

renamed_df.printSchema()

# COMMAND ----------

#12 - compute dataframe using sql command via string
renamed_df.createOrReplaceTempView("rc_data_05")
## Filter out the data during RC S/D
renamed_df = spark.sql("SELECT BE_SD_TYPE,YEAR, EVENT_ID,MONTH, DAY, HOUR, MM, RUL, OBJECT_ID FROM rc_data_05 where EVENT_ID <> 9999999 and EVENT_ID <>0 and RUL<>0 ORDER BY BE_SD_TYPE,YEAR, EVENT_ID,MONTH, DAY, HOUR, MM, OBJECT_ID")

# COMMAND ----------

pdIndexFile =renamed_df.toPandas()

# COMMAND ----------

pdIndexFile.to_csv("INDEX_RC_DATA_1.csv")

# COMMAND ----------

# MAGIC %sh
# MAGIC ls -al

# COMMAND ----------

# MAGIC %fs
# MAGIC mv file:/databricks/driver/INDEX_RC_DATA_1.csv /RC/MSATER_DATA_FILES

# COMMAND ----------

renamed_df2 = spark.sql("select EVENT_ID, COUNT(*) as STEP from rc_data_05 where EVENT_ID <> 9999999 and RUL <> 9999999 and RUL <> 0 GROUP BY  EVENT_ID")
renamed_df2.createOrReplaceTempView("T_EVENT_COUNT")

# COMMAND ----------

display(spark.sql("select * from T_EVENT_COUNT ").toPandas())

# COMMAND ----------

# pdUCMSet1 = renamed_df.where(" YEAR = 2016 and BE_SD_TYPE='UCM' ").toPandas()
# pdUCMSet2 = renamed_df.where(" YEAR = 2017 and BE_SD_TYPE='UCM' ").toPandas()
# pdUCMSet3 = renamed_df.where(" YEAR = 2018 and BE_SD_TYPE='UCM' ").toPandas()

# pdNMDSet1 = renamed_df.where(" YEAR = 2016 and BE_SD_TYPE='NMD' ").toPandas()
# pdNMDSet2 = renamed_df.where(" YEAR = 2017 and BE_SD_TYPE='NMD' ").toPandas()
# pdNMDSet3 = renamed_df.where(" YEAR = 2018 and BE_SD_TYPE='NMD' ").toPandas()

# pdUCMSet1 = pdUCMSet1.sort_values(['YEAR','MONTH','DAY','HOUR','MM','EVENT_ID','OBJECT_ID'])
# pdUCMSet2 = pdUCMSet2.sort_values(['YEAR','MONTH','DAY','HOUR','MM','EVENT_ID','OBJECT_ID'])
# pdUCMSet3 = pdUCMSet3.sort_values(['YEAR','MONTH','DAY','HOUR','MM','EVENT_ID','OBJECT_ID'])

# pdNMDSet1 = pdNMDSet1.sort_values(['YEAR','MONTH','DAY','HOUR','MM','EVENT_ID','OBJECT_ID'])
# pdNMDSet2 = pdNMDSet2.sort_values(['YEAR','MONTH','DAY','HOUR','MM','EVENT_ID','OBJECT_ID'])
# pdNMDSet3 = pdNMDSet3.sort_values(['YEAR','MONTH','DAY','HOUR','MM','EVENT_ID','OBJECT_ID'])

# COMMAND ----------

spDFCountEvent = spark.sql("SELECT EVENT_ID, YEAR, COUNT(*) NUM FROM rc_data_05 where BE_SD_TYPE = 'UCM' and EVENT_ID <> 9999999 and RUL <> 0 GROUP BY EVENT_ID, YEAR order by YEAR, EVENT_ID ")
display(spDFCountEvent.toPandas())

# COMMAND ----------

## sdType = {train/test}
## dsType = {UCM, NMD}
## maxSeqLength = 1440 * 5 (5 days)
## trainRatio = 75%, so testRatio = 100% - trainRatio
# retrun series of event_id in string

def getDSEventIds(sdType, minSeqLength, maxSeqLength, pYear):
  spDF = spark.sql("SELECT EVENT_ID, COUNT(*) NUM FROM rc_data_05 where  BE_SD_TYPE='"+ sdType +"' and RUL BETWEEN 1 AND "+ str(maxSeqLength) +" AND EVENT_ID in (SELECT EVENT_ID FROM T_EVENT_COUNT WHERE STEP >=  "+ str(minSeqLength) +" and STEP <= "+str(maxSeqLength) + ") and YEAR ="+str(pYear) + " GROUP BY EVENT_ID ORDER BY EVENT_ID ")
  return spDF


# COMMAND ----------

# MAGIC %md # Random
# MAGIC Random Subsample From a List
# MAGIC We may be interested in repeating the random selection of items from a list to create a randomly chosen subset.
# MAGIC 
# MAGIC Importantly, once an item is selected from the list and added to the subset, it should not be added again. This is called selection without replacement because once an item from the list is selected for the subset, it is not added back to the original list (i.e. is not made available for re-selection).
# MAGIC 
# MAGIC This behavior is provided in the sample() function that selects a random sample from a list without replacement. The function takes both the list and the size of the subset to select as arguments. Note that items are not actually removed from the original list, only selected into a copy of the list.
# MAGIC 
# MAGIC The example below demonstrates selecting a subset of five items from a list of 20 integers.

# COMMAND ----------

# select a random sample without replacement
from random import seed
from random import sample
  
def randomEventId(dataset, randomStep, targetNumEvent,tryRandom):
  eventList = dataset['EVENT_ID']
  print(eventList)
  if eventList.count()>0 and eventList.count()>=targetNumEvent:
    # seed random number generator
    seed(1)
    # prepare a sequence
    sequence = [i for i in eventList]
    print(sequence)
    # select a subset without replacement
    
    x = 1
    for x in range(tryRandom):
      subset = sample(sequence, targetNumEvent)
      print(subset)
    
    print("Selected Set = ")
    print(subset)
    return subset
  else:
    return 0
  
def round_down_int(n, decimals=0):
    multiplier = 10 ** decimals
    return int(math.floor(n * multiplier) / multiplier)


# COMMAND ----------

pdDF = spDFCountEvent.toPandas()

#pdDF.hist(column="NUM")
import numpy as np
import matplotlib.pyplot as plt
import pandas
df = pandas.DataFrame(pdDF, columns=['NUM', 'YEAR','EVENT_ID'])
pandas.tools.plotting.scatter_matrix(df, alpha=0.2)
#plt.show()
display()

# COMMAND ----------

pd.scatter_matrix(pdDF, diagonal='kde')
display()

# COMMAND ----------

from plotly.offline import plot
from plotly.graph_objs import *
import numpy as np

x = pdDF["YEAR"]
y = pdDF["NUM"]

# Instead of simply calling plot(...), store your plot as a variable and pass it to displayHTML().
# Make sure to specify output_type='div' as a keyword argument.
# (Note that if you call displayHTML() multiple times in the same cell, only the last will take effect.)

p = plot(
  [
    Histogram2dContour(x=x, y=y, contours=Contours(coloring='heatmap')),
    Scatter(x=x, y=y, mode='markers', marker=Marker(color='white', size=3, opacity=0.3))
  ],
  output_type='div'
)

displayHTML(p)

# COMMAND ----------

# from ggplot import *

# ggplot(aes(x='date', y='beef'), data=meat) +\
#     geom_line() +\
#     stat_smooth(colour='blue', span=0.2)

# COMMAND ----------

# %sh
# #ls -al /databricks/python3/bin/pip
# # sudo /databricks/python3/bin/pip install ggplot
# # sudo /databricks/python3/bin/pip install plotly
# %sh
# /databricks/python/bin/pip install plotnine matplotlib==2.2.2

# COMMAND ----------

# %sh
# ls -al /databricks/python/lib/python3.6/site-packages/ggplot/stats/smoothers.py
# cat /databricks/python/lib/python3.6/site-packages/ggplot/stats/smoothers.py

# COMMAND ----------

import pandas as trainDFSet# #dbutils.fs.cp('file:/databricks/python/lib/python3.6/site-packages/ggplot/stats/smoothers.py', "dbfs:/mnt/Exploratory/WCLD/")
# dbutils.fs.cp("dbfs:/mnt/Exploratory/WCLD/smoothers.py", 'file:/databricks/python/lib/python3.6/site-packages/ggplot/stats')

# COMMAND ----------

def getSampleTraingSeries(sdType, minRUL, maxRUL,yearTrainSet, lastYear, wColName,numSample,randomSate):
  for aYear in yearTrainSet:
    spDF = getDSEventIds(sdType, minRUL, maxRUL,aYear)
    spPD = spDF.toPandas()
    nCount = []
    nCount = spPD.count()
    print("Query from Spark >> "+str(nCount[1]) +" events.")
    ##print(nCount[1])
    countRow = nCount[1]
    
    if int(aYear)==lastYear and countRow<numSample:
      numSample=countRow
    elif countRow<numSample:
      numSample=countRow
    else:
      numSample=round_down_int(numSample*yearSampleRatio)
      
    spPD = spPD.sample(n=numSample, weights=wColName, random_state=randomSate)
    nCount = spPD.count()
    countRow = nCount[1]
    
    print("Selected samples of year "+str(aYear)+" = "+str(countRow)+" events.")
    if len(yearTrainSet)>1:
      spPD = spPD.append(spPD)
      
  return spPD


# COMMAND ----------

####
minRUL = 720
maxRUL = 36000
yearTrainSet = [2016,2017]
yearTestSet = [2018]
lastYear = 2018
lastYearSelectRatio = 1
yearSampleRatio = 1
sdType = "UCM"
numSample = 100
randomSate = 1
wColName = "NUM"

print("Loading Train Dataset...")
trainDataSet = getSampleTraingSeries(sdType, minRUL, maxRUL,yearTrainSet, lastYear, wColName,numSample,randomSate)
print("Total Train Samples ="+str(trainDataSet["EVENT_ID"].count()))
print("Loading Test Dataset....")
testDataSet = getSampleTraingSeries(sdType, minRUL, maxRUL,yearTestSet, lastYear, wColName,numSample,randomSate)

if numSample>testDataSet["EVENT_ID"].count():
  numSample = numSample-int(testDataSet["EVENT_ID"].count())
  testDataSetNMD = getSampleTraingSeries("NMD", minRUL, maxRUL, yearTestSet, lastYear, wColName,numSample,randomSate)
  testDataSet = testDataSet.append(testDataSetNMD)
  
print("Total Test Samples =",str(testDataSet["EVENT_ID"].count()))

# COMMAND ----------

display(testDataSet)

# COMMAND ----------

trainSet1IDs = randomEventId(trainSet1,1,20,5)
spDF = getDSEventIds("UCM", 720, 10000,2018)
trainSetIDs= spPD.append(spDF.toPandas())
trainSet2IDs = randomEventId(trainSet1,1,10,3)

# COMMAND ----------

display(spPD)

# COMMAND ----------



# COMMAND ----------

import seaborn as sns
sns.set(style="white")
pdAll = renamed_df.toPandas()
#df = sns.load_dataset("iris")
g = sns.PairGrid(pdAll[['NUM','OBJECT_ID','YEAR']], diag_sharey=False)
g.map_lower(sns.kdeplot)
g.map_diag(sns.kdeplot, lw=3)

g.map_upper(sns.regplot)

display(g.fig)

# COMMAND ----------

# MAGIC %sh
# MAGIC /databricks/python/bin/pip install plotnine matplotlib==2.2.2

# COMMAND ----------

from plotnine import *
from plotnine.data import meat
from mizani.breaks import date_breaks
from mizani.formatters import date_format

spkDF = spark.sql("SELECT DAYTIME,EVENT_ID, LUBE_OIL_PRESS FROM rc_data_05 WHERE RUL BETWEEN 1 AND 1440 and MM in (10,20,30,40,50) " )
spkDFPD = spkDF.toPandas()

#series = pdAll[['DAYTIME','LUBE_OIL_PRESS']]

pn = ggplot(spkDFPD, aes('DAYTIME','LUBE_OIL_PRESS')) + \
    geom_line(color='blue') + \
    scale_x_date(breaks=date_breaks('1 years'), labels=date_format('%b %Y')) + \
    scale_y_continuous() + theme_bw() + theme(figure_size=(12, 8))

# COMMAND ----------

display(pn.draw())

# COMMAND ----------

### This step is for cleaning data using simple median value of each colum to the missing or unknown value of sensors.
from pyspark.sql.functions import when

def replaceByMedian(pySparkDF, columnList):
    for colName in columnList:
        med = pySparkDF.approxQuantile(colName, [0.5], 0.25)
        pySparkDF = pySparkDF.withColumn(colName, when(pySparkDF[colName].isNotNull(),pySparkDF[colName]).otherwise(med[0]))
        print("Relaced [",colName,"] by ", med[0])
    return pySparkDF

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
 'FUEL_GAS_PRESS',
 'LUBE_OIL_PRESS_ENGINE',
 'SPEED',
 'VIBRA_ENGINE']

renamed_df = replaceByMedian(renamed_df,columnList)

# COMMAND ----------

# MAGIC %sh
# MAGIC pip install plotly

# COMMAND ----------

from plotly.offline import plot
from plotly.graph_objs import *
import numpy as np

x = np.random.randn(2000)
y = np.random.randn(2000)

# Instead of simply calling plot(...), store your plot as a variable and pass it to displayHTML().
# Make sure to specify output_type='div' as a keyword argument.
# (Note that if you call displayHTML() multiple times in the same cell, only the last will take effect.)

p = plot(
  [
    Histogram2dContour(x=x, y=y, contours=Contours(coloring='heatmap')),
    Scatter(x=x, y=y, mode='markers', marker=Marker(color='white', size=3, opacity=0.3))
  ],
  output_type='div'
)

displayHTML(p)